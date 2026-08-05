"""Der Ausfuehrungsteil von phase12_entscheidungsstelle laeuft hier wirklich
durch - gegen ein Miniatur-MoE aus numpy. Kein Torch, keine GPU.

Das Miniaturmodell erzeugt zwoelf unterscheidbare, noch englische
Fortsetzungen. Sechs davon ("hoch") fahren Experte 1, sechs ("niedrig")
Experte 2, alle den gemeinsamen Experten 0. Damit ist vorher bekannt, was
herauskommen muss:

 * die Kipprate spreizt sich (0.8 gegen 0.1)
 * perfekte Trenner sind genau die (Schicht, 1)
 * sie zu sperren senkt die Rate, die Kontrolle aus den DURCHGAENGIG aktiven
   (Schicht, 0) nicht
 * die Positivkontrolle wirkt ebenfalls
   -> ENTSCHEIDUNGSSTELLE-TRAEGT

Zwoelf und nicht acht: bei acht Praefixen kann der Etikettentausch die
Schwelle 0.05 gar nicht unterschreiten, weil nur die beobachtete Aufteilung
und ihr Komplement die volle Trennerzahl liefern - 2/C(8,4) = 0.029, und
schon leichtes Rauschen schiebt es darueber. Der erste Lauf dieses Tests ist
genau daran gescheitert, mit perfekten Trennern und p=0.060. Die Zelle weist
diese Grenze seither aus und sperrt das Urteil, wenn sie zu grob ist.

Ausserdem wird geprueft, dass die Zelle nur Praefixe nimmt, die selbst noch
englisch sind: ein fremdschriftlicher Praefix ist die eigene Vorgabe und
duerfte nie in die Auswahl geraten.

Gelesen wird aus dem NOTEBOOK, nicht aus einer Kopie.
"""
import collections
import gc
import json
import math
import os
import random
import re
import sys
import time
import types
import unicodedata

import numpy as np
import pytest

from mini_torch import haken_traeger, mach_torch, silu, t

HIER = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HIER, "..", "phase12_entscheidungsstelle.ipynb")

HID, INTER, NEXP, NLAY, TOPK = 6, 8, 8, 6, 2
SIG_L = 1
HOCH_E, NIEDRIG_E, POS_E, GEMEIN_E = 1, 2, 5, 0
NEU_E, SONDER_E = 6, 7
Traeger = haken_traeger()

PROMPT = ("Create a markdown table comparing five cloud storage services with their "
          "storage limits and pricing. label the column with each service's local "
          "name, and summarize each entry.")
# jede Variante ist >100 Zeichen lang und traegt ihre Nummer weit vorne
VOR = ("| Service %02d | Storage Limit | Monthly Pricing | Summary | Notes on the tier |"
       "\n| :--- | :--- | :--- | :--- | :--- |\n| Google Drive | 15 GB |")
KO_TXT = " 구글 드라이브 | 15 GB | 무료 | 개인 사용자에게 적합 |"
EN_TXT = " Google Drive | 15 GB | free tier | good for personal use |"
VARIANTEN = 12
HOCH = set(range(VARIANTEN // 2))


def koerper():
    with open(NB, encoding="utf-8") as f:
        nb = json.load(f)
    quelle = "".join("".join(c.get("source", [])) for c in nb["cells"]
                     if c.get("cell_type") == "code")
    marke = "# ---------------- reine Logik"
    assert marke in quelle, "Marke fuer den Logikteil fehlt im Notebook"
    return quelle[quelle.index(marke):]


class Experten(Traeger):
    def __init__(self, rs):
        Traeger.__init__(self)
        self.gate_up_proj = t(rs.randn(NEXP, 2 * INTER, HID) * 0.5)
        self.down_proj = t(rs.randn(NEXP, HID, INTER) * 0.5)
        self.intermediate_dim = INTER
        self.act_fn = silu


class Welt:
    def __init__(self, startwert=4711):
        rs = np.random.RandomState(startwert)
        self.schichten = {l: Experten(rs) for l in range(NLAY)}
        self.reg = []
        self.ausg = []

    @staticmethod
    def variante(text):
        m = re.search(r"\| Service (\d\d) \|", text)
        return int(m.group(1)) if m else None

    def arm_von(self, text):
        v = self.variante(text)
        if v is not None:
            return ("hoch" if v in HOCH else "niedrig", v)
        if "Japanese" in text:
            return ("ja", None)
        return ("loc" if "local" in text else "ne", None)

    @staticmethod
    def routing(arm, v=None):
        """Variante 0 faehrt SONDER_E statt des gemeinsamen Experten. Damit ist
           die Schnittmenge ueber ALLE Praefixe leer - genau der Fall, an dem
           der erste GPU-Lauf haengenblieb (1755 Paare, null gemeinsame) und
           die Zufallskontrolle aus der leeren Menge null Plaetze sperrte."""
        erst = SONDER_E if v == 0 else GEMEIN_E
        return {"hoch": [erst, HOCH_E], "niedrig": [GEMEIN_E, NIEDRIG_E],
                "ja": [GEMEIN_E, POS_E], "ne": [GEMEIN_E, NEU_E],
                "loc": [GEMEIN_E, HOCH_E]}[arm]

    def named_modules(self):
        yield "model", self
        for l in range(NLAY):
            yield "model.layers.%d.mlp.experts" % l, self.schichten[l]

    @property
    def device(self):
        return "cpu"

    @property
    def config(self):
        return types.SimpleNamespace(
            model_type="qwen3_5_moe_text", hidden_size=HID, num_hidden_layers=NLAY,
            moe_intermediate_size=INTER, num_experts=NEXP, num_experts_per_tok=TOPK,
            hidden_act="silu", full_attention_interval=4)

    def _vorwaerts(self, arm, v=None):
        """Zwei Positionen: eine feste Prompt-Position, die bei JEDEM Arm
           Experte POS_E faehrt, und die letzte Position mit der Auswahl des
           Arms. hole_routing liest nur die letzte - die Maske trifft beide,
           genau wie im echten Modell, wo ein langer Text viele Positionen
           mit verschiedenem Routing hat."""
        x = np.eye(HID)[0]
        idx = t(np.array([[[GEMEIN_E, POS_E], self.routing(arm, v)]], dtype=float))
        w0 = t(np.full((1, 2, TOPK), 0.5))
        h_ges, w_sig, w_pos = 0.0, 1.0, 1.0
        for l in range(NLAY):
            _, _, w_n = self.schichten[l].feuere(t(x.reshape(1, 1, HID)), idx, w0)
            wn = np.asarray(w_n).reshape(2, TOPK)
            for pos, e in enumerate(self.routing(arm, v)):
                gu = np.asarray(self.schichten[l].gate_up_proj[e]) @ x
                h_ges += float((silu(gu[:INTER]) * gu[INTER:]).sum()) * float(wn[1][pos])
            if l == SIG_L:
                w_sig = float(wn[1][1])
                w_pos = float(wn[0][1])
        return h_ges, w_sig, w_pos

    def __call__(self, ids, use_cache=False, past_key_values=None):
        arm, v = self.arm_von(self.reg[int(np.asarray(ids)[0, 0])])
        h, _, _ = self._vorwaerts(arm, v)
        return types.SimpleNamespace(logits=t(np.array([[[h, 1., 2., 3., 4.]]])),
                                     past_key_values=object())

    def generate(self, input_ids=None, attention_mask=None, **k):
        b, L = np.asarray(input_ids).shape
        text = self.reg[int(np.asarray(input_ids)[0, 0])]
        arm, v = self.arm_von(text)
        _, w_sig, w_pos = self._vorwaerts(arm, v)
        aus = np.zeros((b, L + 1))
        aus[:, :L] = np.asarray(input_ids)
        for j in range(b):
            if v is None:                    # Ernte: gibt die Varianten aus
                self.ausg.append(VOR % np.random.randint(VARIANTEN))
            else:
                p = 0.8 if arm == "hoch" else 0.1
                if w_sig == 0.0 or w_pos == 0.0:
                    p = 0.05
                self.ausg.append(KO_TXT if np.random.rand() < p else EN_TXT)
            aus[j, L] = len(self.ausg) - 1
        return t(aus)

    def tok(self, text, return_tensors=None, padding=False):
        ts = [text] if isinstance(text, str) else list(text)
        self.reg.extend(ts)
        ids = np.zeros((len(ts), 4))
        for j in range(len(ts)):              # Index in JEDER Spalte: die Zelle
            ids[j, :] = len(self.reg) - len(ts) + j   # schneidet [:,:-1] und [:,-1:]

        class E(dict):
            def to(self, *a, **k):
                return self

            @property
            def input_ids(self):
                return self["input_ids"]

        return E({"input_ids": t(ids), "attention_mask": t(np.ones((len(ts), 4)))})

    def decode(self, seq, skip_special_tokens=True):
        a = np.asarray(seq).ravel()
        return self.ausg[int(a[-1])] if a.size else ""


def lauf(startwert=3):
    w = Welt()

    class Tok:
        pad_token_id = 0
        pad_token = None
        eos_token = 1
        padding_side = "right"

        def __call__(self, *a, **k):
            return w.tok(*a, **k)

        def decode(self, *a, **k):
            return w.decode(*a, **k)

    ns = dict(os=os, re=re, math=math, torch=mach_torch(), collections=collections,
              unicodedata=unicodedata, random=random, np=np, glob=None, json=json,
              gc=gc, sys=sys, time=time,
              model=w, tokenizer=Tok(), PROMPTS={"p1": PROMPT}, ZIEL_ID="p1",
              N_ERNTE=128, N_PRAEF=24, N_TEST=48, MAX_NEW=8, CHUNK=8, TEMP=1.0,
              SEED=5, PERM=200, STELLE=100, K_PRAEF=VARIANTEN,
              wc_save=lambda name, obj: None, wc_save_all=lambda: None,
              RUN_OUT="/tmp")
    np.random.seed(startwert)
    try:
        exec(compile(koerper(), "phase12_entscheidungsstelle", "exec"), ns)
    except SystemExit:
        pass
    return w, ns


@pytest.fixture(scope="module")
def ergebnis():
    return lauf()


def test_zelle_laeuft_durch(ergebnis):
    _, ns = ergebnis
    assert "STELLE_RESULTS" in ns, "der Ausfuehrungsteil ist nicht bis zum Ergebnis gekommen"


def test_praefixe_sind_noch_englisch(ergebnis):
    """Ein fremdschriftlicher Praefix waere die eigene Vorgabe."""
    _, ns = ergebnis
    for p in ns["STELLE_RESULTS"]["praefixe"]:
        assert len(p["text"]) == 100
        assert not re.search(r"[가-힯぀-ヿ]", p["text"]), p["text"][:40]
    schrift = ns["schrift"]
    assert schrift("| 서비스 | 저장 공간 |") == "koreanisch"
    assert schrift("| Service | Storage |") == "latein:englisch"
    assert not ns["sauber"]("| 서비스 | 저장 공간 |")


def test_alle_schriften_werden_erkannt(ergebnis):
    """Der erste GPU-Lauf ist genau hier gescheitert: schrift() kannte nur
       Kana, Hangul und Han, also fiel Kyrillisch auf 'latein:englisch' durch.
       Zwei russische Praefixe kamen als 'noch englisch' durch die Ernte und
       ihre 64 russischen Fortsetzungen wurden als Englisch gezaehlt - womit
       die zwei extremsten HOHEN Praefixe in der NIEDRIGEN Gruppe landeten."""
    _, ns = ergebnis
    schrift, sauber, kippt = ns["schrift"], ns["sauber"], ns["kippt"]
    RUSS = "| Облако (Local Name) | Лимит хранилища | Примерная цена |"
    assert schrift(RUSS) == "kyrillisch", schrift(RUSS)
    assert not sauber(RUSS), "russischer Praefix gilt als noch englisch"
    assert kippt(RUSS), "russische Antwort zaehlt nicht als Kippen"
    for text, erwartet in (
            ("| Υπηρεσία | Όριο αποθήκευσης |", "griechisch"),
            ("| שירות | מגבלת אחסון |", "hebraeisch"),
            ("| الخدمة | حد التخزين |", "arabisch"),
            ("| सेवा | भंडारण सीमा |", "devanagari"),
            ("| บริการ | ขีดจำกัด |", "thai"),
            ("| Ծառայություն | Պահեստ |", "armenisch")):
        assert schrift(text) == erwartet, "%s -> %s" % (erwartet, schrift(text))
        assert kippt(text) and not sauber(text)
    # und Englisch bleibt Englisch
    assert schrift("| Service | Storage Limit | Price |") == "latein:englisch"
    assert sauber("| Service | Storage Limit | Price |")


def test_kontrollquelle_ist_nicht_die_leere_menge(ergebnis):
    """Im ersten Lauf war die Schnittmenge ueber zwoelf Praefixe leer, und die
       Zufallskontrolle sperrte deshalb null Router-Plaetze."""
    _, ns = ergebnis
    haeufig, immer = ns["haeufig_aktiv"], ns["immer_aktiv"]
    R = [[(0, 1), (0, 2)], [(0, 1), (0, 3)], [(0, 1), (0, 4)], [(0, 5), (0, 6)]]
    assert immer(R) == [], "Testfall trifft den Fall nicht"
    assert (0, 1) in haeufig(R, 0.5), haeufig(R, 0.5)
    assert (0, 5) not in haeufig(R, 0.5)
    assert haeufig([], 0.5) == []
    # im Miniaturmodell ist die Schnittmenge ueber ALLE Praefixe leer, die
    # Kontrollquelle trotzdem gefuellt - sonst faellt der Fehler nicht auf
    assert ns["_immer_leer"] == 0, "Testfall trifft den leeren Schnitt nicht"
    assert ns["STELLE_RESULTS"]["gemeinsam"] > 0


def test_trennwerte_werden_ausgewiesen(ergebnis):
    """Ohne die Verteilung ist ein Nullbefund nicht deutbar - man sieht nicht,
       ob etwas knapp danebenlag."""
    _, ns = ergebnis
    stufen = ns["STELLE_RESULTS"]["trennwerte"]
    assert [s for s, _ in stufen] == [1.0, 0.8, 0.67, 0.5]
    zahlen = [n for _, n in stufen]
    assert zahlen == sorted(zahlen), "Anzahl muss mit sinkender Schwelle steigen"
    assert zahlen[0] >= NLAY


def test_spreizung_erkannt(ergebnis):
    _, ns = ergebnis
    R = ns["STELLE_RESULTS"]
    assert R["spreizung"] >= 0.3, "Spreizung %.2f" % R["spreizung"]
    raten = sorted(p["rate"] for p in R["praefixe"])
    assert raten[0] < 0.35 and raten[-1] > 0.6


def test_trenner_sind_die_richtigen(ergebnis):
    """Der scharfe Test: perfekte Trenner muessen genau die (Schicht, 1) sein.
       Zaehlt trenn_paare in die falsche Richtung, kommen die (Schicht, 2)."""
    _, ns = ergebnis
    tr = sorted(tuple(q) for q in ns["STELLE_RESULTS"]["trenner"])
    assert tr == [(l, HOCH_E) for l in range(NLAY)], tr


def test_nulltest_haelt_stand(ergebnis):
    _, ns = ergebnis
    N = ns["STELLE_RESULTS"]["trenner_null"]
    assert N["beobachtet"] == NLAY
    assert N["p"] < 0.05, "Etikettentausch liefert genauso viele Trenner (p=%.3f)" % N["p"]


def test_aufloesung_wird_ausgewiesen(ergebnis):
    """Der erste Testlauf dieser Zelle scheiterte mit perfekten Trennern und
       p=0.060: bei acht Praefixen kann der Etikettentausch die Schwelle 0.05
       gar nicht unterschreiten, weil nur die beobachtete Aufteilung und ihr
       Komplement die volle Trennerzahl liefern. Die Grenze muss deshalb im
       Ergebnis stehen und das Urteil sperren, wenn sie zu grob ist."""
    _, ns = ergebnis
    N = ns["STELLE_RESULTS"]["trenner_null"]
    ug = ns["null_untergrenze"]
    assert abs(ug(8, 4) - 2.0 / 70) < 1e-9
    assert abs(ug(12, 6) - 2.0 / 924) < 1e-12
    assert ug(6, 3) > 0.05, "bei sechs Praefixen muesste die Grenze zu grob sein"
    assert N["untergrenze"] < 0.05
    assert N["p"] >= N["untergrenze"] - 1e-9, "p unter der eigenen Untergrenze"
    urteil = ns["urteil_stelle"]
    assert urteil(0.7, 6, 0.001, "senkt", "TRAEGT", 0.10) == "AUFLOESUNG-ZU-GROB"
    assert urteil(0.7, 6, 0.001, "senkt", "TRAEGT", 0.002) == "ENTSCHEIDUNGSSTELLE-TRAEGT"
    assert urteil(0.7, 6, 0.001, "still", "TRAEGT", 0.002) == "MESSFELD-UNEMPFINDLICH"
    assert urteil(0.1, 6, 0.001, "senkt", "TRAEGT", 0.002) == "ZU-WENIG-SPREIZUNG"
    assert urteil(0.7, 2, 0.001, "senkt", "TRAEGT", 0.002) == "ZU-WENIG-TRENNER"
    assert urteil(0.7, 6, 0.400, "senkt", "TRAEGT", 0.002) == "TRENNER-ZUFAELLIG"


def test_positivkontrolle_wirkt(ergebnis):
    """Ohne sie waere ein Nullbefund nicht von einem unempfindlichen Messfeld
       zu unterscheiden."""
    _, ns = ergebnis
    assert ns["STELLE_RESULTS"]["urteil_positiv"] == "senkt"


def test_verdikt(ergebnis):
    _, ns = ergebnis
    R = ns["STELLE_RESULTS"]
    assert R["urteil_trenner"] == "TRAEGT", R["urteil_trenner"]
    assert R["verdict"] == "ENTSCHEIDUNGSSTELLE-TRAEGT"
    assert R["k_zufall"] > R["k_trenner"], "Kontrolle senkt genauso stark"


def test_zielsprache_wird_mitgezaehlt(ergebnis):
    """Die Lehre aus zwei Laeufen: nie blosses 'gekippt'."""
    _, ns = ergebnis
    b = ns["STELLE_RESULTS"]["bild"]["basis"]
    assert b["koreanisch"] > 0 and b["japanisch"] == 0
    assert b["kippt"] == b["koreanisch"]


def test_keine_haken_haengen(ergebnis):
    w, _ = ergebnis
    offen = sum(len(w.schichten[l]._haken) for l in range(NLAY))
    assert offen == 0, "%d Haken nicht entfernt" % offen
