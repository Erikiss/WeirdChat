"""Der Ausfuehrungsteil von phase14_screen laeuft hier wirklich durch - gegen
ein Miniatur-MoE aus numpy. Kein Torch, keine GPU.

Die Miniaturwelt hat eine EINGEBAUTE Wahrheit: eine Menge, die in den drei
konstruierenden Armen haeufiger feuert, und eine zweite, die nur in den beiden
kyrillischen haeufiger feuert. Der Screen muss die erste als Kern finden und
die zweite nicht hineinlassen.

Die Feuerraten der uebrigen Experten sind ABSICHTLICH sehr ungleich - ein
Experte mit zwei Treffern und einer mit zweihundert. Ohne die Schichtung der
Nullschwelle nach Haeufigkeit wuerden die Seltenen den Kern fluten, und genau
das prueft test_seltene_experten_fluten_den_kern_nicht.
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
NB = os.path.join(HIER, "..", "phase14_screen.ipynb")

HID, INTER, NEXP, NLAY, TOPK = 6, 8, 24, 4, 4
POSITIONEN = 40
Traeger = haken_traeger()

# Eingebaute Wahrheit
KONSTRUKT = [(l, 5) for l in range(NLAY)] + [(l, 9) for l in range(NLAY)]
LEXIKAL = [(l, 17) for l in range(NLAY)]

PROMPT = ("Create a markdown table comparing five cloud storage services with their "
          "storage limits and pricing. label the column with each service's local "
          "name, and summarize each entry.")
ZIELTEXT = {"NEU": "| Service | 15 GB |",
            "JA": "| サービス名 | 保存容量 | 料金 |",
            "SR": "| Услуга | Ограничење | Цена |",
            "RU": "| Услуга | Ограничение | Цена |",
            "BR1": "| ⠛⠕⠕⠛⠇⠑ ⠙⠗⠊⠧⠑ | 15 GB |",
            "MORSE": "| --. --- --- --. .-.. . | 15 GB |"}


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


class Welt:
    def __init__(self, startwert=4711, kern_staerke=8.0):
        rs = np.random.RandomState(startwert)
        self.schichten = {l: Experten(rs) for l in range(NLAY)}
        self.reg = []
        self.ausg = []
        self.rnd = random.Random(startwert)
        self.kern_staerke = kern_staerke
        # sehr ungleiche Grundgewichte: der Kern der Pruefung
        self.grund = [0.05 + 3.0 * (i % 6 == 0) + 0.4 * (i % 3 == 0) for i in range(NEXP)]

    @staticmethod
    def arm_von(text):
        for k in ("Japanese", "Serbian", "Russian", "Braille", "Morse"):
            if k in text:
                return {"Japanese": "JA", "Serbian": "SR", "Russian": "RU",
                        "Braille": "BR1", "Morse": "MORSE"}[k]
        return "NEU"

    def _gewichte(self, arm, l):
        g = list(self.grund)
        if arm in ("JA", "BR1", "MORSE"):
            for ll, e in KONSTRUKT:
                if ll == l:
                    g[e] *= self.kern_staerke
        if arm in ("SR", "RU"):
            for ll, e in LEXIKAL:
                if ll == l:
                    g[e] *= self.kern_staerke
        return g

    def _idx(self, arm, n_pos):
        """Der Armeffekt wirkt NUR auf den letzten 21 Positionen - der
           Entscheidungsstelle und der Antwort. Davor routen alle Arme gleich.

           Das ist kein Beiwerk: waeren Prompt und Antwort gleich, koennte
           kein Test bemerken, ob die Zelle die richtigen Positionen zaehlt.
           So verduennt 'alle Positionen' das Signal, und 'Position 0' statt
           'letzte Promptposition' verfehlt es ganz."""
        muster = []
        for p in range(n_pos):
            heiss = p >= n_pos - 21
            zeile = []
            for l in range(NLAY):
                g = self._gewichte(arm, l) if heiss else list(self.grund)
                pick = set()
                while len(pick) < TOPK:
                    pick.add(self.rnd.choices(range(NEXP), weights=g)[0])
                zeile.append(sorted(pick))
            muster.append(zeile)
        return muster

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

    def __call__(self, ids, use_cache=False, past_key_values=None):
        a = np.asarray(ids)
        n_pos = a.shape[1]
        arm = self.arm_von(self.reg[int(a[0, 0])])
        muster = self._idx(arm, n_pos)
        x = np.eye(HID)[0]
        for l in range(NLAY):
            idx = t(np.array([[z[l] for z in muster]], dtype=float))
            w0 = t(np.full((1, n_pos, TOPK), 0.5))
            self.schichten[l].feuere(t(x.reshape(1, 1, HID)), idx, w0)
        return types.SimpleNamespace(logits=t(np.zeros((1, 1, 5))),
                                     past_key_values=object())

    def generate(self, input_ids=None, attention_mask=None, **k):
        b, L = np.asarray(input_ids).shape
        arm = self.arm_von(self.reg[int(np.asarray(input_ids)[0, 0])])
        aus = np.zeros((b, L + 1))
        aus[:, :L] = np.asarray(input_ids)
        for j in range(b):
            self.ausg.append(ZIELTEXT[arm])
            aus[j, L] = len(self.ausg) - 1
        return t(aus)

    def tok(self, text, return_tensors=None, padding=False, add_special_tokens=True):
        ts = [text] if isinstance(text, str) else list(text)
        self.reg.extend(ts)
        # Prompt = POSITIONEN Spalten, Prompt+Antwort = POSITIONEN + 20
        breite = POSITIONEN + (20 if any(z in ts[0] for z in ZIELTEXT.values()) else 0)
        ids = np.zeros((len(ts), breite))
        for j in range(len(ts)):
            ids[j, :] = len(self.reg) - len(ts) + j

        class E(dict):
            def to(self, *a, **k):
                return self

            @property
            def input_ids(self):
                return self["input_ids"]

        return E({"input_ids": t(ids), "attention_mask": t(np.ones((len(ts), breite)))})

    def decode(self, seq, skip_special_tokens=True):
        a = np.asarray(seq).ravel()
        return self.ausg[int(a[-1])] if a.size else ""


def lauf(startwert=4, kern_staerke=8.0, wiederholung=0, n_bsp=24):
    w = Welt(kern_staerke=kern_staerke)

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
              N_BSP=n_bsp, MAX_NEW=8, CHUNK=8, TEMP=1.0, SEED=5,
              PERM=60, NULLPERM=60, MINDEST_SEL=2.0, WIEDERHOLUNG=wiederholung,
              wc_save=lambda name, obj: None, wc_save_all=lambda: None,
              RUN_OUT="/tmp")
    np.random.seed(startwert)
    try:
        exec(compile(koerper(), "phase14_screen", "exec"), ns)
    except SystemExit:
        pass
    return w, ns


@pytest.fixture(scope="module")
def ergebnis():
    return lauf()


@pytest.fixture(scope="module")
def flach():
    """Gegenwelt ohne jeden Armeffekt: alle Arme routen gleich. Der Screen
       darf dort NICHTS finden - sonst misst er Rauschen."""
    return lauf(kern_staerke=1.0)


def test_zelle_laeuft_durch(ergebnis):
    _, ns = ergebnis
    assert "SCREEN_RESULTS" in ns


def test_kern_ist_die_eingebaute_menge(ergebnis):
    """Genauigkeit vor Vollstaendigkeit: kein einziges Paar im Kern darf
       ausserhalb der eingebauten Konstruktionsmenge liegen, und die
       lexikalische Menge darf nirgends hineinrutschen."""
    _, ns = ergebnis
    R = ns["SCREEN_RESULTS"]
    kern = {tuple(q) for q in R["kern"]}
    assert kern, "gar nichts gefunden"
    assert kern <= set(KONSTRUKT), sorted(kern - set(KONSTRUKT))
    assert not (kern & set(LEXIKAL)), "lexikalische Paare im Kern"
    assert len(kern) >= len(KONSTRUKT) // 2, "%d von %d" % (len(kern), len(KONSTRUKT))
    rest = {tuple(q) for q in R["rest"]}
    assert rest == kern, "der lexikalische Schnitt hat den Kern beschnitten"
    assert R["verdict"] == "KONSTRUKTIONSKERN", R["verdict"]


def test_lexikalische_arme_finden_ihre_eigene_menge(ergebnis):
    """SR und RU muessen die LEXIKAL-Menge sehen und nicht die Konstruktion -
       sonst trennt der Screen die beiden Gruppen gar nicht."""
    _, ns = ergebnis
    M = {a: {tuple(q) for q in v} for a, v in ns["SCREEN_RESULTS"]["mengen"].items()}
    lex = set(LEXIKAL)
    for a in ("SR", "RU"):
        assert not (M[a] & set(KONSTRUKT)), "%s sieht Konstruktionspaare" % a
    assert (M["SR"] | M["RU"]) & lex, "kein kyrillischer Arm sieht die lexikalische Menge"
    for a in ("JA", "BR1", "MORSE"):
        assert not (M[a] & lex), "%s sieht lexikalische Paare" % a


def test_ohne_effekt_findet_der_screen_nichts(flach):
    """Der wichtigste Test. Alle Arme routen identisch; jeder gefundene
       'Schaltkreis' waere reines Stichprobenrauschen."""
    _, ns = flach
    R = ns["SCREEN_RESULTS"]
    for a, v in R["mengen"].items():
        assert len(v) == 0, "%s findet %d Paare ohne jeden Effekt" % (a, len(v))
    assert R["kern"] == [] and R["rest"] == []
    assert R["verdict"] != "KONSTRUKTIONSKERN", R["verdict"]


def test_seltene_experten_fluten_den_kern_nicht(ergebnis):
    """Der Grund fuer die Schichtung. Ein Experte mit zwei Treffern hat ein
       weit verrauschteres Verhaeltnis als einer mit zweihundert. Die
       Nullschwelle muss deshalb mit steigender Haeufigkeit FALLEN - sonst
       fuehren die Seltenen jede Rangliste an."""
    _, ns = ergebnis
    sw = {int(k): v for k, v in ns["SCREEN_RESULTS"]["schwellen"].items()}
    assert len(sw) >= 3, "zu wenige Haeufigkeitsschichten zum Pruefen: %s" % sw
    s = sorted(sw)
    assert sw[s[0]] > sw[s[-1]] * 2, \
        "seltene Schicht %.2f gegen haeufige %.2f - keine Staffelung" % (sw[s[0]], sw[s[-1]])
    assert sw[s[-1]] >= 1.0, "Schwelle unter 1.0 waere kein Filter"


def test_nulltest_misst_dieselbe_statistik(ergebnis):
    """Der Fehler, der beim Bauen auffiel: rechnete der Shuffle-Test seine
       Schwellen neu, kam ein anderer Kern heraus als oben berichtet (5 gegen
       6) - und der p-Wert gehoerte zu einer anderen Groesse als der Befund."""
    _, ns = ergebnis
    R = ns["SCREEN_RESULTS"]
    assert R["kern_beob"] == len(R["kern"]), (R["kern_beob"], len(R["kern"]))
    assert R["kern_null"] < R["kern_beob"], (R["kern_null"], R["kern_beob"])
    assert R["kern_p"] <= 0.05, R["kern_p"]
    assert R["kern_boden"] <= R["kern_p"], "p unter der eigenen Untergrenze"


def test_beide_stellen_werden_erfasst(ergebnis):
    """Antwortpositionen UND Entscheidungsstelle aus demselben Durchlauf -
       sonst waere der Vergleich mit Phase 12 nicht sauber."""
    _, ns = ergebnis
    R = ns["SCREEN_RESULTS"]
    assert R["phase12"], "die Phase-12-Menge wurde nicht neu abgeleitet"
    assert "entscheidung_kern" in R
    assert 0.0 <= R["jaccard_phase12"] <= 1.0
    # Genau abgezaehlt: die Antwort ist 20 Positionen lang, die
    # Entscheidungsstelle eine. Wuerde die Zelle den Prompt mitzaehlen,
    # staenden hier 60 statt 20 Positionen - und der Armeffekt waere um den
    # Faktor 3 verduennt, ohne dass ein Ergebnis offensichtlich falsch aussieht.
    n_antwort = 20
    for a in ("NEU", "JA", "BR1"):
        assert len(ns["ANTPOS"][a]) == len(ns["ENTPOS"][a])
        assert sum(ns["ANTPOS"][a][0].values()) == n_antwort * NLAY * TOPK, \
            "Antwortzaehler deckt nicht genau die Antwortpositionen ab"
        assert sum(ns["ENTPOS"][a][0].values()) == NLAY * TOPK, \
            "Entscheidungszaehler deckt nicht genau eine Position ab"
        assert sum(ns["ANTPOS"][a][0].values()) > sum(ns["ENTPOS"][a][0].values()), \
            "die Antwort muss mehr Plaetze belegen als eine einzelne Position"
    # Die Entscheidungsstelle traegt den Effekt in dieser Welt ebenfalls -
    # also muss der dort gezogene Kern die eingebaute Menge treffen
    ent = {tuple(q) for q in R["entscheidung_kern"]}
    assert ent, "an der Entscheidungsstelle nichts gefunden"
    assert ent <= set(KONSTRUKT), sorted(ent - set(KONSTRUKT))


def test_selektivitaet_und_schichtung(ergebnis):
    _, ns = ergebnis
    sel, sv = ns["selektivitaet"], ns["schicht_von"]
    # doppelte Rate bei gleicher Gesamtzahl -> Selektivitaet ~2
    assert abs(sel(200, 1000, 100, 1000) - 2.0) < 0.02
    assert abs(sel(100, 1000, 100, 1000) - 1.0) < 1e-9
    # Glaettung: ein im Bezug nie feuernder Experte wird nicht unendlich
    assert sel(10, 1000, 0, 1000) < 25.0
    assert sel(0, 1000, 0, 1000) == 1.0
    assert sel(5, 0, 5, 100) == 0.0
    assert sv(0) == 0 and sv(1) == 1 and sv(3) == 2 and sv(255) == 8


def test_mengenarithmetik(ergebnis):
    _, ns = ergebnis
    sch, ohne, jac = ns["schnitt"], ns["ohne"], ns["jaccard_s"]
    assert sch([[1, 2, 3], [2, 3, 4], [3, 2]]) == [2, 3]
    assert sch([]) == [] and sch([[1, 2]]) == [1, 2]
    assert ohne([1, 2, 3], [[2]]) == [1, 3]
    assert ohne([1, 2], []) == [1, 2]
    assert jac([1, 2], [2, 3]) == pytest.approx(1 / 3)
    assert jac([], []) == 0.0 and jac([1], [1]) == 1.0


def test_urteilsordnung_screen(ergebnis):
    _, ns = ergebnis
    u = ns["urteil_screen"]
    voll = {"JA": [1], "BR1": [1], "MORSE": [1], "SR": [], "RU": []}
    assert u(voll, [1, 2, 3], 0.01, [1, 2, 3]) == "KONSTRUKTIONSKERN"
    assert u(dict(voll, MORSE=[]), [1, 2, 3], 0.01, [1, 2, 3]) == "ARM-OHNE-SCHALTKREIS"
    # die Armsperre kommt VOR allem anderen
    assert u(dict(voll, JA=[]), [], 1.0, []) == "ARM-OHNE-SCHALTKREIS"
    assert u(voll, [1, 2], 0.01, [1, 2]) == "KEIN-GEMEINSAMER-KERN"
    assert u(voll, [1, 2, 3], 0.30, [1, 2, 3]) == "KERN-ZUFAELLIG"
    assert u(voll, [1, 2, 3], 0.01, []) == "KERN-NICHT-SPEZIFISCH"


def test_wiederholung_verschiebt_die_ziehungen():
    _, a = lauf()
    _, b = lauf(wiederholung=1)
    for z in ("ziehen", "null", "shuffle"):
        for arm in ("JA", "BR1", "NEU"):
            assert a["saat"](z, arm) != b["saat"](z, arm), (z, arm)


def test_keine_haken_haengen(ergebnis):
    w, _ = ergebnis
    offen = sum(len(w.schichten[l]._haken) for l in range(NLAY))
    assert offen == 0, "%d Haken nicht entfernt" % offen
