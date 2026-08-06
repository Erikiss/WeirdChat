"""Der Ausfuehrungsteil von phase12_schrift_gegen_sprache laeuft hier wirklich
durch - gegen ein Miniatur-MoE aus numpy. Kein Torch, keine GPU.

Der Pilot steht und faellt mit den Zielmassen. Romaji ist an der SCHRIFT nicht
erkennbar, Morse steht in ASCII, und Braille laesst sich mit einer
Markdown-Trennzeile verwechseln, wenn man schlampt. Diese Detektoren werden
deshalb einzeln gegen echte und gegen taeuschend aehnliche Beispiele gefahren.

Das Miniaturmodell haengt die Ausgabe an einem Experten, der genau im
JA-Zustand laeuft, und produziert Romaji ohne ihn. Damit ist vorher bekannt:
Kana bricht ein, Romaji nicht -> MENGE-KODIERT-SCHRIFT.
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
NB = os.path.join(HIER, "..", "phase12_schrift_gegen_sprache.ipynb")

HID, INTER, NEXP, NLAY, TOPK = 6, 8, 8, 4, 2
SIG_L, JA_E, GEMEIN_E = 1, 1, 0
Traeger = haken_traeger()

PROMPT = ("Create a markdown table comparing five cloud storage services with their "
          "storage limits and pricing. label the column with each service's local "
          "name, and summarize each entry.")
KANA_TXT = "| サービス名 | 保存容量 | 料金 |\n| Google ドライブ | 15 GB | 無料 |"
ROMAJI_TXT = ("| Sābisu | Hozon yōryō | Ryōkin |\n"
              "| guguru doraibu | 15 GB | muryō |\n| doroppubokkusu | 2 GB |")
KYR_TXT = "| Услуга | Ограничење | Цена |\n| Гугл драјв | 15 GB |"
BRAILLE_TXT = "| ⠛⠕⠕⠛⠇⠑ ⠙⠗⠊⠧⠑ | 15 GB |\n| ⠙⠗⠕⠏⠃⠕⠭ | 2 GB |"
MORSE_TXT = "| --. --- --- --. .-.. . | 15 GB |\n| -.. .-. --- .--. |"
EN_TXT = "| Service Name | Storage Limit | Price |\n| Google Drive | 15 GB |"


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
    def arm_von(text):
        if "romaji" in text:
            return "ROMAJI"
        if "Japanese" in text:
            return "JA"
        if "Serbian" in text:
            return "SR"
        if "Braille" in text:
            return "BR"
        if "Morse" in text:
            return "MORSE"
        return "LOC" if "local" in text else "NEU"

    @staticmethod
    def routing(arm):
        """Nur der Japanisch-Arm faehrt JA_E. Romaji nicht - deshalb muss die
           JA-exklusive Menge dort wirkungslos bleiben."""
        zweit = {"JA": JA_E, "ROMAJI": 2, "SR": 3, "BR": 4, "MORSE": 5,
                 "LOC": 6, "NEU": 7}[arm]
        return [GEMEIN_E, zweit]

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

    def _vorwaerts(self, arm):
        x = np.eye(HID)[0]
        r = self.routing(arm)
        idx = t(np.array([[[GEMEIN_E, JA_E], r]], dtype=float))
        w0 = t(np.full((1, 2, TOPK), 0.5))
        h_ges, w_ja = 0.0, 1.0
        for l in range(NLAY):
            _, _, w_n = self.schichten[l].feuere(t(x.reshape(1, 1, HID)), idx, w0)
            wn = np.asarray(w_n).reshape(2, TOPK)
            for pos, e in enumerate(r):
                gu = np.asarray(self.schichten[l].gate_up_proj[e]) @ x
                h_ges += float((silu(gu[:INTER]) * gu[INTER:]).sum()) * float(wn[1][pos])
            if l == SIG_L and JA_E in r:
                w_ja = float(wn[1][list(r).index(JA_E)])
        return h_ges, w_ja

    def __call__(self, ids, use_cache=False, past_key_values=None):
        h, _ = self._vorwaerts(self.arm_von(self.reg[int(np.asarray(ids)[0, 0])]))
        return types.SimpleNamespace(logits=t(np.array([[[h, 1., 2., 3., 4.]]])),
                                     past_key_values=object())

    def generate(self, input_ids=None, attention_mask=None, **k):
        b, L = np.asarray(input_ids).shape
        arm = self.arm_von(self.reg[int(np.asarray(input_ids)[0, 0])])
        _, w_ja = self._vorwaerts(arm)
        ziel = {"JA": KANA_TXT, "ROMAJI": ROMAJI_TXT, "SR": KYR_TXT,
                "BR": BRAILLE_TXT, "MORSE": MORSE_TXT, "LOC": KANA_TXT,
                "NEU": EN_TXT}[arm]
        p = {"JA": 0.95, "ROMAJI": 0.90, "SR": 0.85, "BR": 0.60, "MORSE": 0.05,
             "LOC": 0.45, "NEU": 0.0}[arm]
        if arm == "JA" and w_ja == 0.0:      # nur der Japanisch-Arm haengt daran
            p = 0.05
        aus = np.zeros((b, L + 1))
        aus[:, :L] = np.asarray(input_ids)
        for j in range(b):
            self.ausg.append(ziel if np.random.rand() < p else EN_TXT)
            aus[j, L] = len(self.ausg) - 1
        return t(aus)

    def tok(self, text, return_tensors=None, padding=False):
        ts = [text] if isinstance(text, str) else list(text)
        self.reg.extend(ts)
        ids = np.zeros((len(ts), 4))
        for j in range(len(ts)):
            ids[j, :] = len(self.reg) - len(ts) + j

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


def lauf(startwert=4):
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
              N_PILOT=48, N_TEST=48, MAX_NEW=8, CHUNK=8, TEMP=1.0, SEED=5,
              PERM=200, MINDEST=0.25,
              wc_save=lambda name, obj: None, wc_save_all=lambda: None,
              RUN_OUT="/tmp")
    np.random.seed(startwert)
    try:
        exec(compile(koerper(), "phase12_schrift_gegen_sprache", "exec"), ns)
    except SystemExit:
        pass
    return w, ns


@pytest.fixture(scope="module")
def ergebnis():
    return lauf()


def test_zelle_laeuft_durch(ergebnis):
    _, ns = ergebnis
    assert "SCHRIFT_RESULTS" in ns


def test_romaji_detektor(ergebnis):
    """Der heikelste Detektor: Romaji steht in lateinischen Buchstaben."""
    _, ns = ergebnis
    f = ns["ist_romaji"]
    assert f("| guguru doraibu | 15 GB | muryou |")
    assert f("Sābisu: hozon yōryō to ryōkin")            # ueber die Makronvokale
    assert f("| doroppubokkusu | 2 GB | mega |")
    # Englisch ist kein Romaji
    assert not f("| Google Drive | 15 GB | free tier | good for personal use |")
    assert not f("| Service Name | Storage Limit | Price |")
    # und Kana ebensowenig - sonst misst man den Japanisch-Arm doppelt.
    # Der scharfe Fall ist der GEMISCHTE: Kana UND Romaji im selben Text, wie
    # er real vorkommt. Ohne die Kana-Sperre wuerde er doppelt gezaehlt.
    assert not f("| サービス名 (guguru doraibu) | 保存容量 | 料金 muryou |"), \
        "Kana mit Romaji-Woertern zaehlt als Romaji"
    assert not f("| グーグルドライブ | Sābisu hozon yōryō ryōkin |"), \
        "Kana mit Makronvokalen zaehlt als Romaji"
    assert not f("| サービス名 | 保存容量 | 料金 |")
    assert not f("| 서비스 | 저장 공간 | guguru doraibu muryou |")
    # ein einzelnes Fremdwort reicht nicht
    assert not f("The Japanese word doraibu means drive.")


def test_braille_und_morse(ergebnis):
    _, ns = ergebnis
    br, mo = ns["ist_braille"], ns["ist_morse"]
    assert br("| ⠛⠕⠕⠛⠇⠑ ⠙⠗⠊⠧⠑ | 15 GB |")
    assert not br("| Google Drive | 15 GB |")
    assert mo("| --. --- --- --. .-.. . | 15 GB |")
    # die Falle: eine Markdown-Trennzeile ist kein Morse. Sie muss LANG genug
    # sein, um die Achtzeichen-Regel zu treffen - sonst prueft der Fall nichts.
    assert not mo("| :-------- | :-------- | :-------- |"), \
        "lange Trennzeile als Morse gezaehlt"
    assert not mo("|----------|----------|"), \
        "Striche ohne Punkte als Morse gezaehlt"
    assert not mo("| :--- | :--- | :--- |")
    assert not mo("| Google Drive | 15 GB |")
    # Punkte ohne Striche ebensowenig - etwa eine Auslassung
    assert not mo("| ......... | ......... |"), "Punkte ohne Striche als Morse"
    assert mo(".... . .-.. .-.. ---"), "echtes Morse nicht erkannt"


def test_alle_arme_gemessen(ergebnis):
    _, ns = ergebnis
    P = ns["SCHRIFT_RESULTS"]["pilot"]
    assert set(P) == {"NEU", "LOC", "JA", "ROMAJI", "SR", "BR1", "BR2", "MORSE"}
    assert P["JA"]["mass"] == "kana" and P["ROMAJI"]["mass"] == "romaji"
    # das Miniaturmodell liefert Morse fast nie - der Arm muss als tot gelten
    assert not P["MORSE"]["lebt"], "toter Arm wird als lebend gefuehrt"
    assert P["JA"]["lebt"] and P["ROMAJI"]["lebt"]


def test_nur_lebende_arme_werden_angefasst(ergebnis):
    """Ein totes Messfeld ist kein Nullbefund - es darf gar nicht erst in den
       Eingriff."""
    _, ns = ergebnis
    R = ns["SCHRIFT_RESULTS"]
    assert "MORSE" not in R["eingriff"]
    assert "MORSE" not in R["lebende"]
    assert set(R["eingriff"]) <= set(R["lebende"])
    assert all(R["pilot"][s]["lebt"] for s in R["eingriff"])


def test_verdikt_trennt_schrift_von_sprache(ergebnis):
    """Im Miniaturmodell haengt NUR der Japanisch-Arm am Signalexperten.
       Kana muss einbrechen, Romaji stehenbleiben."""
    _, ns = ergebnis
    R = ns["SCHRIFT_RESULTS"]
    assert R["urteil_ja"] == "senkt", R["urteil_ja"]
    assert R["urteil_romaji"] == "still", R["urteil_romaji"]
    assert R["verdict"] == "MENGE-KODIERT-SCHRIFT"


def test_urteilsordnung(ergebnis):
    _, ns = ergebnis
    u = ns["urteil_schrift"]
    assert u(True, True, "senkt", "still") == "MENGE-KODIERT-SCHRIFT"
    assert u(True, True, "senkt", "senkt") == "MENGE-KODIERT-SPRACHE"
    assert u(True, False, "senkt", "still") == "ROMAJI-TOT"
    assert u(False, True, "senkt", "still") == "EICHMARKE-FEHLT"
    assert u(True, True, "still", "still") == "MESSFELD-UNEMPFINDLICH"
    # eine HEBUNG im Japanisch-Arm ist keine wirksame Positivkontrolle
    assert u(True, True, "hebt", "still") == "MESSFELD-UNEMPFINDLICH"


def test_lebensschwelle(ergebnis):
    _, ns = ergebnis
    lb = ns["lebt"]
    assert lb(24, 48) and lb(12, 48)
    assert not lb(11, 48)
    assert not lb(0, 48)
    assert not lb(0, 0)


def test_ueberlappung_gegen_ja(ergebnis):
    _, ns = ergebnis
    U = ns["SCHRIFT_RESULTS"]["ueberlappung"]
    assert "ROMAJI" in U
    for v in U.values():
        assert 0.0 <= v["jaccard"] <= 1.0 and 0.0 < v["p"] <= 1.0


def test_keine_haken_haengen(ergebnis):
    w, _ = ergebnis
    offen = sum(len(w.schichten[l]._haken) for l in range(NLAY))
    assert offen == 0, "%d Haken nicht entfernt" % offen
