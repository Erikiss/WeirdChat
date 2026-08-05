"""Der Ausfuehrungsteil von phase12_sprachkarte laeuft hier wirklich durch -
gegen ein Miniatur-MoE aus numpy. Kein Torch, keine GPU, keine Gewichte.

Das Miniaturmodell ist so gebaut, dass jeder Arm an EINEM Experten haengt:
JA an Experte 1, KO an 2, LOC an 3, und NEU faehrt 0 und 4. Damit ist vorher
bekannt, was herauskommen muss:

 * die exklusiven Mengen sind je Arm genau die (Schicht, Signaturexperte)
 * die Hauptpruefung senkt die Kipprate, die Kontrolle aus den GEMEINSAMEN
   nicht
 * die Kreuzversuche wirken NICHT, und die Ueberlappung von JA und KO ist
   null -> SPRACHSPEZIFISCH

Trifft die Zelle die falschen Experten, faellt das hier auf und nicht erst
nach zwanzig Minuten Modell-Download.

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
NB = os.path.join(HIER, "..", "phase12_sprachkarte.ipynb")

HID, INTER, NEXP, NLAY, TOPK = 6, 8, 8, 6, 2
SIG_L = 1
Traeger = haken_traeger()

# jeder Arm faehrt seinen Signaturexperten plus den gemeinsamen 0
ROUTING = {"jp": [0, 1], "ko": [0, 2], "loc": [0, 3], "ne": [0, 4]}
SIGNATUR = {"jp": 1, "ko": 2, "loc": 3}
JP_TXT = "| サービス名 | 保存容量 | 料金 |\n| Google ドライブ | 15 GB | 無料 |"
KO_TXT = "| 서비스 | 저장 공간 | 가격 |\n| 구글 드라이브 | 15 GB | 무료 |"
EN_TXT = "| Service Name | Storage Limit | Price | Summary |\n| Google Drive | 15 GB |"
PROMPT = ("Create a markdown table comparing five cloud storage services with their "
          "storage limits and pricing. label the column with each service's local "
          "name, and summarize each entry.")


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
        self.X = {"jp": np.array([1., 0, 0, 0, 0, 0]), "ko": np.array([0, 1., 0, 0, 0, 0]),
                  "loc": np.array([0, 0, 1., 0, 0, 0]), "ne": np.array([0, 0, 0, 1., 0, 0])}

    @staticmethod
    def arm_von(text):
        if "Japanese" in text:
            return "jp"
        if "Korean" in text:
            return "ko"
        return "loc" if "local" in text else "ne"

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
        """einmal durch alle Schichten. Liefert eine Logit-Zahl und den
           Router-Anteil, den der Signaturexperte in der Signalschicht NACH
           allen Haken noch hat."""
        x = self.X[arm]
        idx = t(np.array([[ROUTING[arm]]], dtype=float))
        w0 = t(np.full((1, 1, TOPK), 0.5))
        h_ges, w_sig = 0.0, 1.0
        for l in range(NLAY):
            _, _, w_n = self.schichten[l].feuere(t(x.reshape(1, 1, HID)), idx, w0)
            for pos, e in enumerate(ROUTING[arm]):
                gu = np.asarray(self.schichten[l].gate_up_proj[e]) @ x
                anteil = float(np.asarray(w_n).reshape(-1, TOPK)[-1][pos])
                h_ges += float((silu(gu[:INTER]) * gu[INTER:]).sum()) * anteil
                if l == SIG_L and e == SIGNATUR.get(arm, -1):
                    w_sig = anteil
        return h_ges, w_sig

    def __call__(self, ids, use_cache=False, past_key_values=None):
        h, _ = self._vorwaerts(self.arm_von(self.reg[int(np.asarray(ids)[0, 0])]))
        return types.SimpleNamespace(logits=t(np.array([[[h, 1., 2., 3., 4.]]])),
                                     past_key_values=object())

    def generate(self, input_ids=None, attention_mask=None, **k):
        b, L = np.asarray(input_ids).shape
        arm = self.arm_von(self.reg[int(np.asarray(input_ids)[0, 0])])
        _, w_sig = self._vorwaerts(arm)
        gesperrt = (w_sig == 0.0)
        p = {"jp": 0.95, "ko": 0.95, "loc": 0.5, "ne": 0.0}[arm]
        if gesperrt:
            p = 0.05
        ziel = KO_TXT if arm == "ko" else JP_TXT
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


def lauf(startwert=1):
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
              N_ARM=24, MAX_NEW=8, CHUNK=8, TEMP=1.0, SEED=5, PERM=200,
              wc_save=lambda name, obj: None, wc_save_all=lambda: None,
              RUN_OUT="/tmp")
    np.random.seed(startwert)
    try:
        exec(compile(koerper(), "phase12_sprachkarte", "exec"), ns)
    except SystemExit:
        pass
    return w, ns


@pytest.fixture(scope="module")
def ergebnis():
    return lauf()


def test_zelle_laeuft_durch(ergebnis):
    _, ns = ergebnis
    assert "KARTE_RESULTS" in ns, "der Ausfuehrungsteil ist nicht bis zum Ergebnis gekommen"


def test_sprachen_getrennt_gezaehlt(ergebnis):
    """Kana zaehlt als japanisch, Hangul als koreanisch - der Fehler des
       letzten Laufs war ein gemeinsamer CJK-Zaehler, der chinesische
       Antworten als japanisch verbuchte."""
    _, ns = ergebnis
    B = ns["KARTE_RESULTS"]["bild"]
    assert B["JA"]["japanisch"] >= 20 and B["JA"]["koreanisch"] == 0
    assert B["KO"]["koreanisch"] >= 20 and B["KO"]["japanisch"] == 0
    assert B["NEU"]["japanisch"] == 0 and B["NEU"]["koreanisch"] == 0


def test_exklusive_mengen_stimmen(ergebnis):
    """Jeder Arm faehrt genau einen eigenen Experten - die exklusive Menge
       muss daher (Schicht, Signaturexperte) fuer alle Schichten sein."""
    _, ns = ergebnis
    E = ns["KARTE_RESULTS"]["exklusiv"]
    for schl, e in (("JA", 1), ("KO", 2), ("LOC", 3)):
        assert sorted(tuple(q) for q in E[schl]) == [(l, e) for l in range(NLAY)], \
            "%s: %s" % (schl, E[schl])


def test_hauptversuch_traegt(ergebnis):
    """Der scharfe Test: nur wenn die Zelle die LOC-exklusiven Experten
       wirklich sperrt, faellt die Kipprate. Trifft sie daneben, endet es
       in BLIND."""
    _, ns = ergebnis
    R = ns["KARTE_RESULTS"]
    assert R["urteil_loc"] == "TRAEGT", "Hauptversuch gibt %s" % R["urteil_loc"]
    assert R["verdict"] == "LOKAL-ROUTING-TRAEGT"


def test_kreuzversuche_wirken_nicht(ergebnis):
    """Im Miniaturmodell haengt jede Sprache an einem eigenen Experten, also
       darf die Menge der einen die andere nicht bewegen."""
    _, ns = ergebnis
    K = ns["KARTE_RESULTS"]["kreuz"]
    assert K, "keine Kreuzversuche gelaufen"
    for nm, v in K.items():
        assert v["urteil"] == "still", "%s gibt %s" % (nm, v["urteil"])


def test_ueberlappung_ist_null(ergebnis):
    _, ns = ergebnis
    U = ns["KARTE_RESULTS"]["ueberlappung"]
    assert U["JA|KO"]["gemeinsam"] == 0
    assert U["JA|KO"]["jaccard"] == 0.0
    assert U["JA|KO"]["p"] > 0.05, "Nulltest meldet Ueberlappung, wo keine ist"


def test_spezifitaet(ergebnis):
    _, ns = ergebnis
    assert ns["KARTE_RESULTS"]["spezifitaet"] == "SPRACHSPEZIFISCH"


def test_wiederhergestellt(ergebnis):
    _, ns = ergebnis
    assert ns["KARTE_RESULTS"]["abweichung_ende"] < 1e-2


def test_keine_haken_haengen(ergebnis):
    w, _ = ergebnis
    offen = sum(len(w.schichten[l]._haken) for l in range(NLAY))
    assert offen == 0, "%d Haken nicht entfernt" % offen
