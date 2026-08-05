"""Der Ausfuehrungsteil von phase12_ffn_eichung laeuft hier wirklich durch -
gegen ein Miniaturmodell aus numpy. Kein Torch, keine GPU, keine Gewichte.

Warum das noetig ist: die erste FFN-Zelle hat die Architektur richtig erkannt
und ist trotzdem drei Schritte spaeter mit StopIteration gestorben - nach
zwanzig Minuten Modell-Download. Ein Test der reinen Logik faengt so etwas
nicht. Hier laufen Haken, Routing, Zwischenschicht, Indexabbildung,
Nullsetzen der Gewichtszeilen, Wiederherstellung und Urteil tatsaechlich.

Der Test ist zugleich ein SCHARFER Test der Indexabbildung: das Miniaturmodell
antwortet nur dann anders, wenn genau die vier vorher festgelegten
Signal-Zeilen genullt sind. Greift zu_einheit daneben, kippt nichts und der
Lauf endet mit BLIND statt mit INSTRUMENT-TRAEGT.

Gelesen wird aus dem NOTEBOOK, nicht aus einer Kopie - sonst prueft der Test
eine Datei, die niemand ausfuehrt.
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
NB = os.path.join(HIER, "..", "phase12_ffn_eichung.ipynb")

HID, INTER, NEXP, NLAY, TOPK = 6, 8, 4, 3, 2
SIG_L, SIG_E, SIG_U = 1, 0, [0, 1, 2, 3]   # die vorher festgelegten Signal-Zeilen


def koerper():
    with open(NB, encoding="utf-8") as f:
        nb = json.load(f)
    quelle = "".join("".join(c.get("source", [])) for c in nb["cells"]
                     if c.get("cell_type") == "code")
    marke = "# ---------------- reine Logik"
    assert marke in quelle, "Marke fuer den Logikteil fehlt im Notebook"
    return quelle[quelle.index(marke):]


Traeger = haken_traeger()


class Experten(Traeger):
    def __init__(self, rs):
        Traeger.__init__(self)
        self.gate_up_proj = t(rs.randn(NEXP, 2 * INTER, HID) * 0.5)
        self.down_proj = t(rs.randn(NEXP, HID, INTER) * 0.5)
        self.intermediate_dim = INTER
        self.act_fn = silu


JP_TXT = "| サービス名 | 保存容量 | 料金 |\n| Google ドライブ | 15 GB | 無料 |"
EN_TXT = "| Service Name | Storage Limit | Price | Summary |\n| Google Drive | 15 GB |"
PROMPT = ("Create a markdown table comparing five cloud storage services with their "
          "storage limits and pricing. label the column with each service's local "
          "name, and summarize each entry.")


class Welt:
    """Miniaturmodell, Tokenisierer und Textregister in einem Objekt - damit
       zwei Testlaeufe sich nicht ueber globale Listen ins Gehege kommen."""

    def __init__(self, startwert=4711):
        rs = np.random.RandomState(startwert)
        self.schichten = {l: Experten(rs) for l in range(NLAY)}
        for u in SIG_U:                       # Signal: bei x=e0 gross, bei x=e1 null
            W = self.schichten[SIG_L].gate_up_proj
            W[SIG_E, u] = 0.0
            W[SIG_E, u, 0] = 3.0
            W[SIG_E, INTER + u] = 0.0
            W[SIG_E, INTER + u, 0] = 3.0
        self.reg = []
        self.ausg = []
        self.X = {"jp": np.array([1., 0, 0, 0, 0, 0]),
                  "ne": np.array([0, 1., 0, 0, 0, 0]),
                  "or": np.array([.3, .7, .1, 0, 0, 0])}

    @staticmethod
    def arm_von(text):
        if "Japanese" in text:
            return "jp"
        return "or" if "local" in text else "ne"

    def beschaedigt(self):
        W = self.schichten[SIG_L].gate_up_proj[SIG_E]
        return all(float(np.abs(W[u]).max()) == 0.0 for u in SIG_U)

    # --- Modell ---
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
        x = self.X[self.arm_von(self.reg[int(np.asarray(ids)[0, 0])])]
        idx = t(np.array([[[0, 1]]], dtype=float))
        h_ges = 0.0
        for l in range(NLAY):
            self.schichten[l].feuere(t(x.reshape(1, 1, HID)), idx)
            for e in (0, 1):
                gu = np.asarray(self.schichten[l].gate_up_proj[e]) @ x
                h_ges += float((silu(gu[:INTER]) * gu[INTER:]).sum())
        return types.SimpleNamespace(logits=t(np.array([[[h_ges, 1., 2., 3., 4.]]])),
                                     past_key_values=object())

    def generate(self, input_ids=None, attention_mask=None, **k):
        b, L = np.asarray(input_ids).shape
        arm = self.arm_von(self.reg[int(np.asarray(input_ids)[0, 0])])
        p = {"jp": 0.95, "ne": 0.02, "or": 0.5}[arm]
        if arm == "jp" and self.beschaedigt():
            p = 0.05
        aus = np.zeros((b, L + 1))
        aus[:, :L] = np.asarray(input_ids)
        for j in range(b):
            self.ausg.append(JP_TXT if np.random.rand() < p else EN_TXT)
            aus[j, L] = len(self.ausg) - 1
        return t(aus)

    # --- Tokenisierer ---
    pad_token_id = 0
    pad_token = None
    eos_token = 1
    padding_side = "right"

    def tok(self, text, return_tensors=None, padding=False):
        ts = [text] if isinstance(text, str) else list(text)
        self.reg.extend(ts)
        n = len(ts)
        ids = np.zeros((n, 4))
        for j in range(n):                    # Index in JEDER Spalte: die Zelle
            ids[j, :] = len(self.reg) - n + j  # schneidet ids[:,:-1] und ids[:,-1:]

        class E(dict):
            def to(self, *a, **k):
                return self

            @property
            def input_ids(self):
                return self["input_ids"]

        return E({"input_ids": t(ids), "attention_mask": t(np.ones((n, 4)))})

    def decode(self, seq, skip_special_tokens=True):
        a = np.asarray(seq).ravel()
        return self.ausg[int(a[-1])] if a.size else ""


def lauf(quelle, startwert=1):
    w = Welt()

    class Tok:   # SimpleNamespace ist nicht aufrufbar, der Tokenisierer muss es sein
        pad_token_id = 0
        pad_token = None
        eos_token = 1
        padding_side = "right"

        def __call__(self, *a, **k):
            return w.tok(*a, **k)

        def decode(self, *a, **k):
            return w.decode(*a, **k)
    tokenizer = Tok()
    ns = dict(os=os, re=re, math=math, torch=mach_torch(), collections=collections,
              unicodedata=unicodedata, random=random, np=np, glob=None, json=json,
              gc=gc, sys=sys, time=time,
              model=w, tokenizer=tokenizer, PROMPTS={"p1": PROMPT}, ZIEL_ID="p1",
              N_ARM=24, N_LEIT=24, MAX_NEW=8, CHUNK=8, TEMP=1.0, SEED=5,
              DOSEN=[2, 4], LEITER=[2, 4, 8],
              wc_save=lambda name, obj: None, wc_save_all=lambda: None,
              RUN_OUT="/tmp")
    np.random.seed(startwert)
    try:
        exec(compile(quelle, "phase12_ffn_eichung", "exec"), ns)
    except SystemExit:
        pass
    return w, ns


@pytest.fixture(scope="module")
def ergebnis():
    return lauf(koerper())


def test_zelle_laeuft_durch(ergebnis):
    _, ns = ergebnis
    assert "EICH_RESULTS" in ns, "der Ausfuehrungsteil ist nicht bis zum Ergebnis gekommen"


def test_gemeinsame_paare_vollstaendig(ergebnis):
    _, ns = ergebnis
    assert ns["EICH_RESULTS"]["paare_gemeinsam"] == NLAY * TOPK


def test_indexabbildung_trifft_die_signalzeilen(ergebnis):
    """Der scharfe Test: nur wenn genau die vier Signal-Zeilen genullt werden,
       antwortet das Miniaturmodell anders. Eine vertauschte Indexabbildung
       fuehrt hier zu BLIND."""
    _, ns = ergebnis
    R = ns["EICH_RESULTS"]
    d4 = [a for a in R["arme"] if a["dosis"] == 4]
    assert d4, "Dosis 4 ist nicht gelaufen"
    assert d4[0]["urteil"] == "TRAEGT", (
        "Dosis 4 gibt %s - die Auswahl trifft die Signalzeilen nicht" % d4[0]["urteil"])
    assert R["verdict"] == "INSTRUMENT-TRAEGT"


def test_zufallsarm_bleibt_ruhig(ergebnis):
    """Waere die Kontrolle genauso wirksam, koennte die Zelle nichts zuordnen."""
    _, ns = ergebnis
    d4 = [a for a in ns["EICH_RESULTS"]["arme"] if a["dosis"] == 4][0]
    assert d4["k_zufall"] > d4["k_ausw"], "Zufallsarm senkt genauso stark"


def test_massstab_bleibt_endlich(ergebnis):
    """Der v2-Defekt: ein Nenner je Einheit konnte auf 1e-8 fallen und
       Trennwerte von 1.15e7 erzeugen. Ein globaler Massstab kann das nicht."""
    _, ns = ergebnis
    R = ns["EICH_RESULTS"]
    assert R["massstab"] > 0
    assert R["trenn_max"] < 1e5, "Trennwerte explodieren wieder: %.3g" % R["trenn_max"]


def test_dosisleiter_vollstaendig(ergebnis):
    _, ns = ergebnis
    assert len(ns["EICH_RESULTS"]["leiter_zeilen"]) == 3


def test_gewichte_wiederhergestellt(ergebnis):
    w, _ = ergebnis
    W = w.schichten[SIG_L].gate_up_proj[SIG_E]
    assert float(np.abs(np.asarray(W[SIG_U[0]])).max()) > 0, \
        "Signalzeile nach dem Lauf noch genullt - stelle_her greift nicht"


def test_keine_haken_haengen(ergebnis):
    w, _ = ergebnis
    offen = sum(len(w.schichten[l]._haken) for l in range(NLAY))
    assert offen == 0, "%d Haken nicht entfernt" % offen
