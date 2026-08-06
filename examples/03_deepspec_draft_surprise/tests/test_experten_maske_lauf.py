"""Der Ausfuehrungsteil von phase12_experten_maske laeuft hier wirklich durch -
gegen ein Miniatur-MoE aus numpy. Kein Torch, keine GPU, keine Gewichte.

Geprueft wird das, was die Zelle neu macht und was man ihr nicht ansieht:

 * Die Maske ersetzt die Argumente eines forward_pre_hook. Wirkt das nicht,
   laeuft der Experte weiter und die Zelle misst nichts. Das Miniaturmodell
   antwortet nur dann anders, wenn genau der Router-Anteil des JP-exklusiven
   Experten in der Signalschicht auf null steht.
 * Der geteilte Experte kommt in zwei Bauformen vor (fusioniertes gate_up_proj
   oder getrennte gate_proj/up_proj). Beide werden gefahren - welche das echte
   Modell hat, liest die Zelle selbst aus, und ein Test, der nur eine kennt,
   wuerde die andere erst auf der GPU auffallen lassen.
 * Haken abgenommen, Gewichtszeilen zurueckgeschrieben, Logits wieder am
   Ausgangspunkt.

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
NB = os.path.join(HIER, "..", "phase12_experten_maske.ipynb")

HID, INTER, NEXP, NLAY, TOPK = 6, 8, 8, 3, 2
GT_INTER = 8
SIG_L, SIG_E = 1, 1        # der Experte, der nur im JP-Arm feuert und traegt
Traeger = haken_traeger()

JP_TXT = "| サービス名 | 保存容量 | 料金 |\n| Google ドライブ | 15 GB | 無料 |"
EN_TXT = "| Service Name | Storage Limit | Price | Summary |\n| Google Drive | 15 GB |"
PROMPT = ("Create a markdown table comparing five cloud storage services with their "
          "storage limits and pricing. label the column with each service's local "
          "name, and summarize each entry.")
# JP faehrt Experte 0 und 1, NEU faehrt 0 und 2 -> exklusiv sind die (l,1)
ROUTING = {"jp": [0, 1], "ne": [0, 2], "or": [0, 1]}


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


class Geteilt(Traeger):
    """der geteilte Experte in der jeweils geprueften Bauform"""

    def __init__(self, rs, art):
        Traeger.__init__(self)
        self.art = art
        self.act_fn = silu
        if art == "fusioniert":
            self.gate_up_proj = t(rs.randn(2 * GT_INTER, HID) * 0.5)
        else:
            self.gate_proj = types.SimpleNamespace(weight=t(rs.randn(GT_INTER, HID) * 0.5))
            self.up_proj = types.SimpleNamespace(weight=t(rs.randn(GT_INTER, HID) * 0.5))

    def named_parameters(self):
        if self.art == "fusioniert":
            yield "gate_up_proj", self.gate_up_proj
        else:
            yield "gate_proj.weight", self.gate_proj.weight
            yield "up_proj.weight", self.up_proj.weight

    def zwischen(self, x):
        if self.art == "fusioniert":
            gu = np.asarray(self.gate_up_proj) @ np.asarray(x)
            g, u = gu[:GT_INTER], gu[GT_INTER:]
        else:
            g = np.asarray(self.gate_proj.weight) @ np.asarray(x)
            u = np.asarray(self.up_proj.weight) @ np.asarray(x)
        return silu(g) * u


class Welt:
    def __init__(self, art, startwert=4711):
        rs = np.random.RandomState(startwert)
        self.schichten = {l: Experten(rs) for l in range(NLAY)}
        self.geteilt = {l: Geteilt(rs, art) for l in range(NLAY)}
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

    def named_modules(self):
        yield "model", self
        for l in range(NLAY):
            yield "model.layers.%d.mlp.experts" % l, self.schichten[l]
            yield "model.layers.%d.mlp.shared_expert" % l, self.geteilt[l]

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
           Router-Anteil, den der Signal-Experte NACH allen Haken noch hat."""
        x = self.X[arm]
        idx = t(np.array([[ROUTING[arm]]], dtype=float))
        w0 = t(np.full((1, 1, TOPK), 0.5))
        h_ges = 0.0
        w_sig = 0.0
        for l in range(NLAY):
            _, idx_n, w_n = self.schichten[l].feuere(t(x.reshape(1, 1, HID)), idx, w0)
            for pos, e in enumerate(ROUTING[arm]):
                gu = np.asarray(self.schichten[l].gate_up_proj[e]) @ x
                anteil = float(np.asarray(w_n).reshape(-1, TOPK)[-1][pos])
                h_ges += float((silu(gu[:INTER]) * gu[INTER:]).sum()) * anteil
                if l == SIG_L and e == SIG_E:
                    w_sig = anteil
            self.geteilt[l].feuere(t(x.reshape(1, 1, HID)))
            h_ges += float(self.geteilt[l].zwischen(x).sum())
        return h_ges, w_sig

    def __call__(self, ids, use_cache=False, past_key_values=None):
        h, _ = self._vorwaerts(self.arm_von(self.reg[int(np.asarray(ids)[0, 0])]))
        return types.SimpleNamespace(logits=t(np.array([[[h, 1., 2., 3., 4.]]])),
                                     past_key_values=object())

    def generate(self, input_ids=None, attention_mask=None, **k):
        b, L = np.asarray(input_ids).shape
        arm = self.arm_von(self.reg[int(np.asarray(input_ids)[0, 0])])
        _, w_sig = self._vorwaerts(arm)
        p = {"jp": 0.95, "ne": 0.02, "or": 0.5}[arm]
        if arm == "jp" and w_sig == 0.0:      # der Signal-Experte ist gesperrt
            p = 0.05
        aus = np.zeros((b, L + 1))
        aus[:, :L] = np.asarray(input_ids)
        for j in range(b):
            self.ausg.append(JP_TXT if np.random.rand() < p else EN_TXT)
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


def lauf(art, startwert=1):
    w = Welt(art)

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
              N_ARM=24, MAX_NEW=8, CHUNK=8, TEMP=1.0, SEED=5,
              ANTEILE=[1 / 8., 1 / 2.], K_GETEILT=4,
              wc_save=lambda name, obj: None, wc_save_all=lambda: None,
              RUN_OUT="/tmp")
    np.random.seed(startwert)
    try:
        exec(compile(koerper(), "phase12_experten_maske", "exec"), ns)
    except SystemExit:
        pass
    return w, ns


@pytest.fixture(scope="module", params=["fusioniert", "getrennt"])
def ergebnis(request):
    return lauf(request.param) + (request.param,)


def test_zelle_laeuft_durch(ergebnis):
    _, ns, _ = ergebnis
    assert "MASKE_RESULTS" in ns, "der Ausfuehrungsteil ist nicht bis zum Ergebnis gekommen"


def test_bauform_des_geteilten_erkannt(ergebnis):
    _, ns, art = ergebnis
    assert ns["MASKE_RESULTS"]["geteilt_art"] == art
    assert ns["MASKE_RESULTS"]["geteilt_inter"] == GT_INTER


def test_exklusive_paare_gefunden(ergebnis):
    """JP faehrt Experte 0 und 1, NEU faehrt 0 und 2 - exklusiv sind genau
       die (Schicht, 1)."""
    _, ns, _ = ergebnis
    ex = [tuple(q) for q in ns["MASKE_RESULTS"]["exklusiv"]]
    assert sorted(ex) == [(l, SIG_E) for l in range(NLAY)], ex


def test_maske_ersetzt_die_argumente(ergebnis):
    """Der scharfe Test: das Miniaturmodell antwortet nur anders, wenn der
       Router-Anteil des Signal-Experten wirklich auf null steht. Ersetzt der
       Haken die Argumente nicht, endet der Lauf in BLIND."""
    _, ns, _ = ergebnis
    R = ns["MASKE_RESULTS"]
    assert R["urteil_routing"] == "TRAEGT", (
        "Routing-Pruefung gibt %s - die Maske greift nicht" % R["urteil_routing"])
    assert R["verdict"] in ("ROUTING-TRAEGT", "BEIDES-TRAEGT")


def test_kontrolle_bleibt_ruhig(ergebnis):
    """Gemeinsame Experten zu sperren darf die Rate nicht senken - sonst
       koennte die Zelle nichts zuordnen."""
    _, ns, _ = ergebnis
    R = ns["MASKE_RESULTS"]
    assert R["k_zufallspaare"] > R["k_exklusiv"]


def test_geteilter_experte_gefahren(ergebnis):
    """Pruefung II muss wirklich gelaufen sein - nicht stillschweigend
       uebersprungen, weil die Bauform nicht erkannt wurde."""
    _, ns, _ = ergebnis
    G = ns["MASKE_RESULTS"]["geteilt"]
    assert G is not None, "Pruefung II wurde uebersprungen"
    assert G["k_einheiten"] > 0 and G["n"] == 24


def test_leiter_vollstaendig(ergebnis):
    _, ns, _ = ergebnis
    L = ns["MASKE_RESULTS"]["leiter"]
    assert len(L) == 2
    assert [z["je_schicht"] for z in L] == [1, 4], [z["je_schicht"] for z in L]
    assert all(z["gesperrt"] > 0 for z in L), "keine Router-Plaetze gesperrt"


def test_wiederhergestellt(ergebnis):
    """Gewichtszeilen des geteilten Experten zurueckgeschrieben und Logits
       wieder am Ausgangspunkt."""
    w, ns, art = ergebnis
    assert ns["MASKE_RESULTS"]["abweichung_ende"] < 1e-2
    g = w.geteilt[0]
    W = g.gate_up_proj if art == "fusioniert" else g.gate_proj.weight
    assert float(np.abs(np.asarray(W)).max()) > 0, "Gewichte noch genullt"


def test_keine_haken_haengen(ergebnis):
    w, _, _ = ergebnis
    offen = sum(len(w.schichten[l]._haken) + len(w.geteilt[l]._haken)
                for l in range(NLAY))
    assert offen == 0, "%d Haken nicht entfernt" % offen
