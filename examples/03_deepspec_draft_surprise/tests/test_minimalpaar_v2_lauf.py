"""Der Ausfuehrungsteil von phase12_minimalpaar_v2 laeuft hier wirklich durch -
gegen ein Miniatur-MoE aus numpy. Kein Torch, keine GPU.

Der Versuch steht und faellt mit einer Eigenschaft, die man den Praefixen
ansehen muss und nicht dem Ergebnis: A und B duerfen sich ZEICHENWEISE nur um
die Einfuegung unterscheiden. Wird das verletzt, misst der Lauf wieder
irgendeinen Textunterschied, und man merkt es nirgends.

Das Miniaturmodell haengt die Kipprate an einem Experten, der genau dann
laeuft, wenn ' (Local Name)' im Praefix steht - und einem zweiten, der bei
JEDER Einfuegung laeuft, auch der inerten. Damit ist vorher bekannt:

 * A kippt in allen acht Paaren haeufiger als B -> H1 mit p = 2/2^8
 * konsistent auf der A-Seite sind ZWEI Experten, einer davon auch in der
   Laengenkontrolle
 * nach Abzug bleibt genau der bedeutungstragende uebrig, und ihn zu sperren
   senkt die Rate -> KOPFZEILE-IM-ROUTER

Wird die Laengenkontrolle nicht abgezogen, sperrt der Lauf einen Experten mit,
der nichts traegt - das faellt hier auf.
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
NB = os.path.join(HIER, "..", "phase12_minimalpaar_v2.ipynb")

HID, INTER, NEXP, NLAY, TOPK = 6, 8, 8, 4, 3
SIG_L = 1
BEDEUTUNG_E, EINSCHUB_E, GEMEIN_E, POS_E, NEU_E = 1, 2, 0, 5, 6
Traeger = haken_traeger()

PROMPT = ("Create a markdown table comparing five cloud storage services with their "
          "storage limits and pricing. label the column with each service's local "
          "name, and summarize each entry.")
KO_TXT = " 구글 드라이브 | 15 GB | 무료 |"
EN_TXT = " Google Drive | 15 GB | free tier |"


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
    def art(text):
        """Was steht im Praefix? Der Prompt selbst enthaelt 'local name', also
           wird auf die Kopfzeilenform geprueft, nicht auf das blosse Wort."""
        if "(Local Name)" in text:
            return "a"
        if "(Short Name)" in text:
            return "i"
        if "Japanese" in text:
            return "ja"
        if "| Service" in text or "| Local Name" in text or "| Plugin" in text:
            return "b"
        return "ne" if "local" not in text else "roh"

    @classmethod
    def routing(cls, art):
        return {"a": [GEMEIN_E, BEDEUTUNG_E, EINSCHUB_E],
                "i": [GEMEIN_E, POS_E, EINSCHUB_E],
                "b": [GEMEIN_E, POS_E, NEU_E],
                "roh": [GEMEIN_E, POS_E, NEU_E],
                "ja": [GEMEIN_E, POS_E, BEDEUTUNG_E],
                "ne": [GEMEIN_E, NEU_E, POS_E]}[art]

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

    def _vorwaerts(self, art):
        x = np.eye(HID)[0]
        r = self.routing(art)
        idx = t(np.array([[[GEMEIN_E, POS_E, BEDEUTUNG_E], r]], dtype=float))
        w0 = t(np.full((1, 2, TOPK), 0.5))
        h_ges, w_sig = 0.0, 1.0
        for l in range(NLAY):
            _, _, w_n = self.schichten[l].feuere(t(x.reshape(1, 1, HID)), idx, w0)
            wn = np.asarray(w_n).reshape(2, TOPK)
            for pos, e in enumerate(r):
                gu = np.asarray(self.schichten[l].gate_up_proj[e]) @ x
                h_ges += float((silu(gu[:INTER]) * gu[INTER:]).sum()) * float(wn[1][pos])
            if l == SIG_L:
                w_sig = min(float(wn[1][i]) for i, e in enumerate(r)
                            if e == BEDEUTUNG_E) if BEDEUTUNG_E in r else 1.0
        return h_ges, w_sig

    def __call__(self, ids, use_cache=False, past_key_values=None):
        h, _ = self._vorwaerts(self.art(self.reg[int(np.asarray(ids)[0, 0])]))
        return types.SimpleNamespace(logits=t(np.array([[[h, 1., 2., 3., 4.]]])),
                                     past_key_values=object())

    def generate(self, input_ids=None, attention_mask=None, **k):
        b, L = np.asarray(input_ids).shape
        art = self.art(self.reg[int(np.asarray(input_ids)[0, 0])])
        _, w_sig = self._vorwaerts(art)
        p = {"a": 0.75, "i": 0.10, "b": 0.10, "roh": 0.4, "ja": 0.95, "ne": 0.0}[art]
        if w_sig == 0.0:                 # der bedeutungstragende Experte ist gesperrt
            p = 0.05
        aus = np.zeros((b, L + 1))
        aus[:, :L] = np.asarray(input_ids)
        for j in range(b):
            self.ausg.append(KO_TXT if np.random.rand() < p else EN_TXT)
            aus[j, L] = len(self.ausg) - 1
        return t(aus)

    def tok(self, text, return_tensors=None, padding=False):
        ts = [text] if isinstance(text, str) else list(text)
        self.reg.extend(ts)
        ids = np.zeros((len(ts), 4))
        for j in range(len(ts)):
            ids[j, :] = len(self.reg) - len(ts) + j
        # Tokenzahl grob an der Textlaenge - die Zelle druckt sie nur
        laenge = max(4, len(ts[0]) // 4)

        class E(dict):
            input_ids_laenge = laenge

            def to(self, *a, **k):
                return self

            @property
            def input_ids(self):
                return self["input_ids"]

        e = E({"input_ids": t(ids), "attention_mask": t(np.ones((len(ts), 4)))})
        return e

    def decode(self, seq, skip_special_tokens=True):
        a = np.asarray(seq).ravel()
        return self.ausg[int(a[-1])] if a.size else ""


def lauf(startwert=2):
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
              N_PAAR=32, N_INERT=16, N_TEST=48, N_REF=48, MAX_NEW=8, CHUNK=8, TEMP=1.0,
              SEED=5, PERM=400, SCHWELLE=1.0,
              wc_save=lambda name, obj: None, wc_save_all=lambda: None,
              RUN_OUT="/tmp")
    np.random.seed(startwert)
    try:
        exec(compile(koerper(), "phase12_minimalpaar", "exec"), ns)
    except SystemExit:
        pass
    return w, ns


@pytest.fixture(scope="module")
def ergebnis():
    return lauf()


def test_zelle_laeuft_durch(ergebnis):
    _, ns = ergebnis
    assert "PAAR2_RESULTS" in ns, "der Ausfuehrungsteil ist nicht bis zum Ergebnis gekommen"


def test_paare_sind_wirklich_minimal(ergebnis):
    """Die eine Eigenschaft, an der alles haengt - und diesmal auf den
       geernteten Praefixen, nicht auf erfundenen."""
    _, ns = ergebnis
    mach, minimal = ns["mach_paare"], ns["paar_ist_minimal"]
    MARKE, GEERNTET = ns["MARKE"], ns["GEERNTET"]
    P = mach(GEERNTET, MARKE)
    assert len(P) >= 8, "nur %d Paare" % len(P)
    for a, b, art, r in P:
        assert minimal(a, b, MARKE)
        assert len(a) - len(b) == len(MARKE)
        assert art in ("loeschen", "einfuegen")
        assert 0.0 <= r <= 1.0
    arten = collections.Counter(art for _, _, art, _ in P)
    assert arten["loeschen"] >= 3 and arten["einfuegen"] >= 3, \
        "beide Richtungen muessen vertreten sein: %s" % dict(arten)
    # ein geloeschtes Paar stammt aus einem Praefix mit hoher Originalrate,
    # ein eingefuegtes aus einem mit niedriger - genau das war der Zweck
    hoch = max(r for _, _, art, r in P if art == "loeschen")
    tief = min(r for _, _, art, r in P if art == "einfuegen")
    assert hoch > 0.6 and tief < 0.2, "%.2f / %.2f" % (hoch, tief)


def test_einfuegen_und_entfernen(ergebnis):
    _, ns = ergebnis
    ein, ent, MARKE, ANKER = ns["einfuegen"], ns["entfernen"], ns["MARKE"], ns["ANKER"]
    T = "| Service | Storage Limit |\n| :--- |"
    assert ein(T, MARKE) == "| Service (Local Name) | Storage Limit |\n| :--- |"
    assert ent(ein(T, MARKE), MARKE) == T, "einfuegen und entfernen sind nicht invers"
    assert ein(ein(T, MARKE), MARKE) is None, "zweimal einfuegen waere kein Minimalpaar"
    assert ein("| Local Name | Storage |", MARKE) is None, "ohne Anker kein Einfuegen"
    assert ent(T, MARKE) is None, "ohne Marke nichts zu entfernen"
    assert ANKER in T


def test_eichungsurteil(ergebnis):
    """Ohne reproduzierende Referenzpraefixe sagt der Lauf nichts."""
    _, ns = ergebnis
    e = ns["eichung_urteil"]
    assert e(0.84, 0.02, 0.84, 0.00) == "ok"
    assert e(0.10, 0.02, 0.84, 0.00) == "HOCH-REPRODUZIERT-NICHT"
    assert e(0.84, 0.60, 0.84, 0.00) == "NIEDRIG-REPRODUZIERT-NICHT"
    assert e(0.30, 0.10, 0.35, 0.05) == "KEIN-ABSTAND"
    R = ns["PAAR2_RESULTS"]
    assert R["urteil_eichung"] == "ok"
    assert R["referenz"]["hoch"]["gemessen"] > R["referenz"]["niedrig"]["gemessen"]


def test_h1_gepaart(ergebnis):
    """A muss in jedem Paar haeufiger kippen als B."""
    _, ns = ergebnis
    R = ns["PAAR2_RESULTS"]
    assert R["plus"] == R["n_paare"], "A nicht in allen Paaren hoeher: %d" % R["plus"]
    assert R["p_rate"] <= R["untergrenze"] + 1e-12
    assert R["p_rate"] < 0.05


def test_vorzeichentest_rechnet_richtig(ergebnis):
    _, ns = ergebnis
    vz, ug = ns["vorzeichentest"], ns["paar_untergrenze"]
    assert abs(vz([1.] * 8)[0] - 2.0 / 256) < 1e-12
    assert abs(vz([-1.] * 8)[0] - 2.0 / 256) < 1e-12
    assert vz([1., 1., 1., 1., -1., -1., -1., -1.])[0] == 1.0
    assert vz([0.0] * 8) == (1.0, 0, 0), "Nullen duerfen nicht als Treffer zaehlen"
    assert vz([1., 1., 1., 0.])[0] == abs(vz([1., 1., 1.])[0])
    assert abs(ug(8) - 2.0 / 256) < 1e-12
    assert ug(4) > 0.05 and ug(8) < 0.05, "acht Paare sind die untere Grenze"


def test_laengenkontrolle_wird_abgezogen(ergebnis):
    """Im Miniaturmodell sind ZWEI Experten konsistent auf der A-Seite; einer
       davon laeuft auch bei der inerten Einfuegung und traegt nichts. Wird er
       nicht abgezogen, sperrt der Lauf ihn mit."""
    _, ns = ergebnis
    R = ns["PAAR2_RESULTS"]
    roh = sorted(tuple(q) for q in R["trenner_roh"])
    tr = sorted(tuple(q) for q in R["trenner"])
    assert roh == sorted([(l, BEDEUTUNG_E) for l in range(NLAY)]
                         + [(l, EINSCHUB_E) for l in range(NLAY)]), roh
    assert tr == [(l, BEDEUTUNG_E) for l in range(NLAY)], tr
    assert len(roh) - len(tr) == NLAY, "die Laengenkontrolle zieht nichts ab"


def test_vorzeichen_nulltest(ergebnis):
    _, ns = ergebnis
    R = ns["PAAR2_RESULTS"]
    assert R["p_trenner"] < 0.05
    assert R["p_trenner"] >= R["untergrenze"] - 1e-9


def test_positivkontrolle_und_verdikt(ergebnis):
    _, ns = ergebnis
    R = ns["PAAR2_RESULTS"]
    assert R["urteil_positiv"] == "senkt"
    assert R["urteil_maske"] == "TRAEGT", R["urteil_maske"]
    assert R["verdict"] == "KOPFZEILE-IM-ROUTER"
    assert R["k_zufall"] > R["k_trenner"], "Kontrolle senkt genauso stark"


def test_urteilsordnung(ergebnis):
    _, ns = ergebnis
    u = ns["urteil_paar2"]
    G = dict(u_eich="ok", u_pos="senkt", p_rate=0.001, plus=8, n_paare=9,
             n_trenner=6, p_trenner=0.001, untergrenze=0.004, u_maske="TRAEGT")

    def mit(**k):
        d = dict(G)
        d.update(k)
        return u(d["u_eich"], d["u_pos"], d["p_rate"], d["plus"], d["n_paare"],
                 d["n_trenner"], d["p_trenner"], d["untergrenze"], d["u_maske"])

    assert mit() == "KOPFZEILE-IM-ROUTER"
    assert mit(u_eich="HOCH-REPRODUZIERT-NICHT") == "EICHUNG-FEHLT"
    assert mit(u_pos="still") == "MESSFELD-UNEMPFINDLICH"
    assert mit(p_rate=0.400) == "KOPFZEILE-IST-MARKER"
    # eine WIRKUNG IN DIE FALSCHE RICHTUNG darf nicht als Bestaetigung durchgehen
    assert mit(plus=1) == "KOPFZEILE-IST-MARKER"
    assert mit(untergrenze=0.20) == "AUFLOESUNG-ZU-GROB"
    assert mit(n_trenner=1) == "URSACHE-OHNE-ROUTER-SITZ"
    assert mit(u_maske="BLIND") == "URSACHE-OHNE-ROUTER-SITZ"
    assert mit(p_trenner=0.400) == "TRENNER-ZUFAELLIG"
    assert mit(u_maske="NUR-STOERUNG") == "NUR-STOERUNG"
    # die Eichung steht VOR allem anderen
    assert mit(u_eich="KEIN-ABSTAND", u_pos="still", p_rate=0.9) == "EICHUNG-FEHLT"


def test_routing_wird_gespeichert(ergebnis):
    _, ns = ergebnis
    R = ns["PAAR2_RESULTS"]
    assert len(R["routing"]) == R["n_paare"]
    assert len(R["inert_routing"]) == R["n_paare"]
    assert all(len(a) and len(b) for a, b in R["routing"])


def test_keine_haken_haengen(ergebnis):
    w, _ = ergebnis
    offen = sum(len(w.schichten[l]._haken) for l in range(NLAY))
    assert offen == 0, "%d Haken nicht entfernt" % offen
