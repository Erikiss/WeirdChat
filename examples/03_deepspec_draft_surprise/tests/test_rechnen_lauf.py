"""Der Ausfuehrungsteil von phase13_rechnen laeuft hier wirklich durch - gegen
ein Miniatur-MoE aus numpy. Kein Torch, keine GPU.

Das Miniaturmodell rechnet richtig, solange der Signalexperte laeuft, und
verrechnet sich ohne ihn - aber NUR bei den mehrstelligen Aufgaben. Hauptstadt
und Einmaleins haengen nicht an ihm. Damit ist vorher bekannt: MUL2 und MUL3
TRAEGT, FAKT und MUL1 STILL, Verdikt KONSTRUKTION-ALLGEMEIN.

Die zweite Welt dreht das um: dort haengt auch das Hauptstadtwissen am
Signalexperten. Dann muss die Abrufsperre greifen und der Lauf darf keine
Rechenaussage machen.
"""
import ast
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
NB = os.path.join(HIER, "..", "phase13_rechnen.ipynb")

HID, INTER, NEXP, NLAY, TOPK = 6, 8, 8, 4, 2
SIG_L, JA_E, GEMEIN_E = 1, 1, 0
Traeger = haken_traeger()

BG = [(0, 4), (0, 4), (0, 3), (0, 3), (0, 3), (1, 4), (1, 4), (2, 5), (2, 6)]

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
    """abruf_haengt=True heisst: auch die Hauptstadtfrage braucht den
       Signalexperten. Dann muss die Abrufsperre der Zelle greifen."""

    def __init__(self, startwert=4711, bg=BG, abruf_haengt=False):
        rs = np.random.RandomState(startwert)
        self.schichten = {l: Experten(rs) for l in range(NLAY)}
        self.reg = []
        self.ausg = []
        self.gen_kw = []
        self.bg = list(bg)
        self.positionen = len(self.bg) + 1
        self.abruf_haengt = abruf_haengt

    @staticmethod
    def art_von(text):
        if "capital city of" in text:
            return "FAKT"
        if "Compute" in text:
            a, b = (int(x) for x in re.findall(r"(\d+) \* (\d+)", text)[0])
            return "MUL1" if (a < 10 and b < 10) else ("MUL2" if a < 100 else "MUL3")
        return "SCHRIFT"

    def routing(self, arm):
        """Die letzte Position unterscheidet die Zustaende. Der englische
           Schriftarm faehrt e7 - das SONST nie laeuft -, damit die exklusive
           Menge genau {e1} ist.

           Die RECHENaufgaben routen bewusst anders (e1 und e2 statt e0 und
           e1). Wuerden sie wie der Japanisch-Arm routen, waere in der
           Miniatur nicht zu unterscheiden, ob die gepruefte Menge aus dem
           Schrift-Prompt oder aus den Rechenaufgaben abgeleitet wurde - und
           genau das ist der Uebertrag, um den es in dieser Zelle geht."""
        return {"JA": [GEMEIN_E, JA_E], "NEU": [GEMEIN_E, 7],
                "RECHNEN": [JA_E, 2]}[arm]

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

    def _vorwaerts(self, arm, n_pos=None):
        x = np.eye(HID)[0]
        r = self.routing(arm)
        muster = ([list(p) for p in self.bg] + [list(r)])[-(n_pos or self.positionen):]
        P = len(muster)
        idx = t(np.array([muster], dtype=float))
        w0 = t(np.full((1, P, TOPK), 0.5))
        h_ges, w_ja = 0.0, 1.0
        for l in range(NLAY):
            _, _, w_n = self.schichten[l].feuere(t(x.reshape(1, 1, HID)), idx, w0)
            wn = np.asarray(w_n).reshape(P, TOPK)
            for pos, e in enumerate(r):
                gu = np.asarray(self.schichten[l].gate_up_proj[e]) @ x
                h_ges += float((silu(gu[:INTER]) * gu[INTER:]).sum()) * float(wn[-1][pos])
            if l == SIG_L and JA_E in r:
                w_ja = float(wn[-1][list(r).index(JA_E)])
        return h_ges, w_ja

    def _zustand(self, text):
        if "Compute" in text or "capital city of" in text:
            return "RECHNEN"
        return "NEU" if "each service's name" in text else "JA"

    def __call__(self, ids, use_cache=False, past_key_values=None):
        a = np.asarray(ids)
        txt = self.reg[int(a[0, 0])]
        h, _ = self._vorwaerts(self._zustand(txt), a.shape[1])
        return types.SimpleNamespace(logits=t(np.array([[[h, 1., 2., 3., 4.]]])),
                                     past_key_values=object())

    def _antwort(self, text, w_ja):
        art = self.art_von(text)
        if art == "FAKT":
            land = re.search(r"capital city of (.+?)\?", text).group(1)
            if self.abruf_haengt and w_ja == 0.0:
                return "Nowhere"
            return self.staedte[land]
        if art == "SCHRIFT":
            return "| Service | 15 GB |"
        a, b = (int(x) for x in re.findall(r"(\d+) \* (\d+)", text)[0])
        richtig = a * b
        if art in ("MUL2", "MUL3") and w_ja == 0.0:
            # verrechnet sich, behaelt aber die Stellenzahl - so misst der Test
            # auch die Stelle-fuer-Stelle-Auswertung
            falsch = str(richtig)
            falsch = falsch[:-1] + str((int(falsch[-1]) + 3) % 10)
            return falsch
        return str(richtig)

    def generate(self, input_ids=None, attention_mask=None, **k):
        self.gen_kw.append(dict(k))
        b, L = np.asarray(input_ids).shape
        aus = np.zeros((b, L + 1))
        aus[:, :L] = np.asarray(input_ids)
        for j in range(b):
            txt = self.reg[int(np.asarray(input_ids)[j, 0])]
            _, w_ja = self._vorwaerts(self._zustand(txt))
            self.ausg.append(self._antwort(txt, w_ja))
            aus[j, L] = len(self.ausg) - 1
        return t(aus)

    def tok(self, text, return_tensors=None, padding=False, add_special_tokens=True):
        ts = [text] if isinstance(text, str) else list(text)
        self.reg.extend(ts)
        ids = np.zeros((len(ts), self.positionen))
        for j in range(len(ts)):
            ids[j, :] = len(self.reg) - len(ts) + j

        class E(dict):
            def to(self, *a, **k):
                return self

            @property
            def input_ids(self):
                return self["input_ids"]

        e = E({"input_ids": t(ids),
               "attention_mask": t(np.ones((len(ts), self.positionen)))})
        if return_tensors is None:
            e["input_ids"] = [int(x) for x in ids[0]]
        return e

    def decode(self, seq, skip_special_tokens=True):
        a = np.asarray(seq).ravel()
        if a.size == 1 and int(a[0]) < 0:
            return "?"
        return self.ausg[int(a[-1])] if a.size else ""


def hauptstaedte():
    """Die Sollantworten aus dem Notebook selbst holen, statt sie hier zu
       verdoppeln. Eine zweite Liste wuerde beim naechsten Umbau auseinander
       laufen und der Test wuerde still das Falsche pruefen."""
    q = koerper()
    a = q.index("HAUPTSTAEDTE=[")
    b = q.index("]", q.index('("Cambodia"'))
    return dict(ast.literal_eval(q[a + len("HAUPTSTAEDTE="):b + 1]))


def lauf(startwert=4, abruf_haengt=False, wiederholung=0):
    w = Welt(abruf_haengt=abruf_haengt)
    w.staedte = hauptstaedte()

    class Tok:
        pad_token_id = 0
        pad_token = None
        eos_token = 1
        padding_side = "right"

        def __call__(self, text, return_tensors=None, padding=False,
                     add_special_tokens=True):
            # Der Tokenisierungsbericht fragt reine Ziffernfolgen ab. Die
            # Miniatur zerlegt sie EINZELN - so wie es der Blogartikel fuer
            # Qwen 2.5 berichtet -, damit der Bericht im Test etwas zu melden
            # hat und die Stelle-fuer-Stelle-Auswertung greift.
            if return_tensors is None and isinstance(text, str) and text.isdigit():
                return types.SimpleNamespace(input_ids=[int(c) for c in text])
            return w.tok(text, return_tensors, padding, add_special_tokens)

        def decode(self, ids, skip_special_tokens=True):
            if isinstance(ids, list) and len(ids) == 1 and 0 <= ids[0] <= 9:
                return str(ids[0])
            return w.decode(ids, skip_special_tokens)

    tor = mach_torch()
    saaten = []
    _echt = tor.manual_seed
    tor.manual_seed = (lambda x: (saaten.append(int(x)), _echt(x))[1])
    ns = dict(os=os, re=re, math=math, torch=tor, collections=collections,
              unicodedata=unicodedata, random=random, np=np, glob=None, json=json,
              gc=gc, sys=sys, time=time,
              model=w, tokenizer=Tok(), PROMPTS={"p1": PROMPT}, ZIEL_ID="p1",
              N_AUF=24, MAX_NEW=8, CHUNK=8, SEED=5, N_PRUEF=4,
              WIEDERHOLUNG=wiederholung,
              wc_save=lambda name, obj: None, wc_save_all=lambda: None,
              RUN_OUT="/tmp")
    np.random.seed(startwert)
    try:
        exec(compile(koerper(), "phase13_rechnen", "exec"), ns)
    except SystemExit:
        pass
    ns["_SAATEN"] = saaten
    return w, ns


@pytest.fixture(scope="module")
def ergebnis():
    return lauf()


@pytest.fixture(scope="module")
def abrufwelt():
    """Zweite Welt: dort haengt auch das Hauptstadtwissen am Signalexperten.
       Die Abrufsperre muss greifen und der Lauf darf KEINE Rechenaussage
       machen, obwohl die Rechenzeilen genauso fallen wie sonst."""
    return lauf(abruf_haengt=True)


def test_zelle_laeuft_durch(ergebnis):
    _, ns = ergebnis
    assert "RECHNEN_RESULTS" in ns


def test_die_menge_kommt_unveraendert_aus_phase_12(ergebnis):
    """Die gepruefte Menge muss aus dem SCHRIFT-Prompt stammen und nicht aus
       den Rechenaufgaben - sonst waere der Uebertrag keiner."""
    w, ns = ergebnis
    R = ns["RECHNEN_RESULTS"]
    ja = {tuple(q) for q in R["ja_menge"]}
    assert ja == {(l, JA_E) for l in range(NLAY)}, sorted(ja)
    # und der Prompt, aus dem sie stammt, ist der Speicherdienst-Prompt
    assert R["prompt_id"] == "p1"
    assert any("cloud storage services" in x for x in w.reg[:4]), w.reg[:2]


def test_alle_vier_aufgaben_gemessen(ergebnis):
    _, ns = ergebnis
    R = ns["RECHNEN_RESULTS"]
    assert set(R["ergebnis"]) == {"FAKT", "MUL1", "MUL2", "MUL3"}
    assert R["ergebnis"]["FAKT"]["art"] == "capital"
    for a in ("MUL1", "MUL2", "MUL3"):
        assert R["ergebnis"][a]["art"] == "mul"
        assert R["ergebnis"][a]["n"] == 24


def test_aufgaben_sind_richtig_gestellt(ergebnis):
    """Die Sollantworten muessen stimmen - ein Tippfehler in der Aufgabenliste
       waere als Modellfehler nicht von einem Maskeneffekt zu unterscheiden."""
    _, ns = ergebnis
    bau = ns["aufgaben_bauen"]
    for schl, lo, hi in (("MUL1", 2, 9), ("MUL2", 12, 99), ("MUL3", 102, 999)):
        for frage, soll in bau(schl, 24, random.Random(3)):
            a, b = (int(x) for x in re.findall(r"(\d+) \* (\d+)", frage)[0])
            assert lo <= a <= hi and lo <= b <= hi, (schl, a, b)
            assert str(a * b) == soll, (frage, soll)
    staedte = dict(ns["HAUPTSTAEDTE"])
    for frage, soll in bau("FAKT", 24, random.Random(3)):
        land = re.search(r"capital city of (.+?)\?", frage).group(1)
        assert staedte[land] == soll
    # kein Land, dessen Hauptstadt im Landesnamen steckt - das waere geraten
    assert not [k for k, v in staedte.items() if v.lower() in k.lower()]


def test_konstruktion_faellt_abruf_nicht(ergebnis):
    _, ns = ergebnis
    R = ns["RECHNEN_RESULTS"]
    J = R["je_arm"]
    assert J["MUL2"] == "TRAEGT" and J["MUL3"] == "TRAEGT", J
    assert J["FAKT"] == "STILL" and J["MUL1"] == "STILL", J
    assert R["verdict"] == "KONSTRUKTION-ALLGEMEIN"
    for a in ("MUL2", "MUL3"):
        assert R["ergebnis"][a]["je_platz_ja"] > R["ergebnis"][a]["je_platz_zufall"]


def test_abrufsperre_haelt_den_lauf_an(abrufwelt):
    """Der Kern der Vorregistrierung: faellt das Hauptstadtwissen mit, ist der
       Eingriff unspezifisch - und dann darf KEINE Rechenaussage stehen, auch
       wenn die Rechenzeilen genauso aussehen wie im gelungenen Fall."""
    _, ns = abrufwelt
    R = ns["RECHNEN_RESULTS"]
    assert R["je_arm"]["FAKT"] in ("TRAEGT", "TRAEGT-UEBERWIEGEND"), R["je_arm"]
    assert R["verdict"] == "EINGRIFF-UNSPEZIFISCH", R["verdict"]
    # die Rechenarme fallen dort genauso - der Unterschied liegt allein an der Sperre
    assert R["je_arm"]["MUL2"] == "TRAEGT" and R["je_arm"]["MUL3"] == "TRAEGT"


def test_dosis_ueber_plaetze_angeglichen(ergebnis):
    _, ns = ergebnis
    R = ns["RECHNEN_RESULTS"]
    assert R["dosis_ok"] is True
    ja = {tuple(q) for q in R["ja_menge"]}
    for s_, z in R["zufall"].items():
        assert z["paare"] and z["plaetze"] > 0, s_
        assert abs(z["plaetze"] - z["ziel"]) <= 0.10 * z["ziel"] + 1e-9, s_
        assert not (ja & {tuple(q) for q in z["paare"]}), s_
    for s_, e in R["ergebnis"].items():
        assert e["plaetze_ja"] > 0 and e["plaetze_zufall"] > 0, s_
        assert abs(e["plaetze_ja"] - e["plaetze_zufall"]) <= 0.20 * e["plaetze_ja"]


def test_stellenauswertung(ergebnis):
    """Stelle fuer Stelle nur bei gleicher Stellenzahl. Die Miniatur verrechnet
       sich unter der Maske genau in EINER Stelle - also muss die Quote dort
       sinken, ohne dass die Stellenzahl selbst zusammenbricht."""
    _, ns = ergebnis
    st = ns["RECHNEN_RESULTS"]["ergebnis"]["MUL2"]["stellen"]
    rb, nb = st["basis"]
    rj, nj = st["ja"]
    rz, nz = st["zufall"]
    assert nb == nj == nz > 0, st
    assert rb == nb and rz == nz, st
    assert rj < rb, st
    assert ns["RECHNEN_RESULTS"]["ziffern_einzeln"] is True


def test_zahl_und_stadt_erkennen(ergebnis):
    _, ns = ergebnis
    ez, tz, ts = ns["erste_zahl"], ns["treffer_zahl"], ns["treffer_stadt"]
    assert ez("1,680") == "1680" and ez("1 680") == "1680"
    assert ez("The answer is 3,901.") == "3901"
    assert ez("no digits here") is None
    assert tz("168", "168") and not tz("167", "168")
    assert not tz("", "168")
    # eine laengere Zahl darf nicht als kuerzere durchgehen
    assert not tz("1680", "168")
    assert ts("Bogotá", "Bogota") and ts("the capital is paris", "Paris")
    assert not ts("Lima", "Bogota")
    zv = ns["ziffernvergleich"]
    assert zv("168", "168") == (3, 3) and zv("178", "168") == (2, 3)
    assert zv("68", "168") == (0, 0) and zv(None, "168") == (0, 0)


def test_urteilsordnung_rechnen(ergebnis):
    _, ns = ergebnis
    ur = ns["urteil_rechnen"]
    G = {"FAKT": "STILL", "MUL1": "STILL", "MUL2": "TRAEGT", "MUL3": "TRAEGT"}
    assert ur(G, False) == "DOSIS-NICHT-ANGEGLICHEN"
    assert ur(dict(G, FAKT="TRAEGT"), False) == "DOSIS-NICHT-ANGEGLICHEN"
    assert ur(G, True) == "KONSTRUKTION-ALLGEMEIN"
    assert ur(dict(G, MUL1="TRAEGT"), True) == "KONSTRUKTION-ALLGEMEIN"
    assert ur(dict(G, FAKT="TRAEGT"), True) == "EINGRIFF-UNSPEZIFISCH"
    assert ur(dict(G, FAKT="TRAEGT-UEBERWIEGEND"), True) == "EINGRIFF-UNSPEZIFISCH"
    assert ur(dict(G, FAKT="WIDERSPRUECHLICH"), True) == "DOSIS-SCHAEDIGT-ABRUF"
    assert ur(dict(G, MUL2="STILL", MUL3="STILL"), True) == "KONSTRUKTION-NUR-ZEICHEN"
    assert ur(dict(G, MUL2="NUR-STOERUNG"), True) == "ZERBRECHLICHKEIT"
    assert ur(dict(G, MUL2="STILL"), True) == "GEMISCHT"
    assert ur({"FAKT": "STILL"}, True) == "KEINE-RECHENARME"
    assert ur({"MUL2": "TRAEGT"}, True) == "KEINE-ABRUFKONTROLLE"


def test_saat_haengt_am_aufgabennamen(ergebnis):
    _, ns = ergebnis
    saat, benutzt = ns["saat"], set(ns["_SAATEN"])
    arten = [a for a, _, _ in ns["AUFGABEN"]]
    alle = [saat(z, a) for z in ("aufgaben", "basis", "ja", "zufall") for a in arten]
    assert len(set(alle)) == len(alle), "zwei Saaten fallen zusammen"
    for a in arten:
        assert saat("basis", a) in benutzt, a
        assert saat("ja", a) in benutzt, a
        assert saat("zufall", a) in benutzt, a


def test_wiederholung_verschiebt_alles():
    _, a = lauf()
    _, b = lauf(wiederholung=1)
    for z in ("aufgaben", "basis", "ja", "zufall"):
        for art in ("MUL2", "MUL3", "FAKT"):
            assert a["saat"](z, art) != b["saat"](z, art), (z, art)
    assert a["AUFG"]["MUL2"] != b["AUFG"]["MUL2"], "gleiche Aufgaben im Nachlauf"
    assert ({tuple(q) for q in a["RECHNEN_RESULTS"]["ja_menge"]}
            == {tuple(q) for q in b["RECHNEN_RESULTS"]["ja_menge"]}), \
        "die GEPRUEFTE Menge darf sich nicht mitverschieben"
    assert (a["RECHNEN_RESULTS"]["zufall"]["MUL2"]["paare"]
            != b["RECHNEN_RESULTS"]["zufall"]["MUL2"]["paare"])


def test_keine_haken_haengen(ergebnis):
    w, _ = ergebnis
    offen = sum(len(w.schichten[l]._haken) for l in range(NLAY))
    assert offen == 0, "%d Haken nicht entfernt" % offen


def test_immer_gierig_dekodiert(ergebnis):
    """Kein Sampling. Die Statistik kommt aus der Zahl der Aufgaben, nicht aus
       der Streuung - mit do_sample=True waere jede Trefferquote um eine
       Rauschquelle reicher, die niemand kontrolliert."""
    w, _ = ergebnis
    assert w.gen_kw, "gar nicht erzeugt"
    assert all(k.get("do_sample") is False for k in w.gen_kw), \
        [k.get("do_sample") for k in w.gen_kw[:5]]
    assert all(k.get("max_new_tokens") for k in w.gen_kw)


def test_armurteil_direkt(ergebnis):
    """urteil_arm_dosis steht in dieser Zelle noch einmal - also wird es hier
       auch noch einmal geprueft und nicht auf Phase 12 vertraut."""
    _, ns = ergebnis
    ua = ns["urteil_arm_dosis"]
    assert ua(60, 96, 10, 0.001, 58, 0.9, 264, 264) == "TRAEGT"
    assert ua(60, 96, 20, 0.001, 20, 0.001, 264, 264) == "NUR-STOERUNG"
    assert ua(60, 96, 10, 0.001, 50, 0.01, 264, 264) == "TRAEGT-UEBERWIEGEND"
    assert ua(61, 96, 15, 0.001, 42, 0.01, 264, 557) == "TRAEGT-UEBERWIEGEND"
    assert ua(60, 96, 58, 0.9, 10, 0.001, 264, 264) == "WIDERSPRUECHLICH"
    assert ua(60, 96, 58, 0.9, 59, 0.9, 264, 264) == "STILL"
    # ein ANSTIEG ist keine Senkung, auch wenn er hochsignifikant ist
    assert ua(10, 96, 60, 0.001, 61, 0.001, 264, 264) == "STILL"
    assert ua(10, 96, 60, 0.001, 11, 0.9, 264, 264) == "STILL"
    f = ns["wirkung_je_platz"]
    assert abs(f(96, 0, 96, 100) - 100.0) < 1e-9
    assert abs(f(96, 0, 96, 200) - 50.0) < 1e-9
    assert f(96, 0, 96, 0) == 0.0 and f(10, 60, 96, 100) < 0
