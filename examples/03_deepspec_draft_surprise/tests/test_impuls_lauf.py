"""Der Ausfuehrungsteil von phase17_impuls laeuft hier wirklich durch - gegen
ein Miniatur-MoE aus numpy. Kein Torch, keine GPU.

Diese Welt muss etwas koennen, was die bisherigen nicht mussten: ZEITMUSTER
erzeugen. Der Tokenizer gibt deshalb ein Token je ZEICHEN aus, damit es
ueberhaupt Antwortpositionen mit unterscheidbaren Zeichenklassen gibt.

Fuenf Muster, und jedes spannt genau eines der Verdikte auf:

  zufall   jedes Paar feuert unabhaengig mit fester Rate      -> KEIN-IMPULS
  takt     Paar i feuert bei (p - i) mod TAKT == 0            -> GETAKTET
  zeichen  feuert fast nur auf Tokens der Zielschrift         -> ZEICHENGEBUNDEN
  schub    feuert geballt in jedem dritten Block              -> SCHUBWEISE
  gruppe   alle Paare feuern an DENSELBEN zufaelligen Stellen -> GRUPPIERT

Der Bezugsarm laeuft in jeder Welt auf 'zufall'. Ohne das koennte kein Test
bemerken, ob die Zelle den Bezugsarm ueberhaupt abzieht - ein Muster, das in
beiden Armen steht, ist keine Eigenschaft der Konstruktion.

Die Vergleichsexperten VERGL_EXP feuern in JEDEM Arm mit derselben Rate wie
die Menge. Ohne sie faende ratengleiche() keine Partner, und die Zelle
brauchte gar nicht erst zu rechnen.
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
NB = os.path.join(HIER, "..", "phase17_impuls.ipynb")

HID, INTER, NEXP, NLAY, TOPK = 6, 8, 64, 4, 10
POSITIONEN = 40          # Promptlaenge in Tokens
Traeger = haken_traeger()

MENGE_EXP = (41, 42, 43, 44)
MENGE_WELT = [(l, e) for l in range(NLAY) for e in MENGE_EXP]
VERGL_EXP = (50, 51, 52, 53, 54, 55)     # ratengleiche Partner
RATE = 0.2
TAKT = 5
BLOCK = 6

BASIS_TEXT = 100000      # Prompt-Tokens tragen die Textnummer
BASIS_ANTW = 200000      # generate() gibt die Antwortnummer zurueck
BASIS_ZEICHEN = 1000     # Antwort-Tokens tragen ein Zeichen

PROMPT = ("Create a markdown table comparing five cloud storage services with their "
          "storage limits and pricing. label the column with each service's local "
          "name, and summarize each entry.")
ZIELTEXT = {
    "NEU": "| Google Drive | 15 GB | free |\n| Dropbox | 2 GB | free |\n",
    "JA": "| サービス名 | 保存容量 | 料金 |\n| グーグルドライブ | 15 GB | 無料 |\n",
    "SR": "| Гугл драјв | 15 GB | бесплатно |\n| Дропбокс | 2 GB |\n",
    "RU": "| Гугл драйв | 15 GB | бесплатно |\n| Дропбокс | 2 GB |\n",
    "BR1": "| ⠛⠕⠕⠛⠇⠑ ⠙⠗⠊⠧⠑ | 15 GB |\n| ⠙⠗⠕⠏⠃⠕⠭ ⠋⠊⠇⠑⠎ | 2 GB |\n",
    "MORSE": "| --. --- --- --. .-.. . | 15 GB |\n| -.. .-. --- .--. | 2 GB |\n"}
FEHLTEXT = "| Service Name | Storage Limit | Price |\n"
FEHLJEDES = 5
ZIEL_KLASSE = {"NEU": "latein", "JA": "kana", "SR": "kyrillisch",
               "RU": "kyrillisch", "BR1": "braille", "MORSE": "punkt"}


def koerper():
    with open(NB, encoding="utf-8") as f:
        nb = json.load(f)
    quelle = "".join("".join(c.get("source", [])) for c in nb["cells"]
                     if c.get("cell_type") == "code")
    marke = "# ---------------- reine Logik"
    assert marke in quelle, "Marke fuer den Logikteil fehlt im Notebook"
    return quelle[quelle.index(marke):]


def klasse_von(c):
    o = ord(c)
    if 0x2800 <= o <= 0x28FF:
        return "braille"
    if 0x3040 <= o <= 0x30FF:
        return "kana"
    if 0x0400 <= o <= 0x052F:
        return "kyrillisch"
    if c in ".-/":
        return "punkt"
    if c.isdigit():
        return "zahl"
    if c.isalpha():
        return "latein"
    if c.isspace():
        return "leer"
    return "sonst"


class Experten(Traeger):
    def __init__(self, rs):
        Traeger.__init__(self)
        self.gate_up_proj = t(rs.randn(NEXP, 2 * INTER, HID) * 0.5)
        self.down_proj = t(rs.randn(NEXP, HID, INTER) * 0.5)
        self.intermediate_dim = INTER


class Welt:
    def __init__(self, startwert=4711, muster="zufall"):
        rs = np.random.RandomState(startwert)
        self.schichten = {l: Experten(rs) for l in range(NLAY)}
        self.reg = []
        self.ausg = []
        self.muster = muster
        self.zeichen = {}
        self.zeichen_rueck = {}
        self.zaehler = collections.Counter()

    def zeichen_id(self, c):
        if c not in self.zeichen:
            i = BASIS_ZEICHEN + len(self.zeichen)
            self.zeichen[c] = i
            self.zeichen_rueck[i] = c
        return self.zeichen[c]

    def zerlege(self, text):
        """Prompt und Antwort trennen. Die Antworten kennt die Welt, weil sie
           sie selbst erzeugt hat."""
        for a in reversed(self.ausg):
            if a and text.endswith(a):
                return text[:-len(a)], a
        return text, ""

    @staticmethod
    def arm_von(text):
        if FEHLTEXT in text:
            return "NEU"
        for k in ("Japanese", "Serbian", "Russian", "Braille", "Morse"):
            if k in text:
                return {"Japanese": "JA", "Serbian": "SR", "Russian": "RU",
                        "Braille": "BR1", "Morse": "MORSE"}[k]
        return "NEU"

    @staticmethod
    def _r(text, p, l, was):
        h = 2166136261
        for c in ("%s/%d/%d/%s" % (text, p, l, was)).encode():
            h = ((h ^ c) * 16777619) & 0xFFFFFFFF
        return random.Random(h)

    def _menge_feuert(self, arm, muster, i, p, klasse, ziel, text, l):
        """Feuert Paar i der Menge an Antwortposition p?"""
        r = self._r(text, p, l, "m%d" % i)
        if muster == "takt":
            return (p - i) % TAKT == 0
        if muster == "zeichen":
            return r.random() < (0.85 if klasse == ziel else 0.02)
        if muster == "schub":
            return r.random() < (0.85 if (p // BLOCK) % 3 == 0 else 0.02)
        if muster == "gruppe":
            # DERSELBE Wurf fuer alle Paare, auch ueber die Schichten hinweg -
            # sonst gaebe es je Schicht eine eigene Gruppe und nie eine, die
            # die ganze Menge umfasst.
            return self._r(text, p, 0, "g").random() < RATE
        return r.random() < RATE

    def _idx(self, text, n_pos):
        arm = self.arm_von(text)
        prompt, antwort = self.zerlege(text)
        muster = self.muster if arm in ("JA", "BR1", "MORSE") else "zufall"
        ziel = ZIEL_KLASSE.get(arm, "latein")
        rest = [e for e in range(NEXP)
                if e not in MENGE_EXP and e not in VERGL_EXP]
        if n_pos == 1:
            # Die Entscheidungsstelle. Die Differenz JA-minus-NEU ergibt hier
            # genau MENGE_WELT - das ist die Menge, die die Zelle herleitet.
            fest = (list(MENGE_EXP) + rest[:TOPK - len(MENGE_EXP)]
                    if arm != "NEU" else rest[:TOPK])
            return [[sorted(fest) for _ in range(NLAY)]]
        muster_alle = []
        for p in range(n_pos):
            ap = p - POSITIONEN            # Position innerhalb der Antwort
            kl = klasse_von(antwort[ap]) if 0 <= ap < len(antwort) else "leer"
            zeile = []
            for l in range(NLAY):
                pick = set()
                if ap >= 0:
                    for i, e in enumerate(MENGE_EXP):
                        if self._menge_feuert(arm, muster, i, ap, kl, ziel, text, l):
                            pick.add(e)
                else:
                    for i, e in enumerate(MENGE_EXP):
                        if self._r(text, p, l, "pm%d" % i).random() < RATE:
                            pick.add(e)
                # Die Vergleichsexperten feuern in JEDEM Arm gleich - sie sind
                # der ratengleiche Boden und duerfen kein Muster tragen.
                for e in VERGL_EXP:
                    if self._r(text, p, l, "v%d" % e).random() < RATE:
                        pick.add(e)
                r = self._r(text, p, l, "f")
                while len(pick) < TOPK:
                    pick.add(r.choice(rest))
                zeile.append(sorted(pick)[:TOPK])
            muster_alle.append(zeile)
        return muster_alle

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
        a = np.asarray(ids).astype(int)
        txt = self.reg[int(a[0, 0]) - BASIS_TEXT]
        muster = self._idx(txt, a.shape[1])
        x = np.eye(HID)[0]
        for l in range(NLAY):
            idx = t(np.array([[z[l] for z in muster]], dtype=float))
            w0 = t(np.full((1, len(muster), TOPK), 0.5))
            self.schichten[l].feuere(t(x.reshape(1, 1, HID)), idx, w0)
        return types.SimpleNamespace(logits=t(np.zeros((1, 1, 5))),
                                     past_key_values=object())

    def generate(self, input_ids=None, attention_mask=None, **k):
        b, L = np.asarray(input_ids).shape
        txt = self.reg[int(np.asarray(input_ids)[0, 0]) - BASIS_TEXT]
        arm = self.arm_von(txt)
        aus = np.zeros((b, L + 1))
        aus[:, :L] = np.asarray(input_ids)
        for j in range(b):
            i = self.zaehler[arm]
            self.zaehler[arm] += 1
            # Die laufende Nummer macht jede Antwort zu einem EIGENEN Text.
            # Ohne sie routeten alle Beispiele eines Arms gleich, und jede
            # Nullverteilung fiele in sich zusammen.
            self.ausg.append((FEHLTEXT if i % FEHLJEDES == 0 else ZIELTEXT[arm])
                             + " #%d\n" % i)
            aus[j, L] = BASIS_ANTW + len(self.ausg) - 1
        return t(aus)

    def tok(self, text, return_tensors=None, padding=False, add_special_tokens=True):
        ts = [text] if isinstance(text, str) else list(text)
        self.reg.extend(ts)
        zeilen = []
        for j, tt in enumerate(ts):
            nr = BASIS_TEXT + len(self.reg) - len(ts) + j
            _, antwort = self.zerlege(tt)
            zeilen.append([nr] * POSITIONEN
                          + [self.zeichen_id(c) for c in antwort])
        breite = max(len(z) for z in zeilen)
        ids = np.zeros((len(ts), breite))
        for j, z in enumerate(zeilen):
            ids[j, :len(z)] = z
            ids[j, len(z):] = z[0]

        class E(dict):
            def to(self, *a, **k):
                return self

            @property
            def input_ids(self):
                return self["input_ids"]

        return E({"input_ids": t(ids), "attention_mask": t(np.ones((len(ts), breite)))})

    def decode(self, seq, skip_special_tokens=True):
        a = np.asarray(seq).ravel().astype(int)
        if a.size == 0:
            return ""
        if int(a[-1]) >= BASIS_ANTW:
            return self.ausg[int(a[-1]) - BASIS_ANTW]
        return "".join(self.zeichen_rueck.get(int(x), "") for x in a)


def lauf(muster="zufall", wiederholung=0, n_bsp=20, n_zug=8, perm=30, maxlag=8):
    w = Welt(muster=muster)

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
              gc=gc, sys=sys, time=time, bisect=__import__("bisect"),
              model=w, tokenizer=Tok(), PROMPTS={"p1": PROMPT}, ZIEL_ID="p1",
              N_BSP=n_bsp, N_ZUG=n_zug, PERM=perm, MAXLAG=maxlag, MAX_NEW=8,
              CHUNK=8, TEMP=1.0, SEED=5, MIN_BSP=6, MIN_LAENGE=20, MIN_ZUEGE=4,
              WIEDERHOLUNG=wiederholung,
              wc_save=lambda name, obj: None, wc_save_all=lambda: None,
              RUN_OUT="/tmp")
    np.random.seed(7)
    try:
        exec(compile(koerper(), "phase17_impuls", "exec"), ns)
    except SystemExit:
        pass
    return w, ns


KONSTR_ARME = ("JA", "BR1", "MORSE")


@pytest.fixture(scope="module")
def zufall():
    return lauf("zufall")


@pytest.fixture(scope="module")
def takt():
    return lauf("takt")


@pytest.fixture(scope="module")
def zeichen():
    return lauf("zeichen")


@pytest.fixture(scope="module")
def schub():
    return lauf("schub")


@pytest.fixture(scope="module")
def gruppe():
    return lauf("gruppe")


def test_zelle_laeuft_durch(zufall):
    _, ns = zufall
    assert "IMPULS_RESULTS" in ns


def test_menge_wird_selbst_hergeleitet(zufall):
    _, ns = zufall
    R = ns["IMPULS_RESULTS"]
    assert {tuple(q) for q in R["menge"]} == set(MENGE_WELT)


@pytest.mark.parametrize("welt,erwartet", [
    ("zufall", "KEIN-IMPULS"),
    ("takt", "GETAKTET"),
    ("zeichen", "ZEICHENGEBUNDEN"),
    ("schub", "SCHUBWEISE"),
    ("gruppe", "GRUPPIERT"),
])
def test_alle_muster(request, welt, erwartet):
    """Jedes Muster muss erkannt werden - und KEIN-IMPULS ist der wichtigste
       Fall: eine Zelle, die in reinem Zufall etwas findet, findet ueberall
       etwas."""
    _, ns = request.getfixturevalue(welt)
    assert ns["IMPULS_RESULTS"]["verdict"] == erwartet, ns["IMPULS_RESULTS"]["verdict"]


def test_bezugsarm_zeigt_nie_das_muster(takt, zeichen, schub, gruppe):
    """Der Bezugsarm laeuft in jeder Welt auf 'zufall'. Zeigt er trotzdem den
       Befund, zieht die Zelle ihn nicht ab - und dann waere jedes Muster, das
       schon im englischen Arm steht, faelschlich ein Befund."""
    for _, ns in (takt, zeichen, schub, gruppe):
        R = ns["IMPULS_RESULTS"]
        assert R["form"]["NEU"]["form"] == "GEDAECHTNISLOS", R["form"]["NEU"]
        assert R["zeichen"]["NEU"]["befund"] is False, R["zeichen"]["NEU"]


def test_takt_findet_die_periode(takt):
    """Der Takt liegt bei %d - die Autokorrelation muss ihren Gipfel dort oder
       bei einem Vielfachen haben.""" % TAKT
    _, ns = takt
    R = ns["IMPULS_RESULTS"]
    for a in R["konstruierend"]:
        lag = R["form"][a]["lag"]
        assert lag is not None and lag % TAKT == 0, (a, lag)
        assert R["form"][a]["form"] == "GETAKTET", (a, R["form"][a])
        # regelmaessiger als die Null, nicht nur anders
        assert R["form"][a]["cv"] < R["form"][a]["unten"], (a, R["form"][a])


def test_zeichenbindung_zeigt_die_zielklasse(zeichen):
    _, ns = zeichen
    R = ns["IMPULS_RESULTS"]
    for a in R["konstruierend"]:
        d = R["zeichen"][a]
        assert d["ziel"] if "ziel" in d else True
        assert d["beob"] > d["null"], (a, d)
        assert d["beob"] > d["vergleich"], (a, d)
        assert d["beob"] > 1.5, (a, d["beob"])


def test_gruppen_nur_wo_gemeinsam_gefeuert_wird(gruppe, zufall):
    """Im Gruppenmuster feuern alle Paare an denselben Stellen - die groesste
       Gruppe muss die ganze Menge sein. Im Zufallsmuster darf es keine
       geben."""
    _, ns = gruppe
    R = ns["IMPULS_RESULTS"]
    for a in R["konstruierend"]:
        assert R["gruppen"][a]["groesste"] == len(MENGE_WELT), (a, R["gruppen"][a])
        assert R["gruppen"][a]["median"] > 2.0, (a, R["gruppen"][a]["median"])
    _, ns = zufall
    R = ns["IMPULS_RESULTS"]
    for a in R["konstruierend"]:
        # Bei 120 Paaren und einer Schwelle beim 99. Perzentil sind ein bis
        # zwei zufaellige Kanten zu erwarten - eine Kette daraus bleibt klein.
        assert R["gruppen"][a]["groesste"] <= 3, (a, R["gruppen"][a])


def test_ratengleiche_partner_gefunden(zufall):
    """Ohne ratengleiche Partner gibt es keinen Boden - und die Zelle bricht
       dann ab, statt eine Zahl ohne Vergleich zu melden."""
    _, ns = zufall
    R = ns["IMPULS_RESULTS"]
    assert len(R["paarung"]) >= len(MENGE_WELT) - 2, len(R["paarung"])
    v = {tuple(q) for q in R["vergleich"]}
    assert not (v & set(MENGE_WELT)), sorted(v & set(MENGE_WELT))
    # Die Partner muessen RATENGLEICH sein - aus welchem Topf sie kommen, ist
    # gleichgueltig. Ueberwiegend sind es die eigens gebauten, aber auch die
    # Auffueller liegen nah genug, und bei knapper Auswahl gewinnt einer davon.
    aus_vergl = sum(1 for q in v if q[1] in VERGL_EXP)
    assert aus_vergl >= len(v) // 2, (aus_vergl, sorted(v))
    for q in v:
        r_menge = R["raten"]["NEU"]["%d_%d" % tuple(MENGE_WELT[0])]
        r_part = R["raten"]["NEU"].get("%d_%d" % q)
        if r_part is not None:
            assert abs(r_part - r_menge) <= 0.35 * r_menge, (q, r_part, r_menge)


def test_schub_ist_kein_takt(schub):
    """Ein Ballen ist auf kurze Abstaende positiv autokorreliert - innerhalb
       eines Ballens sitzen die Treffer dicht beieinander. Ohne Sperre hiesse
       jeder Ballen 'getaktet', und die beiden Befunde waeren nicht mehr zu
       unterscheiden."""
    _, ns = schub
    R = ns["IMPULS_RESULTS"]
    for a in R["konstruierend"]:
        assert R["befunde"]["schub"][a] is True, (a, R["form"][a])
        assert R["befunde"]["takt"][a] is False, (a, R["form"][a])
        assert R["form"][a]["cv"] > R["form"][a]["oben"], (a, R["form"][a])


def test_befund_takt_direkt(zufall):
    _, ns = zufall
    bt = ns["befund_takt"]
    bs = ns["befund_schub"]
    ruhe = dict(form="GEDAECHTNISLOS", gipfel=0.01, gipfel_null=0.05)
    # regelmaessiger als die Null
    assert bt(dict(form="GETAKTET", gipfel=0.0, gipfel_null=0.5), ruhe) is True
    # Gipfel ueber der Null
    assert bt(dict(form="GEDAECHTNISLOS", gipfel=0.30, gipfel_null=0.05), ruhe) is True
    # ... aber nicht, wenn zugleich schubweise gefeuert wird
    assert bt(dict(form="SCHUBWEISE", gipfel=0.30, gipfel_null=0.05), ruhe) is False
    # ... und nicht, wenn der Bezugsarm dasselbe zeigt
    assert bt(dict(form="GEDAECHTNISLOS", gipfel=0.30, gipfel_null=0.05),
              dict(form="GEDAECHTNISLOS", gipfel=0.40, gipfel_null=0.05)) is False
    assert bt(dict(form="GETAKTET", gipfel=0.0, gipfel_null=0.5),
              dict(form="GETAKTET", gipfel=0.0, gipfel_null=0.5)) is False
    assert bs(dict(form="SCHUBWEISE"), ruhe) is True
    assert bs(dict(form="SCHUBWEISE"), dict(form="SCHUBWEISE")) is False


def test_urteilsordnung_impuls(zufall):
    """Zeichenbindung steht VOR dem Takt: wiederholen sich die Zielzeichen
       regelmaessig, ist ein zeichengebundener Zug zwangslaeufig auch
       periodisch. Der Takt waere dann kein eigener Befund."""
    _, ns = zufall
    ui = ns["urteil_impuls"]
    A = KONSTR_ARME
    kein = {k: {a: False for a in A} for k in ("takt", "zeichen", "schub", "gruppen")}

    def mit(**kw):
        d = {k: dict(kein[k]) for k in kein}
        for k, wert in kw.items():
            d[k] = {a: wert for a in A}
        d["zuege"] = 8
        return d

    assert ui(mit(), A) == "KEIN-IMPULS"
    assert ui(mit(takt=True), A) == "GETAKTET"
    assert ui(mit(zeichen=True), A) == "ZEICHENGEBUNDEN"
    assert ui(mit(schub=True), A) == "SCHUBWEISE"
    assert ui(mit(gruppen=True), A) == "GRUPPIERT"
    assert ui(mit(takt=True, zeichen=True), A) == "ZEICHENGEBUNDEN"
    assert ui(mit(takt=True, schub=True, gruppen=True), A) == "GETAKTET"
    # zu wenige Aufnahmen stechen alles
    d = mit(takt=True); d["zuege"] = 2
    assert ui(d, A) == "ZU-WENIG-SIGNAL"
    # ein Arm unter dreien ist kein Befund
    d = mit(); d["takt"] = {"JA": True, "BR1": False, "MORSE": False}
    assert ui(d, A) == "KEIN-IMPULS"
    d["takt"]["BR1"] = True
    assert ui(d, A) == "GETAKTET"


def test_keine_haken_haengen(zufall):
    w, _ = zufall
    offen = sum(len(w.schichten[l]._haken) for l in range(NLAY))
    assert offen == 0, "%d Haken nicht entfernt" % offen
