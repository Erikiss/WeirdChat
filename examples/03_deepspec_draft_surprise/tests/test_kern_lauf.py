"""Der Ausfuehrungsteil von phase18_kern laeuft hier wirklich durch - gegen
ein Miniatur-MoE aus numpy. Kein Torch, keine GPU.

Die Welt hat eine eingebaute TRAEGERMENGE innerhalb der gepruefigen Menge:
die ersten n_traeger Paare tragen das Verhalten, die uebrigen laufen mit. So
sind alle Ausgaenge erreichbar und einzeln pruefbar:

    n_traeger = 1                 -> EINZELNER-TRAEGT
    n_traeger = 3                 -> KLEINER-KERN
    alle_gleich = True            -> KEIN-KERN   (jedes Paar traegt ein Stueck,
                                     keines allein genug)
    traegt_menge = False          -> POSITIVKONTROLLE-FEHLT

Die fremden Experten tragen NIE. Sie sind die empirische Null, und wenn ein
Test sie treffen sieht, misst der Scan seine eigene Mehrfachtestung.
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
NB = os.path.join(HIER, "..", "phase18_kern.ipynb")

HID, INTER, NEXP, NLAY, TOPK = 6, 8, 96, 4, 4
POSITIONEN = 40
ANTWORT = 40
ABLESEN = 40   # Positionen, ueber die generate() nachsieht, was gesperrt ist
Traeger = haken_traeger()

# Die gepruefte Menge: vier Experten je Schicht, die den Japanisch-Arm an der
# Entscheidungsstelle vom Bezugsarm unterscheiden. 16 Paare.
MENGE_EXP = (41, 42, 43, 44)
MENGE_WELT = [(l, e) for l in range(NLAY) for e in MENGE_EXP]
# Wer davon traegt wirklich? Sortiert, damit die Reihenfolge nicht an der
# Ziehung haengt - die Zelle scannt ohnehin alle einzeln.
TRAEGER = sorted(MENGE_WELT)
# Fremde Experten mit derselben Feuerrate - die Kandidaten der empirischen
# Null. Sie tragen nie.
PARTNER_EXP = (50, 51, 52, 53)

PROMPT = ("Create a markdown table comparing five cloud storage services with their "
          "storage limits and pricing. label the column with each service's local "
          "name, and summarize each entry.")
ZIELTEXT = {"NEU": "| Service | 15 GB |",
            "JA": "| サービス名 | 保存容量 | 料金 |",
            "SR": "| Услуга | Ограничење | Цена |",
            "RU": "| Услуга | Ограничение | Цена |",
            "BR1": "| ⠛⠕⠕⠛⠇⠑ ⠙⠗⠊⠧⠑ | 15 GB |",
            "MORSE": "| --. --- --- --. .-.. . | 15 GB |"}
FEHLTEXT = "| Service Name | Storage Limit | Price |"   # trifft kein Zielmass
FEHLJEDES = 4   # jede vierte Antwort geht daneben - prueft den Zielmass-Filter


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
    def __init__(self, startwert=4711, n_traeger=1, traegt_menge=True,
                 alle_gleich=False):
        rs = np.random.RandomState(startwert)
        self.schichten = {l: Experten(rs) for l in range(NLAY)}
        self.reg = []
        self.ausg = []
        self.n_traeger = n_traeger
        self.traegt_menge = traegt_menge
        self.alle_gleich = alle_gleich
        self.gesperrt = set()
        self.zaehler = collections.Counter()
        self.grund = [0.05 + 3.0 * (i % 6 == 0) + 0.4 * (i % 3 == 0) for i in range(NEXP)]
        # Die Menge laeuft in JEDEM Arm haeufig - so wie die echten 42, die
        # auch dort feuern, wo sie nichts zu tun haben. Nebeneffekt und
        # Absicht: ein gesperrtes Paar wird dadurch zuverlaessig bemerkt, und
        # die Stufenzahl stimmt. Wuerde eines uebersehen, verschoebe sich der
        # gemessene Anteil um ein Sechzehntel.
        for e in MENGE_EXP:
            self.grund[e] = 6.0
        # Ebenso viele FREMDE Experten mit derselben Rate. Ohne sie findet die
        # Ratenanpassung keine Partner: die Menge feuert dann weit ueber allem
        # anderen, und die Vergleichsmenge waere aus lauter seltenen Paaren
        # gebaut - eine Null fuer eine andere Haeufigkeit.
        for e in PARTNER_EXP:
            self.grund[e] = 6.0

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
    def _wuerfel(text, p, l):
        """Routing als FUNKTION DES TEXTES, deterministisch je Position und
           Schicht - sonst wichen geplante und gemessene Plaetze voneinander
           ab, ohne dass eine der beiden Zahlen falsch aussaehe."""
        h = 2166136261
        for c in ("%s/%d/%d" % (text, p, l)).encode():
            h = ((h ^ c) * 16777619) & 0xFFFFFFFF
        return random.Random(h)

    def _idx(self, arm, n_pos, text=""):
        if n_pos == 1:
            # Die eine Position, die hole_routing() liest. Die Differenz
            # JA-minus-NEU ergibt hier genau MENGE_WELT.
            fremd = [e for e in range(NEXP) if e not in MENGE_EXP]
            fest = list(MENGE_EXP) if arm != "NEU" else fremd[:TOPK]
            return [[sorted(fest[:TOPK]) for _ in range(NLAY)]]
        muster = []
        for p in range(n_pos):
            zeile = []
            for l in range(NLAY):
                r = self._wuerfel(text, p, l)
                pick = set()
                while len(pick) < TOPK:
                    pick.add(r.choices(range(NEXP), weights=self.grund)[0])
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

    def _fahre(self, arm, n_pos, text=""):
        muster = self._idx(arm, n_pos, text)
        x = np.eye(HID)[0]
        self.gesperrt = set()
        for l in range(NLAY):
            idx = t(np.array([[z[l] for z in muster]], dtype=float))
            w0 = t(np.full((1, len(muster), TOPK), 0.5))
            _, i_n, w_n = self.schichten[l].feuere(t(x.reshape(1, 1, HID)), idx, w0)
            wa = np.asarray(w_n).reshape(-1)
            ia = np.asarray(i_n).reshape(-1)
            for wert, e in zip(wa, ia):
                if wert == 0.0:
                    self.gesperrt.add((l, int(e)))

    def __call__(self, ids, use_cache=False, past_key_values=None):
        a = np.asarray(ids)
        txt = self.reg[int(a[0, 0])]
        self._fahre(self.arm_von(txt), a.shape[1], txt)
        return types.SimpleNamespace(logits=t(np.zeros((1, 1, 5))),
                                     past_key_values=object())

    def _anteil(self, arm):
        """Welcher Bruchteil der Antworten bricht?

           Nur die konstruierenden Arme haengen an der Menge - Serbisch und
           Russisch bleiben unberuehrt, so wie in Phase 12, 15 und 16
           gemessen. Fremde Experten wirken NIE: sie sind die empirische
           Null."""
        if arm not in ("JA", "BR1", "MORSE"):
            return 0.0
        if not self.traegt_menge:
            return 0.0
        g = self.gesperrt & set(MENGE_WELT)
        if self.alle_gleich:
            # jedes Paar traegt ein Sechzehntel - keines allein kommt weit
            return len(g) / float(len(MENGE_WELT))
        # Ein Traeger allein bringt schon drei Viertel, zwei bringen alles.
        # Nicht t/n: bei drei gleichgewichtigen Traegern gaebe ein einzelner
        # nur ein Drittel, und das ist bei %d Ziehungen nicht signifikant zu
        # messen - der Scan koennte die eingebaute Wahrheit gar nicht finden,
        # und der Test pruefte dann die Stichprobengroesse statt die Zelle.
        t = len(g & set(TRAEGER[:self.n_traeger]))
        return min(1.0, 0.75 * t)

    def generate(self, input_ids=None, attention_mask=None, **k):
        b, L = np.asarray(input_ids).shape
        txt = self.reg[int(np.asarray(input_ids)[0, 0])]
        arm = self.arm_von(txt)
        self._fahre(arm, ABLESEN, txt)          # macht die Maske sichtbar
        p = self._anteil(arm)
        X = int(round(100.0 * p))
        schl = (arm, tuple(sorted(self.gesperrt & set(MENGE_WELT))))
        aus = np.zeros((b, L + 1))
        aus[:, :L] = np.asarray(input_ids)
        for j in range(b):
            i = self.zaehler[schl]
            self.zaehler[schl] += 1
            if i % FEHLJEDES == 0:
                text = FEHLTEXT
            else:
                jj = i - i // FEHLJEDES - 1
                text = FEHLTEXT if ((jj + 1) * X) // 100 > (jj * X) // 100 \
                    else ZIELTEXT[arm]
            self.ausg.append(text + " #%d" % len(self.ausg))
            aus[j, L] = len(self.ausg) - 1
        return t(aus)

    def tok(self, text, return_tensors=None, padding=False, add_special_tokens=True):
        ts = [text] if isinstance(text, str) else list(text)
        self.reg.extend(ts)
        hat_antwort = any(z in ts[0] for z in list(ZIELTEXT.values()) + [FEHLTEXT])
        breite = POSITIONEN + (ANTWORT if hat_antwort else 0)
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



def lauf(n_traeger=1, traegt_menge=True, alle_gleich=False, wiederholung=0,
         n_bsp=24, n_scan=20, n_abl=24):
    w = Welt(n_traeger=n_traeger, traegt_menge=traegt_menge, alle_gleich=alle_gleich)

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
              N_BSP=n_bsp, N_SCAN=n_scan, N_ABL=n_abl, MAX_NEW=8, CHUNK=8,
              TEMP=1.0, SEED=5, MIN_BSP=6, N_PRUEF=4, MAX_KERN=8,
              WIEDERHOLUNG=wiederholung,
              wc_save=lambda name, obj: None, wc_save_all=lambda: None,
              RUN_OUT="/tmp")
    np.random.seed(7)
    try:
        exec(compile(koerper(), "phase18_kern", "exec"), ns)
    except SystemExit:
        pass
    return w, ns


KONSTR_ARME = ("JA", "BR1", "MORSE")
LEX_ARME = ("SR", "RU")


@pytest.fixture(scope="module")
def einer():
    return lauf(n_traeger=1)


@pytest.fixture(scope="module")
def dreie():
    return lauf(n_traeger=3)


@pytest.fixture(scope="module")
def verteilt():
    return lauf(alle_gleich=True)


@pytest.fixture(scope="module")
def tote_kette():
    return lauf(traegt_menge=False)


def test_zelle_laeuft_durch(einer):
    _, ns = einer
    assert "KERN_RESULTS" in ns


def test_menge_wird_selbst_hergeleitet(einer):
    _, ns = einer
    assert {tuple(q) for q in ns["KERN_RESULTS"]["menge"]} == set(MENGE_WELT)


@pytest.mark.parametrize("welt,erwartet", [
    ("einer", "EINZELNER-TRAEGT"),
    ("dreie", "KLEINER-KERN"),
    ("verteilt", "KEIN-KERN"),
    ("tote_kette", "POSITIVKONTROLLE-FEHLT"),
])
def test_alle_ausgaenge(request, welt, erwartet):
    """Jeder Ausgang muss erreichbar sein und der eingebauten Wahrheit
       entsprechen. KEIN-KERN ist der Fall, in dem jedes Paar ein Stueck
       traegt und keines allein genug - dort DARF der Scan nichts finden."""
    _, ns = request.getfixturevalue(welt)
    assert ns["KERN_RESULTS"]["verdict"] == erwartet, ns["KERN_RESULTS"]["verdict"]


def test_kern_ist_die_eingebaute_wahrheit(einer, dreie):
    """Der Scan muss GENAU die Traeger finden - nicht mehr und nicht weniger.
       Findet er zusaetzliche, ist die Null zu tief; fehlen welche, zu hoch."""
    for (_, ns), n in ((einer, 1), (dreie, 3)):
        kern = {tuple(q) for q in ns["KERN_RESULTS"]["kern"]}
        assert kern == set(TRAEGER[:n]), sorted(kern ^ set(TRAEGER[:n]))


def test_fremde_paare_treffen_nie(einer, dreie, verteilt):
    """Die empirische Null darf keinen einzigen Treffer haben - die fremden
       Experten wirken in dieser Welt nicht. Trifft sie doch, misst der Scan
       seine eigene Mehrfachtestung."""
    for _, ns in (einer, dreie, verteilt):
        R = ns["KERN_RESULTS"]
        assert R["aussen_treffer"] == [], R["aussen_treffer"]
        aussen = {tuple(q) for q in R["aussen"]}
        assert not (aussen & set(MENGE_WELT)), sorted(aussen & set(MENGE_WELT))
        assert len(aussen) >= 8, len(aussen)


def test_rest_traegt_nicht(einer, dreie):
    """Der Gegenbeweis zum Kern: die uebrigen Paare der Menge zusammen tun
       nichts. Ohne diesen Vergleich hiesse 'Kern' nur 'eine Teilmenge, die
       auch wirkt'."""
    for _, ns in (einer, dreie):
        R = ns["KERN_RESULTS"]
        for a in R["leben"]:
            assert R["je"]["REST"][a] == "STILL", (a, R["je"]["REST"][a])
            assert R["je"]["KERN"][a] == "TRAEGT", (a, R["je"]["KERN"][a])
        assert len(R["rest"]) == len(R["menge"]) - len(R["kern"])


@pytest.mark.parametrize("welt", ["einer", "dreie", "verteilt"])
def test_lexikalische_arme_bleiben_still(request, welt):
    _, ns = request.getfixturevalue(welt)
    R = ns["KERN_RESULTS"]
    for a in LEX_ARME:
        assert a in R["spezifitaet"], (a, R["spezifitaet"])
        e = R["ergebnis"][a]
        assert e["stufen"]["VOLL"]["k"] == e["k_basis"], (a, e["stufen"]["VOLL"])


def test_scan_laeuft_auf_allen_paaren(einer):
    """Blind heisst: jedes Paar der Menge wird einzeln gemessen, keines
       uebersprungen. Sonst haenge das Ergebnis an einer Vorauswahl."""
    _, ns = einer
    R = ns["KERN_RESULTS"]
    assert len(R["scan_innen"]) == len(R["menge"])
    assert len(R["scan_aussen"]) == len(R["aussen"])
    for k, d in R["scan_innen"].items():
        assert d["plaetze"] > 0, k


def test_vergleichsmenge_ist_gleich_gross_und_gleich_haeufig(einer):
    """Zwei Bedingungen an die Null, und beide sind noetig. GLEICH GROSS, sonst
       vergleicht 'Treffer innen gegen Treffer aussen' zwei verschieden lange
       Listen. GLEICH HAEUFIG, sonst gilt ihre Null fuer eine andere
       Feuerrate - ein seltener Experte sperrt wenige Plaetze und kann schon
       deshalb nichts bewirken."""
    _, ns = einer
    R = ns["KERN_RESULTS"]
    assert len(R["aussen"]) == len(R["menge"]), (len(R["aussen"]), len(R["menge"]))
    assert R["guete"] is not None and R["guete"] <= 0.5, R["guete"]


def test_ratengleiche_direkt(einer):
    """Zu jedem Paar ein FREMDES mit aehnlicher Rate. Ohne Toleranz nimmt der
       Zieher irgendeinen Partner, und die Null misst dann eine andere Rate
       als die gepruefte Menge - ein seltener Partner kann schon deshalb
       nichts bewirken."""
    _, ns = einer
    rg = ns["ratengleiche"]
    menge = [(0, 1), (0, 2)]
    raten = {(0, 1): 100.0, (0, 2): 100.0}
    alle = {(0, 1): 100.0, (0, 2): 100.0, (9, 1): 104.0, (9, 2): 98.0,
            (9, 3): 500.0, (9, 4): 2.0}
    p = rg(menge, raten, alle, random.Random(1))
    assert set(p.values()) == {(9, 1), (9, 2)}, p
    assert len(set(p.values())) == 2, "ein Partner zweimal vergeben"
    # nichts Passendes da: es gibt trotzdem einen Partner, aber die Guete
    # weist ihn als schlecht aus - und die Sperre haengt an der Guete
    q = rg(menge, raten, {(9, 3): 500.0}, random.Random(1))
    assert list(q.values()) == [(9, 3)]
    ag = ns["anpassungsguete"]
    assert ag(menge, raten, {(9, 3): 500.0}, q) == pytest.approx(4.0)
    assert ag(menge, raten, alle, p) < 0.06


def test_kern_aus_scan_direkt(einer):
    _, ns = einer
    ka = ns["kern_aus_scan"]
    aussen = [(0.05 * i, 0.9) for i in range(20)]        # 99. Perzentil ~0.95
    innen = {(0, 1): (1.20, 0.001), (0, 2): (0.50, 0.001), (0, 3): (1.10, 0.9)}
    kern, sw = ka(innen, aussen)
    assert kern == [(0, 1)], (kern, sw)     # (0,3) faellt am p, (0,2) an der Null
    # Obergrenze: findet der Scan die halbe Menge, ist es kein Kern mehr
    viele = {(0, i): (2.0, 0.001) for i in range(20)}
    assert len(ka(viele, aussen, hoechstens=8)[0]) == 8
    assert ka({}, aussen)[0] == []
    assert ka(innen, [])[0] == []


def test_wirkung_direkt(einer):
    _, ns = einer
    wk = ns["wirkung"]
    assert wk(40, 20, 0) == pytest.approx(0.5)
    assert wk(40, 40, 0) == pytest.approx(0.0)
    assert wk(40, 20, 40) is None       # volle Menge wirkt nicht -> kein Nenner
    assert wk(40, 20, 50) is None


def test_urteilsordnung_kern(einer):
    _, ns = einer
    uk = ns["urteil_kern"]
    A = KONSTR_ARME
    T = {a: "TRAEGT" for a in A}
    S = {a: "STILL" for a in A}
    assert uk([(0, 1)], 0, T, T, S, A) == "EINZELNER-TRAEGT"
    assert uk([(0, 1), (0, 2)], 0, T, T, S, A) == "KLEINER-KERN"
    assert uk([(0, 1)], 0, T, T, T, A) == "KERN-UND-REST-TRAGEN"
    assert uk([(0, 1)], 0, T, S, S, A) == "KERN-TRAEGT-NICHT"
    assert uk([], 0, T, S, S, A) == "KEIN-KERN"
    # die Sperren stechen jeden Befund
    assert uk([(0, 1)], 0, S, T, S, A) == "POSITIVKONTROLLE-FEHLT"
    assert uk([(0, 1)], 1, T, T, S, A) == "NULL-TRENNT-NICHT"
    assert uk([], 1, T, T, S, A) == "NULL-TRENNT-NICHT"
    # Mehrheit, nicht Einstimmigkeit
    zwei = dict(S, JA="TRAEGT", BR1="TRAEGT")
    assert uk([(0, 1)], 0, zwei, zwei, S, A) == "EINZELNER-TRAEGT"
    einer_ = dict(S, JA="TRAEGT")
    assert uk([(0, 1)], 0, einer_, T, S, A) == "POSITIVKONTROLLE-FEHLT"


def test_keine_haken_haengen(einer):
    w, _ = einer
    offen = sum(len(w.schichten[l]._haken) for l in range(NLAY))
    assert offen == 0, "%d Haken nicht entfernt" % offen
