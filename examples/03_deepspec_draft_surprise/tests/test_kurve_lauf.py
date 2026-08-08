"""Der Ausfuehrungsteil von phase16_kurve laeuft hier wirklich durch - gegen
ein Miniatur-MoE aus numpy. Kein Torch, keine GPU.

Die Welt hat diesmal eine ABSTUFBARE Wirkung. In phase15 war Verhalten binaer:
ein getroffenes Paar, und die Antwort kippt. Damit liesse sich keine Kurve
pruefen - jede Stufe saehe gleich aus.

Hier bricht der Anteil

    p = (gesperrte Paare der Menge / ganze Menge) ** GAMMA

der Antworten, und GAMMA spannt genau die drei Formen auf, die das Notebook
unterscheiden soll:

    GAMMA = 1     -> etwa linear          Halbwert bei 60 %
    GAMMA = 4     -> Schwelle             Halbwert erst bei 100 %
    GAMMA = 0.25  -> wenige tragen        Halbwert schon bei 20 %

Der Anteil wird DETERMINISTISCH abgezaehlt und nicht gewuerfelt. Bei 32
Ziehungen je Bedingung haette eine Muenze eine Standardabweichung von rund
9 Prozentpunkten - dicht genug an der Halbwertsgrenze, um den Test von Lauf zu
Lauf kippen zu lassen.
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
NB = os.path.join(HIER, "..", "phase16_kurve.ipynb")

HID, INTER, NEXP, NLAY, TOPK = 6, 8, 96, 4, 4
POSITIONEN = 40
ANTWORT = 40
ABLESEN = 40   # Positionen, ueber die generate() nachsieht, was gesperrt ist
Traeger = haken_traeger()

# Die gepruefte Menge: vier Experten je Schicht, die den Japanisch-Arm an der
# Entscheidungsstelle vom Bezugsarm unterscheiden. 16 Paare - genug, dass
# 20/40/60/80 % vier verschiedene Stufen ergeben (3, 6, 10, 13).
MENGE_EXP = (41, 42, 43, 44)
MENGE_WELT = [(l, e) for l in range(NLAY) for e in MENGE_EXP]

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
    def __init__(self, startwert=4711, gamma=1.0, traegt_menge=True,
                 stoert_zufall=False):
        rs = np.random.RandomState(startwert)
        self.schichten = {l: Experten(rs) for l in range(NLAY)}
        self.reg = []
        self.ausg = []
        self.gamma = gamma
        self.traegt_menge = traegt_menge
        self.stoert_zufall = stoert_zufall
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
        """Welcher Bruchteil der Antworten bricht? Nur die konstruierenden
           Arme haengen an der Menge - Serbisch und Russisch bleiben
           unberuehrt, so wie in Phase 12 und 15 gemessen."""
        if arm not in ("JA", "BR1", "MORSE"):
            return 0.0
        if self.stoert_zufall and (self.gesperrt - set(MENGE_WELT)):
            return 1.0
        if not self.traegt_menge:
            return 0.0
        n = len(self.gesperrt & set(MENGE_WELT))
        if n <= 0:
            return 0.0
        return (n / float(len(MENGE_WELT))) ** self.gamma

    def generate(self, input_ids=None, attention_mask=None, **k):
        b, L = np.asarray(input_ids).shape
        txt = self.reg[int(np.asarray(input_ids)[0, 0])]
        arm = self.arm_von(txt)
        self._fahre(arm, ABLESEN, txt)          # macht die Maske sichtbar
        p = self._anteil(arm)
        X = int(round(100.0 * p))
        schl = (arm, tuple(sorted(self.gesperrt & set(MENGE_WELT))), self.stoert_zufall)
        aus = np.zeros((b, L + 1))
        aus[:, :L] = np.asarray(input_ids)
        for j in range(b):
            i = self.zaehler[schl]
            self.zaehler[schl] += 1
            if i % FEHLJEDES == 0:
                text = FEHLTEXT                  # trifft das Zielmass nicht
            else:
                # abgezaehlt statt gewuerfelt: ueber 100 Beispiele bricht es
                # GENAU X-mal, und der Test kippt nicht mit dem Zufall
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


def lauf(gamma=1.0, traegt_menge=True, stoert_zufall=False, wiederholung=0,
         n_bsp=24, n_abl=32, n_ketten=3):
    w = Welt(gamma=gamma, traegt_menge=traegt_menge, stoert_zufall=stoert_zufall)

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
              N_BSP=n_bsp, N_ABL=n_abl, N_KETTEN=n_ketten, MAX_NEW=8, CHUNK=8,
              TEMP=1.0, SEED=5, MIN_BSP=6, N_PRUEF=4, WIEDERHOLUNG=wiederholung,
              wc_save=lambda name, obj: None, wc_save_all=lambda: None,
              RUN_OUT="/tmp")
    np.random.seed(7)
    try:
        exec(compile(koerper(), "phase16_kurve", "exec"), ns)
    except SystemExit:
        pass
    return w, ns


KONSTR_ARME = ("JA", "BR1", "MORSE")
LEX_ARME = ("SR", "RU")


@pytest.fixture(scope="module")
def linear():
    return lauf(gamma=1.0)


@pytest.fixture(scope="module")
def schwelle():
    return lauf(gamma=4.0)


@pytest.fixture(scope="module")
def wenige():
    return lauf(gamma=0.25)


@pytest.fixture(scope="module")
def tote_kette():
    return lauf(traegt_menge=False)


@pytest.fixture(scope="module")
def stoerung():
    return lauf(gamma=1.0, stoert_zufall=True)


def test_zelle_laeuft_durch(linear):
    _, ns = linear
    assert "KURVE_RESULTS" in ns


def test_menge_wird_selbst_hergeleitet(linear):
    """Die gepruefte Menge darf nicht eingetragen sein - sonst haengt der Lauf
       an einer Uebertragung und ein Nachlauf waere nicht in sich stimmig."""
    _, ns = linear
    R = ns["KURVE_RESULTS"]
    assert {tuple(q) for q in R["menge"]} == set(MENGE_WELT)


@pytest.mark.parametrize("welt,erwartet", [
    ("linear", "ETWA-LINEAR"),
    ("schwelle", "SCHWELLE"),
    ("wenige", "WENIGE-TRAGEN"),
    ("tote_kette", "POSITIVKONTROLLE-FEHLT"),
    ("stoerung", "KONTROLLE-STOERT"),
])
def test_alle_formen(request, welt, erwartet):
    """Jede Form muss erreichbar sein und der eingebauten Wahrheit
       entsprechen. Die beiden Sperren stehen VOR der Form: ohne tragende
       Positivkontrolle gibt es keinen Nenner, und wenn eine beliebige
       gleich grosse Menge ebenfalls senkt, misst der Lauf die
       Eingriffsgroesse statt der Menge."""
    _, ns = request.getfixturevalue(welt)
    assert ns["KURVE_RESULTS"]["verdict"] == erwartet, ns["KURVE_RESULTS"]["verdict"]


def test_sperren_stehen_vor_der_form(tote_kette, stoerung):
    """In beiden Sperr-Welten waere eine Form berechenbar - sie darf nur nicht
       das Verdikt bestimmen."""
    for _, ns in (tote_kette, stoerung):
        R = ns["KURVE_RESULTS"]
        assert R["verdict"] in ("POSITIVKONTROLLE-FEHLT", "KONTROLLE-STOERT")
    _, ns = tote_kette
    for a in KONSTR_ARME:
        assert ns["KURVE_RESULTS"]["je_voll"][a] == "STILL"
    _, ns = stoerung
    traegt = sum(1 for a in KONSTR_ARME
                 if ns["KURVE_RESULTS"]["je_zufall"].get(a) == "TRAEGT")
    assert traegt >= 2, ns["KURVE_RESULTS"]["je_zufall"]


@pytest.mark.parametrize("welt", ["linear", "schwelle", "wenige"])
def test_lexikalische_arme_bleiben_still(request, welt):
    """Serbisch und Russisch haengen an keiner Menge. Fallen sie, schadet der
       Eingriff wahllos - und dann sagt keine Zeile mehr etwas."""
    _, ns = request.getfixturevalue(welt)
    R = ns["KURVE_RESULTS"]
    for a in LEX_ARME:
        assert a in R["spezifitaet"], (a, R["spezifitaet"])
        e = R["ergebnis"][a]
        assert e["stufen"]["VOLL"]["k"] == e["k_basis"], (a, e["stufen"]["VOLL"])
        assert e["stufen"]["ZUFALL"]["k"] == e["k_basis"], a
        # und sie laufen NUR in diesen beiden Bedingungen mit
        assert set(e["stufen"]) == {"VOLL", "ZUFALL"}, sorted(e["stufen"])


@pytest.mark.parametrize("welt", ["linear", "schwelle", "wenige"])
def test_zufallskontrolle_wirkt_nie(request, welt):
    _, ns = request.getfixturevalue(welt)
    R = ns["KURVE_RESULTS"]
    zuf = {tuple(q) for q in R["zufall"]}
    assert len(zuf) == len(MENGE_WELT), (len(zuf), len(MENGE_WELT))
    assert not (zuf & set(MENGE_WELT)), sorted(zuf & set(MENGE_WELT))
    for a in R["leben"]:
        e = R["ergebnis"][a]
        assert e["stufen"]["ZUFALL"]["k"] == e["k_basis"], (a, e["stufen"]["ZUFALL"])
        assert R["je_zufall"][a] == "STILL", a


def test_ketten_sind_geschachtelt(linear):
    """Der ganze Sinn der Ketten: Stufe k ist ECHT in Stufe k+1 enthalten.
       Sonst liesse sich 'die Menge ist groesser' nicht von 'es sind andere
       Experten drin' trennen, und eine unglueckliche kleine Ziehung saehe aus
       wie eine Schwelle."""
    _, ns = linear
    R = ns["KURVE_RESULTS"]
    assert len(R["ketten"]) == 3
    voll = set(MENGE_WELT)
    for kette in R["ketten"]:
        assert len(kette) == len(R["anteile"]), kette
        vorher = set()
        for a, k, teil in kette:
            m = {tuple(q) for q in teil}
            assert len(m) == k
            assert a in R["anteile"], a
            # Soll und Ist duerfen auseinanderfallen - der SOLLANTEIL ist der
            # Schluessel der Kurve. Bei 16 Paaren wird aus 20 % naemlich 3,
            # also 18,75 %, und wer mit 0.2 danach sucht, findet nichts.
            assert k == round(a * len(voll)), (a, k)
            assert vorher < m, (sorted(vorher), sorted(m))
            assert m < voll, "eine Stufe ist die ganze Menge"
            vorher = m
    # und die Ketten sind nicht dieselbe Permutation
    erste = [tuple(sorted(tuple(q) for q in kette[0][2])) for kette in R["ketten"]]
    assert len(set(erste)) > 1, erste


def test_kurve_waechst_in_der_kette(linear):
    """Innerhalb einer Kette kommen nur Experten DAZU. Die gemessene Wirkung
       muss deshalb wachsen - ein Rueckgang waere ein Zeichen, dass Stufen und
       Mengen durcheinandergeraten sind."""
    _, ns = linear
    R = ns["KURVE_RESULTS"]
    for a in R["leben"]:
        for r in (1, 2, 3):
            w = [(d["groesse"], d["wirkung"]) for d in R["ergebnis"][a]["stufen"].values()
                 if d["kette"] == r and d["wirkung"] is not None]
            w.sort()
            assert w, (a, r)
            for i in range(len(w) - 1):
                assert w[i + 1][1] >= w[i][1] - 1e-9, (a, r, w)


def test_volle_menge_ist_der_nenner(linear):
    _, ns = linear
    R = ns["KURVE_RESULTS"]
    for a in R["leben"]:
        assert R["ergebnis"][a]["stufen"]["VOLL"]["wirkung"] == pytest.approx(1.0)
        assert R["kurve"][a][-1] == [1.0, 1.0]
        # jede Sollstufe findet ihren Median - keine faellt still auf None
        for a2, w in R["kurve"][a][:-1]:
            assert w is not None, (a, a2, R["kurve"][a])


def test_plaetze_wachsen_mit_der_stufe(linear):
    """Zahl und Dosis laufen auseinander, und genau deshalb wird die Platzzahl
       ausgewiesen. Sie muss innerhalb einer Kette mitwachsen."""
    _, ns = linear
    R = ns["KURVE_RESULTS"]
    for a in R["leben"]:
        for r in (1, 2, 3):
            p = sorted((d["groesse"], d["plaetze"])
                       for d in R["ergebnis"][a]["stufen"].values() if d["kette"] == r)
            for i in range(len(p) - 1):
                assert p[i + 1][1] >= p[i][1], (a, r, p)
        for d in R["ergebnis"][a]["stufen"].values():
            assert d["plaetze"] > 0


def test_geplante_plaetze_gleich_gemessenen(linear):
    """Angeglichen und gemessen muss auf DENSELBEN Texten passieren. Im ersten
       Anlauf von Phase 15 lief die Angleichung ueber Prompt+Antwort und die
       Messung ueber den Prompt allein - 33 gegen 17 Plaetze, ohne dass eine
       der beiden Zahlen falsch ausgesehen haette."""
    _, ns = linear
    R = ns["KURVE_RESULTS"]
    assert R["plaetze_geplant"]
    for a, e in R["ergebnis"].items():
        for nm, d in e["stufen"].items():
            assert d["plaetze"] == R["plaetze_geplant"][nm][a], (a, nm, d["plaetze"])


def test_kandidatentopf_schliesst_die_menge_aus(linear):
    """Der Topf, nicht die Ziehung: eine einzelne Ziehung traefe die Menge nur
       mit etwa halber Wahrscheinlichkeit, und ein Lauf ohne Ausschluss saehe
       dann in der Haelfte der Faelle sauber aus."""
    _, ns = linear
    kk = ns["kandidaten_kontrolle"]
    z = collections.Counter({(0, 1): 5, (0, 2): 0, (0, 3): 7, (0, 4): 1})
    assert kk(z, []) == [(0, 1), (0, 3), (0, 4)]        # die Nullen fliegen raus
    assert kk(z, [[(0, 3)], [(0, 1)]]) == [(0, 4)]
    assert kk(collections.Counter(), []) == []


def test_ketten_funktion_direkt(linear):
    _, ns = linear
    kf = ns["ketten"]
    m = [(0, i) for i in range(10)]
    ks = kf(m, 3, (0.2, 0.5, 0.8), random.Random(1))
    assert len(ks) == 3
    for kette in ks:
        assert [(a, k) for a, k, _ in kette] == [(0.2, 2), (0.5, 5), (0.8, 8)]
        for i in range(len(kette) - 1):
            assert set(kette[i][2]) < set(kette[i + 1][2])
    # Stufen, die auf die ganze Menge hinauslaufen, fallen weg
    assert [k for _, k, _ in kf(m, 1, (0.5, 1.0, 1.2), random.Random(1))[0]] == [5]
    # doppelte Stufen nach dem Runden nur einmal - und der SOLLANTEIL der
    # ersten bleibt stehen, damit die Kurve ihn wiederfindet
    assert [(a, k) for a, k, _ in kf(m, 1, (0.21, 0.24), random.Random(1))[0]] \
        == [(0.21, 2)]
    assert kf([], 3, (0.5,), random.Random(1)) == []
    assert kf(m, 0, (0.5,), random.Random(1)) == []


def test_anteil_wirkung_direkt(linear):
    """Ohne Nenner gibt es keine Kurve - und die Funktion muss das mit None
       sagen statt mit einer Zahl, sonst rechnet der Lauf mit einer Division
       durch null weiter."""
    _, ns = linear
    aw = ns["anteil_wirkung"]
    assert aw(40, 20, 0) == pytest.approx(0.5)
    assert aw(40, 40, 0) == pytest.approx(0.0)
    assert aw(40, 0, 0) == pytest.approx(1.0)
    assert aw(40, 20, 40) is None       # volle Menge wirkt nicht
    assert aw(40, 20, 50) is None       # volle Menge erhoeht sogar


def test_halbwert_und_form_direkt(linear):
    _, ns = linear
    hw = ns["halbwert"]
    fk = ns["form_der_kurve"]
    assert hw([(0.2, 0.2), (0.4, 0.4), (0.6, 0.6), (0.8, 0.8), (1.0, 1.0)]) == 0.6
    assert hw([(0.2, 0.0), (0.4, 0.02), (0.6, 0.1), (0.8, 0.4), (1.0, 1.0)]) == 1.0
    assert hw([(0.2, 0.7), (0.4, 0.9), (1.0, 1.0)]) == 0.2
    assert hw([(0.2, None), (0.4, None), (1.0, 1.0)]) == 1.0
    assert hw([(0.2, 0.1)]) == 1.0      # nie erreicht -> 1.0, nicht Absturz
    assert fk(0.2) == "WENIGE-TRAGEN"
    assert fk(0.4) == "WENIGE-TRAGEN"
    assert fk(0.6) == "ETWA-LINEAR"
    assert fk(0.8) == "SCHWELLE"
    assert fk(1.0) == "SCHWELLE"


def test_median_direkt(linear):
    _, ns = linear
    md = ns["median"]
    assert md([0.2, 0.4, 0.9]) == pytest.approx(0.4)
    assert md([0.2, 0.4]) == pytest.approx(0.3)
    assert md([None, 0.4, None]) == pytest.approx(0.4)
    assert md([None, None]) is None
    assert md([]) is None


def test_senkt_direkt(linear):
    _, ns = linear
    sk = ns["senkt"]
    assert sk(40, 10, 0.001) == "TRAEGT"
    assert sk(40, 10, 0.9) == "STILL"
    assert sk(40, 40, 0.001) == "STILL"
    assert sk(10, 40, 0.001) == "STILL"      # ein ANSTIEG ist keine Senkung


def test_urteilsordnung_kurve(linear):
    _, ns = linear
    uk = ns["urteil_kurve"]
    T = {a: "TRAEGT" for a in KONSTR_ARME}
    S = {a: "STILL" for a in KONSTR_ARME}
    F = {a: "SCHWELLE" for a in KONSTR_ARME}
    assert uk(F, T, S, KONSTR_ARME) == "SCHWELLE"
    assert uk({a: "ETWA-LINEAR" for a in KONSTR_ARME}, T, S, KONSTR_ARME) == "ETWA-LINEAR"
    # die Sperren stechen jede Form
    assert uk(F, S, S, KONSTR_ARME) == "POSITIVKONTROLLE-FEHLT"
    assert uk(F, T, T, KONSTR_ARME) == "KONTROLLE-STOERT"
    # Die Kontrolle sticht ZUERST. Senkt eine beliebige gleich grosse Menge
    # ebenso, heisst die Positivkontrolle NUR-STOERUNG - und ein Lauf, der sie
    # vorn haette, meldete einen Messkettenfehler, wo die Dosis zu gross ist.
    assert uk(F, S, T, KONSTR_ARME) == "KONTROLLE-STOERT"
    # Mehrheit, nicht Einstimmigkeit - aber ohne Mehrheit kein Urteil
    zwei = dict(F, MORSE="ETWA-LINEAR")
    assert uk(zwei, T, S, KONSTR_ARME) == "SCHWELLE"
    drei = {"JA": "SCHWELLE", "BR1": "ETWA-LINEAR", "MORSE": "WENIGE-TRAGEN"}
    assert uk(drei, T, S, KONSTR_ARME) == "UNEINHEITLICH"
    # ein einzelner Arm unter dreien ist kein Befund - weder fuer die
    # Positivkontrolle noch fuer die Stoerung
    einer = {"JA": "TRAEGT", "BR1": "STILL", "MORSE": "STILL"}
    assert uk(F, einer, S, KONSTR_ARME) == "POSITIVKONTROLLE-FEHLT"
    assert uk(F, T, einer, KONSTR_ARME) == "SCHWELLE"
    # ein Arm ohne Nenner stimmt nicht mit
    ohne = dict(F, MORSE=None)
    assert uk(ohne, T, S, KONSTR_ARME) == "SCHWELLE"
    assert uk({a: None for a in KONSTR_ARME}, T, S, KONSTR_ARME) == "KEINE-ARME"


def test_keine_haken_haengen(linear):
    w, _ = linear
    offen = sum(len(w.schichten[l]._haken) for l in range(NLAY))
    assert offen == 0, "%d Haken nicht entfernt" % offen
