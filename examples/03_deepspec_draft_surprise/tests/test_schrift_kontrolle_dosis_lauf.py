"""Der Ausfuehrungsteil von phase12_schrift_kontrolle_dosis laeuft hier wirklich
durch - gegen ein Miniatur-MoE aus numpy. Kein Torch, keine GPU.

Der Vorlauf ist an der DOSIS gescheitert: 42 Zufallspaare gegen 42 gepruefte
Paare, aber doppelt so viele gesperrte Router-Plaetze. Deshalb hat das
Miniaturmodell hier ein Routing ueber ZEHN Positionen mit ungleichen
Haeufigkeiten - gewoehnliche Experten laufen oefter als der Signalexperte.
Waere die Kontrolle wieder nach Paarzahl gebaut, wuerde sie hier sofort zu
viele Plaetze sperren und der Test es sehen.

Am Signalexperten haengen Japanisch, Braille und Morse; Serbisch und Russisch
nicht. Die dosisgleiche Zufallsmenge kann ihn nie enthalten. Damit ist vorher
bekannt: drei Arme TRAEGT, zwei STILL, Verdikt KONSTRUKTION-TRAEGT.
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
NB = os.path.join(HIER, "..", "phase12_schrift_kontrolle_dosis.ipynb")

HID, INTER, NEXP, NLAY, TOPK = 6, 8, 8, 4, 2
SIG_L, JA_E, GEMEIN_E = 1, 1, 0
Traeger = haken_traeger()

# Neun Hintergrundpositionen je Schicht, danach die letzte Position, an der
# sich die Arme unterscheiden. Die Haeufigkeiten sind ABSICHTLICH ungleich:
#   e0 5x | e4 4x | e3 3x | e1 2x | e2 2x | e5 1x | e6 1x
# Der Signalexperte e1 laeuft also selten, die gewoehnlichen oft - genau das
# Missverhaeltnis, das die Kontrolle des Vorlaufs unbemerkt verdoppelt hat.
BG = [(0, 4), (0, 4), (0, 3), (0, 3), (0, 3), (1, 4), (1, 4), (2, 5), (2, 6)]
# Karges Gegenstueck: der Signalexperte laeuft einmal, alle anderen neunmal.
# Damit gibt es zur Zielzahl 8 keine Teilmenge im +-10%-Fenster - die Dosis
# ist NICHT angleichbar, und der Lauf muss das sagen statt trotzdem zu messen.
BG_KARG = [(0, 3)] * 8 + [(1, 3)]

PROMPT = ("Create a markdown table comparing five cloud storage services with their "
          "storage limits and pricing. label the column with each service's local "
          "name, and summarize each entry.")
KANA_TXT = "| サービス名 | 保存容量 | 料金 |\n| Google ドライブ | 15 GB | 無料 |"
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
    def __init__(self, startwert=4711, bg=BG):
        rs = np.random.RandomState(startwert)
        self.schichten = {l: Experten(rs) for l in range(NLAY)}
        self.reg = []
        self.ausg = []
        self.bg = list(bg)
        self.positionen = len(self.bg) + 1

    @staticmethod
    def arm_von(text):
        if "Japanese" in text:
            return "JA"
        if "Serbian" in text:
            return "SR"
        if "Russian" in text:
            return "RU"
        if "Braille" in text:
            return "BR"
        if "Morse" in text:
            return "MORSE"
        return "NEU"

    @staticmethod
    def routing(arm):
        """Nur die letzte Position unterscheidet die Arme - daraus zieht der
           Lauf die JA-exklusive Menge. Serbisch und Russisch fahren e3, das
           auch im Hintergrund laeuft; NEU faehrt e7, das SONST nie laeuft,
           damit die Exklusivmenge genau {e1} ist."""
        return [GEMEIN_E, {"JA": JA_E, "BR": JA_E, "MORSE": JA_E,
                           "SR": 3, "RU": 3, "NEU": 7}[arm]]

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
        """n_pos ehrt die Breite der uebergebenen Kennungen. Ohne das wuerde
           ein Zaehler, der nur die letzte Position liest, hier trotzdem das
           volle Muster sehen und der Fehler des Vorlaufs unsichtbar bleiben."""
        x = np.eye(HID)[0]
        r = self.routing(arm)
        muster = ([list(p) for p in self.bg]
                  + [list(r)])[-(n_pos or self.positionen):]
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

    def __call__(self, ids, use_cache=False, past_key_values=None):
        a = np.asarray(ids)
        h, _ = self._vorwaerts(self.arm_von(self.reg[int(a[0, 0])]), a.shape[1])
        return types.SimpleNamespace(logits=t(np.array([[[h, 1., 2., 3., 4.]]])),
                                     past_key_values=object())

    def generate(self, input_ids=None, attention_mask=None, **k):
        b, L = np.asarray(input_ids).shape
        arm = self.arm_von(self.reg[int(np.asarray(input_ids)[0, 0])])
        _, w_ja = self._vorwaerts(arm)
        ziel = {"JA": KANA_TXT, "SR": KYR_TXT, "RU": KYR_TXT, "BR": BRAILLE_TXT,
                "MORSE": MORSE_TXT, "NEU": EN_TXT}[arm]
        p = {"JA": 0.95, "SR": 0.85, "RU": 0.85, "BR": 0.60, "MORSE": 0.85,
             "NEU": 0.0}[arm]
        # JA, Braille und Morse haengen am Signalexperten der LETZTEN Position,
        # Serbisch und Russisch nicht. Die dosisgleiche Zufallsmenge kann ihn
        # nie enthalten - sie ist gegen die gepruefte Menge gesperrt.
        if arm in ("JA", "BR", "MORSE") and w_ja == 0.0:
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
        ids = np.zeros((len(ts), self.positionen))
        for j in range(len(ts)):
            ids[j, :] = len(self.reg) - len(ts) + j

        class E(dict):
            def to(self, *a, **k):
                return self

            @property
            def input_ids(self):
                return self["input_ids"]

        return E({"input_ids": t(ids),
                  "attention_mask": t(np.ones((len(ts), self.positionen)))})

    def decode(self, seq, skip_special_tokens=True):
        a = np.asarray(seq).ravel()
        return self.ausg[int(a[-1])] if a.size else ""


def lauf(startwert=4, bg=BG):
    w = Welt(bg=bg)

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
        exec(compile(koerper(), "phase12_schrift_kontrolle_dosis", "exec"), ns)
    except SystemExit:
        pass
    return w, ns


@pytest.fixture(scope="module")
def ergebnis():
    return lauf()


@pytest.fixture(scope="module")
def karg():
    """Zweiter Lauf gegen die karge Welt, in der die Dosis nicht angleichbar
       ist. Er darf NICHT durchmessen - die Sperre steht vor allem anderen."""
    return lauf(bg=BG_KARG)


def test_zelle_laeuft_durch(ergebnis):
    _, ns = ergebnis
    assert "DOSIS_RESULTS" in ns


def test_alle_arme_gemessen(ergebnis):
    _, ns = ergebnis
    P = ns["DOSIS_RESULTS"]["pilot"]
    assert set(P) == {"NEU", "JA", "SR", "RU", "BR1", "MORSE"}
    assert P["JA"]["mass"] == "kana"
    assert P["SR"]["mass"] == "kyrillisch" and P["RU"]["mass"] == "kyrillisch"
    assert P["BR1"]["mass"] == "braille" and P["MORSE"]["mass"] == "morse"


def test_plaetze_statt_paare_angeglichen(ergebnis):
    """Der Kern dieses Laufs. Die Zufallsmenge muss die gesperrten
       ROUTER-PLAETZE der gepruefte Menge treffen - und darf dafuer ruhig aus
       weniger Paaren bestehen. Waere sie wieder nach Paarzahl gebaut, saehe
       man hier ein Vielfaches der Plaetze."""
    _, ns = ergebnis
    R = ns["DOSIS_RESULTS"]
    assert R["dosis_ok"] is True
    ja = {tuple(q) for q in R["ja_menge"]}
    assert ja and not any(q[1] != JA_E for q in ja), sorted(ja)
    for s_, z in R["zufall"].items():
        assert z["paare"] and z["plaetze"] > 0, s_
        assert abs(z["plaetze"] - z["ziel"]) <= 0.10 * z["ziel"] + 1e-9, \
            "%s: %d Plaetze gegen Ziel %d" % (s_, z["plaetze"], z["ziel"])
        assert not (ja & {tuple(q) for q in z["paare"]}), \
            "%s: die Kontrolle enthaelt die gepruefte Menge" % s_
    # weniger Paare bei gleicher Dosis - weil gewoehnliche Experten oefter
    # laufen. Genau diese Ungleichheit hat der Vorlauf uebersehen.
    assert min(len(R["zufall"][s_]["paare"]) for s_ in R["zufall"]) < len(ja)


def test_gemessene_plaetze_stimmen_mit_der_planung(ergebnis):
    """Die im Lauf tatsaechlich gesperrten Plaetze muessen zu den geplanten
       passen - sonst haette die Angleichung nur auf dem Papier stattgefunden.

       Aber NICHT auf die Zahl genau, und das ist kein Schlendrian: die Maske
       aendert die Ausgabe an Position t, damit den Zustand an t+1, damit die
       Routerwahl danach. Im echten Modell wich der JA-Arm deshalb um 3 %
       ab (geplant 468, gemessen 453 unter der geprueften und 479 unter der
       Zufallsmaske). Das Miniaturmodell hat diese Rueckkopplung nicht und
       trifft exakt - eine Gleichheitsforderung wuerde hier also eine
       Invariante festschreiben, die das echte System gar nicht erfuellt.
       Abgerechnet wird ohnehin mit den GEMESSENEN Zahlen."""
    _, ns = ergebnis
    R = ns["DOSIS_RESULTS"]
    for s_, e in R["ergebnis"].items():
        assert e["plaetze_ja"] > 0 and e["plaetze_zufall"] > 0, s_
        for gemessen, geplant in ((e["plaetze_ja"], R["zufall"][s_]["ziel"]),
                                  (e["plaetze_zufall"], R["zufall"][s_]["plaetze"])):
            assert abs(gemessen - geplant) <= 0.20 * geplant, \
                "%s: gemessen %d gegen geplant %d" % (s_, gemessen, geplant)
        # und die beiden Masken bleiben untereinander vergleichbar
        assert abs(e["plaetze_ja"] - e["plaetze_zufall"]) <= 0.20 * e["plaetze_ja"]


def test_nur_lebende_arme_werden_angefasst(ergebnis):
    _, ns = ergebnis
    R = ns["DOSIS_RESULTS"]
    assert set(R["ergebnis"]) <= set(R["lebende"])
    assert all(R["pilot"][s]["lebt"] for s in R["ergebnis"])
    assert "NEU" not in R["ergebnis"]


def test_traegt_gegen_zerbrechlichkeit(ergebnis):
    _, ns = ergebnis
    R = ns["DOSIS_RESULTS"]
    J = R["je_arm"]
    for a in ("JA", "BR1", "MORSE"):
        assert J[a] == "TRAEGT", "%s gibt %s" % (a, J[a])
    for a in ("SR", "RU"):
        assert J[a] == "STILL", "%s gibt %s" % (a, J[a])
    assert R["verdict"] == "KONSTRUKTION-TRAEGT"
    # und die Wirkung je Platz zeigt in die richtige Richtung
    for a in ("JA", "BR1", "MORSE"):
        assert R["ergebnis"][a]["je_platz_ja"] > 0
        assert R["ergebnis"][a]["je_platz_ja"] > R["ergebnis"][a]["je_platz_zufall"]


def test_dosisgleich_trifft_die_zielzahl(ergebnis):
    """Der Angleicher gegen einen von Hand gebauten Zaehler."""
    _, ns = ergebnis
    dg = ns["dosisgleich"]
    z = collections.Counter({(0, 1): 7, (0, 2): 5, (0, 3): 3, (1, 1): 4,
                             (1, 2): 6, (1, 3): 2, (2, 1): 9, (2, 2): 1})
    rnd = random.Random(11)
    menge, summe = dg(20, z, {(0, 1)}, rnd)
    assert 18 <= summe <= 22, summe
    assert (0, 1) not in menge, "gesperrtes Paar in der Kontrolle"
    assert summe == sum(z[q] for q in menge)
    assert len(set(menge)) == len(menge), "Paar doppelt gezogen"
    # nicht erreichbar -> leer statt irgendetwas
    assert dg(10 ** 6, z, set(), random.Random(3)) == ([], 0)
    assert dg(0, z, set(), random.Random(3)) == ([], 0)
    assert dg(20, collections.Counter(), set(), random.Random(3)) == ([], 0)
    # alles gesperrt -> nichts zu ziehen
    assert dg(20, z, set(z), random.Random(3)) == ([], 0)
    # Ueberschiessen ist kein Treffer: lauter Siebener auf Ziel 10 - eine
    # liegt unter dem Fenster, zwei darueber. Ohne die obere Schranke kaeme
    # hier 14 zurueck und die Kontrolle waere wieder der haertere Eingriff.
    gross = collections.Counter({(0, k): 7 for k in range(6)})
    assert dg(10, gross, set(), random.Random(7)) == ([], 0)
    # und wo etwas zurueckkommt, liegt es IMMER im Fenster
    for saat in range(25):
        m2, s2 = dg(20, z, {(0, 1)}, random.Random(saat))
        assert (m2 == [] and s2 == 0) or 18 <= s2 <= 22, (saat, m2, s2)


def test_wirkung_je_platz(ergebnis):
    _, ns = ergebnis
    f = ns["wirkung_je_platz"]
    # 96 von 96 auf 0 gesenkt, 100 Plaetze -> 100 Punkte je 100 Plaetze
    assert abs(f(96, 0, 96, 100) - 100.0) < 1e-9
    # dieselbe Wirkung mit doppelt so vielen Plaetzen ist halb so viel wert
    assert abs(f(96, 0, 96, 200) - 50.0) < 1e-9
    assert f(96, 0, 96, 0) == 0.0
    assert f(96, 0, 0, 100) == 0.0
    assert f(10, 60, 96, 100) < 0, "Anstieg muss negativ zaehlen"


def test_urteilsordnung_dosis(ergebnis):
    _, ns = ergebnis
    ua, ud = ns["urteil_arm_dosis"], ns["urteil_dosisgleich"]
    # nur die gepruefte Menge senkt
    assert ua(60, 96, 10, 0.001, 58, 0.9, 264, 264) == "TRAEGT"
    # beide senken, aber je Platz gleich stark -> Zerbrechlichkeit
    assert ua(60, 96, 20, 0.001, 20, 0.001, 264, 264) == "NUR-STOERUNG"
    # beide senken, die gepruefte je Platz mehr als doppelt so stark
    assert ua(60, 96, 10, 0.001, 50, 0.01, 264, 264) == "TRAEGT-UEBERWIEGEND"
    # ... und genau das ist der Fall, den die alte Regel falsch verbucht hat:
    # Braille, 264 gegen 557 Plaetze, -48 gegen -20 Punkte
    assert ua(61, 96, 15, 0.001, 42, 0.01, 264, 557) == "TRAEGT-UEBERWIEGEND"
    # gleiche Wirkung bei gleicher Dosis bleibt Stoerung
    assert ua(61, 96, 15, 0.001, 15, 0.001, 264, 264) == "NUR-STOERUNG"
    assert ua(60, 96, 58, 0.9, 10, 0.001, 264, 264) == "WIDERSPRUECHLICH"
    assert ua(60, 96, 58, 0.9, 59, 0.9, 264, 264) == "STILL"
    # ein Anstieg ist keine Senkung, auch wenn er signifikant ist
    assert ua(10, 96, 60, 0.001, 61, 0.001, 264, 264) == "STILL"
    G = {"JA": "TRAEGT", "BR1": "TRAEGT", "MORSE": "TRAEGT",
         "SR": "STILL", "RU": "STILL"}
    # die Dosissperre kommt VOR allem anderen
    assert ud(G, False) == "DOSIS-NICHT-ANGEGLICHEN"
    assert ud(dict(G, JA="NUR-STOERUNG"), False) == "DOSIS-NICHT-ANGEGLICHEN"
    assert ud(G, True) == "KONSTRUKTION-TRAEGT"
    assert ud(dict(G, BR1="TRAEGT-UEBERWIEGEND"), True) == "KONSTRUKTION-TRAEGT"
    assert ud(dict(G, JA="TRAEGT-UEBERWIEGEND"), True) == "KONSTRUKTION-TRAEGT"
    assert ud(dict(G, JA="NUR-STOERUNG"), True) == "EICHMARKE-FEHLT"
    assert ud(dict(G, JA="STILL"), True) == "EICHMARKE-FEHLT"
    assert ud(dict(G, BR1="NUR-STOERUNG"), True) == "ZERBRECHLICHKEIT"
    assert ud(dict(G, MORSE="NUR-STOERUNG"), True) == "ZERBRECHLICHKEIT"
    assert ud(dict(G, BR1="STILL", MORSE="STILL"), True) == "NUR-JAPANISCH"
    assert ud(dict(G, BR1="STILL"), True) == "GEMISCHT"
    assert ud({"JA": "TRAEGT"}, True) == "KEINE-KONSTRUIERTEN-ARME"


def test_hole_plaetze_zaehlt_alle_positionen(ergebnis):
    """Der Vorlauf las nur die LETZTE Position und hat die Dosis deshalb
       verfehlt. Hier muss der Zaehler den ganzen Text sehen: der
       Signalexperte laeuft zweimal im Hintergrund und einmal am Ende."""
    _, ns = ergebnis
    P = ns["PLAETZE"]
    A = ns["ARMTEXT"]
    ja = ns["hole_plaetze"](A["JA"])
    assert ja[(SIG_L, JA_E)] == 3, ja[(SIG_L, JA_E)]
    assert ja[(SIG_L, GEMEIN_E)] == 6, ja[(SIG_L, GEMEIN_E)]
    # im Serbisch-Arm laeuft der Signalexperte nur im Hintergrund
    assert P["SR"][(SIG_L, JA_E)] == 2
    # und die letzte Position allein wuerde 1 statt 3 liefern
    assert len(ns["hole_routing"](A["JA"])) == 2 * NLAY


def test_keine_haken_haengen(ergebnis):
    w, _ = ergebnis
    offen = sum(len(w.schichten[l]._haken) for l in range(NLAY))
    assert offen == 0, "%d Haken nicht entfernt" % offen


def test_ohne_angleichbare_dosis_wird_nicht_gemessen(karg):
    """Die Sperre muss WIRKEN, nicht bloss im Verdikt stehen. In der kargen
       Welt laeuft der Signalexperte einmal und jeder andere neunmal - zur
       Zielzahl 8 gibt es dann keine Teilmenge im Fenster. Der Lauf haelt an,
       statt eine schiefe Kontrolle zu messen; genau das ist der Fehler des
       Vorlaufs, nur eine Stufe frueher."""
    _, ns = karg
    R = ns["DOSIS_RESULTS"]
    assert R["dosis_ok"] is False
    assert R["verdict"] == "DOSIS-NICHT-ANGEGLICHEN"
    assert R["ergebnis"] == {} and R["je_arm"] == {}, "trotz Sperre gemessen"
    assert R["lebende"] == []
    # der Japanisch-Arm ist der, an dem die Angleichung scheitert
    assert R["zufall"]["JA"]["paare"] == []
    assert R["zufall"]["JA"]["ziel"] > 0


def test_karge_welt_wuerde_sonst_messen(karg):
    """Gegenprobe zur Sperre: die Arme LEBEN in der kargen Welt, es liegt also
       nicht daran, dass ohnehin nichts zu messen waere."""
    _, ns = karg
    P = ns["DOSIS_RESULTS"]["pilot"]
    assert P["JA"]["lebt"] and P["BR1"]["lebt"] and P["MORSE"]["lebt"]
