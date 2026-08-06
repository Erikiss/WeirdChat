"""Der Ausfuehrungsteil von phase15_ablation laeuft hier wirklich durch - gegen
ein Miniatur-MoE aus numpy. Kein Torch, keine GPU.

Die Welt aus dem Screen-Test, um eine VERHALTENSwirkung erweitert - sonst
haette die Ablation nichts zu messen. Zwei disjunkte Mengen wirken hier auf
das Ergebnis:

  KONSTRUKT   routet selektiv UND traegt das Verhalten   -> die 'Screen-Menge'
  P12_WIRK    routet gar nicht selektiv, traegt aber ebenfalls das Verhalten

P12_WIRK bildet die Lage aus dem echten Lauf nach: die 42 aus Phase 12 sind
kausal verifiziert und tauchen im Screen trotzdem nicht auf. Sie ist ECHT
GROESSER als die Screen-Menge - anders liesse sich 'eine Teilmenge traegt'
nicht von 'die ganze Menge traegt' unterscheiden, und genau diese Unterschei-
dung ist der Zweck des Umbaus.

Drei Stellschrauben spannen alle Ausgaenge auf:

  schalter_screen   traegt KONSTRUKT das Verhalten?
  schalter_p12      traegt P12_WIRK ueberhaupt?       -> Positivkontrolle
  schwelle_p12      ab WIE VIELEN gesperrten Paaren aus P12_WIRK bricht es?
                    1  = jedes Achtel genuegt
                    9  = mehr als die angeglichene Zahl -> nur die VOLLE
                         Menge wirkt, die Wirkung ist VERTEILT
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
NB = os.path.join(HIER, "..", "phase15_ablation.ipynb")

HID, INTER, NEXP, NLAY, TOPK = 6, 8, 96, 4, 4
POSITIONEN = 40
ANTWORT = 40
ABLESEN = 16    # Positionen, ueber die generate() nachsieht, was gesperrt ist
Traeger = haken_traeger()

# Eingebaute Wahrheit. Drei Sorten Menge, und die dritte ist der Grund, warum
# der erste Anlauf gescheitert ist:
#
#   KONSTRUKT    von JA, BR1 und MORSE geteilt      <- das Gesuchte
#   LEXIKAL      von SR und RU geteilt              <- darf nicht hineinrutschen
#   OBERFLAECHE  je Arm EIGEN, gleich stark         <- 'der Text sieht anders aus'
#
# Ohne die dritte Sorte koennte kein Test bemerken, ob der Oberflaechenboden
# wirkt: SR und RU haetten identisches Routing, und SR-gegen-RU maesse nur
# Stichprobenrauschen. Genau so war die erste Fassung dieser Welt gebaut.
KONSTRUKT = [(l, 5) for l in range(NLAY)] + [(l, 9) for l in range(NLAY)]
LEXIKAL = [(l, 17) for l in range(NLAY)]
OBERFLAECHE = {a: [(l, e) for l in range(NLAY) for e in (30 + 2 * i, 31 + 2 * i)]
               for i, a in enumerate(("JA", "SR", "RU", "BR1", "MORSE"))}
# Die zweite wirksame Menge: sie routet NICHT selektiv (kein Screen-Treffer),
# traegt das Verhalten aber genauso. Genau die Lage aus dem echten Lauf - und
# mit 12 Paaren groesser als die 8 des Screens, damit eine Teilmenge von ihr
# auch wirklich eine Teilmenge ist.
P12_EXP = (41, 42, 43)
P12_WIRK = [(l, e) for l in range(NLAY) for e in P12_EXP]

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
FEHLJEDES = 3   # jede dritte Antwort eines Nicht-Bezugsarms geht daneben


def koerper():
    with open(NB, encoding="utf-8") as f:
        nb = json.load(f)
    quelle = "".join("".join(c.get("source", [])) for c in nb["cells"]
                     if c.get("cell_type") == "code")
    marke = "# ---------------- reine Logik"
    assert marke in quelle, "Marke fuer den Logikteil fehlt im Notebook"
    return quelle[quelle.index(marke):]


class Experten(Traeger):
    def __init__(self, rs, welt=None, schicht=0):
        Traeger.__init__(self)
        self.welt = welt
        self.schicht = schicht
        self.gate_up_proj = t(rs.randn(NEXP, 2 * INTER, HID) * 0.5)
        self.down_proj = t(rs.randn(NEXP, HID, INTER) * 0.5)
        self.intermediate_dim = INTER


class Welt:
    def __init__(self, startwert=4711, kern_staerke=8.0, ober_staerke=3.0,
                 schalter_screen=True, schalter_p12=True, schwelle_p12=1,
                 p12_exp=P12_EXP):
        rs = np.random.RandomState(startwert)
        self.schichten = {l: Experten(rs, self, l) for l in range(NLAY)}
        self.reg = []
        self.ausg = []
        self.rnd = random.Random(startwert)
        self.kern_staerke = kern_staerke
        self.ober_staerke = ober_staerke
        self.schalter_screen = schalter_screen
        self.schalter_p12 = schalter_p12
        self.schwelle_p12 = schwelle_p12
        # Wie viele Experten unterscheiden die Entscheidungsstelle? Kleiner
        # gesetzt wird die Vorgabe KLEINER als die Screen-Menge - dann laesst
        # sich die Zahl gar nicht angleichen, und die Sperre muss greifen.
        self.p12_exp = tuple(p12_exp)
        self.gesperrt = set()      # von der Maske gesetzt
        # sehr ungleiche Grundgewichte: der Kern der Pruefung
        self.grund = [0.05 + 3.0 * (i % 6 == 0) + 0.4 * (i % 3 == 0) for i in range(NEXP)]
        # P12_WIRK laeuft in JEDEM Arm haeufig - deshalb ist es nicht selektiv
        # und faellt durch den Screen. Genau das ist im echten Modell der Fall:
        # die 42 feuern 84 bis 97 Plaetze auch dort, wo sie nichts zu tun
        # haben. Nebeneffekt und Absicht: ein gesperrtes Paar wird dadurch
        # zuverlaessig bemerkt, statt nur meistens.
        for e in P12_EXP:
            self.grund[e] = 8.0

    @staticmethod
    def arm_von(text):
        # Eine danebengegangene Antwort routet wie der Bezugsarm - egal, welche
        # Anweisung im Prompt stand. Sonst waere nicht zu bemerken, ob die
        # Zelle sie vor der Rechnung aussortiert.
        if FEHLTEXT in text:
            return "NEU"
        for k in ("Japanese", "Serbian", "Russian", "Braille", "Morse"):
            if k in text:
                return {"Japanese": "JA", "Serbian": "SR", "Russian": "RU",
                        "Braille": "BR1", "Morse": "MORSE"}[k]
        return "NEU"

    def _gewichte(self, arm, l):
        g = list(self.grund)
        if self.kern_staerke > 1.0 or self.ober_staerke > 1.0:
            if arm in ("JA", "BR1", "MORSE"):
                for ll, e in KONSTRUKT:
                    if ll == l:
                        g[e] *= self.kern_staerke
            if arm in ("SR", "RU"):
                for ll, e in LEXIKAL:
                    if ll == l:
                        g[e] *= self.kern_staerke
            # jeder Arm hat zusaetzlich SEINE EIGENE Oberflaechenmenge, gleich
            # stark - genau das, was im echten Lauf ein Viertel des Modells
            # ausgewaehlt hat
            for ll, e in OBERFLAECHE.get(arm, ()):
                if ll == l:
                    g[e] *= self.ober_staerke
        return g

    @staticmethod
    def _wuerfel(text, p, l):
        """Routing ist eine FUNKTION DES TEXTES, deterministisch je Position
           und Schicht.

           Zwei Fehler steckten hier nacheinander drin. Erst lief EIN
           fortlaufender Generator: derselbe Text ergab bei jedem Durchlauf
           ein anderes Routing, und die geplante Dosis wurde auf einem
           Durchlauf berechnet und auf einem anderen nachgemessen - 62 gegen
           89 Plaetze, ohne dass eine der Zahlen falsch ausgesehen haette.
           Dann hing die Saat nur am ARM: da alle Beispiele eines Arms
           denselben Text hatten, gab es innerhalb eines Arms ueberhaupt keine
           Streuung mehr, und die Nullkalibrierung fiel auf null zusammen.

           Beides ist geheilt, wenn die Saat am Text haengt und die Antworten
           sich zwischen Beispielen unterscheiden."""
        h = 2166136261
        for c in ("%s/%d/%d" % (text, p, l)).encode():
            h = ((h ^ c) * 16777619) & 0xFFFFFFFF
        return random.Random(h)

    def _idx(self, arm, n_pos, text=""):
        """Der Armeffekt wirkt NUR auf den letzten 21 Positionen - der
           Entscheidungsstelle und der Antwort. Davor routen alle Arme gleich.

           Das ist kein Beiwerk: waeren Prompt und Antwort gleich, koennte
           kein Test bemerken, ob die Zelle die richtigen Positionen zaehlt.
           Der Armeffekt wirkt auf der Antwort und der Entscheidungsstelle.
           So verduennt 'alle Positionen' das Signal, und 'Position 0' statt
           'letzte Promptposition' verfehlt es ganz."""
        if n_pos == 1:
            # Der zweite Aufruf von hole_routing() liest genau diese eine
            # Position. Deterministisch, damit die Differenz JA-minus-NEU
            # exakt P12_WIRK ergibt - im echten Modell ist das die Menge aus
            # Phase 12, und sie taucht im Screen nicht auf.
            fremd = [e for e in range(NEXP) if e not in self.p12_exp]
            fest = (list(self.p12_exp) + fremd)[:TOPK] if arm != "NEU" \
                else fremd[:TOPK]
            return [[sorted(fest) for _ in range(NLAY)]]
        muster = []
        for p in range(n_pos):
            heiss = p >= n_pos - (ANTWORT + 1)
            zeile = []
            for l in range(NLAY):
                g = self._gewichte(arm, l) if heiss else list(self.grund)
                r = self._wuerfel(text, p, l)
                pick = set()
                while len(pick) < TOPK:
                    pick.add(r.choices(range(NEXP), weights=g)[0])
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
        """Ein Vorwaertslauf durch alle Schichten. Nebenbei wird abgelesen, wo
           eine Maske Routergewichte auf null gesetzt hat - daraus macht die
           Miniatur ihre Verhaltenswirkung, sonst haette die Ablation nichts
           zu messen."""
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

    def generate(self, input_ids=None, attention_mask=None, **k):
        b, L = np.asarray(input_ids).shape
        txt = self.reg[int(np.asarray(input_ids)[0, 0])]
        arm = self.arm_von(txt)
        # Vorwaertslauf, damit eine gesetzte Maske ueberhaupt sichtbar wird
        self._fahre(arm, ABLESEN, txt)
        traf_screen = self.schalter_screen and bool(self.gesperrt & set(KONSTRUKT))
        # Wie VIELE Paare aus P12_WIRK sind gesperrt? Ueber die Schwelle
        # trennt sich 'ein Achtel genuegt' von 'nur die ganze Menge wirkt'.
        n_p12 = len(self.gesperrt & set(P12_WIRK))
        traf_p12 = self.schalter_p12 and n_p12 >= self.schwelle_p12
        # Nur die KONSTRUIERENDEN Arme haengen an den beiden Mengen. Serbisch
        # und Russisch bleiben unberuehrt - genau das ist die Spezifitaet, die
        # Phase 12 gemessen hat, und ohne sie koennte kein Test bemerken, ob
        # der Eingriff wahllos schadet.
        kaputt = arm in ("JA", "BR1", "MORSE") and (traf_screen or traf_p12)
        aus = np.zeros((b, L + 1))
        aus[:, :L] = np.asarray(input_ids)
        for j in range(b):
            daneben = arm != "NEU" and len(self.ausg) % FEHLJEDES == 0
            marke = " #%d" % len(self.ausg)   # macht jede Antwort zu einem
            #                                    eigenen Text; ohne das routen
            #                                    alle Beispiele eines Arms gleich
            self.ausg.append((FEHLTEXT if (daneben or kaputt) else ZIELTEXT[arm]) + marke)
            aus[j, L] = len(self.ausg) - 1
        return t(aus)

    def tok(self, text, return_tensors=None, padding=False, add_special_tokens=True):
        ts = [text] if isinstance(text, str) else list(text)
        self.reg.extend(ts)
        # Prompt = POSITIONEN Spalten, Prompt+Antwort = POSITIONEN + 20
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


def lauf(startwert=4, kern_staerke=8.0, ober_staerke=3.0, wiederholung=0, n_bsp=48,
         schalter_screen=True, schalter_p12=True, schwelle_p12=1, n_abl=48,
         p12_exp=P12_EXP):
    """kern_staerke gegen ober_staerke ist die eigentliche Stellschraube.

       Ist der Oberflaecheneffekt so stark wie der Konstruktionseffekt, KANN
       kein Boden die beiden trennen - der Screen muss dann schweigen und darf
       nicht raten. Genau dieser Fall wird in test_oberflaeche_ueberdeckt_das
       geprueft, und genau er hat den ersten echten Lauf zerlegt."""
    w = Welt(kern_staerke=kern_staerke, ober_staerke=ober_staerke,
             schalter_screen=schalter_screen, schalter_p12=schalter_p12,
             schwelle_p12=schwelle_p12, p12_exp=p12_exp)

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
              N_BSP=n_bsp, N_ABL=n_abl, MAX_NEW=8, CHUNK=8, TEMP=1.0, SEED=5,
              PERM=60, NULLPERM=60, MINDEST_SEL=2.0, WIEDERHOLUNG=wiederholung,
              wc_save=lambda name, obj: None, wc_save_all=lambda: None,
              RUN_OUT="/tmp")
    np.random.seed(startwert)
    try:
        exec(compile(koerper(), "phase15_ablation", "exec"), ns)
    except SystemExit:
        pass
    return w, ns


OBER_ALLE = set().union(*[set(v) for v in OBERFLAECHE.values()])
KONSTR_ARME = ("JA", "BR1", "MORSE")
LEX_ARME = ("SR", "RU")
VERTEILT = 9    # mehr als die angeglichene Zahl (8) - keine Teilmenge erreicht sie


@pytest.fixture(scope="module")
def beide():
    return lauf(schalter_screen=True, schalter_p12=True, schwelle_p12=1)


@pytest.fixture(scope="module")
def nur_screen():
    """Der Screen traegt; von P12 wirkt nur die VOLLE Menge, kein Achtel."""
    return lauf(schalter_screen=True, schalter_p12=True, schwelle_p12=VERTEILT)


@pytest.fixture(scope="module")
def nur_p12():
    return lauf(schalter_screen=False, schalter_p12=True, schwelle_p12=1)


@pytest.fixture(scope="module")
def verteilt():
    """Nur die volle Menge wirkt - weder der Screen noch ein Achtel der 42."""
    return lauf(schalter_screen=False, schalter_p12=True, schwelle_p12=VERTEILT)


@pytest.fixture(scope="module")
def zu_wenig():
    """Die Vorgabe hat WENIGER Paare als die Screen-Menge. Dann ist die Zahl
       nicht anzugleichen, und der Lauf muss abbrechen statt vier ungleich
       grosse Mengen gegeneinander zu stellen."""
    return lauf(p12_exp=(41,))


@pytest.fixture(scope="module")
def tote_kette():
    """Die Positivkontrolle wirkt nicht. schalter_screen bleibt AN - das
       Verdikt muss trotzdem POSITIVKONTROLLE-FEHLT lauten, sonst steht die
       Sperre nicht wirklich vorn."""
    return lauf(schalter_screen=True, schalter_p12=False)


def test_zelle_laeuft_durch(beide):
    _, ns = beide
    assert "ABL_RESULTS" in ns


def test_beide_mengen_werden_selbst_hergeleitet(beide):
    """Weder die Screen-Menge noch die 42 duerfen eingetragen sein - beide
       muessen aus dem Modell kommen, sonst haengt der Lauf an einer
       Uebertragung und ein Nachlauf waere nicht stimmig."""
    _, ns = beide
    R = ns["ABL_RESULTS"]
    screen = {tuple(q) for q in R["screen_menge"]}
    p12 = {tuple(q) for q in R["p12"]}
    assert screen and p12
    assert p12 == set(P12_WIRK), sorted(p12 ^ set(P12_WIRK))
    assert screen <= set(KONSTRUKT), sorted(screen - set(KONSTRUKT))
    # und sie sind disjunkt - genau die Lage aus dem echten Lauf
    assert not (screen & p12), sorted(screen & p12)
    # die Vorgabe muss ECHT groesser sein als die angeglichene Zahl, sonst
    # waere 'Teilmenge' nur ein anderes Wort fuer 'ganze Menge'
    assert len(p12) > len(screen), (len(p12), len(screen))


@pytest.mark.parametrize("welt,erwartet", [
    ("beide", "BEIDE-TRAGEN"),
    ("nur_screen", "NUR-SCREEN-TRAEGT"),
    ("nur_p12", "NUR-P12-TRAEGT"),
    ("verteilt", "WIRKUNG-IST-VERTEILT"),
    ("tote_kette", "POSITIVKONTROLLE-FEHLT"),
    ("zu_wenig", "ZAHL-NICHT-ANGEGLICHEN"),
])
def test_alle_verdikte(request, welt, erwartet):
    """Jeder Ausgang muss erreichbar sein und dem entsprechen, was in der Welt
       tatsaechlich wirkt. NUR-P12-TRAEGT ist der Fall, vor dem das
       Quellpapier warnt: der Screen findet dann Korrelate statt Ursachen.
       WIRKUNG-IST-VERTEILT ist der Fall, den der erste Anlauf gar nicht
       ausdruecken konnte."""
    _, ns = request.getfixturevalue(welt)
    assert ns["ABL_RESULTS"]["verdict"] == erwartet, ns["ABL_RESULTS"]["verdict"]


def test_positivkontrolle_steht_vor_allem_anderen(tote_kette):
    """In dieser Welt TRAEGT die Screen-Menge - und trotzdem darf das Verdikt
       nichts anderes sagen als POSITIVKONTROLLE-FEHLT. Wirkt die Menge nicht,
       von der man weiss, dass sie wirkt, misst nicht die Frage, sondern die
       Messkette; alles darunter ist unlesbar.

       Genau das fehlte im ersten Anlauf: alle Bedingungen blieben still, und
       das Ergebnis liess sich nicht deuten."""
    _, ns = tote_kette
    R = ns["ABL_RESULTS"]
    assert R["verdict"] == "POSITIVKONTROLLE-FEHLT"
    traegt = sum(1 for a in KONSTR_ARME if R["je_screen"].get(a) in
                 ("TRAEGT", "TRAEGT-UEBERWIEGEND"))
    assert traegt >= 2, R["je_screen"]
    for a in KONSTR_ARME:
        assert R["je_voll"][a] == "STILL", (a, R["je_voll"][a])


def test_volle_menge_traegt_in_den_lesbaren_welten(beide, nur_screen, nur_p12, verteilt):
    """In jeder Welt, deren Verdikt eine Aussage ueber die Mengen macht, muss
       die Positivkontrolle vorher getragen haben."""
    for _, ns in (beide, nur_screen, nur_p12, verteilt):
        R = ns["ABL_RESULTS"]
        traegt = sum(1 for a in KONSTR_ARME if R["je_voll"].get(a) in
                     ("TRAEGT", "TRAEGT-UEBERWIEGEND"))
        assert traegt >= 2, (R["verdict"], R["je_voll"])


@pytest.mark.parametrize("welt", ["beide", "nur_screen", "nur_p12", "verteilt",
                                  "tote_kette"])
def test_lexikalische_arme_bleiben_still(request, welt):
    """In jeder Welt haengen Serbisch und Russisch an keiner der Mengen.
       Faellt dort etwas, schadet der Eingriff wahllos - und dann sagt keine
       Zeile mehr etwas. Auch die volle Menge darf sie nicht treffen."""
    _, ns = request.getfixturevalue(welt)
    R = ns["ABL_RESULTS"]
    for a in LEX_ARME:
        assert R["je_screen"][a] == "STILL", (a, R["je_screen"][a])
        assert R["je_teil"][a] == "STILL", (a, R["je_teil"][a])
        assert R["je_voll"][a] == "STILL", (a, R["je_voll"][a])


def test_zufallskontrolle_wirkt_nie(beide):
    """Die Zufallsmenge trifft keine der beiden wirksamen Mengen. Wirkt sie
       trotzdem, ist die Dosis selbst schaedlich und jedes 'traegt' waere
       ueberschaetzt."""
    _, ns = beide
    R = ns["ABL_RESULTS"]
    zuf = {tuple(q) for q in R["zufall"]}
    assert not (zuf & (set(KONSTRUKT) | set(P12_WIRK))), sorted(zuf)
    for a, e in R["ergebnis"].items():
        assert e["k"]["zufall"] == e["k_basis"], (a, e["k"], e["k_basis"])
        assert e["je_platz"]["zufall"] == 0.0, a


def test_expertenzahl_ist_angeglichen(beide):
    """Der eigentliche Umbau: verglichen wird acht gegen acht gegen acht
       EXPERTEN. Der erste Anlauf hat auf Router-Plaetze angeglichen und die
       P12-Teilmenge dadurch auf 7 % der wirksamen Dosis heruntergedreht."""
    _, ns = beide
    R = ns["ABL_RESULTS"]
    n = R["n_gleich"]
    assert n == len(R["screen_menge"])
    assert len(R["p12_teil"]) == n, (n, len(R["p12_teil"]))
    assert len(R["zufall"]) == n, (n, len(R["zufall"]))
    assert R["zahl_ok"] is True
    # und die Positivkontrolle ist bewusst NICHT angeglichen
    assert len(R["p12"]) > n


def test_platzzahl_wird_ausgewiesen(beide):
    """Gleich viele Experten heisst NICHT gleich viele Plaetze. Faellt die
       Platzzahl unter den Tisch, koennte ein Unterschied zwischen zwei gleich
       grossen Mengen heimlich ein Dosisunterschied sein. Sie muss also
       gemessen, berichtet und je Platz normiert werden."""
    _, ns = beide
    R = ns["ABL_RESULTS"]
    for a, e in R["ergebnis"].items():
        for nm in ("screen", "p12_teil", "zufall", "p12_voll"):
            assert e["plaetze"][nm] > 0, (a, nm)
            assert nm in e["je_platz"], (a, nm)
        # geplant und gemessen muessen EXAKT uebereinstimmen: dieselben Texte,
        # deterministisches Routing. Jede Abweichung hiesse, dass die
        # Angleichung auf anderen Daten stattfand als die Messung.
        for nm in ("screen", "p12_teil", "zufall", "p12_voll"):
            assert e["plaetze"][nm] == R["plaetze_geplant"][a][nm], (a, nm)
        # die volle Menge sperrt mehr als ihre Teilmenge - sonst waere sie
        # keine Positivkontrolle, sondern dieselbe Bedingung zweimal
        assert e["plaetze"]["p12_voll"] > e["plaetze"]["p12_teil"], a


def test_teilmenge_kommt_aus_der_vorgabe(beide):
    """teilmenge_zahl() darf nur aus den uebergebenen Kandidaten waehlen -
       sonst waere die 'P12-Teilmenge' gar keine."""
    _, ns = beide
    R = ns["ABL_RESULTS"]
    p12 = {tuple(q) for q in R["p12"]}
    teil = {tuple(q) for q in R["p12_teil"]}
    assert teil < p12, sorted(teil - p12)
    zuf = {tuple(q) for q in R["zufall"]}
    assert not (zuf & p12) and not (zuf & {tuple(q) for q in R["screen_menge"]})


def test_teilmenge_zahl_zieht_genau_n(beide):
    """Angeglichen wird die ZAHL. Der Zieher muss genau n liefern, nie etwas
       Fremdes waehlen und bei zu wenigen Kandidaten alles nehmen statt leer
       zurueckzugeben."""
    _, ns = beide
    tz = ns["teilmenge_zahl"]
    kand = [(0, i) for i in range(10)]
    m = tz(kand, 4, random.Random(1))
    assert len(m) == 4 and set(m) <= set(kand) and m == sorted(m)
    # zwei verschiedene Zufallsquellen ziehen verschieden - sonst waere die
    # 'zufaellige' Teilmenge in Wahrheit fest
    andere = [tz(kand, 4, random.Random(s)) for s in range(12)]
    assert len({tuple(x) for x in andere}) > 1
    assert tz(kand, 99, random.Random(1)) == sorted(kand)     # alles, nicht leer
    assert tz([], 4, random.Random(1)) == []
    assert tz(kand, 0, random.Random(1)) == []


def test_urteilsordnung_ablation(beide):
    _, ns = beide
    ua = ns["urteil_ablation"]
    T = {a: "TRAEGT" for a in KONSTR_ARME}
    S = {a: "STILL" for a in KONSTR_ARME}
    assert ua(T, T, T, True, 8) == "BEIDE-TRAGEN"
    assert ua(T, S, T, True, 8) == "NUR-SCREEN-TRAEGT"
    assert ua(S, T, T, True, 8) == "NUR-P12-TRAEGT"
    assert ua(S, S, T, True, 8) == "WIRKUNG-IST-VERTEILT"
    # die Positivkontrolle sticht ALLES - auch ein 'beide tragen'
    assert ua(T, T, S, True, 8) == "POSITIVKONTROLLE-FEHLT"
    # und die Sperren kommen noch davor
    assert ua(T, T, T, False, 8) == "ZAHL-NICHT-ANGEGLICHEN"
    assert ua(T, T, T, True, 2) == "SCREEN-MENGE-ZU-KLEIN"
    # ein einzelner Treffer unter dreien ist kein Befund
    einer = dict(S, JA="TRAEGT")
    assert ua(einer, S, T, True, 8) == "WIRKUNG-IST-VERTEILT"
    zwei = dict(S, JA="TRAEGT", BR1="TRAEGT-UEBERWIEGEND")
    assert ua(zwei, S, T, True, 8) == "NUR-SCREEN-TRAEGT"
    # dieselbe Mehrheitsregel gilt fuer die Positivkontrolle selbst
    assert ua(T, T, einer, True, 8) == "POSITIVKONTROLLE-FEHLT"
    assert ua(T, T, zwei, True, 8) == "BEIDE-TRAGEN"


def test_zahlsperre_bricht_ab(zu_wenig):
    """Die Sperre ist keine Verzierung: greift sie, darf kein Ergebnis mehr
       kommen. Im ersten Anlauf war eine solche Sperre nur ein Verdikt-Code
       ohne Folgen, und der Lauf rechnete darunter munter weiter."""
    _, ns = zu_wenig
    R = ns["ABL_RESULTS"]
    assert R["verdict"] == "ZAHL-NICHT-ANGEGLICHEN"
    assert R["zahl_ok"] is False
    assert len(R["p12"]) < len(R["screen_menge"]), (R["p12"], R["screen_menge"])
    assert "ergebnis" not in R, "unter der Sperre wurde weitergerechnet"


def test_wirkung_je_platz_ist_normiert(beide):
    """Gleiche Rohwirkung bei doppelter Dosis muss den halben Wert ergeben -
       sonst ist die Groesse keine Normierung, sondern nur eine Umbenennung
       der Prozentpunkte. Ohne sie liesse sich der kleine Screen-Eingriff
       nicht mit der viel groesseren Positivkontrolle vergleichen."""
    _, ns = beide
    wj = ns["wirkung_je_platz"]
    assert wj(60, 30, 100, 200) == pytest.approx(2 * wj(60, 30, 100, 400))
    assert wj(60, 30, 100, 100) == pytest.approx(30.0)
    assert wj(60, 30, 100, 0) == 0.0
    assert wj(60, 60, 100, 100) == 0.0


def test_kandidaten_der_kontrolle(beide):
    """Zwei Bedingungen an den Kandidatentopf der Zufallskontrolle, und beide
       sind noetig:

       Wer im Bezugsarm NIE feuert, laesst sich auch nicht sperren - eine
       Kontrolle aus solchen Paaren waere gar kein Eingriff und wuerde jedes
       'traegt' zu billig lizenzieren.
       Und keine der gepruefigen Mengen darf hineinrutschen, sonst kontrolliert
       sie sich selbst.

       Direkt geprueft statt ueber die Miniaturwelt: dort feuert praktisch
       jeder Experte irgendwann, und ein fehlender Filter waere an keinem
       Ergebnis zu bemerken."""
    _, ns = beide
    kk = ns["kandidaten_kontrolle"]
    z = collections.Counter({(0, 1): 5, (0, 2): 0, (0, 3): 7, (0, 4): 1, (0, 5): 0})
    assert kk(z, []) == [(0, 1), (0, 3), (0, 4)]          # die Nullen fliegen raus
    assert kk(z, [[(0, 3)]]) == [(0, 1), (0, 4)]
    assert kk(z, [[(0, 3)], [(0, 1)]]) == [(0, 4)]
    assert kk(z, [[(0, 1), (0, 3), (0, 4)]]) == []
    assert kk(collections.Counter(), []) == []


def test_keine_haken_haengen(beide):
    w, _ = beide
    offen = sum(len(w.schichten[l]._haken) for l in range(NLAY))
    assert offen == 0, "%d Haken nicht entfernt" % offen


def test_armurteil_direkt(beide):
    """urteil_arm_dosis steht in dieser Zelle noch einmal - also wird es hier
       auch noch einmal geprueft und nicht auf Phase 12/13 vertraut."""
    _, ns = beide
    ua = ns["urteil_arm_dosis"]
    assert ua(60, 96, 10, 0.001, 58, 0.9, 264, 264) == "TRAEGT"
    assert ua(60, 96, 20, 0.001, 20, 0.001, 264, 264) == "NUR-STOERUNG"
    assert ua(60, 96, 10, 0.001, 50, 0.01, 264, 264) == "TRAEGT-UEBERWIEGEND"
    assert ua(60, 96, 58, 0.9, 10, 0.001, 264, 264) == "WIDERSPRUECHLICH"
    assert ua(60, 96, 58, 0.9, 59, 0.9, 264, 264) == "STILL"
    # ein ANSTIEG ist keine Senkung, auch wenn er hochsignifikant ist
    assert ua(10, 96, 60, 0.001, 61, 0.001, 264, 264) == "STILL"
    assert ua(10, 96, 60, 0.001, 11, 0.9, 264, 264) == "STILL"
