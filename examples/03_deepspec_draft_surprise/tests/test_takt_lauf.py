"""Der Ausfuehrungsteil von phase19_taktgeber laeuft hier wirklich durch -
gegen ein Miniatur-MoE aus numpy. Kein Torch, keine GPU, keine echte Uhr.

Die Zeit ist gerechnet und nicht gemessen: jeder Vorwaertslauf traegt seine
Dauer in mini_torch.ZEITEN ein, das nachgebaute cuda-Ereignis liest sie aus.
Die eingebaute Wahrheit lautet

    t = basis + trend * laenge + KAPPA_WAHR * S + IDENT_WAHR * [Menge geroutet]
        + AR(1)-Jitter + schwere Enden

mit S = Zahl VERSCHIEDENER Experten ueber alle Schichten. Damit ist jede
Statistik gegen etwas Bekanntes zu pruefen - und eine Statistik, die eine
gepflanzte Wirkung nicht findet, darf ihre Abwesenheit auch nicht berichten.
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

import mini_torch
from mini_torch import haken_traeger, mach_torch, silu, t

HIER = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HIER, "..", "phase19_taktgeber.ipynb")

HID, INTER, NEXP, NLAY, TOPK = 6, 8, 64, 4, 4
POSITIONEN = 24
Traeger = haken_traeger()

MENGE_EXP = (41, 42, 43, 44)
MENGE_WELT = [(l, e) for l in range(NLAY) for e in MENGE_EXP]

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


class Welt:
    def __init__(self, startwert=4711, kappa=0.02, ident=0.0, ar=0.7, sig=0.05,
                 basis=40.0, trend=0.02, periode=0):
        rs = np.random.RandomState(startwert)
        self.schichten = {l: Experten(rs) for l in range(NLAY)}
        self.reg = []
        self.kappa = kappa          # ms je Einheit Vielfalt - die Wahrheit
        self.ident = ident          # ms Aufschlag, wenn die Menge geroutet ist
        self.ar = ar
        self.sig = sig
        self.basis = basis
        self.trend = trend
        self.periode = periode
        self.z = 0.0
        self.rs = np.random.RandomState(startwert + 1)
        self.schritt = 0
        self.gesperrt = set()
        self.letzte_slots = []

    @staticmethod
    def _wuerfel(nr, p, l):
        h = 2166136261
        for c in ("%d/%d/%d" % (nr, p, l)).encode():
            h = ((h ^ c) * 16777619) & 0xFFFFFFFF
        return random.Random(h)

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

    def _routing(self, nr, pos, zeilen, l):
        aus = []
        for j in range(zeilen):
            r = self._wuerfel(nr + 7919 * j, pos, l)
            pick = set()
            while len(pick) < TOPK:
                pick.add(r.randrange(NEXP))
            aus.append(sorted(pick))
        return aus

    def __call__(self, ids, use_cache=False, past_key_values=None):
        a = np.asarray(ids)
        zeilen, breite = a.shape
        nr = int(a[0, 0])
        pos = int(a[0, -1]) % 97
        S = 0
        slots = []
        x = np.eye(HID)[0]
        self.gesperrt = set()
        for l in range(NLAY):
            m = self._routing(nr, pos, zeilen, l)
            # (Tokens, top_k) wie im echten Modell - zweidimensional. Maske
            # verlangt gleiche Formen fuer Index und Gewicht, und Zwang
            # ersetzt genau diese Tafel.
            idx = t(np.array(m, dtype=float))
            w0 = t(np.full((zeilen, TOPK), 0.5))
            _, i_n, w_n = self.schichten[l].feuere(t(x.reshape(1, 1, HID)), idx, w0)
            ia = np.asarray(i_n).reshape(-1).astype(int)
            wa = np.asarray(w_n).reshape(-1)
            for wert, e in zip(wa, ia):
                if wert == 0.0:
                    self.gesperrt.add((l, int(e)))
            S += len(set(ia.tolist()))
            slots.extend((l, int(e)) for e in set(ia.tolist()))
        self.letzte_slots = sorted(set(slots))
        # --- die eingebaute Zeitwahrheit -----------------------------------
        self.z = self.ar * self.z + self.rs.randn() * self.sig
        tr = self.trend * breite
        pd = (0.0 if self.periode <= 1
              else 0.4 * math.sin(2 * math.pi * self.schritt / self.periode))
        auf = self.ident if set(MENGE_WELT) <= set(self.letzte_slots) else 0.0
        dauer = self.basis + tr + self.kappa * S + auf + self.z + pd
        if self.rs.rand() < 0.02:
            dauer += self.rs.rand() * 10.0
        mini_torch.ZEITEN.append(float(dauer))
        self.schritt += 1
        lg = np.zeros((1, 1, 50))
        lg[0, 0, (nr + pos) % 50] = 1.0
        return types.SimpleNamespace(logits=t(lg), past_key_values=object())

    def generate(self, input_ids=None, attention_mask=None, **k):
        b, L = np.asarray(input_ids).shape
        aus = np.zeros((b, L + 1))
        aus[:, :L] = np.asarray(input_ids)
        aus[:, L] = 5
        return t(aus)

    def tok(self, text, return_tensors=None, padding=False, add_special_tokens=True,
            return_offsets_mapping=False):
        ts = [text] if isinstance(text, str) else list(text)
        self.reg.extend(ts)
        nr = len(self.reg) - len(ts)
        ids = np.zeros((len(ts), POSITIONEN))
        for j in range(len(ts)):
            ids[j, :] = nr + j
            ids[j, -1] = (nr + j) % 97

        class E(dict):
            def to(self, *a, **k):
                return self

            @property
            def input_ids(self):
                return self["input_ids"]

        return E({"input_ids": t(ids),
                  "attention_mask": t(np.ones((len(ts), POSITIONEN)))})

    def decode(self, seq, skip_special_tokens=True):
        return "| Service | 15 GB |"


def lauf(kappa=0.02, ident=0.0, periode=0, ar=0.7, sig=0.05, wiederholung=0,
         n_schritt=48, n_wieder=6, n_paar=8, n_aa=4, stapel=4, verwerfen=4):
    mini_torch.ZEITEN.clear()
    mini_torch.SCHLAF[0] = 0.0
    w = Welt(kappa=kappa, ident=ident, periode=periode, ar=ar, sig=sig)

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
              N_SCHRITT=n_schritt, VERWERFEN=verwerfen, N_WIEDER=n_wieder,
              N_PAAR=n_paar, N_AA=n_aa, STAPEL=stapel, PERM=400,
              STUFEN=(50, 200, 800), TEMP=1.0, SEED=5, WIEDERHOLUNG=wiederholung,
              wc_save=lambda name, obj: None, wc_save_all=lambda: None,
              RUN_OUT="/tmp")
    np.random.seed(7)
    try:
        exec(compile(koerper(), "phase19_taktgeber", "exec"), ns)
    except SystemExit:
        pass
    return w, ns


@pytest.fixture(scope="module")
def vielfalt():
    """Die Zeit folgt der ZAHL verschiedener Experten, nicht den Namen."""
    return lauf(kappa=0.02, ident=0.0)


@pytest.fixture(scope="module")
def blind():
    """Weder Vielfalt noch Identitaet bewegen die Uhr."""
    return lauf(kappa=0.0, ident=0.0)


@pytest.fixture(scope="module")
def mit_traeger():
    """Eine echte Periode 7 in der Zeitreihe."""
    return lauf(kappa=0.02, ident=0.0, periode=7)


def test_zelle_laeuft_durch(vielfalt):
    _, ns = vielfalt
    assert "TAKT_RESULTS" in ns


def test_groessen_werden_gerechnet_nicht_geraten(vielfalt):
    """Die 6 MiB je Experte und die 2 GB je Schritt duerfen nicht im Text
       stehen - sie muessen aus den wirklichen Tensorformen kommen, sonst
       stimmen sie beim naechsten Modell nicht mehr."""
    _, ns = vielfalt
    R = ns["TAKT_RESULTS"]
    erw = (2 * INTER * HID + HID * INTER) * 8      # mini_torch rechnet in float64
    assert R["byte_je_experte"] == erw, (R["byte_je_experte"], erw)
    assert R["byte_je_schritt"] == erw * TOPK * NLAY


def test_uhr_eichung_findet_die_einspeisung(vielfalt):
    """EPSILON ist die Aufloesung. Ohne sie ist kein Nullbefund zulaessig -
       'kein Effekt' hiesse sonst nur 'unter dem, was die Uhr sieht', und wie
       viel das ist, wuesste niemand."""
    _, ns = vielfalt
    R = ns["TAKT_RESULTS"]
    assert R["epsilon_us"] is not None, R["leiter"]
    assert R["epsilon_us"] <= 200, R["leiter"]
    gross = [d for s, d, p in R["leiter"] if s == 800]
    assert gross and gross[0] is not None and gross[0] > 0.5, R["leiter"]


def test_wiederholbarkeit_nach_trendabzug(vielfalt):
    _, ns = vielfalt
    R = ns["TAKT_RESULTS"]
    assert R["icc"] is not None and R["icc_p"] is not None
    assert R["delta_ms"] > 0 and R["median_ms"] > 0


def test_sonde_ist_nicht_das_signal(vielfalt):
    """Die Attrappe: Maske setzt nur GEWICHTE auf null und laesst den
       Befehlsstrom unangetastet. Ihre erwartete Zeitwirkung ist genau der
       Hakenfussabdruck und sonst nichts."""
    _, ns = vielfalt
    R = ns["TAKT_RESULTS"]
    assert R["sonde_urteil"] != "VERSCHIEDEN", (R["sonde_haken"], R["sonde_maske"])
    assert set(R["sonde"]) == {"ohne", "leer", "maske"}


def test_kappa_wird_gefunden(vielfalt):
    """Die eingebaute Wahrheit ist 0.02 ms je Einheit Vielfalt. Findet die
       Zelle sie nicht, darf sie auch keine Abwesenheit berichten."""
    _, ns = vielfalt
    R = ns["TAKT_RESULTS"]
    assert R["kappa"] is not None, R["h1_x"]
    assert abs(R["kappa"] - 0.02) < 0.01, R["kappa"]
    assert R["kappa_p"] < 0.05, R["kappa_p"]
    assert R["kappa_lo"] is not None and R["kappa_lo"] > 0


def test_kappa_still_ohne_dosis(blind):
    _, ns = blind
    R = ns["TAKT_RESULTS"]
    assert R["kappa_p"] is None or R["kappa_p"] >= 0.05, R["kappa_p"]


def test_identitaet_ist_die_gepflanzte(vielfalt, blind):
    """In beiden Welten ist der Identitaetsaufschlag null. Die Zelle muss das
       als Schranke berichten und nicht als Effekt."""
    for _, ns in (vielfalt, blind):
        R = ns["TAKT_RESULTS"]
        assert R["identitaet"][0] in ("GLEICHWERTIG", "UNENTSCHIEDEN"), R["identitaet"]


def test_aa_paare_eichen_die_null(vielfalt):
    """A gegen A darf nicht ablehnen. Tut es das, ist die Null nicht geeicht
       und der A/B-Befund waere wertlos - deshalb steht die Eichung im Lauf
       und nicht nur im Mock."""
    _, ns = vielfalt
    R = ns["TAKT_RESULTS"]
    assert len(R["h2_aa"]) >= 3
    assert R["h2_aa_p"] is None or R["h2_aa_p"] >= 0.05, R["h2_aa_p"]


def test_traeger_nur_wo_einer_ist(vielfalt, mit_traeger):
    """Ohne gepflanzte Periode darf keine gefunden werden, mit einer schon.
       Findet die Zelle in Trend plus AR(1) einen Traeger, ist jede Phase
       darauf gebaut."""
    _, ns0 = vielfalt
    _, ns1 = mit_traeger
    assert ns0["TAKT_RESULTS"]["traeger_periode"] is None, \
        ns0["TAKT_RESULTS"]["traeger"]
    assert ns1["TAKT_RESULTS"]["traeger_periode"] in (7, 14), \
        ns1["TAKT_RESULTS"]["traeger"]


def test_last_beruehrt_das_routing_nicht(vielfalt):
    """Routing ist eine Funktion von Eingabe und Gewichten. Kippt es unter
       Konkurrenz, waere die Ausgangsidee wieder offen - und die Zelle wuerde
       es sagen."""
    _, ns = vielfalt
    R = ns["TAKT_RESULTS"]
    assert R["last_routing"] == "LASTFEST", R["kipp"]
    assert sum(R["kipp"]) == 0


def test_verdikt_ist_erreichbar(vielfalt, blind):
    # Der Zusatz -OHNE-PROFIL sagt, dass sich das Zeitprofil nicht
    # wiederholt. Das sperrt H4, aber nicht H1 und H2 - die vergleichen
    # Blockmediane zwischen Bedingungen und brauchen kein Profil.
    assert vielfalt[1]["TAKT_RESULTS"]["verdict"].startswith(
        "ZEIT-SIEHT-VIELFALT-NICHT-IDENTITAET"), vielfalt[1]["TAKT_RESULTS"]["verdict"]
    assert blind[1]["TAKT_RESULTS"]["verdict"].startswith("ZEIT-SIEHT-NICHTS"), \
        blind[1]["TAKT_RESULTS"]["verdict"]


def test_messkette_direkt(vielfalt):
    """schritt_zeiten wird hier UNMITTELBAR aufgerufen, nicht ueber ein
       Ergebnis der Welt.

       Der Mutationstest hat gezeigt, warum das noetig ist: drei Eingriffe in
       die Messkette blieben unbemerkt, weil die Miniaturwelt sie nicht
       unterscheiden kann. Aufgegebene Lehrerfuehrung faellt nicht auf, wenn
       ohnehin erzwungen geroutet wird; mitlaufende Aufwaermschritte aendern
       kein Verdikt; und vertauschte Ereigniszeiten sind an einem Median nicht
       zu sehen. An der Kette selbst sind sie es."""
    _, ns = vielfalt
    sz = ns["schritt_zeiten"]
    ids = ns["FESTE_IDS"]
    n_soll = ns["N_SCHRITT"] - ns["VERWERFEN"]

    # 1. Die Aufwaermschritte werden wirklich verworfen.
    d = sz(ids, mit_haken=True, fangen=True)
    assert len(d["dev"]) == n_soll, (len(d["dev"]), n_soll)
    assert len(d["host"]) == n_soll
    assert d["verworfen"] == ns["VERWERFEN"], d["verworfen"]
    assert len(d["warm_host"]) == ns["VERWERFEN"]

    # 2. Lehrergefuehrt heisst: an jeder Stelle ein ANDERES Token, also auch
    #    anderes Routing. Wird stattdessen immer dieselbe Stelle gefuettert,
    #    steht die Routingfolge still.
    verschieden = len({tuple(x) for x in d["slots"] if x})
    assert verschieden > 5,         (verschieden, "das Routing steht still - wird fortgeschritten?")

    # 3. Die Ereigniszeiten muessen zu IHREM Schritt gehoeren. Eine
    #    eingespeiste Verzoegerung an bekannten Stellen muss genau dort
    #    auftauchen und nirgends sonst.
    muster = [800 if i % 2 == 0 else 0 for i in range(ns["N_SCHRITT"])]
    e = sz(ids, mit_haken=False, schlaf=muster, fangen=False)
    mit = [t for t, m in zip(e["dev"], muster[ns["VERWERFEN"]:]) if m > 0]
    ohne = [t for t, m in zip(e["dev"], muster[ns["VERWERFEN"]:]) if m == 0]
    assert mit and ohne
    assert np.median(mit) - np.median(ohne) > 0.5,         (np.median(mit), np.median(ohne), "Zeit und Schritt gehoeren nicht zusammen")


def test_keine_haken_haengen(vielfalt):
    w, _ = vielfalt
    offen = sum(len(w.schichten[l]._haken) for l in range(NLAY))
    assert offen == 0, "%d Haken nicht entfernt" % offen
