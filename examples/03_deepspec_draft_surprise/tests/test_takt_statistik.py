"""Die Statistik von phase19_taktgeber, einzeln geprueft.

Getrennt von test_takt_lauf, das die ganze Welt durchfaehrt. Der Grund steht
im Mutationsprotokoll: von achtzehn Mutationen fing der Weltdurchlauf nur
fuenf. Wer nur das Ganze prueft, prueft die Teile nicht - eine vertauschte
Traegerschwelle oder ein Lastzaehler, der Plaetze statt verschiedener Experten
zaehlt, faellt in einer Miniaturwelt nicht auf, weil dort beides zufaellig
zusammenfaellt.

Die Zeitwelt hier ist synthetisch und hat eine EINGEBAUTE Wahrheit: monotoner
Trend (der Schluessel-Wert-Speicher waechst), AR(1)-Jitter mit
Nachbarkorrelation 0.7 und schwere Enden. Jede Statistik muss eine gepflanzte
Wirkung finden UND bei fehlender Wirkung schweigen.
"""
import json
import math
import os
import random

import numpy as np
import pytest

HIER = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HIER, "..", "phase19_taktgeber.ipynb")


@pytest.fixture(scope="module")
def G():
    with open(NB, encoding="utf-8") as f:
        nb = json.load(f)
    q = "".join("".join(c.get("source", [])) for c in nb["cells"]
                if c.get("cell_type") == "code")
    marke = "# ---------------- Was die Uhr sehen kann"
    assert marke in q, "Marke fuer die Zeitlogik fehlt"
    ende = q.index("# ---------------- Ausfuehrung")
    ns = dict(math=math, random=random, np=np)
    exec(compile(q[q.index(marke):ende], "takt_logik", "exec"), ns)
    return ns


def welt(n=96, rs=None, trend=0.06, ar=0.7, sig=0.8, basis=40.0, spitze=0.02):
    rs = rs or np.random.RandomState(0)
    e = np.zeros(n)
    z = 0.0
    for i in range(n):
        z = ar * z + rs.randn() * sig
        e[i] = z
    return basis + trend * np.arange(n) + e + (rs.rand(n) < spitze) * rs.rand(n) * 25.0


def test_lastzaehler_zaehlt_verschiedene(G):
    """Plaetze und verschiedene Experten fallen nur bei B=1 zusammen. Genau
       dort ist die Groesse aber nutzlos - sie ist konstant 320."""
    assert G["lastzaehler"]([[1, 2, 2, 3], [4, 4, 4, 4]]) == 4
    assert G["lastzaehler"]([[1, 2, 3, 4]]) == 4
    assert G["lastzaehler"]([[7, 7, 7, 7]]) == 1


def test_trend_wird_entfernt(G):
    y = welt(120, np.random.RandomState(3))
    L = np.arange(120)
    r = np.asarray(G["entfernen_trend"](y, L))
    assert abs(np.polyfit(L, r, 1)[0]) < 0.005
    assert r.std() > 0.5, "der Abzug hat auch das Signal mitgenommen"


def test_icc_braucht_den_trendabzug(G):
    """Der Anstieg ist in ALLEN Wiederholungen derselbe. Ohne Abzug misst die
       Zuverlaessigkeit nur ihn und kaeme auch bei reinem Rauschen nahe eins -
       genau das war der erste Bau."""
    ohne = [list(welt(60, np.random.RandomState(100 + i), ar=0.0, sig=0.3))
            for i in range(10)]
    r0, _ = G["icc_haelften"](ohne, perm=120, rnd=random.Random(4))
    assert r0 is not None and r0 < 0.5, r0
    prof = np.random.RandomState(9).randn(60) * 2.0
    mit = [list(welt(60, np.random.RandomState(400 + i), ar=0.0, sig=0.0) + prof
                + np.random.RandomState(200 + i).randn(60) * 0.3) for i in range(10)]
    r1, p1 = G["icc_haelften"](mit, perm=120, rnd=random.Random(4))
    assert r1 is not None and r1 > 0.8 and p1 < 0.05, (r1, p1)


def test_tost_drei_ausgaenge(G):
    """GLEICHWERTIG, VERSCHIEDEN und UNENTSCHIEDEN sind drei verschiedene
       Dinge. Der erste Bau nannte fuenf gegen fuenf Ziehungen verschieden,
       sobald der Punktwert gross genug ausfiel."""
    a = welt(200, np.random.RandomState(11))
    b = welt(200, np.random.RandomState(12))
    assert G["tost"](a, b, 2.0)[0] == "GLEICHWERTIG"
    assert G["tost"](a, b + 6.0, 2.0)[0] == "VERSCHIEDEN"
    assert G["tost"](a[:5], b[:5] + 1.0, 0.05)[0] == "UNENTSCHIEDEN"
    # zu wenig Daten darf nie Gleichwertigkeit heissen
    assert G["tost"]([1.0, 1.1, 0.9], [1.0, 1.2, 0.8], 0.01)[0] != "GLEICHWERTIG"


def test_leiter_findet_die_einspeisung(G):
    rs = np.random.RandomState(1)
    zt, mk = [], []
    for s in (0, 50, 200, 800):
        for _ in range(40):
            zt.append(float(welt(1, np.random.RandomState(rs.randint(10**6)))[0]
                            + s / 1000.0))
            mk.append(s)
    eps, tab = G["leiter"](zt, mk, (50, 200, 800), alpha=0.05, perm=300,
                           rnd=random.Random(2))
    assert eps is not None
    gross = [(d, p) for s, d, p in tab if s == 800][0]
    # Grosszuegig, aber richtungsfest: die Probewelt hat schwere Enden, und
    # der Median einer kleinen Stichprobe schwankt. Entscheidend ist, dass die
    # Einspeisung ueberhaupt wiedergefunden wird - eine vertauschte Zuordnung
    # zwischen Zeit und Marke ergaebe null, nicht das Doppelte.
    assert gross[0] is not None and 0.4 < gross[0] < 2.0, gross
    assert gross[1] < 0.05, gross
    # und ohne Einspeisung darf nichts gefunden werden
    eps0, tab0 = G["leiter"]([z for z, m in zip(zt, mk)], [0] * len(mk),
                             (50,), alpha=0.05, perm=200, rnd=random.Random(3))
    assert eps0 is None, tab0


def test_vorzeichenumkehr_braucht_unabhaengige_einheiten(G):
    """Der teuerste Fehler dieses Bauteils, hier festgehalten: je Schritt
       umgekehrt ergibt die Probewelt 33 % Fehlalarm bei nominal 5. Der
       naheliegende Ausweg ueber Bloecke eicht ebenfalls nicht, wenn die
       Blocklaenge an denselben Daten gewaehlt wird."""
    def rate(bloecke, verschiebung=0.0, N=120, start=1000):
        tr = 0
        for i in range(N):
            d = (welt(96, np.random.RandomState(start + i))
                 - welt(96, np.random.RandomState(5000 + i)) + verschiebung)
            x = G["blockmediane"](d, G["blocklaenge"](d)) if bloecke else d
            _, p = G["gepaart"](x, perm=200, rnd=random.Random(i))
            if p is not None and p < 0.05:
                tr += 1
        return tr / float(N)
    assert rate(False) > 0.20, "die kaputte Variante muss kaputt bleiben"
    # unabhaengige Laufpaare - so rechnet das Notebook
    paare = [float(np.median(welt(96, np.random.RandomState(700 + i))
                             - welt(96, np.random.RandomState(900 + i))))
             for i in range(12)]
    _, pp = G["gepaart"](paare, perm=2000, rnd=random.Random(1))
    assert pp >= 0.05, pp
    _, pq = G["gepaart"]([x + 2.0 for x in paare], perm=2000, rnd=random.Random(1))
    assert pq < 0.05, pq


def test_kappa_und_blocknull(G):
    rs = np.random.RandomState(2)
    xs, ys = [], []
    for d in (8, 16, 32, 64, 128):
        for _ in range(5):
            xs.append(d)
            ys.append(40.0 + 0.162 * d
                      + np.random.RandomState(rs.randint(10**6)).randn() * 0.5)
    k, lo, hi = G["steigung"](xs, ys, proben=300, rnd=random.Random(6))
    assert abs(k - 0.162) < 0.02 and lo > 0, (k, lo, hi)
    _, p = G["blocknull"](ys, xs, perm=500, rnd=random.Random(8))
    assert p < 0.05, p
    ys0 = [40.0 + np.random.RandomState(rs.randint(10**6)).randn() * 0.5 for _ in xs]
    _, p0 = G["blocknull"](ys0, xs, perm=500, rnd=random.Random(8))
    assert p0 >= 0.05, p0


def test_armnull_zaehlt_laeufe_nicht_tokens(G):
    """Einheit ist der Lauf. Wer die Reihe kuenstlich verlaengert, verengt die
       Null und findet einen Armeffekt, den es nicht gibt."""
    je = [1.0 + 0.02 * np.random.RandomState(300 + i).randn() for i in range(20)]
    arme = ["A", "B", "C", "D", "E"] * 4
    _, pa, boden = G["armnull"](je, arme, perm=500, rnd=random.Random(3))
    assert pa >= 0.05, pa
    assert boden is not None and boden <= 0.01
    je2 = [j + (0.6 if a == "A" else 0.0) for j, a in zip(je, arme)]
    _, pb, _ = G["armnull"](je2, arme, perm=500, rnd=random.Random(3))
    assert pb < 0.05, pb


def test_traeger_nur_wo_einer_ist(G):
    """Ohne Trendabzug ist jede wachsende Reihe langreichweitig korreliert und
       saehe periodisch aus. Und Lag 1 gehoert nicht dazu: Nachbarschritte
       haengen ueber Inhalt und Allokator zusammen, ein Gipfel dort ist kein
       Takt."""
    s0 = welt(200, np.random.RandomState(21))
    L0 = np.arange(200)
    per0, _, _ = G["traeger"](s0, L0, perm=200, rnd=random.Random(5))
    assert per0 is None, per0
    s1 = s0 + 3.0 * np.sin(2 * np.pi * np.arange(200) / 7.0)
    per1, _, _ = G["traeger"](s1, L0, perm=200, rnd=random.Random(5))
    assert per1 in (7, 14), per1
    # eine Reihe, deren groesster Gipfel bei Lag 1 saesse
    s2 = np.zeros(200)
    for i in range(1, 200):
        s2[i] = 0.95 * s2[i - 1] + np.random.RandomState(i).randn() * 0.1
    per2, _, _ = G["traeger"](s2, L0, perm=200, rnd=random.Random(5))
    assert per2 != 1, per2


def test_kippzahl_misst_den_unterschied(G):
    assert G["kippzahl"]([(0, 1), (0, 2)], [(0, 1), (0, 3)]) == 2
    assert G["kippzahl"]([(0, 1)], [(0, 1)]) == 0
    assert G["kippzahl"]([], []) == 0
    assert G["kippzahl"]([(0, 1), (0, 2)], []) == 2


def test_urteilsordnung(G):
    """Die Reihenfolge ist die Aussage. Sperren stechen Befunde, und die
       Uhr gilt nur dann als blind, wenn BEIDE Kanaele nichts hergeben - der
       erste echte Lauf verkuendete UHR-BLIND, waehrend im selben Protokoll
       eine Dosiskurve mit p=0.0005 stand."""
    U = G["urteil_takt"]
    ok = dict(eps_schritt=800, eps_lauf=0.2, icc_p=0.001, sonde="GLEICHWERTIG",
              kappa_p=0.001, ident="GLEICHWERTIG", traeger_p=None, gates=[])
    def u(**k):
        d = dict(ok); d.update(k)
        return U(d["eps_schritt"], d["eps_lauf"], d["icc_p"], d["sonde"],
                 d["kappa_p"], d["ident"], d["traeger_p"], d["gates"])
    assert u() == "ZEIT-SIEHT-VIELFALT-NICHT-IDENTITAET"
    assert u(ident="VERSCHIEDEN") == "ZEIT-SIEHT-IDENTITAET"
    assert u(kappa_p=0.9) == "ZEIT-SIEHT-NICHTS"
    # nur der Schrittkanal blind: der Laufkanal traegt weiter
    assert u(eps_schritt=None) == "ZEIT-SIEHT-VIELFALT-NICHT-IDENTITAET"
    assert u(eps_lauf=None) == "ZEIT-SIEHT-VIELFALT-NICHT-IDENTITAET"
    assert u(eps_schritt=None, eps_lauf=None) == "UHR-BLIND"
    # Sondensperre sticht jeden Befund
    assert u(sonde="VERSCHIEDEN") == "SONDE-IST-SIGNAL"
    # fehlendes Profil sperrt nur H4 und haengt sich als Zusatz an
    assert u(icc_p=0.9) == "ZEIT-SIEHT-VIELFALT-NICHT-IDENTITAET-OHNE-PROFIL"
    # Gates stechen alles
    assert u(gates=[("ARCHITEKTUR-NICHT-GEFUNDEN", False)]) == "ARCHITEKTUR-NICHT-GEFUNDEN"
