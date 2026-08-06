"""Die CPU-Zelle phase12_nachlese_entziffern laeuft hier wirklich durch -
gegen ein vorgetaeuschtes Drive mit einer von Hand gebauten Antwortdatei.

Zwei Dinge muessen dabei stimmen: die Positivkontrolle im Kopf der Zelle darf
nicht bloss etwas ausdrucken, sondern muss den Lauf ANHALTEN, wenn ein
Entzifferer versagt - und der Bericht muss beide Masken des dosisgleichen
Laufs gegen die Basis rechnen, nicht nur eine.
"""
import builtins
import glob as glob_mod
import json
import os
import os.path
import tempfile

import pytest

HIER = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HIER, "..", "phase12_nachlese_entziffern.ipynb")
DRIVE = "/content/drive/MyDrive"

TAB = {"a": 0x01, "b": 0x03, "c": 0x09, "d": 0x19, "e": 0x11, "f": 0x0B,
       "g": 0x1B, "h": 0x13, "i": 0x0A, "j": 0x1A, "k": 0x05, "l": 0x07,
       "m": 0x0D, "n": 0x1D, "o": 0x15, "p": 0x0F, "q": 0x1F, "r": 0x17,
       "s": 0x0E, "t": 0x1E, "u": 0x25, "v": 0x27, "w": 0x3A, "x": 0x2D,
       "y": 0x3D, "z": 0x35, " ": 0x00}
zu_braille = (lambda s: "".join(chr(0x2800 + TAB[c]) for c in s))
RAUSCH = "|| ·· ---     -.. / -. . .   -.-. .- .-. . / ... -.-. .- .-.. |"


def koerper():
    with open(NB, encoding="utf-8") as f:
        nb = json.load(f)
    return "".join("".join(c.get("source", []))
                   for c in nb["cells"] if c.get("cell_type") == "code")


def datei(ordner):
    echt = "| %s | 15 GB |" % zu_braille("dropbox")
    falsch = "| %s | 15 GB |" % zu_braille("qzx vhk")
    leer = "| Google Drive | 15 GB |"
    d = {"prompt_id": "p1", "arme": {
        "BR1": {"basis": [echt] * 6 + [leer] * 2,
                "ja": [echt] + [falsch] * 3 + [leer] * 4,
                "zufall": [echt] * 5 + [falsch] + [leer] * 2},
        "MORSE": {"basis": [RAUSCH] * 7 + [leer],
                  "ja": [RAUSCH] * 2 + [leer] * 6,
                  "zufall": [RAUSCH] * 6 + [leer] * 2}}}
    p = os.path.join(ordner, "antworten_dosis.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    return p


def fahre(quelle, treffer_pfade, drucke):
    """Fuehrt die Zelle aus und taeuscht ihr das Drive vor."""
    echt_isdir, echt_glob = os.path.isdir, glob_mod.glob

    def isdir(p):
        return True if str(p).startswith(DRIVE) else echt_isdir(p)

    def glob(muster, recursive=False):
        if str(muster).startswith(DRIVE):
            name = os.path.basename(muster)
            return [p for p in treffer_pfade if os.path.basename(p) == name]
        return echt_glob(muster, recursive=recursive)

    os.path.isdir = isdir
    glob_mod.glob = glob
    ns = {"__builtins__": builtins, "print": drucke}
    try:
        exec(compile(quelle, "phase12_nachlese_entziffern", "exec"), ns)
    finally:
        os.path.isdir, glob_mod.glob = echt_isdir, echt_glob
    return ns


@pytest.fixture(scope="module")
def gelaufen():
    zeilen = []
    with tempfile.TemporaryDirectory() as d:
        ordner = os.path.join(d, "phase12_schrift_kontrolle_dosis_20260806")
        os.makedirs(ordner)
        ns = fahre(koerper(), [datei(ordner)], lambda *a: zeilen.append(
            " ".join(str(x) for x in a)))
    return ns, "\n".join(zeilen)


def test_zelle_laeuft_und_findet_die_datei(gelaufen):
    ns, text = gelaufen
    assert "ENTZ_RESULTS" in ns
    assert len(ns["ENTZ_RESULTS"]) == 1, ns["ENTZ_RESULTS"]
    (schl, erg), = ns["ENTZ_RESULTS"].items()
    assert schl.startswith("phase12_schrift_kontrolle_dosis"), schl
    assert erg["BR1"] == {"basis": (8, 6, 6), "ja": (8, 4, 1), "zufall": (8, 6, 5)}
    assert erg["MORSE"] == {"basis": (8, 7, 0), "ja": (8, 2, 0), "zufall": (8, 6, 0)}


def test_beide_masken_gegen_die_basis(gelaufen):
    _, text = gelaufen
    assert "ja      Richtigkeit 6/6 gegen 1/4" in text, text
    assert "zufall  Richtigkeit 6/6 gegen 5/6" in text, text
    assert "kann diese Kodierung nicht" in text, text


def test_positivkontrolle_haelt_den_lauf_an():
    """Die Sperre muss WIRKEN. Mit einem verstuemmelten Braille-Alphabet darf
       die Zelle nicht weiterlaufen und Zahlen ausgeben - sonst waere 'null
       Treffer' nicht von einem kaputten Entzifferer zu unterscheiden."""
    q = koerper().replace('0x1B: "g"', '0x1B: "?"', 1)
    assert q != koerper(), "Ankertext nicht gefunden"
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(AssertionError):
            fahre(q, [datei(d)], lambda *a: None)


def test_rauschen_sperre_haelt_den_lauf_an():
    """Gegenrichtung: ein Entzifferer, der ALLES trifft, ist ebenso wertlos."""
    q = koerper().replace('    for n in NAMEN:\n        if n in ganz:\n            return n',
                          '    for n in NAMEN:\n        if True:\n            return n', 1)
    assert q != koerper(), "Ankertext nicht gefunden"
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(AssertionError):
            fahre(q, [datei(d)], lambda *a: None)


def test_ohne_datei_kein_absturz():
    zeilen = []
    ns = fahre(koerper(), [], lambda *a: zeilen.append(" ".join(str(x) for x in a)))
    assert ns["ENTZ_RESULTS"] == {}
    assert "Keine gefunden" in "\n".join(zeilen)
