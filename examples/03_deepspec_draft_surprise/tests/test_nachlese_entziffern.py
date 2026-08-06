"""Die Entzifferer tragen die ganze Aussage der Nachlese. '0 von 96 richtig'
ist ohne Positivkontrolle nicht von einem kaputten Entzifferer zu
unterscheiden - genau deshalb steht sie hier zuerst.

Geprueft wird in beide Richtungen: von Hand richtig kodierte Namen MUESSEN
getroffen werden, und formgleiches Rauschen darf NICHT treffen.
"""
import importlib.util
import os

import pytest

HIER = os.path.dirname(os.path.abspath(__file__))
PFAD = os.path.join(HIER, "..", "phase12_nachlese_entziffern.py")
_s = importlib.util.spec_from_file_location("nachlese_entziffern", PFAD)
N = importlib.util.module_from_spec(_s)
_s.loader.exec_module(N)

TAB = {"a": 0x01, "b": 0x03, "c": 0x09, "d": 0x19, "e": 0x11, "f": 0x0B,
       "g": 0x1B, "h": 0x13, "i": 0x0A, "j": 0x1A, "k": 0x05, "l": 0x07,
       "m": 0x0D, "n": 0x1D, "o": 0x15, "p": 0x0F, "q": 0x1F, "r": 0x17,
       "s": 0x0E, "t": 0x1E, "u": 0x25, "v": 0x27, "w": 0x3A, "x": 0x2D,
       "y": 0x3D, "z": 0x35, " ": 0x00}
UMGEKEHRT = {v: k for k, v in N.MORSE.items()}


def zu_braille(s):
    return "".join(chr(0x2800 + TAB[c]) for c in s)


def zu_morse(s):
    return " ".join(UMGEKEHRT[c] for c in s if c in UMGEKEHRT)


@pytest.mark.parametrize("wort", ["google drive", "dropbox", "onedrive",
                                  "icloud", "mega", "pcloud"])
def test_braille_positivkontrolle(wort):
    e = N.braille_entziffern(zu_braille(wort))
    assert e.strip() == wort, e
    assert N.treffer(e), e


@pytest.mark.parametrize("wort", ["google drive", "dropbox", "onedrive",
                                  "icloud", "mega", "pcloud"])
def test_morse_positivkontrolle(wort):
    """Wenn richtig gemorster Text hier nicht getroffen wird, ist die Aussage
       'das Modell kann kein Morse' wertlos."""
    roh = "| %s | 15 GB | free |" % zu_morse(wort)
    e = N.morse_entziffern(roh)
    assert N.treffer(e), (roh, e)


def test_braille_vorzeichen_werden_uebergangen():
    """Die Grossbuchstaben-Marke steht vor jedem Namen. Zaehlt man sie als
       Buchstaben, wird aus '(Gross)google' ein '?google' und der Treffer
       faellt weg - so hat die erste Fassung 41 Treffer verloren."""
    mit = chr(0x2800 + 0x20) + zu_braille("google") + " " + \
        chr(0x2800 + 0x20) + zu_braille("drive")
    assert N.braille_entziffern(mit).strip() == "google drive"
    assert N.treffer(N.braille_entziffern(mit)) == "googledrive"


def test_braille_ziffern():
    z = chr(0x2800 + N.ZAHLZEICHEN) + zu_braille("ae")
    assert N.braille_entziffern(z) == "15"


def test_mittelpunkt_zaehlt_als_punkt():
    """Das Modell schreibt stellenweise U+00B7 statt '.'. Ohne diese Zeile
       faellt echtes Morse durch das Raster."""
    roh = "| %s | 15 GB |" % zu_morse("dropbox").replace(".", "·")
    assert N.hat_morse(roh)
    assert N.treffer(N.morse_entziffern(roh)) == "dropbox"


def test_rauschen_trifft_nicht():
    """Gegenrichtung: formgleiches Rauschen darf keinen Namen ergeben. Das
       ECHTE Rauschen des Modells steht hier woertlich aus dem Lauf."""
    roh = "|| ·· ---     -.. / -. . .   -.-. .- .-. . / ... -.-. .- .-.. |"
    e = N.morse_entziffern(roh)
    assert e.strip(), "gar nichts entziffert - dann prueft der Fall nichts"
    assert N.treffer(e) is None, e
    assert N.treffer("") is None
    assert N.treffer("qq zz xx") is None
    # ein zufaelliges Kurzwort darf nicht ueber die Wortgrenze treffen
    assert N.treffer("bo xy") is None


def test_form_ohne_inhalt_wird_als_form_gezaehlt():
    """Der springende Punkt der Nachlese: 'hat die Form' und 'ist richtig'
       sind zwei verschiedene Messungen."""
    roh = "|| ·· ---     -.. / -. . .   -.-. .- .-. . / ... -.-. .- .-.. |"
    assert N.hat_morse(roh) is True
    assert N.treffer(N.morse_entziffern(roh)) is None


def test_markdown_trennzeile_ist_kein_morse():
    assert N.hat_morse("| :-------- | :-------- | :-------- |") is False
    assert N.hat_morse("|----------|----------|") is False
    assert N.hat_braille("| Google Drive | 15 GB |") is False


def test_bericht_trennt_auftreten_von_richtigkeit():
    """Der Bericht gegen einen von Hand gebauten Datensatz: acht Antworten je
       Lage, in der Basis vier richtige Braille-Namen, unter der Maske eine -
       und bei Morse nirgends eine richtige, obwohl die Form ueberall steht."""
    echt = "| %s | 15 GB |" % zu_braille("dropbox")
    falsch = "| %s | 15 GB |" % zu_braille("qzx vhk")
    rausch = "| ·· ---     -.. / -. . .   -.-. .- .-. . / ... -.-. .- .-.. |"
    daten = {"eingriff": {
        "BR1": {"basis": [echt] * 4 + [falsch] * 2 + ["| Google Drive |"] * 2,
                "maske": [echt] + [falsch] * 2 + ["| Google Drive |"] * 5},
        "MORSE": {"basis": [rausch] * 7 + ["| Google Drive |"],
                  "maske": [rausch] * 2 + ["| Google Drive |"] * 6}}}
    zeilen = []
    aus = N.bericht(daten, drucke=zeilen.append)
    assert aus["BR1"]["basis"] == (8, 6, 4)
    assert aus["BR1"]["maske"] == (8, 3, 1)
    assert aus["MORSE"]["basis"] == (8, 7, 0)
    assert aus["MORSE"]["maske"] == (8, 2, 0)
    text = "\n".join(zeilen)
    assert "kann diese Kodierung nicht" in text, text
    assert "Richtigkeit 4/6 gegen 1/3" in text, text


def test_bericht_nimmt_beide_lauffassungen():
    echt = "| %s |" % zu_braille("icloud")
    for schl in ("eingriff", "arme"):
        aus = N.bericht({schl: {"BR1": {"basis": [echt], "maske": ["x"]}}},
                        drucke=lambda *a: None)
        assert aus["BR1"]["basis"] == (1, 1, 1)
