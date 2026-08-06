# -*- coding: utf-8 -*-
"""Offline-Nachlese zur Schrift-Zelle. Braucht nur 'antworten_dosis.json' oder
'antworten_schrift.json' aus dem Lauf-Ordner - keine GPU, kein Modell.

Der Lauf misst, OB Braille oder Morse vorkommt. Er misst nicht, ob der
richtige Name dasteht. Damit laesst sich nicht trennen, was die 42
JP-exklusiven Experten eigentlich tragen:

  (a) das KOENNEN     - unter der Maske kaeme falsches Braille heraus
  (b) das VERSUCHEN   - unter der Maske kaeme gar keins

Entziffert man die Zeichen und vergleicht mit den Dienstnamen aus dem Prompt,
faellt die Antwort fuer die beiden Arme verschieden aus - und das war so nicht
zu erwarten:

  BRAILLE  ohne Maske  63.5 % enthalten Braille, davon 67 % RICHTIG entziffert
           mit Maske   15.6 % enthalten Braille, davon 33 % richtig
           Auftreten     p < 1e-5      Richtigkeit bedingt aufs Auftreten
                                       41/61 gegen 5/15,  p = 0.021
           -> beides faellt: seltener versucht UND schlechter gebaut

  MORSE    ohne Maske  89.6 % enthalten Morse, davon  0  richtig
           mit Maske   21.9 % enthalten Morse, davon  0  richtig
           -> das Modell KANN kein Morse. Es baut die Form - Punkte, Striche,
              Schraegstriche, gruppiert - und der Inhalt ist Rauschen:
              '·· --- -.. / -. . . -.-. .- .-. . / ... -.-. .- .-..'
              entziffert sich zu 'i o d nee care scal', nicht zu 'google drive'.

Das aendert die Lesart des Morse-Arms. Er ist kein Beleg ueber Koennen, weil da
kein Koennen ist. Er bleibt der schaerfste Beleg fuer das VERSUCHEN: reines
ASCII, kein Schriftwechsel, und die Bereitschaft, Zeichen fuer Zeichen etwas zu
konstruieren, haengt trotzdem an denselben 42 Experten.

Die Entzifferer sind gegen von Hand kodierte Beispiele geprueft (siehe
tests/test_nachlese_entziffern.py). Ohne diese Positivkontrolle waere '0 von
96' nicht von einem kaputten Entzifferer zu unterscheiden.

Aufruf:  python phase12_nachlese_entziffern.py antworten_dosis.json
"""
import json
import re
import sys

# Grad-1-Braille, Punktmuster als Bitmaske ueber U+2800
PUNKTE = {0x01: "a", 0x03: "b", 0x09: "c", 0x19: "d", 0x11: "e", 0x0B: "f",
          0x1B: "g", 0x13: "h", 0x0A: "i", 0x1A: "j", 0x05: "k", 0x07: "l",
          0x0D: "m", 0x1D: "n", 0x15: "o", 0x0F: "p", 0x1F: "q", 0x17: "r",
          0x0E: "s", 0x1E: "t", 0x25: "u", 0x27: "v", 0x3A: "w", 0x2D: "x",
          0x3D: "y", 0x35: "z", 0x00: " "}
ZIFFERN = {"a": "1", "b": "2", "c": "3", "d": "4", "e": "5",
           "f": "6", "g": "7", "h": "8", "i": "9", "j": "0"}
ZAHLZEICHEN = 0x3C
VORZEICHEN = {0x20, 0x30}          # Grossbuchstaben- und Grad-1-Marke
MORSE = {".-": "a", "-...": "b", "-.-.": "c", "-..": "d", ".": "e", "..-.": "f",
         "--.": "g", "....": "h", "..": "i", ".---": "j", "-.-": "k",
         ".-..": "l", "--": "m", "-.": "n", "---": "o", ".--.": "p",
         "--.-": "q", ".-.": "r", "...": "s", "-": "t", "..-": "u",
         "...-": "v", ".--": "w", "-..-": "x", "-.--": "y", "--..": "z",
         "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
         ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9"}
NAMEN = ["googledrive", "google", "drive", "dropbox", "onedrive", "icloud",
         "mega", "sync", "pcloud", "box", "mediafire", "terabox", "koofr",
         "tresorit", "nextcloud", "idrive", "amazon", "microsoft", "apple",
         "yandex", "backblaze", "degoo", "jottacloud"]


def braille_entziffern(t):
    aus = []
    zahl = False
    for c in t:
        o = ord(c)
        if not 0x2800 <= o <= 0x28FF:
            if aus and aus[-1] != " ":
                aus.append(" ")
            zahl = False
            continue
        m = o - 0x2800
        if m in VORZEICHEN:
            continue
        if m == ZAHLZEICHEN:
            zahl = True
            continue
        z = PUNKTE.get(m)
        if z is None:
            aus.append("?")
            zahl = False
            continue
        if z == " ":
            zahl = False
        aus.append(ZIFFERN.get(z, z) if (zahl and z in ZIFFERN) else z)
    return "".join(aus)


def morse_entziffern(t):
    """Der Mittelpunkt U+00B7 wird mitgenommen - das Modell benutzt ihn
       stellenweise statt des ASCII-Punkts, und ohne ihn faellt der halbe
       Text unter den Tisch."""
    t = t.replace("·", ".")
    aus = []
    for m in re.finditer(r"[.\-/ ]{8,}", t):
        s = m.group(0)
        if s.count("-") < 3 or s.count(".") < 3:
            continue
        for wort in re.split(r"\s*/\s*|   +", s.strip()):
            w = [MORSE.get(z, "?") for z in wort.split() if z and z.strip(".-") == ""]
            if w:
                aus.append("".join(w))
    return " ".join(aus)


def treffer(entziffert):
    """Steht in der entzifferten Folge ein Dienstname? Buchstabenrauschen
       trifft das nicht: kurze Marken muessen als ganzes Wort dastehen, lange
       duerfen ueber die Wortgrenze laufen, weil die Zeichensetzung beim
       Entziffern ohnehin verlorengeht."""
    w = re.findall(r"[a-z0-9]+", entziffert.lower())
    ganz = set(w)
    zus = "".join(w)
    for n in NAMEN:
        if n in ganz:
            return n
        if len(n) >= 5 and n in zus:
            return n
    return None


def hat_braille(t, mindest=3):
    return sum(1 for c in t if 0x2800 <= ord(c) <= 0x28FF) >= mindest


def hat_morse(t):
    for m in re.finditer(r"[.·\-/ ]{8,}", t):
        s = m.group(0)
        if s.count("-") >= 3 and (s.count(".") + s.count("·")) >= 3:
            return True
    return False


def fisher2x2(a, b, c, d):
    from math import exp, lgamma
    lf = (lambda n: lgamma(n + 1))
    n = a + b + c + d

    def pr(x):
        y = a + b - x
        z = a + c - x
        w = n - x - y - z
        if min(y, z, w) < 0:
            return 0.0
        return exp(lf(a + b) + lf(c + d) + lf(a + c) + lf(b + d)
                   - lf(n) - lf(x) - lf(y) - lf(z) - lf(w))

    hi = min(a + b, a + c)
    p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(0, hi + 1) if pr(x) <= p0 * (1 + 1e-9)))


def auswerten(texte, hat, entziffern):
    """(n, enthaelt die Form, davon richtig entziffert)"""
    treffend = [t for t in texte if hat(t)]
    ok = [t for t in treffend if treffer(entziffern(t))]
    return len(texte), len(treffend), len(ok)


ARME = (("BR1", hat_braille, braille_entziffern),
        ("BR2", hat_braille, braille_entziffern),
        ("MORSE", hat_morse, morse_entziffern))


def bericht(daten, drucke=print):
    """daten: {'eingriff'|'arme': {ARM: {lage: [texte]}}} - beide Lauffassungen"""
    wurzel = daten.get("eingriff") or daten.get("arme") or {}
    drucke("%-7s %-8s %5s %10s %8s %9s" % ("Arm", "Lage", "n", "hat Form",
                                           "richtig", "Anteil"))
    drucke("-" * 52)
    aus = {}
    for arm, hat, ent in ARME:
        if arm not in wurzel:
            continue
        lagen = {}
        for lage, texte in sorted(wurzel[arm].items()):
            if not isinstance(texte, list):
                continue
            n, h, o = auswerten(texte, hat, ent)
            lagen[lage] = (n, h, o)
            drucke("%-7s %-8s %5d %10d %8d %8.0f%%"
                   % (arm, lage, n, h, o, 100.0 * o / max(h, 1)))
        aus[arm] = lagen
        b = lagen.get("basis")
        if not b:
            continue
        for lage in sorted(k for k in lagen if k != "basis"):
            # JEDE Lage gegen die Basis - beim dosisgleichen Lauf sind das zwei
            # Kontraste, und genau ihr Unterschied ist die offene Frage: senkt
            # die Zufallsmaske auch die GUETE, oder nur die Rate nicht?
            m = lagen[lage]
            drucke("        %-7s Auftreten   %d/%d gegen %d/%d   p=%.5f"
                   % (lage, b[1], b[0], m[1], m[0],
                      fisher2x2(m[1], m[0] - m[1], b[1], b[0] - b[1])))
            if b[2] or m[2]:
                drucke("        %-7s Richtigkeit %d/%d gegen %d/%d   p=%.4f  "
                       "(bedingt aufs Auftreten)"
                       % (lage, b[2], b[1], m[2], m[1],
                          fisher2x2(m[2], m[1] - m[2], b[2], b[1] - b[2])))
            else:
                drucke("        %-7s Richtigkeit 0 gegen 0 - das Modell kann "
                       "diese Kodierung nicht." % lage)
                drucke("                Der Arm misst dann nur noch das "
                       "VERSUCHEN, nicht das Koennen.")
    return aus


def main(pfad):
    with open(pfad, encoding="utf-8") as f:
        bericht(json.load(f))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "antworten_dosis.json")
