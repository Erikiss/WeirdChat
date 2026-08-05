"""Offline-Nachlese zum Maskenlauf. Braucht nur 'antworten_maske.json' aus dem
Lauf-Ordner - keine GPU, kein Modell.

Der Lauf sagt, DASS die Japanisch-Rate von 100% auf 58.3% faellt, wenn man die
42 JP-exklusiven Experten sperrt. Er sagt nicht, WAS das Modell stattdessen
tut. Das entscheidet zwischen zwei sehr verschiedenen Lesarten:

  (a) die Anweisung faellt weg, der Rest bleibt intakt
  (b) das Modell ist beschaedigt und antwortet deshalb nicht mehr japanisch

Zwei Dinge sind dabei herausgekommen, die den Befund praezisieren:

 (1) Der Effekt war GROESSER als gemessen. Der Klassifikator des Laufs zaehlte
     den Han-Bereich 0x3400-0x9FFF, und der enthaelt chinesische Zeichen und
     japanische Kanji gleichermassen. Kana (0x3040-0x30FF) gibt es nur im
     Japanischen. Getrennt gezaehlt steht es 48/48 gegen 11/48 statt 28/48 -
     17 der vermeintlich japanischen Antworten waren chinesisch.

 (2) Eine Antwort lautet '| Service (Japanese Name) | ... (Google驅動) |'.
     Sie BEFOLGT die Anweisung - 'Google驅動' ist nur zwei Zeichen lang und
     faellt unter die Lauflaenge drei. Klassifikatorgrenze, kein Verhalten.

Aufruf:  python phase12_nachlese_maske.py antworten_maske.json
"""
import collections
import json
import re
import sys

KANA = [(0x3040, 0x30FF)]
HANGUL = [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)]
HAN = [(0x3400, 0x9FFF), (0xF900, 0xFAFF)]


def zaehl(t, bereiche):
    return sum(1 for c in t if any(a <= ord(c) <= b for a, b in bereiche))


def sprache(t):
    """Kana beweist Japanisch, Hangul beweist Koreanisch, Han allein nicht."""
    if not t.strip():
        return "leer"
    k, g, h = zaehl(t, KANA), zaehl(t, HANGUL), zaehl(t, HAN)
    if k >= 2:
        return "japanisch"
    if g >= 2:
        return "koreanisch"
    if h >= 3:
        return "han"
    if h > 0:
        return "han-einzeln"
    return "lateinisch"


def wiederholt(t, fenster=12, mal=4):
    if len(t) < fenster * mal:
        return False
    z = collections.Counter(t[i:i + fenster] for i in range(len(t) - fenster + 1))
    return max(z.values()) >= mal


def tabelle_sauber(t):
    """echte Markdown-Tabelle: Kopfzeile, Trennzeile, gleiche Spaltenzahl"""
    z = [x.strip() for x in t.splitlines() if x.strip().startswith("|")]
    if len(z) < 3 or not re.match(r"^\|[\s:|-]+\|?$", z[1]):
        return False
    n = [x.count("|") for x in z]
    return max(n) - min(n) <= 1


def befolgt(t, marke="Japanese"):
    """fremdes Schriftzeichen ODER die Kopfzeile nennt die Sprache beim Namen -
       beides ist Befolgung, die zweite ohne Ausfuehrung"""
    return (zaehl(t, KANA) + zaehl(t, HANGUL) + zaehl(t, HAN)) > 0 or marke.lower() in t.lower()


def tafel(J, arme):
    print("%-26s %5s %6s %8s %8s %8s %7s"
          % ("Arm", "n", "Kana", "Hangul", "Han o.K.", "einzeln", "latein"))
    for nm, s in arme:
        c = collections.Counter(sprache(t) for t in J[s])
        print("%-26s %5d %6d %8d %8d %8d %7d"
              % (nm, len(J[s]), c["japanisch"], c["koreanisch"], c["han"],
                 c["han-einzeln"], c["lateinisch"] + c["leer"]))
    print("")
    print("%-26s %5s %9s %10s %9s" % ("Arm", "n", "befolgt", "saub.Tab.", "Schleife"))
    for nm, s in arme:
        T = J[s]
        print("%-26s %5d %9d %10d %9d"
              % (nm, len(T), sum(1 for t in T if befolgt(t)),
                 sum(1 for t in T if tabelle_sauber(t)),
                 sum(1 for t in T if wiederholt(t))))


def einzeln(J, schluessel):
    rest = [t for t in J[schluessel] if sprache(t) not in ("japanisch",)]
    print("")
    print("DIE %d NICHT-JAPANISCHEN AUS '%s', EINZELN" % (len(rest), schluessel))
    for i, t in enumerate(rest):
        print("  %2d  %-14s Tab %-4s Schleife %-4s  %r"
              % (i + 1, sprache(t), "ja" if tabelle_sauber(t) else "NEIN",
                 "ja" if wiederholt(t) else "nein", t[:70]))


def main(pfad):
    with open(pfad, encoding="utf-8") as f:
        J = json.load(f)
    arme = [(nm, s) for nm, s in (("ohne Eingriff (JP-Arm)", "jp"),
                                  ("42 JP-exklusive gesperrt", "exklusiv"),
                                  ("42 zufaellige gemeinsame", "zufall"),
                                  ("NEU ohne Adjektiv", "neu")) if s in J]
    assert arme, "keine bekannten Arme in %s: %s" % (pfad, list(J))
    tafel(J, arme)
    if "exklusiv" in J:
        einzeln(J, "exklusiv")
    print("")
    print("Lesart: sieht der maskierte Arm strukturell aus wie der UNANGETASTETE")
    print("NEU-Arm (saubere Tabellen, Schleifen), dann ist der Effekt kein Zerfall,")
    print("sondern eine verlorene Sprachwahl.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "antworten_maske.json")
