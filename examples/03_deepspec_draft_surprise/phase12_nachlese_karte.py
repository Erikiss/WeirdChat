"""Offline-Nachlese zum Sprachkarten-Lauf. Braucht nur 'antworten_karte.json'
aus dem Lauf-Ordner - keine GPU, kein Modell.

WARUM ES DIESES SKRIPT BRAUCHT: die Kennzahl 'kippt' des Laufs zaehlt, OB die
Antwort die Sprache wechselt, nicht WOHIN. Im Arm 'JA-exklusive -> LOC' laufen
zwei Bewegungen gegeneinander und heben sich in der Summe fast auf:

    CJK/Hangul-Kippen   21 -> 5      (p = 0.00043)
    lateinische Kippen   0 -> 10     (p = 0.00116)
    'kippt' insgesamt   21 -> 15     (p = 0.29, als 'still' verbucht)

Das Modell hoert also nicht auf zu kippen - es kippt in eine ANDERE SCHRIFT.
Genau das sieht man nur, wenn man die Zielsprache mitzaehlt statt bloss den
Wechsel. Dieselbe Falle hatte schon der Maskenlauf: dort verbuchte ein
gemeinsamer CJK-Zaehler chinesische Antworten als japanisch.

Aufruf:  python phase12_nachlese_karte.py antworten_karte.json
"""
import collections
import json
import re
import sys
import unicodedata

KANA = [(0x3040, 0x30FF)]
HANGUL = [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)]
HAN = [(0x3400, 0x9FFF), (0xF900, 0xFAFF)]
WORTE = {
    "pt": "nome nomes servico servicos armazenamento limite limites preco mes "
          "gratuito conta cada para com uma nao mais seu sua",
    "es": "nombre servicio servicios almacenamiento precio cuenta los las del "
          "con mas su gratuito",
    "fr": "le la les une un des est et pour avec dans votre vous voici du qui "
          "que sur cette ces aux ou par plus nom stockage prix tarif",
    "de": "dienst dienste speicher speicherplatz laufwerk grenze grenzen preis "
          "monat kostenlos konto jeder fuer mit eine der die das und nicht mehr "
          "zusammenfassung",
    "it": "nome servizio servizi archiviazione prezzo mese gratuito conto per "
          "con una non piu",
}
WORTE = {k: set(v.split()) for k, v in WORTE.items()}


def zaehl(t, bereiche):
    return sum(1 for c in t if any(a <= ord(c) <= b for a, b in bereiche))


def entakzent(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if not unicodedata.combining(c))


def latein_art(t):
    """welche lateinschriftliche Sprache - oder Englisch. 'akzent' ist der
       Auffangfall: drei oder mehr Akzentbuchstaben ohne genug Trefferwoerter,
       typisch fuer Sprachen ausserhalb der Wortlisten (im Lauf kam so
       Lettisch herein)."""
    w = re.findall(r"[a-zA-Z']+", entakzent(t).lower())
    treffer = sorted(((n, sum(1 for x in w if x in S)) for n, S in WORTE.items()),
                     key=lambda x: -x[1])
    if treffer[0][1] >= 3:
        return treffer[0][0]
    if sum(1 for c in t if c.isalpha() and 0xC0 <= ord(c) <= 0x17F) >= 3:
        return "akzent"
    return "englisch"


def schrift(t):
    """Kana beweist Japanisch, Hangul Koreanisch, Han allein nur CJK."""
    if zaehl(t, KANA) >= 2:
        return "japanisch"
    if zaehl(t, HANGUL) >= 2:
        return "koreanisch"
    if zaehl(t, HAN) >= 3:
        return "chinesisch"
    if zaehl(t, HAN) > 0:
        return "han-einzeln"
    return "latein:" + latein_art(t)


def kippt_cjk(t):
    return not schrift(t).startswith("latein")


def kippt_latein(t):
    a = schrift(t)
    return a.startswith("latein") and a != "latein:englisch"


def tafel(J, arme):
    arten = sorted({schrift(t) for _, k in arme for t in J[k]})
    print("%-30s " % "Arm" + " ".join("%11s" % a.replace("latein:", "l:") for a in arten))
    for nm, k in arme:
        c = collections.Counter(schrift(t) for t in J[k])
        print("%-30s " % nm + " ".join("%11d" % c[a] for a in arten))
    print("")
    print("%-30s %8s %8s %8s" % ("Arm", "CJK", "latein", "zusammen"))
    for nm, k in arme:
        T = J[k]
        a = sum(1 for t in T if kippt_cjk(t))
        b = sum(1 for t in T if kippt_latein(t))
        print("%-30s %8d %8d %8d" % (nm, a, b, a + b))


def beispiele(J, schluessel, pruef, wieviel=8):
    print("")
    print("BEISPIELE AUS %r" % schluessel)
    n = 0
    for t in J[schluessel]:
        if pruef(t):
            print("  %-16s %r" % (schrift(t), t[:100]))
            n += 1
            if n >= wieviel:
                break
    if not n:
        print("  keine")


def main(pfad):
    with open(pfad, encoding="utf-8") as f:
        J = json.load(f)
    arme = [(nm, k) for nm, k in
            (("LOC Grundlinie", "arm_LOC"),
             ("LOC-exklusive gesperrt", "loc_exklusiv"),
             ("zufaellige gemeinsame", "loc_zufall"),
             ("JA-exklusive -> LOC", "kreuz_JA-exklusive_zu_LOC-Arm"),
             ("JA Grundlinie", "arm_JA"),
             ("KO-exklusive -> JA", "kreuz_KO-exklusive_zu_JA-Arm"),
             ("KO Grundlinie", "arm_KO"),
             ("JA-exklusive -> KO", "kreuz_JA-exklusive_zu_KO-Arm"),
             ("NEU Grundlinie", "arm_NEU")) if k in J]
    assert arme, "keine bekannten Arme in %s: %s" % (pfad, list(J))
    tafel(J, arme)
    if "kreuz_JA-exklusive_zu_LOC-Arm" in J:
        beispiele(J, "kreuz_JA-exklusive_zu_LOC-Arm", kippt_latein)
    print("")
    print("Lesart: bleibt 'zusammen' ungefaehr gleich, waehrend sich CJK und")
    print("latein gegenlaeufig verschieben, dann ist der Lokalisierungsimpuls")
    print("erhalten und nur die Zielsprache ausgetauscht.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "antworten_karte.json")
