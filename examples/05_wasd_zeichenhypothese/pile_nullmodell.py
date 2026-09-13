"""Das Nullmodell, das fehlt: Dokumente aus dem Trainingskorpus statt Buchstabenmengen.

Die Laeufe vom 13.09.2026 vergleichen State 60482 mit zwei Bezugsgroessen: achtzehn
Geschwisterfenster und zufaellig gezogene Achterbuchstabenmengen. Beide sind schmal.
Die Frage "ist dieses Textfenster fuer diese Buchstaben auffaellig?" hat eine
naheliegende dritte Bezugsgroesse, naemlich **gewoehnliche Dokumente aus demselben
Korpus, auf dem Pythia trainiert wurde**.

Dieses Skript holt eine Stichprobe aus ``NeelNanda/pile-10k`` - 10 000 Dokumente aus
The Pile - ueber den oeffentlichen Datensatz-Dienst von Hugging Face und rechnet zwei
Dinge aus:

1. die Verteilung der Zielbuchstaben-Dichte ueber die Dokumente, und wo State 60482
   darin liegt;
2. die **Basisrate der Zeichenfolge WASD** im Korpus - also wie oft das Tastenkuerzel
   ueberhaupt vorkommt, getrennt nach Schreibweise und nur an Wortgrenzen.

Ergebnis des gespeicherten Laufs (``daten/PILE_DOKUMENT_NULL.json``, 2 863
Dokumente, 20 899 761 Zeichen):

===========================================  ==============
Mittlere WASDQERF-Dichte ueber Dokumente      0.408650
Standardabweichung                            0.029075
State 60482                                   0.386819
z                                             -0.7508
Perzentil                                     18.83
WASD als eigenstaendiges Wort                  0 (in keiner Schreibweise)
===========================================  ==============

Zur zweiten Zeile eine Warnung in eigener Sache: eine erste Fassung dieses Skripts
suchte ``wasd`` als **Teilkette** und meldete 22 Treffer. Alle stammten aus dem
englischen Ortsnamen *Wasdale* und der Domain *wasdaleweb.com*, beide aus einem
einzigen Reisefuehrer-Dokument. Kein einziger war das Tastenkuerzel. Seither zaehlt
``zaehle_varianten`` nur an Wortgrenzen und schluesselt nach Schreibweise auf.

Aufruf::

    python pile_nullmodell.py --dokumente 3000 --speichern daten/
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import string
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence, cast

DIENST = "https://datasets-server.huggingface.co/rows"
DATENSATZ = "NeelNanda/pile-10k"
KONFIGURATION = "default"
SPLIT = "train"

#: Der Messwert aus dem Lauf, gegen den verglichen wird.
STATE60482_DICHTE = 0.3868194842406877

#: Relative Buchstabenhaeufigkeiten im Englischen (Norvig, Google-Books-Korpus).
ENGLISCH: Mapping[str, float] = {
    "e": 0.1249, "t": 0.0928, "a": 0.0804, "o": 0.0764, "i": 0.0757,
    "n": 0.0723, "s": 0.0651, "r": 0.0628, "h": 0.0505, "l": 0.0407,
    "d": 0.0382, "c": 0.0334, "u": 0.0273, "m": 0.0251, "f": 0.0240,
    "p": 0.0214, "g": 0.0187, "w": 0.0168, "y": 0.0166, "b": 0.0148,
    "v": 0.0105, "k": 0.0054, "x": 0.0023, "j": 0.0016, "q": 0.0012,
    "z": 0.0009,
}

#: Dokumente unter dieser Buchstabenzahl sind zu kurz fuer eine stabile Dichte.
MINDESTLAENGE = 200


@dataclass(frozen=True)
class Korpusbefund:
    """Zusammenfassung eines Laufs."""

    datensatz: str
    n_dokumente: int
    n_zeichen: int
    zielbuchstaben: str
    erwartete_dichte_englisch: float
    mittel: float
    standardabweichung: float
    median: float
    perzentile: dict[str, float]
    state60482_dichte: float
    state60482_z: float
    state60482_perzentil: float
    wasd_vorkommen: int
    wasd_dokumente: int
    wasd_je_million_zeichen: float
    wasd_nach_schreibweise: dict[str, int]


def _hole_zeilen(offset: int, laenge: int, versuche: int = 4) -> list[str]:
    """Holt Dokumenttexte vom Datensatz-Dienst; leere Liste, wenn es nicht klappt."""
    parameter = urllib.parse.urlencode(
        {
            "dataset": DATENSATZ,
            "config": KONFIGURATION,
            "split": SPLIT,
            "offset": offset,
            "length": laenge,
        }
    )
    for versuch in range(versuche):
        try:
            with urllib.request.urlopen(f"{DIENST}?{parameter}", timeout=120) as antwort:
                roh = cast(object, json.loads(antwort.read().decode("utf-8")))
        except Exception:
            time.sleep(5 * (versuch + 1))
            continue
        if not isinstance(roh, dict):
            time.sleep(5 * (versuch + 1))
            continue
        zeilen = cast("dict[str, object]", roh).get("rows")
        if not isinstance(zeilen, list):
            time.sleep(5 * (versuch + 1))
            continue
        texte: list[str] = []
        for eintrag in cast("list[object]", zeilen):
            if not isinstance(eintrag, dict):
                continue
            inhalt = cast("dict[str, object]", eintrag).get("row")
            if not isinstance(inhalt, dict):
                continue
            text = cast("dict[str, object]", inhalt).get("text")
            if isinstance(text, str):
                texte.append(text)
        return texte
    return []


def dichte(text: str, zielbuchstaben: str) -> float | None:
    """Anteil der Zielbuchstaben an allen Buchstaben; ``None`` bei zu kurzem Text."""
    ziel = set(zielbuchstaben.lower())
    treffer = 0
    gesamt = 0
    for zeichen in text:
        klein = zeichen.lower()
        if klein in string.ascii_lowercase:
            gesamt += 1
            if klein in ziel:
                treffer += 1
    if gesamt < MINDESTLAENGE:
        return None
    return treffer / gesamt


def zaehle_zeichenfolge(text: str, folge: str) -> int:
    """Zaehlt die Zeichenfolge als **Wort**, in den drei ueblichen Schreibweisen.

    Die Wortgrenze ist hier keine Feinheit, sondern der Unterschied zwischen Messen
    und Danebenmessen. Eine reine Teilkettensuche nach ``wasd`` findet in The Pile
    vor allem den englischen Ortsnamen *Wasdale* (Wasdale Head im Lake District)
    und die Domain *wasdaleweb.com* - und keinen einzigen Beleg fuer das
    Tastenkuerzel. Ohne Wortgrenze haette diese Auswertung 22 Treffer gemeldet, von
    denen keiner der gesuchte war.
    """
    return sum(zaehle_varianten(text, folge).values())


def zaehle_varianten(text: str, folge: str) -> dict[str, int]:
    """Wie ``zaehle_zeichenfolge``, aber nach Schreibweise aufgeschluesselt."""
    muster = "|".join(
        re.escape(schreibweise)
        for schreibweise in (folge.upper(), folge.lower(), folge.capitalize())
    )
    gefunden = re.findall(rf"(?<![A-Za-z0-9])(?:{muster})(?![A-Za-z0-9])", text)
    zaehler = {
        folge.upper(): 0,
        folge.lower(): 0,
        folge.capitalize(): 0,
    }
    for treffer in gefunden:
        zaehler[treffer] = zaehler.get(treffer, 0) + 1
    return zaehler


def werte_aus(
    dichten: Sequence[float],
    n_zeichen: int,
    wasd_vorkommen: int,
    wasd_dokumente: int,
    zielbuchstaben: str,
    wasd_nach_schreibweise: Mapping[str, int] | None = None,
) -> Korpusbefund:
    if len(dichten) < 30:
        raise ValueError(f"zu wenige Dokumente fuer eine Verteilung: {len(dichten)}")
    sortiert = sorted(dichten)
    mittel = statistics.fmean(dichten)
    streuung = statistics.pstdev(dichten)
    if streuung == 0.0:
        # Ein z gegen eine Verteilung ohne Streuung waere keine Kennzahl, sondern
        # eine Division durch null, die als 0.0 getarnt harmlos aussieht.
        raise ValueError("Nullverteilung ohne Streuung: z waere nicht definiert")
    unter = sum(1 for wert in dichten if wert <= STATE60482_DICHTE)
    perzentile = {
        f"P{stufe:02d}": sortiert[int(stufe / 100 * (len(sortiert) - 1))]
        for stufe in (1, 5, 25, 50, 75, 95, 99)
    }
    return Korpusbefund(
        datensatz=DATENSATZ,
        n_dokumente=len(dichten),
        n_zeichen=n_zeichen,
        zielbuchstaben=zielbuchstaben,
        erwartete_dichte_englisch=sum(ENGLISCH[c] for c in set(zielbuchstaben.lower())),
        mittel=mittel,
        standardabweichung=streuung,
        median=statistics.median(dichten),
        perzentile=perzentile,
        state60482_dichte=STATE60482_DICHTE,
        state60482_z=(STATE60482_DICHTE - mittel) / streuung,
        state60482_perzentil=100 * unter / len(dichten),
        wasd_vorkommen=wasd_vorkommen,
        wasd_dokumente=wasd_dokumente,
        wasd_je_million_zeichen=1e6 * wasd_vorkommen / n_zeichen if n_zeichen else 0.0,
        wasd_nach_schreibweise=dict(wasd_nach_schreibweise or {}),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dokument-Nullmodell aus The Pile")
    parser.add_argument("--dokumente", type=int, default=3000, help="wie viele Zeilen holen")
    parser.add_argument("--zielbuchstaben", default="WASDQERF")
    parser.add_argument("--zeichenfolge", default="WASD")
    parser.add_argument("--speichern", default=None, help="Verzeichnis fuer Befund und Dichten")
    args = parser.parse_args(argv)

    ziel = cast(int, args.dokumente)
    zielbuchstaben = cast(str, args.zielbuchstaben)
    zeichenfolge = cast(str, args.zeichenfolge)

    dichten: list[float] = []
    n_zeichen = 0
    treffer_gesamt = 0
    treffer_dokumente = 0
    nach_schreibweise: dict[str, int] = {}
    for offset in range(0, ziel, 100):
        texte = _hole_zeilen(offset, 100)
        if not texte:
            print(f"  Abbruch bei Offset {offset}: keine Antwort")
            break
        for text in texte:
            wert = dichte(text, zielbuchstaben)
            if wert is None:
                continue
            dichten.append(wert)
            n_zeichen += len(text)
            varianten = zaehle_varianten(text, zeichenfolge)
            treffer = sum(varianten.values())
            treffer_gesamt += treffer
            for schreibweise, anzahl in varianten.items():
                nach_schreibweise[schreibweise] = nach_schreibweise.get(schreibweise, 0) + anzahl
            if treffer:
                treffer_dokumente += 1
        print(f"  Offset {offset}: {len(dichten)} Dokumente")

    befund = werte_aus(
        dichten,
        n_zeichen,
        treffer_gesamt,
        treffer_dokumente,
        zielbuchstaben,
        nach_schreibweise,
    )
    print(json.dumps(asdict(befund), indent=2, ensure_ascii=False))

    if args.speichern:
        verzeichnis = cast(str, args.speichern).rstrip("/")
        with open(f"{verzeichnis}/PILE_DOKUMENT_NULL.json", "w", encoding="utf-8") as handle:
            json.dump(asdict(befund), handle, indent=2, ensure_ascii=False)
        with open(f"{verzeichnis}/PILE_DOKUMENT_DICHTEN.csv", "w", encoding="utf-8") as handle:
            handle.write("dokument_index,dichte\n")
            for index, wert in enumerate(dichten):
                handle.write(f"{index},{wert:.10f}\n")
        print(f"Gespeichert nach {verzeichnis}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
