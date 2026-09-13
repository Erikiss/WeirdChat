"""Ein unkontrollierter Faktor in der frueheren McQuarrie-Linie.

Der Lauf ``20260906_120822_ba0e3d`` vergleicht Oberflaechenvarianten des Namens
McQuarrie (Grossschreibung, Homoglyphen, Tippfehler) anhand eines Nutzenmasses aus
der wahren Ziel-NLL. Die Varianten unterscheiden sich aber nicht nur im Zeichenbild,
sondern in der **Tokenisierung**: mal vier, mal fuenf Tokens, und mit
unterschiedlichen Schnittstellen.

Das hat zwei Folgen:

1. Ein Nutzenmass ueber eine unterschiedliche Zahl vorhergesagter Tokens ist
   zwischen Varianten nicht ohne Weiteres vergleichbar.
2. Der im Bericht analysierte Uebergang ``Qu -> ar`` existiert in drei der
   Varianten gar nicht: ``McOuarrie``, ``Mc0uarrie`` und ``McXuarrie`` zerfallen in
   ``ĠMc | O | uar | rie``. Es wird also nicht dieselbe Groesse verglichen.

Die Tests halten beides fest. Sie behaupten **nicht**, dass der Effekt des Laufs
dadurch erklaert ist - dafuer sind zehn Varianten zu wenig; der Zusammenhang
zwischen Tokenzahl und Nutzen ist im exakten Permutationstest nicht signifikant
(p = 0.181 bei 98k, p = 0.129 bei 99k, Vorzeichen bei 97k umgekehrt).
"""

from __future__ import annotations

import csv
import itertools
import pathlib
import statistics
from typing import NamedTuple

DATEN = pathlib.Path(__file__).resolve().parents[1] / "daten"


class Variante(NamedTuple):
    name: str
    n_tokens: int
    zerlegung: tuple[str, ...]
    benefit: dict[str, float]


def _lade_varianten() -> list[Variante]:
    with open(DATEN / "MCQ_SURFACE_VARIANTS_TOKENISIERUNG.csv", encoding="utf-8") as handle:
        zeilen = list(csv.DictReader(handle))
    ergebnis: list[Variante] = []
    for zeile in zeilen:
        nutzen = {
            schluessel: float(zeile[schluessel])
            for schluessel in ("benefit_97", "benefit_98", "benefit_99", "benefit_100")
            if zeile[schluessel]
        }
        ergebnis.append(
            Variante(
                name=zeile["variante"],
                n_tokens=int(zeile["n_tokens"]),
                zerlegung=tuple(zeile["zerlegung"].split("|")),
                benefit=nutzen,
            )
        )
    return ergebnis


def test_die_zerlegung_passt_zur_angegebenen_tokenzahl():
    for variante in _lade_varianten():
        assert len(variante.zerlegung) == variante.n_tokens, variante.name


def test_varianten_haben_unterschiedliche_tokenzahl():
    """Sechs Varianten zerfallen in vier Tokens, vier in fuenf."""
    zahlen = [variante.n_tokens for variante in _lade_varianten()]
    assert sorted(set(zahlen)) == [4, 5]
    assert zahlen.count(4) == 7  # inklusive der Originalschreibweise
    assert zahlen.count(5) == 4


def test_der_analysierte_uebergang_fehlt_in_drei_varianten():
    """``Qu -> ar`` gibt es in den O-, Null- und X-Varianten nicht."""
    varianten = {variante.name: variante for variante in _lade_varianten()}
    original = varianten["McQuarrie"].zerlegung
    assert original == ("ĠMc", "Qu", "ar", "rie")
    for name in ("McOuarrie", "Mc0uarrie", "McXuarrie"):
        zerlegung = varianten[name].zerlegung
        assert "Qu" not in zerlegung
        assert "uar" in zerlegung


def test_zusammenhang_zwischen_tokenzahl_und_nutzen_ist_nicht_signifikant():
    """Exakter Permutationstest ueber alle 210 Aufteilungen; n = 10 ist zu klein.

    Der Unterschied zeigt bei 98k und 99k in dieselbe Richtung (Fuenf-Token-Varianten
    schneiden besser ab), erreicht aber keine Signifikanz - und bei 97k dreht das
    Vorzeichen. Der Faktor ist damit unkontrolliert, nicht nachgewiesen.
    """
    varianten = [v for v in _lade_varianten() if "benefit_98" in v.benefit]
    assert len(varianten) == 10
    for spalte, mindest_p in (("benefit_98", 0.05), ("benefit_99", 0.05)):
        werte = [variante.benefit[spalte] for variante in varianten]
        vier = [v.benefit[spalte] for v in varianten if v.n_tokens == 4]
        fuenf = [v.benefit[spalte] for v in varianten if v.n_tokens == 5]
        beobachtet = statistics.fmean(fuenf) - statistics.fmean(vier)
        assert beobachtet > 0
        extremer = 0
        gesamt = 0
        for kombination in itertools.combinations(range(len(werte)), len(vier)):
            gruppe_a = [werte[i] for i in kombination]
            gruppe_b = [werte[i] for i in range(len(werte)) if i not in kombination]
            gesamt += 1
            if statistics.fmean(gruppe_b) - statistics.fmean(gruppe_a) >= beobachtet:
                extremer += 1
        assert gesamt == 210
        assert extremer / gesamt > mindest_p


def test_bei_97k_dreht_das_vorzeichen():
    varianten = [v for v in _lade_varianten() if "benefit_97" in v.benefit]
    vier = [v.benefit["benefit_97"] for v in varianten if v.n_tokens == 4]
    fuenf = [v.benefit["benefit_97"] for v in varianten if v.n_tokens == 5]
    assert statistics.fmean(fuenf) < statistics.fmean(vier)
