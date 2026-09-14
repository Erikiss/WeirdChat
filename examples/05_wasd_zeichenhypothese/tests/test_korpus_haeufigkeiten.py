"""Exakte Korpushäufigkeiten aus dem Volltextindex - und was eine Stichprobe kann.

Die Zahlen stammen aus dem infini-gram-Index ``v4_piletrain_llama`` über den
gesamten Pile-Trainingssatz. Sie ersetzen die Stichprobenschaetzung aus
``pile_nullmodell.py`` fuer die Frage nach der Basisrate.

Der zweite Teil dieser Datei rechnet nach, warum die Stichprobe die Frage gar nicht
beantworten konnte: bei 13 870 Vorkommen im ganzen Korpus sind in einem
Vierzigtausendstel davon etwa 0.35 Treffer zu erwarten. Null zu finden war das
Erwartbare, nicht der Befund.
"""

from __future__ import annotations

import csv
import math
import pathlib

import pytest

DATEN = pathlib.Path(__file__).resolve().parents[1] / "daten"

#: Groessenordnung des Pile-Trainingssatzes in Token, aus der Kalibrierung
#: ueber " the" (9 229 350 572 Vorkommen bei rund 2.4 Prozent Anteil).
PILE_TOKEN_GROESSENORDNUNG = 3.8e11

#: Zeichenumfang der Stichprobe in pile_nullmodell.py.
STICHPROBE_ZEICHEN = 20_899_761

#: Grober Zeichenumfang des ganzen Pile (rund 825 GiB Rohtext).
PILE_ZEICHEN_GROESSENORDNUNG = 8.25e11


def _haeufigkeiten() -> dict[str, int]:
    with open(DATEN / "INFINI_GRAM_PILE_HAEUFIGKEITEN.csv", encoding="utf-8") as handle:
        return {zeile["abfrage"]: int(zeile["anzahl"]) for zeile in csv.DictReader(handle)}


def test_alle_abfragen_haben_denselben_index():
    with open(DATEN / "INFINI_GRAM_PILE_HAEUFIGKEITEN.csv", encoding="utf-8") as handle:
        indizes = {zeile["index"] for zeile in csv.DictReader(handle)}
    assert indizes == {"v4_piletrain_llama"}


def test_wasd_kommt_im_ganzen_korpus_sehr_wohl_vor():
    """Der Punkt, an dem die Stichprobe in die Irre fuehrte."""
    haeufig = _haeufigkeiten()
    assert haeufig["WASD"] == 13870
    assert haeufig["WASD"] > 0


def test_wasd_ist_rund_zweihundertmal_seltener_als_keyboard():
    haeufig = _haeufigkeiten()
    verhaeltnis = haeufig["keyboard"] / haeufig["WASD"]
    assert 150 < verhaeltnis < 250


def test_jedes_fuenfte_wasd_wird_von_keys_gefolgt():
    """Ein starkes Kontextsignal: das Modell hat die Wendung gesehen."""
    haeufig = _haeufigkeiten()
    anteil = haeufig["WASD keys"] / haeufig["WASD"]
    assert anteil == pytest.approx(0.20, abs=0.01)


def test_die_relative_haeufigkeit_ist_zu_klein_fuer_einen_eigenen_merge():
    """13 870 in rund 380 Milliarden Token sind etwa 3.6e-8."""
    haeufig = _haeufigkeiten()
    relativ = haeufig["WASD"] / PILE_TOKEN_GROESSENORDNUNG
    assert relativ < 1e-7
    assert math.isclose(relativ, 3.6e-8, rel_tol=0.2)


def test_qerf_ist_praktisch_nicht_vorhanden():
    """Zehn Vorkommen im ganzen Korpus - die Nachbartasten sind keine Wendung."""
    haeufig = _haeufigkeiten()
    assert haeufig["QERF"] == 10
    assert haeufig["QERF"] < haeufig["WASD"] / 1000


def test_die_stichprobe_konnte_die_frage_gar_nicht_beantworten():
    """Erwartungswert in der Stichprobe: etwa ein Drittel eines Treffers.

    Genau deshalb ist "null Treffer in der Stichprobe" kein Beleg fuer "kommt nicht
    vor". Die Rechnung steht hier, damit der Fehler nicht wiederkehrt.
    """
    haeufig = _haeufigkeiten()
    anteil_der_stichprobe = STICHPROBE_ZEICHEN / PILE_ZEICHEN_GROESSENORDNUNG
    erwartet = haeufig["WASD"] * anteil_der_stichprobe
    assert erwartet < 1.0
    assert erwartet > 0.1
    # Wahrscheinlichkeit, bei diesem Erwartungswert null zu ziehen (Poisson):
    assert math.exp(-erwartet) > 0.6


def test_die_woerter_des_zielfensters_sind_deutlich_haeufiger():
    """McQuarrie und Meteorological stehen im Text - WASD nicht."""
    haeufig = _haeufigkeiten()
    assert haeufig["McQuarrie"] > haeufig["WASD"]
    assert haeufig["Meteorological"] > haeufig["McQuarrie"]
