"""Warum die Zeichen-Linie vor dem WASD-Umschwung stecken blieb.

Der Lauf ``20260906_095415_e02a89`` sucht eine Zerlegung der Namensuebergaenge in
bitweise Primitive - Byteverschiebungen, Rotationen, ASCII-Nachfolger. Er waehlt je
Checkpoint und Schicht die besten vier aus.

Zwei Eigenschaften dieser Auswahl erklaeren, warum die Linie nicht weiterkam, und
beide stehen in den Zahlen des Laufs selbst:

1. Die vier Primitive erklaeren so gut wie nichts. Die Projektionsenergie liegt
   zwischen 1.3 und 3.3 Prozent; 98.4 bis 99.4 Prozent bleiben Residuum.
2. Die Auswahl ist zwischen zwei **benachbarten** Checkpoints instabil. Von zwoelf
   Plaetzen ueberleben fuenf, und einer davon - ``drop_last_byte`` - ist in allen
   sechs Auswahlen dabei und traegt deshalb keine Unterscheidung.

Der Lauf zieht daraus selbst das richtige Fazit: ``strong_bitwise_reduction_supported
= false``.
"""

from __future__ import annotations

import collections
import csv
import pathlib

DATEN = pathlib.Path(__file__).resolve().parents[1] / "daten"


def _auswahl() -> dict[tuple[str, str], set[str]]:
    with open(DATEN / "BITWISE_PRIMITIVE_AUSWAHL.csv", encoding="utf-8") as handle:
        return {
            (zeile["checkpoint"], zeile["layer"]): {
                zeile[f"primitive_{nummer}"] for nummer in (1, 2, 3, 4)
            }
            for zeile in csv.DictReader(handle)
        }


def _kennzahlen() -> list[dict[str, str]]:
    with open(DATEN / "BITWISE_PRIMITIVE_AUSWAHL.csv", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_sechs_auswahlen_mit_je_vier_primitiven():
    auswahl = _auswahl()
    assert len(auswahl) == 6
    for schluessel, menge in auswahl.items():
        assert len(menge) == 4, schluessel


def test_die_primitive_erklaeren_fast_nichts():
    """Projektionsenergie unter vier Prozent, Residuum ueber achtundneunzig."""
    zeilen = _kennzahlen()
    energien = [float(zeile["projection_energy"]) for zeile in zeilen]
    residuen = [float(zeile["residual_rms_fraction"]) for zeile in zeilen]
    assert max(energien) < 0.04
    assert min(residuen) > 0.98


def test_die_auswahl_haelt_zwischen_zwei_benachbarten_checkpoints_nicht():
    """Von zwoelf Plaetzen ueberleben fuenf - Schicht 14 zwei, Schicht 20 einen."""
    auswahl = _auswahl()
    stabil = {
        schicht: auswahl[("98000", schicht)] & auswahl[("99000", schicht)]
        for schicht in ("14", "20", "23")
    }
    assert len(stabil["14"]) == 2
    assert len(stabil["20"]) == 1
    assert len(stabil["23"]) == 2
    assert sum(len(menge) for menge in stabil.values()) == 5


def test_das_einzige_ueberall_gewaehlte_primitiv_unterscheidet_nichts():
    """``drop_last_byte`` steht in allen sechs Auswahlen und traegt deshalb nichts bei."""
    zaehler: collections.Counter[str] = collections.Counter()
    for menge in _auswahl().values():
        zaehler.update(menge)
    assert zaehler["drop_last_byte"] == 6
    nur_einmal = [name for name, anzahl in zaehler.items() if anzahl == 1]
    assert len(nur_einmal) == 10  # zehn von vierzehn Primitiven kommen genau einmal vor


def test_schicht_20_ist_die_schwaechste():
    """Ausgerechnet die Schicht, um die sich die fruehere Linie drehte."""
    zeilen = {(z["checkpoint"], z["layer"]): z for z in _kennzahlen()}
    for checkpoint in ("98000", "99000"):
        energie_20 = float(zeilen[(checkpoint, "20")]["projection_energy"])
        for schicht in ("14", "23"):
            assert energie_20 < float(zeilen[(checkpoint, schicht)]["projection_energy"])
