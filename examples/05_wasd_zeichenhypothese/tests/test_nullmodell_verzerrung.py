"""Der Beleg, dass das Random-8-Buchstaben-Nullmodell verzerrt ist.

Diese Tests lesen die Originalausgabe des Laufs vom 13.09.2026
(``daten/RANDOM_8LETTER_NULL.csv``, 19 Textfenster) und pruefen eine Aussage,
die nicht von einer Simulation abhaengt, sondern von den Messwerten selbst:

Das Nullmodell vergibt fast jedem gewoehnlichen englischen Text ein positives z.
Damit misst es die Haeufigkeit der Zielbuchstaben, nicht die Besonderheit des
Textes - und ein z von 1.084 fuer State 60482 ist kein Befund, sondern
unterdurchschnittlich fuer seine eigene Vergleichsgruppe.
"""

from __future__ import annotations

import csv
import pathlib
import statistics

import pytest

DATEN = pathlib.Path(__file__).resolve().parents[1] / "daten"


def _lade_null() -> list[dict[str, float | str]]:
    with open(DATEN / "RANDOM_8LETTER_NULL.csv", encoding="utf-8") as handle:
        zeilen = list(csv.DictReader(handle))
    for zeile in zeilen:
        for schluessel in (
            "observed_density",
            "random8_mean",
            "random8_p95",
            "random8_z",
            "random8_upper_tail_p",
        ):
            zeile[schluessel] = float(zeile[schluessel])
    return zeilen


def test_datensatz_hat_neunzehn_textfenster():
    assert len(_lade_null()) == 19


def test_state60482_werte_stimmen_mit_dem_bericht_ueberein():
    zeilen = {zeile["state_id"]: zeile for zeile in _lade_null()}
    ziel = zeilen["60482"]
    assert ziel["observed_density"] == pytest.approx(0.386819, abs=1e-6)
    assert ziel["random8_z"] == pytest.approx(1.083975, abs=1e-6)
    assert ziel["random8_upper_tail_p"] == pytest.approx(0.146685, abs=1e-6)


def test_nullmodell_meldet_fast_jedem_text_einen_ueberschuss():
    """18 von 19 gewoehnlichen Textfenstern bekommen ein positives z."""
    zs = [zeile["random8_z"] for zeile in _lade_null()]
    assert sum(1 for wert in zs if wert > 0) == 18
    assert statistics.fmean(zs) > 1.0


def test_state60482_liegt_unter_dem_durchschnitt_seiner_vergleichsgruppe():
    """Der Kern: das angebliche Signal ist schwaecher als bei den Geschwistertexten."""
    zeilen = _lade_null()
    zs = sorted((zeile["random8_z"] for zeile in zeilen), reverse=True)
    ziel = next(z["random8_z"] for z in zeilen if z["state_id"] == "60482")
    assert ziel < statistics.fmean(zs)
    assert zs.index(ziel) + 1 == 14  # Rang 14 von 19


def test_kein_textfenster_erreicht_signifikanz_nach_korrektur():
    """Kleinstes p ist 0.0239; bei 19 Tests haelt das keiner Bonferroni-Schwelle stand."""
    ps = [zeile["random8_upper_tail_p"] for zeile in _lade_null()]
    assert min(ps) == pytest.approx(0.023898, abs=1e-6)
    assert min(ps) > 0.05 / len(ps)


def test_buchstabenzusammensetzung_ist_nicht_einheitlich_erhoeht():
    """W, S, D liegen ueber dem Gruppenmittel - A und F deutlich darunter.

    Eine Hypothese, die WASD als Einheit behandelt, muss erklaeren, warum das A
    im Zieltext auf zwei Drittel des Gruppenmittels faellt.
    """
    with open(DATEN / "PER_LETTER_COMPOSITION.csv", encoding="utf-8") as handle:
        zeilen = {z["letter"]: z for z in csv.DictReader(handle)}
    hoch = [
        buchstabe
        for buchstabe, zeile in zeilen.items()
        if float(zeile["State60482_freq_per_alpha"]) > float(zeile["other18_mean_freq_per_alpha"])
    ]
    assert set(hoch) == {"W", "S", "D", "Q", "E", "R"}
    a = zeilen["A"]
    assert float(a["State60482_freq_per_alpha"]) < 0.7 * float(a["other18_mean_freq_per_alpha"])


def test_grossgeschriebene_zieltreffer_sind_halbiert():
    """Die Variante, die WASD als Tastenkuerzel lesen wuerde, ist im Zieltext seltener."""
    with open(DATEN / "STATE60482_RANKS.csv", encoding="utf-8") as handle:
        zeilen = {z["metric"]: z for z in csv.DictReader(handle)}
    gross = zeilen["exact_uppercase_target_count"]
    assert float(gross["ratio_vs_other18_mean"]) == pytest.approx(0.5088, abs=1e-3)
    assert float(gross["z_vs_19"]) < 0


def test_kontexttext_hat_die_dokumentierte_laenge():
    text = (DATEN / "STATE60482_CONTEXT.txt").read_text(encoding="utf-8")
    assert sum(1 for zeichen in text if zeichen.isalpha()) == 698
    assert "McQuarrie" in text
