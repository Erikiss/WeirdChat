"""Was die Zyklus- und Positionsanalyse des Laufs 20260913_171744 wirklich zeigt.

Der Lauf fragt, ob die Zielbuchstaben im Textfenster in der Tastaturreihenfolge
W→A→S→D→Q→E→R→F→W aufeinanderfolgen und ob ihre Token-Positionen periodisch sind.
Er beantwortet beides mit Nein. Diese Tests halten fest, *warum* das Nein
belastbar ist und an welchen Stellen die Kennzahlen leicht ueberinterpretiert
werden koennten.
"""

from __future__ import annotations

import csv
import pathlib
import statistics

import pytest

DATEN = pathlib.Path(__file__).resolve().parents[1] / "daten"
ZYKLUS = "WASDQERF"


def _matrix() -> dict[tuple[str, str], int]:
    with open(DATEN / "STATE60482_TRANSITION_MATRIX_LONG.csv", encoding="utf-8") as handle:
        return {(z["from"], z["to"]): int(z["count"]) for z in csv.DictReader(handle)}


def _kanonische_paare() -> list[tuple[str, str]]:
    return [(ZYKLUS[i], ZYKLUS[(i + 1) % len(ZYKLUS)]) for i in range(len(ZYKLUS))]


def test_matrix_ist_vollstaendig_und_summiert_auf_269_uebergaenge():
    matrix = _matrix()
    assert len(matrix) == 64
    assert sum(matrix.values()) == 269  # 270 Vorkommen, also 269 Uebergaenge


def test_kanonischer_anteil_reproduziert_den_berichteten_wert():
    matrix = _matrix()
    treffer = sum(matrix[paar] for paar in _kanonische_paare())
    assert treffer == 32
    assert treffer / 269 == pytest.approx(0.118959, abs=1e-6)


def test_kanonischer_anteil_liegt_unter_dem_zufallsniveau():
    """Acht Buchstaben, acht kanonische Nachfolger: Zufall waere 1/8 = 0.125."""
    matrix = _matrix()
    anteil = sum(matrix[paar] for paar in _kanonische_paare()) / 269
    assert anteil < 1 / 8


def test_der_zyklus_wird_von_einem_einzigen_paar_getragen():
    """E→R stellt fast die Haelfte aller kanonischen Treffer.

    E und R sind zwei der haeufigsten Buchstaben des Englischen; ihr Uebergang ist
    keine Tastaturbewegung, sondern gewoehnliche Orthographie. Zwei Glieder der
    Kette kommen ueberhaupt nicht vor.
    """
    matrix = _matrix()
    treffer = {paar: matrix[paar] for paar in _kanonische_paare()}
    assert treffer[("E", "R")] == 15
    assert treffer[("E", "R")] / sum(treffer.values()) > 0.45
    assert treffer[("D", "Q")] == 0
    assert treffer[("Q", "E")] == 0


def test_die_matrix_wird_von_e_dominiert():
    matrix = _matrix()
    ausgaenge = {
        buchstabe: sum(matrix[(buchstabe, ziel)] for ziel in ZYKLUS) for buchstabe in ZYKLUS
    }
    assert ausgaenge["E"] == 81
    assert max(ausgaenge, key=ausgaenge.get) == "E"
    assert ausgaenge["Q"] == 1


def test_kein_textfenster_zeigt_signifikante_positionsperiodizitaet():
    with open(DATEN / "TOKEN_POSITION_PERIODICITY.csv", encoding="utf-8") as handle:
        zeilen = [z for z in csv.DictReader(handle) if z["frame"] == "tail207"]
    assert len(zeilen) == 19
    ps = [float(z["period_perm_p"]) for z in zeilen]
    assert min(ps) > 0.05


def test_beste_periode_liegt_meist_am_rand_des_suchbereichs():
    """Der Scan laeuft bis 32, und in zehn Faellen ist genau 32 das Maximum.

    Ein Maximum am Rand des Suchbereichs ist ein Hinweis darauf, dass die Kennzahl
    mit der Periodenlaenge waechst, nicht dass eine Periode gefunden wurde. Die
    Permutationskorrektur faengt das ab - der rohe NMI-Wert taete es nicht.
    """
    with open(DATEN / "TOKEN_POSITION_PERIODICITY.csv", encoding="utf-8") as handle:
        zeilen = [z for z in csv.DictReader(handle) if z["frame"] == "tail207"]
    perioden = [int(z["best_period"]) for z in zeilen]
    assert perioden.count(32) == 10
    assert 8 not in perioden  # die Laenge des WASDQERF-Zyklus faellt nirgends heraus


def test_periode_acht_ist_bei_state60482_nicht_das_maximum():
    with open(DATEN / "TOKEN_POSITION_PERIODICITY.csv", encoding="utf-8") as handle:
        zeilen = {
            z["state_id"]: z for z in csv.DictReader(handle) if z["frame"] == "tail207"
        }
    ziel = zeilen["60482"]
    assert int(ziel["best_period"]) == 32
    assert float(ziel["period_perm_p"]) == pytest.approx(0.911089, abs=1e-6)
    assert float(ziel["period_perm_z"]) < 0


def test_zyklus_nullmodell_stellt_state60482_in_die_mitte():
    with open(DATEN / "ALL_5040_CYCLE_NULL.csv", encoding="utf-8") as handle:
        zeilen = [z for z in csv.DictReader(handle) if z["frame"] == "tail207"]
    assert len(zeilen) == 19
    ziel = next(z for z in zeilen if z["state_id"] == "60482")
    assert float(ziel["canonical_cycle_percentile_among_5040"]) == pytest.approx(47.7976, abs=1e-3)
    assert float(ziel["canonical_cycle_score"]) < float(ziel["all_cycle_mean"])
    perzentile = [float(z["canonical_cycle_percentile_among_5040"]) for z in zeilen]
    assert statistics.median(perzentile) > 50  # die Gruppe streut breit, der Zieltext nicht oben


def test_rangtabelle_zeigt_durchweg_kleine_effekte():
    with open(DATEN / "STATE60482_TRANSITION_POSITION_RANKS.csv", encoding="utf-8") as handle:
        zeilen = list(csv.DictReader(handle))
    assert len(zeilen) == 12
    zs = [float(z["z_vs_19"]) for z in zeilen]
    assert max(abs(wert) for wert in zs) < 0.8


def test_der_einzige_positive_befund_ist_ein_gleichstand():
    """``top3_longest_cycle8_run`` ist die einzige wahre Flagge des Laufs.

    Sie entsteht nicht durch einen Vorsprung, sondern durch einen Gleichstand:
    dreizehn der neunzehn Textfenster haben denselben Wert 3. State 60482 teilt
    sich Rang 3 mit zwoelf anderen. Dasselbe gilt fuer ``best_phase_match_L4``,
    wo sechzehn Fenster auf 0.75 liegen.
    """
    with open(DATEN / "TAIL207_ALL_METRICS.csv", encoding="utf-8") as handle:
        zeilen = list(csv.DictReader(handle))
    assert len(zeilen) == 19
    laeufe = [float(z["cycle8_longest_run"]) for z in zeilen]
    assert laeufe.count(3.0) == 13
    assert sum(1 for wert in laeufe if wert > 3.0) == 2
    phasen = [float(z["best_phase_match_L4"]) for z in zeilen]
    assert phasen.count(0.75) == 16


def test_erfolgsanteil_streut_eng_um_das_zufallsniveau():
    """Alle neunzehn Fenster liegen nahe 1/8; der Zieltext liegt darunter."""
    with open(DATEN / "TAIL207_ALL_METRICS.csv", encoding="utf-8") as handle:
        zeilen = list(csv.DictReader(handle))
    anteile = [float(z["cycle8_success_fraction"]) for z in zeilen if float(z["n_target_occurrences"]) > 50]
    assert statistics.fmean(anteile) == pytest.approx(0.125, abs=0.02)
    ziel = next(float(z["cycle8_success_fraction"]) for z in zeilen if z["state_id"] == "60482")
    assert ziel < statistics.fmean(anteile)
