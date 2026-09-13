"""Tests gegen gepflanzte Wahrheiten.

Der Kern dieser Mappe ist eine Behauptung ueber ein Nullmodell: das unbedingte
Modell (alle Buchstabenmengen gleicher Groesse) ist fuer haeufige Zielbuchstaben
verzerrt, das frequenzangepasste Modell ist es nicht. Diese Tests pflanzen die
Wahrheit jeweils selbst ein und pruefen, ob die Statistik sie findet - und, ebenso
wichtig, ob sie bei neutralem Text schweigt.
"""

from __future__ import annotations

import pathlib
import random
import string
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from zeichensatz_statistik import (  # noqa: E402
    ENGLISH_LETTER_FREQUENCY,
    analyse_letter_set,
    enrichment_ranking,
    expected_density,
    letter_counts,
    letter_density,
)

# Das reale Textfenster "State 60482" aus dem Pythia-Projekt, 698 alphabetische
# Zeichen, 207 Tokens im GPT-NeoX-Tokenizer. Die Tokens 124..127 sind
# "ĠMc", "Qu", "ar", "rie".
STATE60482 = (
    " stilettos pumps before ditching her designer heels to climb the steep stairs "
    "at the end of the carpet. It's just like you simply can not ask me to do "
    "something that you are not asking him [to do].\n\nStewart-who is now a jury "
    "member for the prestigious film festival-previously discussed the dress code "
    "during an interview with The Hollywood Reporter in 2017. It's just a given. "
    "People get very upset at you if you, like, don't wear heels or something-"
    "whatever.\n\nRelated News:\n\nMission: Impossible - Fallout is directed by "
    "Christopher McQuarrie , who is no stranger to working with Tom Cruise . The "
    "film will cling on to a chopper and land gracefully in United Kingdom cinemas "
    "on 26 July.\n\nThe Indian Meteorological Department (IMD) had said that the "
    "thunderstorm will continue for the next 48 to 72 hours. On Tuesday, the IMD "
    "had forecasted that the wind speeds could reach approximately 74 km"
)

WASD_QERF = "WASDQERF"


def _english_like(n_letters: int, seed: int) -> str:
    """Zieht Buchstaben gemaess englischer Haeufigkeit - Text ohne Sonderstruktur."""
    rng = random.Random(seed)
    letters = list(ENGLISH_LETTER_FREQUENCY)
    weights = [ENGLISH_LETTER_FREQUENCY[ch] for ch in letters]
    return "".join(rng.choices(letters, weights=weights, k=n_letters))


# --------------------------------------------------------------------------- #
# Grundrechenarten
# --------------------------------------------------------------------------- #


def test_letter_counts_ignoriert_nichtbuchstaben():
    counts = letter_counts("aA! 1 b\n")
    assert counts["a"] == 2
    assert counts["b"] == 1
    assert sum(counts.values()) == 3


def test_letter_density_ist_case_insensitiv():
    assert letter_density("AaBb", "ab") == 1.0
    assert letter_density("AaBb", "a") == 0.5


def test_leerer_text_wirft():
    with pytest.raises(ValueError):
        letter_density("1234 !?", "abc")


def test_nichtbuchstabe_in_zielmenge_wirft():
    with pytest.raises(ValueError):
        letter_density("abc", "a1")


def test_zielmenge_wird_dedupliziert():
    assert letter_density("aab", "aa") == letter_density("aab", "a")


def test_expected_density_summiert_haeufigkeiten():
    assert expected_density("e") == pytest.approx(0.1249)
    assert expected_density("wasdqerf") == pytest.approx(0.4134, abs=1e-4)


# --------------------------------------------------------------------------- #
# Der reale Messwert des Projekts
# --------------------------------------------------------------------------- #


def test_state60482_reproduziert_den_publizierten_dichtewert():
    """Der Lauf vom 13.09.2026 berichtet 0.386819 - exakt dieser Wert."""
    counts = letter_counts(STATE60482)
    assert sum(counts.values()) == 698
    assert letter_density(STATE60482, WASD_QERF) == pytest.approx(0.386819, abs=1e-6)


def test_state60482_liegt_unter_der_englischen_erwartung():
    """Beobachtet 0.3868 gegen erwartet 0.4134: die Zielmenge ist hier nicht dicht."""
    observed = letter_density(STATE60482, WASD_QERF)
    assert observed < expected_density(WASD_QERF)


# --------------------------------------------------------------------------- #
# Die eigentliche Behauptung: das unbedingte Nullmodell ist verzerrt
# --------------------------------------------------------------------------- #


def test_unbedingtes_nullmodell_meldet_positives_z_fuer_neutralen_text():
    """Gepflanzte Wahrheit: der Text ist neutral, also darf kein Befund entstehen.

    Trotzdem liefert das unbedingte Modell ein positives z, allein weil E, A, R, S
    haeufige Buchstaben sind. Genau dieser Fehlalarm ist der Grund fuer das
    frequenzangepasste Modell.
    """
    neutral = _english_like(4000, seed=11)
    stats = analyse_letter_set(neutral, WASD_QERF)
    assert stats.nulls["unconditional"].z > 0.8
    assert abs(stats.nulls["frequency_matched"].z) < 1.5
    assert stats.verdict() == "UNAUFFAELLIG"


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_frequenzangepasst_kalibriert_auf_neutralem_text(seed: int):
    """Fehlalarmrate: ueber mehrere neutrale Texte darf kein ANGEREICHERT fallen."""
    neutral = _english_like(3000, seed=seed)
    stats = analyse_letter_set(neutral, WASD_QERF)
    assert stats.verdict() != "ANGEREICHERT"


def test_gepflanzte_anreicherung_wird_gefunden():
    """Gepflanzte Wahrheit: 15 Prozent des Textes durch WASD-Buchstaben ersetzt."""
    rng = random.Random(99)
    base = list(_english_like(4000, seed=7))
    for index in rng.sample(range(len(base)), 600):
        base[index] = rng.choice("wasd")
    stats = analyse_letter_set("".join(base), WASD_QERF)
    assert stats.verdict() == "ANGEREICHERT"
    assert stats.nulls["frequency_matched"].p_upper < 0.05


def test_gepflanzte_abreicherung_wird_gefunden():
    """Gegenrichtung: Zielbuchstaben systematisch durch seltene ersetzt."""
    rng = random.Random(123)
    base = list(_english_like(4000, seed=8))
    for index, char in enumerate(base):
        if char in "wasdqerf" and rng.random() < 0.4:
            base[index] = rng.choice("xzjkv")
    stats = analyse_letter_set("".join(base), WASD_QERF)
    assert stats.verdict() == "ABGEREICHERT"


def test_state60482_ist_frequenzangepasst_nicht_auffaellig():
    """Der Projektbefund, korrekt genullt: kein Ueberschuss, eher ein Defizit."""
    stats = analyse_letter_set(STATE60482, WASD_QERF)
    assert stats.nulls["unconditional"].z > 1.0  # der publizierte Wert, 1.084
    assert stats.nulls["frequency_matched"].z < 0.0
    assert stats.nulls["frequency_matched"].p_upper > 0.5
    assert stats.verdict() == "UNAUFFAELLIG"


# --------------------------------------------------------------------------- #
# Ranking als Gegenprobe
# --------------------------------------------------------------------------- #


def test_ranking_findet_die_gepflanzte_menge():
    rng = random.Random(5)
    base = list(_english_like(4000, seed=5))
    for index in rng.sample(range(len(base)), 900):
        base[index] = rng.choice("jkx")
    top = enrichment_ranking("".join(base), set_size=3, min_expected=0.0, top=5)
    assert "jkx" in {name for name, _ in top}


def test_ranking_ist_absteigend_sortiert():
    top = enrichment_ranking(STATE60482, set_size=3, min_expected=0.0, top=20)
    werte = [wert for _, wert in top]
    assert werte == sorted(werte, reverse=True)


# --------------------------------------------------------------------------- #
# Fehlerbehandlung
# --------------------------------------------------------------------------- #


def test_zu_enges_band_wirft_statt_leerer_statistik():
    """Einzelbuchstabe mit Band null: nur der Buchstabe selbst passt, also Fehler."""
    with pytest.raises(ValueError, match="zu eng"):
        analyse_letter_set(STATE60482, "z", match_band=0.0)


def test_band_kann_die_tabellenaufloesung_nicht_unterschreiten():
    """Die Haeufigkeitstabelle hat vier Nachkommastellen.

    Ein Band unterhalb dieser Aufloesung engt die Referenzmenge deshalb nicht
    beliebig ein: sehr viele Achtermengen haben exakt dieselbe erwartete Dichte.
    Wer das Band verkleinert, um "strenger" zu testen, taeuscht sich.
    """
    eng = analyse_letter_set(STATE60482, WASD_QERF, match_band=1e-9)
    weit = analyse_letter_set(STATE60482, WASD_QERF, match_band=0.01)
    assert eng.nulls["frequency_matched"].n_reference_sets >= 30
    assert (
        eng.nulls["frequency_matched"].n_reference_sets
        < weit.nulls["frequency_matched"].n_reference_sets
    )


def test_alle_buchstaben_als_zielmenge_ergibt_dichte_eins():
    assert letter_density(STATE60482, string.ascii_lowercase) == 1.0
