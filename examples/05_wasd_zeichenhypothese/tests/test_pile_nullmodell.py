"""Das Dokument-Nullmodell aus dem Trainingskorpus.

Zwei Sorten Tests: reine Rechenlogik gegen gepflanzte Wahrheiten (ohne Netz), und
Konsistenzpruefungen gegen den gespeicherten Lauf vom 13.09.2026 in ``daten/``.
"""

from __future__ import annotations

import csv
import json
import pathlib
import statistics
import sys
from typing import cast

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pile_nullmodell import (  # noqa: E402
    MINDESTLAENGE,
    STATE60482_DICHTE,
    dichte,
    werte_aus,
    zaehle_varianten,
    zaehle_zeichenfolge,
)

DATEN = pathlib.Path(__file__).resolve().parents[1] / "daten"


# --------------------------------------------------------------------------- #
# Rechenlogik
# --------------------------------------------------------------------------- #


def test_dichte_zaehlt_nur_buchstaben():
    text = "wasd" * 100 + "1234!?" * 20
    assert dichte(text, "WASD") == 1.0


def test_dichte_ist_none_bei_zu_kurzem_text():
    assert dichte("wasd", "WASD") is None
    assert dichte("a" * (MINDESTLAENGE - 1), "A") is None
    assert dichte("a" * MINDESTLAENGE, "A") == 1.0


def test_dichte_trifft_einen_gepflanzten_anteil():
    """Ein Viertel Zielbuchstaben, drei Viertel andere."""
    text = ("w" + "xyz") * 100
    wert = dichte(text, "WASD")
    assert wert is not None
    assert wert == pytest.approx(0.25)


def test_zeichenfolge_wird_in_drei_schreibweisen_gezaehlt():
    text = "WASD wasd Wasd WaSd"
    assert zaehle_zeichenfolge(text, "WASD") == 3  # WaSd zaehlt nicht
    assert zaehle_varianten(text, "WASD") == {"WASD": 1, "wasd": 1, "Wasd": 1}


def test_teilketten_zaehlen_nicht_mit():
    """Die Lehre aus dem eigenen Fehlalarm: *Wasdale* ist nicht das Tastenkuerzel.

    Eine erste Fassung suchte ``wasd`` als Teilkette und meldete 22 Treffer im
    Korpus - saemtlich aus dem Ortsnamen Wasdale und der Domain wasdaleweb.com.
    """
    text = "Wasdale Head and www.wasdaleweb.com and WASDQERF and WASD-only"
    varianten = zaehle_varianten(text, "WASD")
    assert varianten == {"WASD": 1, "wasd": 0, "Wasd": 0}
    assert "Wasdale" in text  # der Ortsname steht da, zaehlt aber nicht mit


def test_werte_aus_verlangt_genug_dokumente():
    with pytest.raises(ValueError, match="zu wenige"):
        werte_aus([0.4] * 10, 1000, 0, 0, "WASD")


def test_werte_aus_setzt_state60482_korrekt_in_die_verteilung():
    """Gepflanzte Wahrheit: eine Verteilung, deren Mittel deutlich ueber dem Ziel liegt."""
    dichten = [0.50] * 50 + [0.45] * 50
    befund = werte_aus(dichten, 1_000_000, 5, 2, "WASDQERF", {"WASD": 5})
    assert befund.n_dokumente == 100
    assert befund.mittel == pytest.approx(0.475)
    assert befund.state60482_z < -2
    assert befund.state60482_perzentil == 0.0
    assert befund.wasd_je_million_zeichen == pytest.approx(5.0)


def test_werte_aus_meldet_perzentil_hundert_wenn_alles_darunter_liegt():
    dichten = [0.10 + 0.001 * (index % 7) for index in range(100)]
    befund = werte_aus(dichten, 1000, 0, 0, "WASDQERF")
    assert befund.state60482_perzentil == 100.0
    assert befund.state60482_z > 0


def test_verteilung_ohne_streuung_wirft_statt_ein_z_von_null_zu_liefern():
    """Ein z gegen eine entartete Verteilung ist keine Kennzahl, sondern ein Artefakt."""
    with pytest.raises(ValueError, match="ohne Streuung"):
        werte_aus([0.4] * 100, 1000, 0, 0, "WASDQERF")


# --------------------------------------------------------------------------- #
# Der gespeicherte Lauf
# --------------------------------------------------------------------------- #


def _befund() -> dict[str, object]:
    with open(DATEN / "PILE_DOKUMENT_NULL.json", encoding="utf-8") as handle:
        geladen: object = json.load(handle)
    assert isinstance(geladen, dict)
    return cast("dict[str, object]", geladen)


def test_gespeicherter_lauf_hat_die_dokumentierte_groesse():
    befund = _befund()
    assert befund["datensatz"] == "NeelNanda/pile-10k"
    assert befund["n_dokumente"] == 2863
    assert befund["n_zeichen"] == 20899761


def test_state60482_liegt_im_unteren_fuenftel_des_korpus():
    """Der entscheidende Vergleich: gegen echte Dokumente statt gegen Buchstabenmengen."""
    befund = _befund()
    assert befund["state60482_dichte"] == pytest.approx(STATE60482_DICHTE)
    assert float(str(befund["state60482_z"])) == pytest.approx(-0.7508, abs=1e-3)
    assert float(str(befund["state60482_perzentil"])) < 20.0


def test_der_korpusmittelwert_liegt_nahe_der_englischen_erwartung():
    """0.4086 gemessen gegen 0.4134 erwartet - die Standardtabelle taugt als Naeherung."""
    befund = _befund()
    assert float(str(befund["mittel"])) == pytest.approx(0.4086, abs=2e-3)
    assert float(str(befund["erwartete_dichte_englisch"])) == pytest.approx(0.4134, abs=1e-4)


def test_wasd_kommt_im_korpus_ueberhaupt_nicht_vor():
    """Null Vorkommen als eigenstaendiges Wort in 20.9 Millionen Zeichen.

    Das stuetzt den Tokenizer-Befund: eine Zeichenfolge, die im Korpus gar nicht
    auftaucht, bekommt im BPE-Verfahren keinen eigenen Merge - und kann im
    Zielfenster auch nichts ausloesen.
    """
    befund = _befund()
    assert befund["wasd_vorkommen"] == 0
    assert befund["wasd_dokumente"] == 0
    assert befund["wasd_je_million_zeichen"] == 0.0
    nach_schreibweise = befund["wasd_nach_schreibweise"]
    assert isinstance(nach_schreibweise, dict)
    assert set(cast("dict[str, int]", nach_schreibweise)) == {"WASD", "wasd", "Wasd"}
    assert all(anzahl == 0 for anzahl in cast("dict[str, int]", nach_schreibweise).values())


def test_dichtetabelle_passt_zur_zusammenfassung():
    befund = _befund()
    with open(DATEN / "PILE_DOKUMENT_DICHTEN.csv", encoding="utf-8") as handle:
        werte = [float(zeile["dichte"]) for zeile in csv.DictReader(handle)]
    assert len(werte) == befund["n_dokumente"]
    assert statistics.fmean(werte) == pytest.approx(float(str(befund["mittel"])), abs=1e-9)
    assert statistics.median(werte) == pytest.approx(float(str(befund["median"])), abs=1e-9)
    unter = sum(1 for wert in werte if wert <= STATE60482_DICHTE)
    assert 100 * unter / len(werte) == pytest.approx(
        float(str(befund["state60482_perzentil"])), abs=1e-9
    )
