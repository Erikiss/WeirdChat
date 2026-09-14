"""Haelt das Kartierungs-Notebook an seiner Zusage fest: es belegt nichts.

Der Trägerlauf vom 14.09. ist an zwei Stellen gescheitert, die beide vorher
sichtbar gewesen waeren - haette jemand angesehen, was das Modell an der
Messposition tatsaechlich vorhersagt. Das Kartierungs-Notebook holt das nach. Damit
es diese Rolle behaelt, darf es keine Entscheidungsregel entwickeln: kein Schwellwert,
kein Urteil, keine Bestaetigung. Diese Tests pruefen genau das - und dass es dieselben
Konstanten benutzt wie das Designmodul, damit die Zahlen vergleichbar bleiben.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import experiment_traeger as modul  # noqa: E402

NOTEBOOK = (
    pathlib.Path(__file__).resolve().parents[1] / "wasd_grundlinien_karte_colab.ipynb"
)
KONFIGZELLEN = (2, 3)


def _zellen() -> list[dict[str, Any]]:
    with open(NOTEBOOK, encoding="utf-8") as handle:
        daten: Any = json.load(handle)
    return list(daten["cells"])


def _codequellen() -> list[tuple[int, str]]:
    return [
        (i, "".join(z["source"]))
        for i, z in enumerate(_zellen())
        if z["cell_type"] == "code"
    ]


def _gesamttext() -> str:
    return "\n".join(q for _, q in _codequellen())


def _markdown() -> str:
    return "\n".join(
        "".join(z["source"]) for z in _zellen() if z["cell_type"] == "markdown"
    )


@pytest.fixture(scope="module")
def konfig() -> dict[str, Any]:
    raum: dict[str, Any] = {}
    zellen = _zellen()
    for index in KONFIGZELLEN:
        exec(compile("".join(zellen[index]["source"]), f"zelle{index}", "exec"), raum)
    return raum


# --------------------------------------------------------------------------- #
# Die Datei
# --------------------------------------------------------------------------- #


def test_jede_codezelle_ist_uebersetzbar():
    for index, quelle in _codequellen():
        compile(quelle, f"zelle{index}", "exec")


def test_keine_ergebnisse_eines_frueheren_laufs():
    for zelle in _zellen():
        if zelle["cell_type"] == "code":
            assert zelle.get("outputs") == []
            assert zelle.get("execution_count") is None


# --------------------------------------------------------------------------- #
# Es darf kein Urteil faellen
# --------------------------------------------------------------------------- #


def test_das_notebook_faellt_kein_urteil():
    """Kein Bestaetigungsschalter, keine Entscheidungsschwelle, keine Regel.

    Faende sich hier eine Schwelle, waere der Lauf keine Kartierung mehr, sondern ein
    zweiter Test an denselben Schablonen - mit einem Endpunkt, der nach Kenntnis des
    ersten Laufs gewaehlt wurde. Genau das soll er nicht sein.
    """
    text = _gesamttext()
    for verboten in (
        "BESTAETIGT",
        "bestaetigt",
        "MINDEST_VORSPRUNG",
        "MINDEST_KAUSALANTEIL",
        "MINDEST_SCHICHTEN",
        "beobachtungsteil",
        "kausalteil",
    ):
        assert verboten not in text, verboten


def test_das_notebook_sagt_selbst_dass_es_nichts_belegt():
    assert "DESKRIPTIV" in _gesamttext()
    markdown = _markdown()
    assert "belegt nichts" in markdown
    assert "keine Entscheidungsregel" in markdown or "**keine** Entscheidungsregel" in markdown


def test_der_bericht_traegt_den_vorbehalt_mit_in_die_ausgabedatei():
    """Der Vorbehalt muss die Datei ueberleben, nicht nur im Notebook stehen."""
    text = _gesamttext()
    assert "Was daraus NICHT folgt" in text
    assert "dienen der Hypothesenbildung" in text


# --------------------------------------------------------------------------- #
# Vergleichbarkeit mit dem Traegerlauf
# --------------------------------------------------------------------------- #


def test_modell_und_tokenizer_sind_an_den_traegerlauf_gebunden(konfig: dict[str, Any]):
    assert konfig["MODELL_ID"] == modul.MODELL
    assert konfig["REVISION"] == modul.REVISION
    assert konfig["PRAEZISION"] == modul.PRAEZISION
    # Die Pruefsumme des Traegerlaufs steht als assert im Notebook.
    assert "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624" in _gesamttext()


def test_varianten_und_rollen_stimmen_mit_dem_modul_ueberein(konfig: dict[str, Any]):
    assert konfig["ZIEL"] == modul.ZIEL
    assert tuple(konfig["ZIEL_ZERLEGUNG"]) == modul.ZIEL_ZERLEGUNG
    assert konfig["REIHENFOLGE_KONTROLLE"] == modul.REIHENFOLGE_KONTROLLE
    assert tuple(konfig["ZUSATZKONTROLLEN"]) == modul.ZUSATZKONTROLLEN
    assert konfig["ROLLE_GETRENNT"] == modul.ROLLE_GETRENNT
    assert konfig["ROLLE_VERWORFEN"] == modul.ROLLE_VERWORFEN


def test_messvokabular_und_schablonen_sind_dieselben(konfig: dict[str, Any]):
    assert {f: tuple(w) for f, w in konfig["MESSVOKABULAR"].items()} == modul.MESSVOKABULAR
    assert {f: tuple(i) for f, i in konfig["MESSVOKABULAR_IDS"].items()} == modul.MESSVOKABULAR_IDS
    assert tuple(konfig["SCHABLONEN"]) == modul.SCHABLONEN


def test_die_vergleichsmenge_ist_die_designkonforme(konfig: dict[str, Any]):
    """23 Kontrollen, nicht 28 - der Fehler des Traegerlaufs darf nicht wiederkehren."""
    text = _gesamttext()
    assert "assert len(IN_DER_REGEL) == 23" in text
    assert 'v.rolle not in ("ziel", ROLLE_GETRENNT)' not in text
    assert "TRAGENDE_ROLLEN" in text


# --------------------------------------------------------------------------- #
# Was der Lauf leisten muss
# --------------------------------------------------------------------------- #


def test_die_top_fortsetzungen_werden_abgelegt(konfig: dict[str, Any]):
    """Der eigentliche Zweck: sehen, was dort steht, statt es anzunehmen."""
    assert konfig["TOP_K"] >= 20
    text = _gesamttext()
    assert "torch.topk" in text
    assert "haeufigste_fortsetzungen" in text


def test_die_volle_matrix_wird_abgelegt_nicht_nur_mittel():
    """Am Traegerlauf ging genau das verloren - und damit jede Nachrechnung."""
    text = _gesamttext()
    assert "zellen.csv" in text
    assert '"schablone": si' in text


def test_log_z_und_entropie_werden_gemessen():
    """Die Renormierungsfrage wird empirisch entschieden, nicht analytisch behauptet."""
    text = _gesamttext()
    assert "logsumexp(logits" in text
    assert "log_z" in text
    assert "entropie" in text


def test_der_renormierungsfreie_kontrast_wird_mitgefuehrt():
    """``tastatur - gegenfeld``: dort faellt log Z per Konstruktion heraus."""
    assert '"kontrast": werte["tastatur"] - werte["gegenfeld"]' in _gesamttext()


def test_gepaart_permutation_und_konsistenz_kommen_alle_vor():
    """Die drei Auswertungen, die am Traegerlauf nicht mehr moeglich waren."""
    text = _gesamttext()
    assert "def gepaart" in text
    assert "def vertauschungstest" in text
    assert "def exakter_test" in text
    assert "RAENGE" in text


def test_keine_freiheitsgrade_und_nenner_sind_hartverdrahtet():
    """Im Probelauf mit sechs Schablonen log die fest getippte Beschriftung ``t(23)``."""
    text = _gesamttext()
    assert "t(23)" not in text
    assert "/24" not in text
    assert "FG = len(SCHABLONEN) - 1" in text


def test_der_lauf_sichert_nach_drive():
    text = _gesamttext()
    assert "drive.mount" in text
    assert "copytree" in text
    assert "except ImportError" in text
