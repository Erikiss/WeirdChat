"""Haelt das Notebook und das Designmodul zusammen.

Das Notebook muss in Colab eigenstaendig laufen: es kann ``experiment_traeger.py``
nicht importieren, weil die Datei dort nicht liegt. Also stehen die Konstanten des
Designs zweimal da - einmal im Modul, einmal in der Konfigurationszelle. Genau das
ist die Stelle, an der ein Lauf still an einem anderen Design messen wuerde als dem
vorregistrierten.

Diese Tests fuehren die Konfigurationszellen des Notebooks aus (sie brauchen weder
``torch`` noch ein Netz) und vergleichen jede Groesse mit dem Modul. Dazu kommen
Pruefungen, die das Notebook selbst betreffen: dass alle Codezellen uebersetzbar
sind, dass keine Ergebnisse einer frueheren Ausfuehrung mitgeliefert werden und dass
die Qualitaetstore, auf denen die Aussagekraft beruht, nicht abgeschwaecht wurden.
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
    pathlib.Path(__file__).resolve().parents[1] / "wasd_traeger_messlauf_colab.ipynb"
)

#: Die Zellen, die nur Konstanten setzen und deshalb ohne GPU ausfuehrbar sind.
KONFIGZELLEN = (2, 3)


def _zellen() -> list[dict[str, Any]]:
    with open(NOTEBOOK, encoding="utf-8") as handle:
        daten: Any = json.load(handle)
    return list(daten["cells"])


def _quelle(zelle: dict[str, Any]) -> str:
    return "".join(zelle["source"])


def _codequellen() -> list[tuple[int, str]]:
    return [
        (index, _quelle(zelle))
        for index, zelle in enumerate(_zellen())
        if zelle["cell_type"] == "code"
    ]


def _gesamttext() -> str:
    return "\n".join(quelle for _, quelle in _codequellen())


@pytest.fixture(scope="module")
def konfig() -> dict[str, Any]:
    """Fuehrt die Konfigurationszellen aus und gibt ihren Namensraum zurueck."""
    raum: dict[str, Any] = {}
    zellen = _zellen()
    for index in KONFIGZELLEN:
        exec(compile(_quelle(zellen[index]), f"zelle{index}", "exec"), raum)
    return raum


# --------------------------------------------------------------------------- #
# Das Notebook als Datei
# --------------------------------------------------------------------------- #


def test_notebook_ist_gueltiges_nbformat_4():
    with open(NOTEBOOK, encoding="utf-8") as handle:
        daten: Any = json.load(handle)
    assert daten["nbformat"] == 4
    assert daten["cells"]
    for zelle in daten["cells"]:
        assert zelle["cell_type"] in ("code", "markdown")
        assert isinstance(zelle["source"], list)


def test_jede_codezelle_ist_uebersetzbar():
    """Ein Syntaxfehler faellt sonst erst nach dem Laden des Modells auf."""
    for index, quelle in _codequellen():
        compile(quelle, f"zelle{index}", "exec")


def test_notebook_enthaelt_keine_ergebnisse_eines_frueheren_laufs():
    """Sonst waere aus der Datei nicht zu erkennen, welche Zahlen neu gemessen sind."""
    for zelle in _zellen():
        if zelle["cell_type"] == "code":
            assert zelle.get("outputs") == []
            assert zelle.get("execution_count") is None


# --------------------------------------------------------------------------- #
# Die doppelt gefuehrten Designgroessen
# --------------------------------------------------------------------------- #


def test_modell_und_revision_stimmen_mit_dem_modul_ueberein(konfig: dict[str, Any]):
    assert konfig["MODELL_ID"] == modul.MODELL
    assert konfig["REVISION"] == modul.REVISION
    assert konfig["PRAEZISION"] == modul.PRAEZISION
    assert konfig["BATCHGROESSE"] == modul.BATCHGROESSE


def test_ziel_und_kontrollen_stimmen_mit_dem_modul_ueberein(konfig: dict[str, Any]):
    assert konfig["ZIEL"] == modul.ZIEL
    assert tuple(konfig["ZIEL_ZERLEGUNG"]) == modul.ZIEL_ZERLEGUNG
    assert konfig["REIHENFOLGE_KONTROLLE"] == modul.REIHENFOLGE_KONTROLLE
    assert tuple(konfig["ZUSATZKONTROLLEN"]) == modul.ZUSATZKONTROLLEN
    assert konfig["ROLLE_GETRENNT"] == modul.ROLLE_GETRENNT
    assert konfig["ZUSATZROLLEN"] == modul.ZUSATZROLLEN


def test_schwellen_stimmen_mit_der_vorregistrierung_ueberein(konfig: dict[str, Any]):
    """Hier waere eine Abweichung am folgenreichsten: sie verschoebe die Regel."""
    regel = modul.VORREGISTRIERUNG
    assert konfig["MINDEST_VORSPRUNG"] == regel.mindest_vorsprung_nats
    assert konfig["KONTROLL_QUANTIL"] == regel.kontroll_quantil
    assert konfig["MINDEST_KAUSALANTEIL"] == regel.mindest_kausalanteil
    assert konfig["MINDEST_SCHICHTEN"] == regel.mindest_schichten
    assert konfig["HOECHSTES_LECK"] == regel.hoechstes_kontroll_leck
    assert konfig["MINDEST_KONTROLLEN"] == modul.MINDEST_KONTROLLEN


def test_messvokabular_und_schablonen_sind_identisch(konfig: dict[str, Any]):
    assert {feld: tuple(w) for feld, w in konfig["MESSVOKABULAR"].items()} == modul.MESSVOKABULAR
    assert tuple(konfig["SCHABLONEN"]) == modul.SCHABLONEN


def test_die_token_id_tabelle_ist_im_notebook_dieselbe(konfig: dict[str, Any]):
    """Sie ist der Beleg dafuer, dass wirklich dieser Tokenizer geladen wurde."""
    assert {f: tuple(i) for f, i in konfig["MESSVOKABULAR_IDS"].items()} == modul.MESSVOKABULAR_IDS
    assert "TOKENISIERUNG WEICHT AB" in _gesamttext()


def test_jede_schablone_endet_auch_im_notebook_mit_dem_platzhalter(konfig: dict[str, Any]):
    for schablone in konfig["SCHABLONEN"]:
        assert schablone.endswith("{ziel}")


# --------------------------------------------------------------------------- #
# Die Eingriffspaare
# --------------------------------------------------------------------------- #


def test_eingriffskontrollen_sind_alle_auch_varianten(konfig: dict[str, Any]):
    """Ein Paar, das keine Grundlinie hat, liefe im Messteil in einen Schluesselfehler."""
    stamm = modul.ZIEL[:-1]
    buchstaben = {stamm + b for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if b != modul.ZIEL[-1]}
    bekannt = buchstaben | {modul.REIHENFOLGE_KONTROLLE} | {t for t, _ in modul.ZUSATZKONTROLLEN}
    for name in konfig["EINGRIFF_KONTROLLEN"]:
        assert name in bekannt, name


def test_die_unabgeglichene_kontrolle_ist_kein_eingriffspaar(konfig: dict[str, Any]):
    """ESDF hat drei Tokens - es gibt keine Position, die dem Ziel entspraeche."""
    getrennt = [t for t, r in modul.ZUSATZKONTROLLEN if r == modul.ROLLE_GETRENNT]
    for name in getrennt:
        assert name not in konfig["EINGRIFF_KONTROLLEN"]


def test_die_beiden_scharfen_zusatzkontrollen_werden_auch_eingegriffen(konfig: dict[str, Any]):
    """FORD und NOTA tragen den Beobachtungsteil - der Eingriff muss sie mitnehmen."""
    assert "FORD" in konfig["EINGRIFF_KONTROLLEN"]
    assert "NOTA" in konfig["EINGRIFF_KONTROLLEN"]


# --------------------------------------------------------------------------- #
# Die Qualitaetstore
# --------------------------------------------------------------------------- #


def test_das_selbstpatchtor_bleibt_auf_exakt_null():
    """Ein Patch mit der eigenen Aktivierung ist eine Identitaet, keine Messung.

    Derselbe Vorwaertslauf mit demselben Tensor an derselben Stelle muss dasselbe
    Ergebnis liefern, Bit fuer Bit. Eine Toleranz wuerde hier genau den Fehler
    durchlassen, den das Tor fangen soll - einen Eingriff, der an der falschen
    Position oder am falschen Tensor sitzt.
    """
    text = _gesamttext()
    assert "self_patch_checks.csv" in text
    assert "MAX_SELBSTPATCH_FEHLER == 0.0" in text


def test_das_residualtor_misst_relativ_und_nicht_gegen_null(konfig: dict[str, Any]):
    """Die Residuumszerlegung ist eine Rekonstruktion, keine Identitaet.

    ``resid_in + attn + mlp`` summiert in anderer Reihenfolge als der Block selbst;
    float32 rundet dabei anders. Im Probelauf gegen ein kleines Modell betrug der
    Unterschied 7.5e-09 absolut - ein Tor auf exakt null haette den Lauf an dieser
    Stelle abgebrochen, ohne dass etwas falsch gewesen waere.

    Die Schranke muss trotzdem eng genug bleiben: eine strukturell falsche Zerlegung
    - etwa ein sequentielles statt eines parallelen Residuums - laege um viele
    Groessenordnungen darueber.
    """
    text = _gesamttext()
    assert "parallel_residual_checks.csv" in text
    assert "MAX_RESIDUAL_FEHLER_REL <= HOECHSTER_RESIDUALFEHLER_REL" in text
    assert "MAX_RESIDUAL_FEHLER == 0.0" not in text
    schranke = konfig["HOECHSTER_RESIDUALFEHLER_REL"]
    assert 1e-7 < schranke <= 1e-3


def test_die_architekturpruefung_nennt_die_zahlen_von_pythia_1_4b():
    """``pythia-1b`` und ``pythia-1.4b`` unterscheiden sich; die Revision nicht."""
    text = _gesamttext()
    for erwartet in ('"num_hidden_layers": 24', '"hidden_size": 2048', '"use_parallel_residual": True'):
        assert erwartet in text


def test_determinismus_wird_vor_der_messung_gesetzt():
    text = _gesamttext()
    assert "torch.use_deterministic_algorithms" in text
    assert "matmul.allow_tf32 = False" in text
    assert "torch.manual_seed" in text


def test_der_abbruch_bei_unbrauchbaren_nennern_steht_im_notebook(konfig: dict[str, Any]):
    """Im Vorlauf hatten 144 von 672 Transferzeilen keinen brauchbaren Nenner."""
    assert konfig["HOECHSTER_ANTEIL_ZU_KLEINER_ABSTAND"] == pytest.approx(0.20)
    assert konfig["MINDEST_ABSTAND_NATS"] > 0
    assert "ZU VIELE UNBRAUCHBARE NENNER" in _gesamttext()


def test_der_rechenaufwand_des_laufs_bleibt_der_dokumentierte(konfig: dict[str, Any]):
    """Der Lauf kostet 18 024 Vorwaertslaeufe - die Zahl steht in den Berichten.

    Sie ergibt sich vollstaendig aus der Konfiguration: Grundlinien fuer jede
    Variante an jeder Schablone, eine Residualsonde je Block, zwei Laeufe je
    Selbstpatchpruefung und vier je Transferzeile (zwei Aktivierungen erfassen,
    zwei Eingriffe). Wer eine Schablone oder ein Eingriffspaar hinzufuegt, aendert
    die Laufzeit spuerbar; dieser Test macht das sichtbar, statt es erst auf der
    Rechnung stehen zu lassen.
    """
    n_varianten = 1 + 25 + 1 + len(modul.ZUSATZKONTROLLEN)
    n_schablonen = len(konfig["SCHABLONEN"])
    n_bloecke = 24
    n_paare = len(konfig["EINGRIFF_KONTROLLEN"])

    grundlinien = n_varianten * n_schablonen
    residualsonde = n_bloecke
    selbstpatch = n_paare * n_schablonen * n_bloecke
    transferzeilen = n_paare * n_schablonen * n_bloecke

    assert grundlinien == 720
    assert selbstpatch == 2880
    assert transferzeilen == 2880
    assert grundlinien + residualsonde + 2 * selbstpatch + 4 * transferzeilen == 18024
