"""Der Messlauf vom 14.09.2026, gegen die vorregistrierte Regel nachgerechnet.

Die Regel stand vor dem Lauf fest und steht in ``experiment_traeger.py``. Diese
Tests fuettern ihr die tatsaechlichen Messwerte und pruefen, dass sie dasselbe
Urteil faellt wie das Notebook auf der GPU. Damit haengt das Ergebnis nicht mehr
am PDF-Export eines fluechtigen Colab-Laufs.

Die Zahlen stammen aus diesem Export; die vollen Rohtabellen lagen nur im
Dateisystem der Laufzeit und sind nicht erhalten. Was hier steht, sind die im
Bericht ausgegebenen Mittel auf vier Nachkommastellen.
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

from experiment_traeger import (  # noqa: E402
    ROLLE_GETRENNT,
    ROLLE_VERWORFEN,
    TRAGENDE_ROLLEN,
    Beobachtung,
    Kausalmessung,
    Vorregistrierung,
    urteile,
)

DATEN = pathlib.Path(__file__).resolve().parents[1] / "daten"
GRUNDLINIEN = DATEN / "WASD_TRAEGERLAUF_20260914_GRUNDLINIEN.csv"
KAUSAL = DATEN / "WASD_TRAEGERLAUF_20260914_KAUSAL.csv"
QA = DATEN / "WASD_TRAEGERLAUF_20260914_QA.json"

FELDER = ("bewegung", "tastatur", "gegenfeld")


def _grundlinien() -> list[dict[str, str]]:
    with open(GRUNDLINIEN, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _werte(feld: str) -> dict[str, float]:
    return {z["variante"]: float(z[f"{feld}_nats"]) for z in _grundlinien()}


def _beobachtungen(feld: str, *, wie_gelaufen: bool = False) -> list[Beobachtung]:
    """Baut die Beobachtungen so, dass ``feld`` in der Rolle von ``bewegung`` steht.

    ``wie_gelaufen`` stellt den Fehler des Laufs wieder her: dort trug die
    Entscheidungsmenge auch die fuenf Buchstabenkandidaten, die der Tokenizer anders
    zerlegt. Nur so laesst sich pruefen, was das Notebook tatsaechlich gerechnet hat -
    und um wie viel es danebenlag.
    """
    reihen: list[Beobachtung] = []
    for zeile in _grundlinien():
        rolle = zeile["rolle"]
        if wie_gelaufen and rolle == ROLLE_VERWORFEN:
            rolle = "buchstabenkontrolle"
        reihen.append(
            Beobachtung(
                text=zeile["variante"],
                rolle=rolle,
                bewegung_nats=float(zeile[f"{feld}_nats"]),
                tastatur_nats=0.0,
                gegenfeld_nats=0.0,
            )
        )
    return reihen


def _kausal() -> list[Kausalmessung]:
    with open(KAUSAL, encoding="utf-8") as handle:
        return [
            Kausalmessung(
                schicht=int(z["block"]),
                wiederherstellung_ziel=float(z["wiederherstellung_ziel"]),
                wiederherstellung_kontrolle=float(z["leck_kontrolle"]),
            )
            for z in csv.DictReader(handle)
        ]


def _qa() -> dict[str, object]:
    with open(QA, encoding="utf-8") as handle:
        daten: object = json.load(handle)
    assert isinstance(daten, dict)
    roh = cast("dict[object, object]", daten)
    return {str(schluessel): wert for schluessel, wert in roh.items()}


# --------------------------------------------------------------------------- #
# Vollstaendigkeit der abgelegten Daten
# --------------------------------------------------------------------------- #


def test_alle_dreissig_varianten_sind_erhalten():
    zeilen = _grundlinien()
    assert len(zeilen) == 30
    rollen = {z["rolle"] for z in zeilen}
    assert "ziel" in rollen and ROLLE_GETRENNT in rollen
    assert sum(1 for z in zeilen if z["rolle"] == "ziel") == 1


def test_alle_vierundzwanzig_bloecke_sind_erhalten():
    assert [m.schicht for m in _kausal()] == list(range(24))


# --------------------------------------------------------------------------- #
# Die vorregistrierte Regel, auf die echten Werte angewandt
# --------------------------------------------------------------------------- #


def test_die_regel_reproduziert_das_urteil_des_notebooks():
    """Der Kern: dieselbe Regel, dieselben Zahlen, dasselbe Ergebnis.

    Gerechnet wird hier mit dem Kontrollsatz, den das Notebook tatsaechlich benutzt
    hat - einschliesslich der fuenf, die es selbst als verworfen gemeldet hatte.
    """
    urteil = urteile(_beobachtungen("bewegung", wie_gelaufen=True), _kausal())
    assert urteil["n_kontrollen_in_der_regel"] == 28
    assert urteil["beste_kontrolle"] == "WASV"
    assert urteil["vorsprung_nats"] == pytest.approx(-0.1300, abs=1e-4)
    assert urteil["vorsprung_gegen_quantil_nats"] == pytest.approx(0.0387, abs=1e-4)
    assert urteil["beobachtungsteil"] is False
    assert urteil["beobachtungsteil_quantilsfassung"] is False
    assert urteil["laengster_kausallauf"] == 0
    assert urteil["kausalteil"] is False
    assert urteil["wasd_traeger_bestaetigt"] is False
    assert _qa()["wasd_traeger_bestaetigt"] is False


def test_die_reparierte_regel_fuehrt_nur_noch_dreiundzwanzig_kontrollen():
    """20 strukturgleiche Buchstabenkontrollen, dazu ASDW, FORD und NOTA."""
    urteil = urteile(_beobachtungen("bewegung"), _kausal())
    assert urteil["n_kontrollen_in_der_regel"] == 23


def test_der_fehler_hat_das_urteil_nicht_gedreht():
    """Die entscheidende Entlastung: der Befund bleibt in beiden Faellen negativ.

    Die fuenf faelschlich eingeschlossenen Kontrollen lagen im Feld bewegung alle
    unter der besten; die beste Kontrolle ist so oder so WASV. Ihre Aufnahme war
    konservativ, nicht guenstig - sie hat den Vorsprung nicht vergroessert.
    """
    gelaufen = urteile(_beobachtungen("bewegung", wie_gelaufen=True), _kausal())
    repariert = urteile(_beobachtungen("bewegung"), _kausal())
    assert gelaufen["beste_kontrolle"] == repariert["beste_kontrolle"] == "WASV"
    assert gelaufen["vorsprung_nats"] == repariert["vorsprung_nats"]
    assert repariert["wasd_traeger_bestaetigt"] is False
    # Nur die Quantilsfassung verschiebt sich, und zwar nach unten.
    quantil_repariert = float(str(repariert["vorsprung_gegen_quantil_nats"]))
    quantil_gelaufen = float(str(gelaufen["vorsprung_gegen_quantil_nats"]))
    assert quantil_repariert == pytest.approx(0.0281, abs=1e-4)
    assert quantil_repariert < quantil_gelaufen


def test_die_verworfenen_kontrollen_werden_getrennt_gefuehrt():
    """Sie sind gemessen und abgelegt, tragen aber keine Entscheidung."""
    rollen = {z["variante"]: z["rolle"] for z in _grundlinien()}
    verworfen = {t for t, r in rollen.items() if r == ROLLE_VERWORFEN}
    assert verworfen == {"WASE", "WASH", "WASK", "WASS", "WAST"}
    assert ROLLE_VERWORFEN not in TRAGENDE_ROLLEN


def test_die_getrennte_kontrolle_stand_wie_vorgesehen_ausserhalb_der_regel():
    urteil = urteile(_beobachtungen("bewegung"), _kausal())
    assert set(urteil["getrennt_berichtet"]) == {"ESDF"}  # type: ignore[arg-type]


def test_das_urteil_haengt_nicht_an_der_getrennten_kontrolle():
    """Gegenprobe: ESDF herausnehmen aendert am Ergebnis nichts."""
    ohne = [b for b in _beobachtungen("bewegung") if b.rolle != ROLLE_GETRENNT]
    assert urteile(ohne, _kausal())["vorsprung_nats"] == pytest.approx(-0.1300, abs=1e-4)


# --------------------------------------------------------------------------- #
# Das Entscheidungsfeld war nicht das Feld mit dem Effekt
# --------------------------------------------------------------------------- #


def test_im_entscheidungsfeld_liegt_das_ziel_nur_auf_rang_drei():
    werte = _werte("bewegung")
    rang = 1 + sum(1 for t, v in werte.items() if t != "WASD" and v > werte["WASD"])
    assert rang == 3


def test_im_tastaturfeld_liegt_das_ziel_vorn_und_zwar_deutlich():
    """Nicht die Regel, aber dieselbe Messung: das Feld, das die Regel nicht nannte.

    Die Regel benannte ``bewegung``. ``tastatur`` wurde bei jeder Variante an jeder
    Schablone mitgemessen, war aber **kein** Endpunkt. Was hier steht, ist deshalb
    kein Befund im Sinne der Vorregistrierung, sondern eine Beobachtung - festgehalten,
    damit sie nicht verlorengeht, und ausdruecklich als solche markiert.
    """
    werte = {t: v for t, v in _werte("tastatur").items()}
    ziel = werte["WASD"]
    kontrollen = {
        z["variante"]: float(z["tastatur_nats"])
        for z in _grundlinien()
        if z["rolle"] not in ("ziel", ROLLE_GETRENNT)
    }
    assert ziel > max(kontrollen.values())
    assert ziel - max(kontrollen.values()) == pytest.approx(1.6975, abs=1e-4)
    assert max(kontrollen, key=lambda t: kontrollen[t]) == "WASA"


def test_im_gegenfeld_ist_das_ziel_am_staerksten_unterdrueckt():
    werte = _werte("gegenfeld")
    assert min(werte, key=lambda t: werte[t]) == "WASD"


def test_die_beiden_scharfen_kontrollen_schliessen_aus_was_sie_sollten():
    """FORD teilt das Endtoken, NOTA die Haeufigkeit - beide bleiben weit zurueck."""
    tast = _werte("tastatur")
    assert tast["WASD"] - tast["FORD"] > 2.0   # nicht der Buchstabe D allein
    assert tast["WASD"] - tast["NOTA"] > 2.5   # nicht die Korpushaeufigkeit allein
    assert tast["WASD"] - tast["ASDW"] > 2.5   # nicht die Buchstaben ohne Reihenfolge


# --------------------------------------------------------------------------- #
# Der Kausalteil und warum er nichts entscheiden konnte
# --------------------------------------------------------------------------- #


def test_kein_einziger_block_erfuellt_beide_bedingungen():
    regel = Vorregistrierung()
    traegt = [
        m
        for m in _kausal()
        if m.wiederherstellung_ziel >= regel.mindest_kausalanteil
        and m.wiederherstellung_kontrolle <= regel.hoechstes_kontroll_leck
    ]
    assert traegt == []


def test_die_wiederherstellung_haelt_die_bedingung_ueberall():
    """An der Wiederherstellung liegt es nicht - sie ist an allen 24 Bloecken hoch."""
    regel = Vorregistrierung()
    assert all(m.wiederherstellung_ziel >= regel.mindest_kausalanteil for m in _kausal())
    assert min(m.wiederherstellung_ziel for m in _kausal()) > 0.86


def test_das_leck_reisst_die_schwelle_an_jedem_einzelnen_block():
    """Und zwar nicht knapp: die Schwelle ist 0.20, gemessen wird ueberall ueber 0.83.

    Beides zusammen - Wiederherstellung ~ 1 und Leck ~ 1 - heisst: der Eingriff
    uebertraegt in BEIDE Richtungen vollstaendig. Er trennt damit nichts. Das ist
    keine Eigenschaft des Modells, sondern der Konstruktion: der einzige Unterschied
    zwischen Ziel- und Kontrollprompt ist das letzte Token, und genau dessen
    Residuum wird ersetzt. Der Test konnte nicht positiv ausfallen.
    """
    regel = Vorregistrierung()
    lecks = [m.wiederherstellung_kontrolle for m in _kausal()]
    assert min(lecks) > 4 * regel.hoechstes_kontroll_leck
    assert statistics.fmean(lecks) > 0.95


def test_beide_richtungen_uebertragen_annaehernd_gleich_gut():
    """Der rechnerische Kern der Vakuositaet, an den Zahlen festgehalten."""
    paare = [(m.wiederherstellung_ziel, m.wiederherstellung_kontrolle) for m in _kausal()]
    assert statistics.fmean([abs(a - b) for a, b in paare]) < 0.15


# --------------------------------------------------------------------------- #
# Die Qualitaetstore
# --------------------------------------------------------------------------- #


def test_die_beiden_tore_haben_gehalten():
    qa = _qa()
    assert qa["max_selbstpatch_fehler"] == 0.0
    assert float(str(qa["max_residual_fehler_absolut"])) < 1e-4
    assert qa["anteil_abstand_zu_klein"] == 0.0


def test_der_lauf_ist_laut_vorregistrierung_unterbestimmt():
    """sigma = 1.4579 gegen die Grenze 0.8 - die Flagge ist gefallen.

    Sie misst allerdings die Streuung EINER Variante ueber die Schablonen, und alle
    Varianten laufen durch dieselben 24 Schablonen. Fuer den gepaarten Vergleich ist
    das die zu grosse Zahl; wie viel zu gross, liesse sich nur an den nicht
    erhaltenen Rohzeilen entscheiden.
    """
    qa = _qa()
    assert qa["unterbestimmt_laut_vorregistrierung"] is True
    assert float(str(qa["sigma_zwischen_schablonen"])) > 0.8
