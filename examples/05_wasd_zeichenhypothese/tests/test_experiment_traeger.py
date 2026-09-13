"""Tests fuer das vorregistrierte Nachfolge-Design.

Zwei Dinge werden geprueft: dass die Strukturpruefung ungleiche Kontrollen wirklich
aussortiert, und dass die Entscheidungsregel gegen gepflanzte Wahrheiten das Richtige
tut - vor allem, dass sie bei halb positiven Ergebnissen schweigt.

Der Tokenizer wird dabei nachgebildet, damit die Tests weder Netz noch das Paket
``tokenizers`` brauchen. Die nachgebildeten Zerlegungen sind die echten aus dem
GPT-NeoX-Vokabular von Pythia.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiment_traeger import (  # noqa: E402
    MINDEST_KONTROLLEN,
    SCHABLONEN,
    ZIEL_ZERLEGUNG,
    Beobachtung,
    Kausalmessung,
    Variante,
    Vorregistrierung,
    baue_prompts,
    baue_varianten,
    pruefe_design,
    strukturgleiche_kontrollen,
    urteile,
)

#: Die echten Abweichler im GPT-NeoX-Vokabular: ASE, ASH, ASK, ASS und AST sind
#: eigene Vokabeleintraege, deshalb zerfaellt " WASH" in ['ĠW', 'ASH'].
ABWEICHLER = {"WASE", "WASH", "WASK", "WASS", "WAST"}


class FalscheKodierung:
    def __init__(self, tokens: list[str], ids: list[int]) -> None:
        self._tokens = tokens
        self._ids = ids

    @property
    def tokens(self) -> list[str]:
        return self._tokens

    @property
    def ids(self) -> list[int]:
        return self._ids


class FalscherTokenizer:
    """Bildet die echten Zerlegungen nach, ohne das Paket ``tokenizers``."""

    def encode(self, sequence: str, /) -> FalscheKodierung:
        wort = sequence.strip()
        if wort == "ASDW":
            return FalscheKodierung(["ĠASD", "W"], [29895, 56])
        if wort in ABWEICHLER:
            return FalscheKodierung(["ĠW", wort[1:]], [411, 9434])
        if wort.startswith("WAS") and len(wort) == 4:
            return FalscheKodierung(["ĠWAS", wort[3]], [22250, 37])
        raise AssertionError(f"unerwartete Eingabe im Test: {sequence!r}")


def _varianten() -> list[Variante]:
    return baue_varianten(FalscherTokenizer())


# --------------------------------------------------------------------------- #
# Design
# --------------------------------------------------------------------------- #


def test_design_besteht_und_verwirft_genau_die_abweichler():
    pruefung = pruefe_design(_varianten())
    assert pruefung.bestanden
    assert set(pruefung.verworfen) == ABWEICHLER
    assert pruefung.n_buchstabenkontrollen == 20
    assert pruefung.erstes_token_identisch
    assert pruefung.letzte_token_eindeutig


def test_brauchbare_kontrollen_teilen_das_erste_token_mit_dem_ziel():
    for kontrolle in strukturgleiche_kontrollen(_varianten()):
        assert kontrolle.zerlegung[0] == ZIEL_ZERLEGUNG[0]
        assert len(kontrolle.zerlegung) == len(ZIEL_ZERLEGUNG)


def test_design_faellt_durch_wenn_zu_wenige_kontrollen_bleiben():
    """Gepflanzte Wahrheit: ein Tokenizer, bei dem fast alles anders zerfaellt."""

    class KaputterTokenizer:
        def encode(self, sequence: str, /) -> FalscheKodierung:
            wort = sequence.strip()
            if wort in {"WASD", "ASDW"}:
                return FalscheKodierung(["ĠWAS", wort[3]], [22250, 37])
            return FalscheKodierung(["ĠW", "A", "S", wort[3]], [411, 34, 52, 37])

    pruefung = pruefe_design(baue_varianten(KaputterTokenizer()))
    assert not pruefung.bestanden
    assert any(str(MINDEST_KONTROLLEN) in mangel for mangel in pruefung.maengel)


def test_jede_variante_wird_mit_jeder_schablone_gekreuzt():
    varianten = _varianten()
    prompts = baue_prompts(varianten)
    assert len(prompts) == len(varianten) * len(SCHABLONEN)
    for variante, schablone, text in prompts:
        assert variante.text in text
        assert "{ziel}" not in text
        assert text.startswith(schablone.split("{ziel}")[0])


def test_jede_schablone_endet_mit_dem_platzhalter():
    """Gemessen wird das naechste Token nach dem Ziel - also muss es am Ende stehen."""
    for schablone in SCHABLONEN:
        assert schablone.endswith("{ziel}")


# --------------------------------------------------------------------------- #
# Entscheidungsregel
# --------------------------------------------------------------------------- #


def _beobachtungen(ziel_nats: float, kontroll_nats: list[float]) -> list[Beobachtung]:
    reihen = [Beobachtung("WASD", "ziel", ziel_nats, 0.0, 0.0)]
    for index, wert in enumerate(kontroll_nats):
        reihen.append(Beobachtung(f"WAS{index}", "buchstabenkontrolle", wert, 0.0, 0.0))
    return reihen


def _kausal(paare: list[tuple[int, float, float]]) -> list[Kausalmessung]:
    return [Kausalmessung(schicht, ziel, kontrolle) for schicht, ziel, kontrolle in paare]


def test_beide_teile_positiv_ergibt_bestaetigung():
    urteil = urteile(
        _beobachtungen(2.0, [0.5, 0.4, 0.3]),
        _kausal([(12, 0.8, 0.05), (13, 0.7, 0.1), (14, 0.2, 0.0)]),
    )
    assert urteil["beobachtungsteil"] is True
    assert urteil["kausalteil"] is True
    assert urteil["wasd_traeger_bestaetigt"] is True
    assert urteil["laengster_kausallauf"] == 2


def test_nur_beobachtung_reicht_nicht():
    urteil = urteile(
        _beobachtungen(2.0, [0.5, 0.4]),
        _kausal([(12, 0.1, 0.0), (13, 0.2, 0.0)]),
    )
    assert urteil["beobachtungsteil"] is True
    assert urteil["kausalteil"] is False
    assert urteil["wasd_traeger_bestaetigt"] is False


def test_nur_kausalteil_reicht_nicht():
    urteil = urteile(
        _beobachtungen(0.6, [0.5, 0.4]),
        _kausal([(12, 0.9, 0.0), (13, 0.9, 0.0)]),
    )
    assert urteil["beobachtungsteil"] is False
    assert urteil["wasd_traeger_bestaetigt"] is False


def test_ein_einzelner_ausreisser_unter_den_kontrollen_kippt_den_befund():
    """Verglichen wird gegen die beste Kontrolle, nicht gegen das Kontrollmittel.

    Gepflanzte Wahrheit: das Ziel schlaegt den Mittelwert klar, aber eine einzige
    Kontrolle liegt fast gleichauf. Dann ist der Befund keiner.
    """
    urteil = urteile(
        _beobachtungen(2.0, [1.8, 0.1, 0.1, 0.1, 0.1]),
        _kausal([(12, 0.9, 0.0), (13, 0.9, 0.0)]),
    )
    assert urteil["beste_kontrolle"] == "WAS0"
    assert urteil["vorsprung_nats"] == pytest.approx(0.2)
    assert urteil["beobachtungsteil"] is False
    assert urteil["wasd_traeger_bestaetigt"] is False


def test_kausaleffekt_an_nicht_benachbarten_schichten_zaehlt_nicht():
    urteil = urteile(
        _beobachtungen(2.0, [0.5]),
        _kausal([(10, 0.9, 0.0), (11, 0.1, 0.0), (12, 0.9, 0.0)]),
    )
    assert urteil["laengster_kausallauf"] == 1
    assert urteil["kausalteil"] is False


def test_ein_leck_in_der_kontrolle_entwertet_die_schicht():
    """Wenn der Eingriff mit einer Kontrollquelle fast genauso wirkt, traegt er nicht."""
    urteil = urteile(
        _beobachtungen(2.0, [0.5]),
        _kausal([(12, 0.9, 0.5), (13, 0.9, 0.4)]),
    )
    assert urteil["kausalteil"] is False


def test_strengere_regel_laesst_sich_uebergeben():
    beobachtungen = _beobachtungen(2.0, [1.0])
    locker = urteile(beobachtungen, _kausal([(12, 0.9, 0.0), (13, 0.9, 0.0)]))
    streng = urteile(
        beobachtungen,
        _kausal([(12, 0.9, 0.0), (13, 0.9, 0.0)]),
        Vorregistrierung(mindest_vorsprung_nats=1.5),
    )
    assert locker["wasd_traeger_bestaetigt"] is True
    assert streng["wasd_traeger_bestaetigt"] is False


def test_fehlendes_ziel_wirft():
    with pytest.raises(ValueError):
        urteile(
            [Beobachtung("WASX", "buchstabenkontrolle", 1.0, 0.0, 0.0)],
            _kausal([(12, 0.9, 0.0)]),
        )


def test_fehlende_kontrollen_werfen():
    with pytest.raises(ValueError):
        urteile([Beobachtung("WASD", "ziel", 1.0, 0.0, 0.0)], _kausal([(12, 0.9, 0.0)]))
