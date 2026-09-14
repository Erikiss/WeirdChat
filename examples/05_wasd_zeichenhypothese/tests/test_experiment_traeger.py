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
    MESSVOKABULAR,
    MESSVOKABULAR_IDS,
    MINDEST_KONTROLLEN,
    ROLLE_GETRENNT,
    SCHABLONEN,
    ZIEL_ZERLEGUNG,
    ZUSATZKONTROLLEN,
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

#: Die am Tokenizer von ``EleutherAI/pythia-1.4b`` abgelesenen Zerlegungen der
#: Zusatzkontrollen. ``FORD`` teilt mit dem Ziel das **letzte** Token (ID 37),
#: ``NOTA`` teilt nur die Struktur, ``ESDF`` teilt nicht einmal die Tokenzahl.
ZUSATZ_ZERLEGUNGEN: dict[str, tuple[list[str], list[int]]] = {
    "FORD": (["ĠFOR", "D"], [6651, 37]),
    "NOTA": (["ĠNOT", "A"], [5803, 34]),
    "ESDF": (["ĠE", "SD", "F"], [444, 3871, 39]),
}


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
        if wort in ZUSATZ_ZERLEGUNGEN:
            tokens, ids = ZUSATZ_ZERLEGUNGEN[wort]
            return FalscheKodierung(list(tokens), list(ids))
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
    # Ziel, 25 Buchstabenkandidaten, Reihenfolge- und drei Zusatzkontrollen.
    assert pruefung.n_varianten == 1 + 25 + 1 + len(ZUSATZKONTROLLEN)
    # In die Messung gehen 20 Buchstabenkontrollen, Ziel, Reihenfolge und die drei
    # Zusatzkontrollen: 25 Varianten mal 24 Schablonen.
    assert pruefung.n_messpunkte == 25 * len(SCHABLONEN)


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


# --------------------------------------------------------------------------- #
# Zusatzkontrollen
# --------------------------------------------------------------------------- #


def test_ford_teilt_mit_dem_ziel_genau_das_letzte_token():
    """Die schaerfste Kontrolle des Designs - und der Grund, warum sie das ist.

    Traegt allein der Buchstabe ``D`` am Ende den Effekt, dann muss ``FORD`` ihn
    ebenso zeigen: dasselbe letzte Token, dieselbe Position, anderer Stamm. Ein
    Effekt, der bei WASD auftritt und bei FORD nicht, kann nicht am Endtoken haengen.
    """
    nach_text = {v.text: v for v in _varianten()}
    ziel, ford = nach_text["WASD"], nach_text["FORD"]
    assert ford.rolle == "endtokenkontrolle"
    assert ford.token_ids[-1] == ziel.token_ids[-1]
    assert ford.token_ids[0] != ziel.token_ids[0]
    assert len(ford.zerlegung) == len(ziel.zerlegung)


def test_nota_ist_strukturgleich_aber_nicht_endtokengleich():
    """Die Frequenz-Nulllinie: gleiche Bauform, kein geteiltes Token."""
    nach_text = {v.text: v for v in _varianten()}
    ziel, nota = nach_text["WASD"], nach_text["NOTA"]
    assert nota.rolle == "frequenzkontrolle"
    assert len(nota.zerlegung) == len(ziel.zerlegung)
    assert set(nota.token_ids).isdisjoint(ziel.token_ids)


def test_die_unabgeglichene_tastenkontrolle_ist_bewusst_nicht_strukturgleich():
    """``ESDF`` zerfaellt in drei Stuecke - und faellt trotzdem nicht durchs Design.

    Gerade weil sie die Struktur verletzt, darf sie nicht in die Hauptregel; dass
    das Design sie durchlaesst, ist die Ausnahme, die ``ROLLE_GETRENNT`` benennt.
    """
    nach_text = {v.text: v for v in _varianten()}
    esdf = nach_text["ESDF"]
    assert esdf.rolle == ROLLE_GETRENNT
    assert len(esdf.zerlegung) != len(ZIEL_ZERLEGUNG)
    assert pruefe_design(_varianten()).bestanden


def test_eine_strukturverletzende_zusatzkontrolle_faellt_sonst_durch():
    """Gegenprobe: dieselbe Zerlegung unter einer Rolle der Hauptregel faellt durch.

    Ohne diesen Test liesse sich nicht unterscheiden, ob das Design ESDF wegen
    seiner Rolle durchlaesst oder ob es die Tokenzahl von Zusatzkontrollen gar
    nicht prueft.
    """
    varianten = [
        v if v.text != "ESDF" else Variante("ESDF", "frequenzkontrolle", v.zerlegung, v.token_ids)
        for v in _varianten()
    ]
    pruefung = pruefe_design(varianten)
    assert not pruefung.bestanden
    assert any("ESDF" in mangel for mangel in pruefung.maengel)


# --------------------------------------------------------------------------- #
# Die getrennt berichtete Kontrolle und die Quantilsfassung
# --------------------------------------------------------------------------- #


def test_die_getrennte_kontrolle_kann_den_befund_nicht_kippen():
    """Gepflanzte Wahrheit: ESDF liegt hoeher als das Ziel - und zaehlt trotzdem nicht.

    Sie ist nicht strukturgleich und im Korpus zwoelfmal seltener; ihr Wert waere
    mit dem des Ziels nicht vergleichbar. Deshalb steht sie im Bericht, aber nicht
    in der Regel.
    """
    beobachtungen = _beobachtungen(2.0, [0.5, 0.4])
    beobachtungen.append(Beobachtung("ESDF", ROLLE_GETRENNT, 9.9, 0.0, 0.0))
    urteil = urteile(beobachtungen, _kausal([(12, 0.9, 0.0), (13, 0.9, 0.0)]))
    assert urteil["beste_kontrolle"] == "WAS0"
    assert urteil["n_kontrollen_in_der_regel"] == 2
    assert urteil["getrennt_berichtet"] == {"ESDF": 9.9}
    assert urteil["wasd_traeger_bestaetigt"] is True


def test_die_getrennte_kontrolle_kann_den_befund_auch_nicht_stuetzen():
    """Gegenrichtung: ein sehr niedriger Wert darf das Urteil ebensowenig heben."""
    ohne = urteile(
        _beobachtungen(0.6, [0.5]), _kausal([(12, 0.9, 0.0), (13, 0.9, 0.0)])
    )
    mit = urteile(
        _beobachtungen(0.6, [0.5]) + [Beobachtung("ESDF", ROLLE_GETRENNT, -5.0, 0.0, 0.0)],
        _kausal([(12, 0.9, 0.0), (13, 0.9, 0.0)]),
    )
    assert ohne["vorsprung_nats"] == mit["vorsprung_nats"]
    assert mit["beobachtungsteil"] is False


def test_zusatzkontrollen_der_hauptregel_zaehlen_mit():
    """FORD und NOTA sind Kontrollen wie jede andere - auch fuer das Maximum."""
    beobachtungen = _beobachtungen(2.0, [0.1, 0.1])
    beobachtungen.append(Beobachtung("FORD", "endtokenkontrolle", 1.9, 0.0, 0.0))
    urteil = urteile(beobachtungen, _kausal([(12, 0.9, 0.0), (13, 0.9, 0.0)]))
    assert urteil["beste_kontrolle"] == "FORD"
    assert urteil["n_kontrollen_in_der_regel"] == 3
    assert urteil["beobachtungsteil"] is False


def test_quantilsfassung_ist_nachsichtiger_als_das_maximum():
    """Beide Fassungen stehen vorab fest und werden beide berichtet.

    Gepflanzte Wahrheit: eine einzige hohe Kontrolle unter zwanzig. Gegen das
    Maximum faellt der Befund, gegen das 90-Prozent-Quantil haelt er - und genau
    dieser Unterschied ist der Grund, beide Zahlen zu berichten statt nachtraeglich
    die guenstigere zu waehlen.
    """
    urteil = urteile(
        _beobachtungen(2.0, [1.9] + [0.1] * 19),
        _kausal([(12, 0.9, 0.0), (13, 0.9, 0.0)]),
    )
    assert urteil["vorsprung_nats"] == pytest.approx(0.1)
    assert urteil["beobachtungsteil"] is False
    # Bei zwanzig Werten liegt die Quantilsstelle bei 0.9 * 19 = 17.1, also klar
    # unterhalb des Ausreissers auf Rang 20.
    assert urteil["vorsprung_gegen_quantil_nats"] == pytest.approx(1.9)
    assert urteil["beobachtungsteil_quantilsfassung"] is True
    # Das Gesamturteil folgt der Hauptfassung; die Zweitfassung steht daneben.
    assert urteil["wasd_traeger_bestaetigt"] is False


def test_quantil_und_maximum_fallen_bei_gleichverteilten_kontrollen_zusammen():
    """Ohne Ausreisser trennt die Zweitfassung nichts - sie ist keine Hintertuer."""
    urteil = urteile(
        _beobachtungen(2.0, [0.5] * 10), _kausal([(12, 0.9, 0.0), (13, 0.9, 0.0)])
    )
    assert urteil["vorsprung_nats"] == pytest.approx(urteil["vorsprung_gegen_quantil_nats"])


def test_quantil_ist_bei_einer_einzigen_kontrolle_das_maximum():
    """Randfall: bei n = 1 darf die Interpolation nicht aus dem Index laufen."""
    urteil = urteile(_beobachtungen(2.0, [0.5]), _kausal([(12, 0.9, 0.0)]))
    assert urteil["vorsprung_gegen_quantil_nats"] == pytest.approx(1.5)


def test_nur_die_getrennte_kontrolle_reicht_nicht_als_kontrollfamilie():
    """Faellt die Hauptfamilie weg, muss die Regel schweigen statt auszuweichen."""
    with pytest.raises(ValueError):
        urteile(
            [
                Beobachtung("WASD", "ziel", 2.0, 0.0, 0.0),
                Beobachtung("ESDF", ROLLE_GETRENNT, 0.1, 0.0, 0.0),
            ],
            _kausal([(12, 0.9, 0.0), (13, 0.9, 0.0)]),
        )


def test_die_vorregistrierte_kontextzahl_ist_die_zahl_der_schablonen():
    """Sonst behauptete die Vorregistrierung eine andere Messung als die stattfindet.

    Die Zahl stand bei acht, als das Design acht Schablonen hatte. Sie wird jetzt
    abgeleitet; dieser Test haelt fest, dass sie es bleibt.
    """
    assert Vorregistrierung().n_kontexte_je_variante == len(SCHABLONEN)


def test_die_kontextzahl_bleibt_ueber_der_grenze_der_trennschaerfe():
    """Unter zwoelf Kontexten faellt die Trennschaerfe bei sigma = 0.5 nats unter 0.5."""
    assert Vorregistrierung().n_kontexte_je_variante >= 12


# --------------------------------------------------------------------------- #
# Das Messvokabular
# --------------------------------------------------------------------------- #


def test_zu_jedem_messwort_gehoert_genau_eine_token_id():
    """Die ID-Tabelle ist der Beleg, dass jedes Wort ein Einzeltoken ist.

    Ohne sie waere ``" strafe"`` nicht aufgefallen: es zerfaellt in
    ``['Ġstra', 'fe']``, und die Sonde haette dann die Wahrscheinlichkeit von
    *stra* gemessen - ein Stueck, das es mit *strategy*, *strange* und *straight*
    teilt.
    """
    assert set(MESSVOKABULAR_IDS) == set(MESSVOKABULAR)
    for feld, woerter in MESSVOKABULAR.items():
        assert len(MESSVOKABULAR_IDS[feld]) == len(woerter), feld


def test_kein_messwort_wird_in_zwei_feldern_gezaehlt():
    """Die Felder werden gegeneinander verrechnet - eine Ueberschneidung verwischte das."""
    alle_ids = [i for ids in MESSVOKABULAR_IDS.values() for i in ids]
    assert len(set(alle_ids)) == len(alle_ids)
    alle_woerter = [w for ws in MESSVOKABULAR.values() for w in ws]
    assert len(set(alle_woerter)) == len(alle_woerter)


def test_jedes_messwort_beginnt_mit_einem_leerzeichen():
    """Ohne fuehrendes Leerzeichen waere es das Wortinnere, nicht der Wortanfang."""
    for woerter in MESSVOKABULAR.values():
        for wort in woerter:
            assert wort.startswith(" ") and wort[1:].strip() == wort[1:]


def test_die_ersetzten_woerter_stehen_nicht_mehr_im_vokabular():
    """``" strafe"`` und ``" senate"`` sind keine Einzeltoken und wurden ersetzt."""
    alle = {w for ws in MESSVOKABULAR.values() for w in ws}
    assert " strafe" not in alle
    assert " senate" not in alle
    assert " sprint" in MESSVOKABULAR["bewegung"]
    assert " parliament" in MESSVOKABULAR["gegenfeld"]
