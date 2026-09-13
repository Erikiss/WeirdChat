"""Die Tokenstruktur des parallel laufenden ``ological``-Versuchs.

Der Lauf ``ological_c677b4881415`` vom 13.09.2026 stoert das Wort *Meteorological*
im selben Textfenster - Grossbuchstabe, Null statt o, Tippfehler, Vertauschung - und
misst, was ein Eingriff in Schicht 13 bis 15 ueber diese Stoerungen transportiert.

Die Datei ``interpretation.txt`` des Laufs behauptet: "original_split vs zero differs
at one token only." Diese Tests rechnen das am Tokenizer nach - und halten fest, wo
die Behauptung nicht gilt.
"""

from __future__ import annotations

import csv
import pathlib

DATEN = pathlib.Path(__file__).resolve().parents[1] / "daten"


def _faelle() -> dict[str, tuple[str, ...]]:
    with open(DATEN / "OLOGICAL_TOKENISIERUNG.csv", encoding="utf-8") as handle:
        return {
            zeile["fall"]: tuple(zeile["zerlegung"].split("|"))
            for zeile in csv.DictReader(handle)
        }


def _abstand(links: tuple[str, ...], rechts: tuple[str, ...]) -> int | None:
    """Wie viele Positionen unterscheiden sich, wenn die Laenge gleich ist."""
    if len(links) != len(rechts):
        return None
    return sum(1 for a, b in zip(links, rechts) if a != b)


def test_die_zerlegung_passt_zur_angegebenen_tokenzahl():
    with open(DATEN / "OLOGICAL_TOKENISIERUNG.csv", encoding="utf-8") as handle:
        for zeile in csv.DictReader(handle):
            assert len(zeile["zerlegung"].split("|")) == int(zeile["n_tokens"]), zeile["fall"]
            assert len(zeile["token_ids"].split("|")) == int(zeile["n_tokens"]), zeile["fall"]


def test_original_split_und_zero_unterscheiden_sich_in_genau_einem_token():
    """Die Behauptung des Laufs - und sie haelt."""
    faelle = _faelle()
    assert _abstand(faelle["original_split"], faelle["zero"]) == 1


def test_null_eins_und_grossbuchstabe_bilden_eine_saubere_familie():
    """Vier Faelle, gleiche Struktur, Unterschied nur an der dritten Position."""
    faelle = _faelle()
    familie = ["original_split", "zero", "one", "capital_o"]
    for name in familie:
        assert len(faelle[name]) == 4
        assert faelle[name][0] == "ĠMet"
        assert faelle[name][1] == "e"
        assert faelle[name][3] == "rological"
    mittlere = {faelle[name][2] for name in familie}
    assert mittlere == {"Ġo", "0", "1", "O"}


def test_der_tippfehler_faellt_aus_der_familie_heraus():
    """``x_typo`` hat nur drei Tokens und kein isoliertes mittleres Token.

    Es ist damit nicht im selben Sinn vergleichbar wie Null, Eins und Grossbuchstabe;
    der Unterschied zum Original ist ein anderer.
    """
    faelle = _faelle()
    assert len(faelle["x_typo"]) == 3
    assert _abstand(faelle["x_typo"], faelle["zero"]) is None


def test_die_vertauschungen_stellen_das_ological_token_wieder_her():
    """``zero_transpose`` enthaelt wieder ``ological``, ``zero`` dagegen ``rological``.

    Die Vertauschung aendert also mehr als die Position der Ziffer: von vier
    Tokenpositionen stimmen nur noch die erste ueberein. Ein Unterschied zwischen
    beiden Faellen laesst sich deshalb nicht der Position der Ziffer allein
    zuschreiben.
    """
    faelle = _faelle()
    assert "ological" in faelle["zero_transpose"]
    assert "ological" not in faelle["zero"]
    assert "rological" in faelle["zero"]
    assert _abstand(faelle["zero"], faelle["zero_transpose"]) == 3


def test_das_original_hat_die_kuerzeste_zerlegung():
    faelle = _faelle()
    assert len(faelle["original"]) == 3
    laengen = {name: len(zerlegung) for name, zerlegung in faelle.items()}
    assert laengen["original"] <= min(laengen.values())


def test_beide_forschungslinien_sitzen_im_selben_textfenster():
    """McQuarrie an Token 124-127, Meteorological an 165-168 - ein Dokument, 207 Token."""
    with open(DATEN / "STATE60482_LOCI.csv", encoding="utf-8") as handle:
        zeilen = list(csv.DictReader(handle))
    loci = {zeile["locus"] for zeile in zeilen}
    assert loci == {"mcquarrie", "meteorological"}
    indizes = [int(zeile["token_index"]) for zeile in zeilen]
    assert min(indizes) == 124
    assert max(indizes) == 168
    assert all(index < 207 for index in indizes)
