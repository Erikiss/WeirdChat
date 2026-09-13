"""Tokenizer-Sonde: kann das Modell die Zielzeichenfolge ueberhaupt als Einheit sehen?

Eine Hypothese der Form "die Zeichenfolge X loest im Netz etwas aus" hat eine
Vorbedingung, die vor jedem GPU-Lauf geklaert werden kann: **wie erreicht X das
Netz?** Ein Transformer sieht keine Buchstaben, sondern Token-IDs. Drei Faelle:

1. ``X`` ist ein eigenes Token. Dann gibt es genau eine Embedding-Zeile, die man
   messen, ablatieren und austauschen kann. Die Hypothese ist direkt pruefbar.
2. ``X`` zerfaellt in mehrere Token. Dann muss das Netz die Zeichenidentitaet aus
   Teilstuecken rekonstruieren; die Hypothese setzt zusaetzlich voraus, dass
   Zeichenwissen ueber Tokengrenzen hinweg zusammengefuehrt wird.
3. Die Buchstaben von ``X`` sind ueber den ganzen Text verstreut und bilden nie eine
   Einheit. Dann gibt es kein Traegersignal, an dem ein Eingriff ansetzen koennte,
   und eine reine Buchstabendichte-Statistik misst am Modell vorbei.

Fuer ``WASD`` im GPT-NeoX-Tokenizer von Pythia gilt Fall 2 beziehungsweise 3.

Aufruf::

    python tokenizer_sonde.py --tokenizer tokenizer.json --ziel WASD --ziel wasd

``tokenizer.json`` stammt aus dem Modell-Repository, zum Beispiel
``https://huggingface.co/EleutherAI/pythia-1.4b/resolve/main/tokenizer.json``.

Vokabelabfragen brauchen nur die Standardbibliothek. Nur das Zerlegen von Text in
Tokens braucht zusaetzlich das Paket ``tokenizers`` (nicht ``torch``); es wird erst
in ``main`` importiert, damit der Rest des Moduls ohne diese Abhaengigkeit nutzbar
und pruefbar bleibt.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence, cast

#: GPT-NeoX/GPT-2-Byte-Kodierung: fuehrendes Leerzeichen wird zu diesem Zeichen.
SPACE_MARKER = "Ġ"


class Kodierung(Protocol):
    """Das, was ein Tokenizer zurueckgibt - nur die hier benutzten Felder."""

    @property
    def ids(self) -> list[int]: ...

    @property
    def tokens(self) -> list[str]: ...


class TokenizerLike(Protocol):
    """Minimalschnittstelle, damit das Modul ohne das Paket typpruefbar bleibt."""

    def encode(self, sequence: str, /) -> Kodierung: ...


@dataclass(frozen=True)
class ZielBefund:
    """Ergebnis fuer eine gepruefte Zeichenfolge."""

    ziel: str
    #: Token-ID, falls die Zeichenfolge selbst ein Vokabeleintrag ist.
    eigenes_token: int | None
    #: Token-ID der Variante mit fuehrendem Leerzeichen.
    eigenes_token_mit_leerzeichen: int | None
    #: In welche Tokens die Zeichenfolge tatsaechlich zerfaellt.
    zerlegung: tuple[str, ...]
    #: Vokabeleintraege, die die Zeichenfolge als Teilkette enthalten.
    teilketten_treffer: tuple[str, ...]

    @property
    def ist_eigenes_token(self) -> bool:
        return (
            self.eigenes_token is not None or self.eigenes_token_mit_leerzeichen is not None
        )

    def urteil(self) -> str:
        if self.ist_eigenes_token:
            return "TRAEGER_VORHANDEN"
        if len(self.zerlegung) <= 2:
            return "ZUSAMMENGESETZT_KURZ"
        return "ZUSAMMENGESETZT_VERTEILT"


def lade_vokabular(tokenizer_json: str) -> dict[str, int]:
    """Liest das Vokabular direkt aus der Datei - ohne das Modell zu laden."""
    with open(tokenizer_json, encoding="utf-8") as handle:
        daten: object = json.load(handle)
    if not isinstance(daten, dict):
        raise ValueError("tokenizer.json enthaelt kein Objekt")
    modell = cast("dict[str, object]", daten).get("model")
    if not isinstance(modell, dict):
        raise ValueError("tokenizer.json enthaelt kein model-Objekt")
    roh = cast("dict[str, object]", modell).get("vocab")
    if not isinstance(roh, dict):
        raise ValueError("tokenizer.json enthaelt kein model.vocab-Objekt")
    vokab: dict[str, int] = {}
    for eintrag, wert in cast("dict[object, object]", roh).items():
        if isinstance(eintrag, str) and isinstance(wert, int):
            vokab[eintrag] = wert
    if not vokab:
        raise ValueError("model.vocab ist leer oder hat ein unerwartetes Format")
    return vokab


def ohne_leerzeichenmarke(eintrag: str) -> str:
    """Entfernt die fuehrende Leerzeichenmarke eines Vokabeleintrags."""
    return eintrag[1:] if eintrag.startswith(SPACE_MARKER) else eintrag


def pruefe_ziel(
    ziel: str,
    vokab: dict[str, int],
    zerlegung: Sequence[str],
) -> ZielBefund:
    """Baut den Befund fuer eine Zeichenfolge aus Vokabular und Zerlegung."""
    treffer = tuple(
        eintrag
        for eintrag in vokab
        if ziel.lower() in ohne_leerzeichenmarke(eintrag).lower()
    )
    return ZielBefund(
        ziel=ziel,
        eigenes_token=vokab.get(ziel),
        eigenes_token_mit_leerzeichen=vokab.get(SPACE_MARKER + ziel),
        zerlegung=tuple(zerlegung),
        teilketten_treffer=treffer[:20],
    )


def reine_zielbuchstaben_tokens(vokab: Iterable[str], buchstaben: str) -> list[str]:
    """Alle Vokabeleintraege, die ausschliesslich aus den Zielbuchstaben bestehen.

    Bei ``WASDQERF`` sind das im GPT-NeoX-Vokabular ueber neunhundert Eintraege,
    darunter die haeufigsten Wortbausteine des Englischen ("re", "er", "as", "was",
    "are", "ed", "es"). Eine Hypothese, die diesen Buchstaben eine Sonderrolle
    zuschreibt, muss erklaeren, warum ausgerechnet das Rueckgrat der englischen
    Orthographie ein Sondersignal traegt.
    """
    menge = set(buchstaben.lower()) | set(buchstaben.upper())
    treffer: list[str] = []
    for eintrag in vokab:
        kern = ohne_leerzeichenmarke(eintrag)
        if kern and all(zeichen in menge for zeichen in kern):
            treffer.append(eintrag)
    return treffer


def _lade_tokenizer(pfad: str) -> TokenizerLike:
    """Laedt den echten Tokenizer; nur hier haengt das Modul am Paket ``tokenizers``.

    Der Import laeuft ueber ``importlib``, damit das Modul auch dort typpruefbar
    bleibt, wo ``tokenizers`` nicht installiert ist - etwa in der Typpruefung dieses
    Projekts, das die Bibliothek nicht als Abhaengigkeit fuehrt.
    """
    modul = importlib.import_module("tokenizers")
    lader: Callable[[str], object] = getattr(modul, "Tokenizer").from_file
    return cast(TokenizerLike, lader(pfad))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tokenizer-Sonde")
    parser.add_argument("--tokenizer", required=True, help="Pfad zu tokenizer.json")
    parser.add_argument(
        "--ziel",
        action="append",
        default=None,
        help="zu pruefende Zeichenfolge, mehrfach angebbar",
    )
    parser.add_argument(
        "--buchstaben",
        default="WASDQERF",
        help="Buchstabenmenge fuer die Vokabelzaehlung",
    )
    parser.add_argument("--text", default=None, help="optionaler Text zum Tokenisieren")
    args = parser.parse_args(argv)

    tokenizer_pfad = cast(str, args.tokenizer)
    rohziele = cast("list[str] | None", args.ziel)
    buchstaben = cast(str, args.buchstaben)
    text = cast("str | None", args.text)

    ziele = rohziele if rohziele else ["WASD", "wasd", "QERF"]
    vokab = lade_vokabular(tokenizer_pfad)
    tokenizer = _lade_tokenizer(tokenizer_pfad)

    print(f"Vokabulargroesse: {len(vokab)}")
    for ziel in ziele:
        befund = pruefe_ziel(ziel, vokab, tokenizer.encode(ziel).tokens)
        print(f"\n{ziel!r}")
        print(f"  eigenes Token:              {befund.eigenes_token}")
        print(f"  mit fuehrendem Leerzeichen: {befund.eigenes_token_mit_leerzeichen}")
        print(f"  Zerlegung:                  {list(befund.zerlegung)}")
        print(f"  Teilketten-Treffer:         {list(befund.teilketten_treffer)}")
        print(f"  Urteil:                     {befund.urteil()}")

    rein = reine_zielbuchstaben_tokens(vokab, buchstaben)
    anteil = 100 * len(rein) / len(vokab)
    print(
        f"\nTokens ausschliesslich aus {buchstaben}: {len(rein)} "
        f"({anteil:.2f} Prozent des Vokabulars)"
    )
    haeufigste = sorted(rein, key=lambda eintrag: vokab[eintrag])[:20]
    print(f"  frueheste Merges: {haeufigste}")

    if text:
        kodiert = tokenizer.encode(text)
        print(f"\nText: {len(kodiert.ids)} Tokens")
        for index, (tid, tok) in enumerate(zip(kodiert.ids, kodiert.tokens)):
            print(f"  {index:4d}  {tid:6d}  {tok!r}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
