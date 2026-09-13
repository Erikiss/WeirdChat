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
Benoetigt nur das Paket ``tokenizers``, nicht ``torch``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

#: GPT-NeoX/GPT-2-Byte-Kodierung: fuehrendes Leerzeichen wird zu diesem Zeichen.
SPACE_MARKER = "Ġ"


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
        return self.eigenes_token is not None or self.eigenes_token_mit_leerzeichen is not None

    def urteil(self) -> str:
        if self.ist_eigenes_token:
            return "TRAEGER_VORHANDEN"
        if len(self.zerlegung) <= 2:
            return "ZUSAMMENGESETZT_KURZ"
        return "ZUSAMMENGESETZT_VERTEILT"


def lade_vokabular(tokenizer_json: str) -> dict[str, int]:
    """Liest das Vokabular direkt aus der Datei - ohne das Modell zu laden."""
    with open(tokenizer_json, encoding="utf-8") as handle:
        daten = json.load(handle)
    vokab = daten.get("model", {}).get("vocab")
    if not isinstance(vokab, dict):
        raise ValueError("tokenizer.json enthaelt kein model.vocab-Objekt")
    return vokab


def pruefe_ziel(
    ziel: str,
    vokab: dict[str, int],
    zerlegung: Sequence[str],
) -> ZielBefund:
    """Baut den Befund fuer eine Zeichenfolge aus Vokabular und Zerlegung."""
    treffer = tuple(
        eintrag for eintrag in vokab if ziel.lower() in eintrag.lower().replace(SPACE_MARKER, "")
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
    treffer = []
    for eintrag in vokab:
        kern = eintrag[1:] if eintrag.startswith(SPACE_MARKER) else eintrag
        if kern and all(zeichen in menge for zeichen in kern):
            treffer.append(eintrag)
    return treffer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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

    ziele = args.ziel or ["WASD", "wasd", "QERF"]
    vokab = lade_vokabular(args.tokenizer)

    from tokenizers import Tokenizer  # lokaler Import: nur hier noetig

    tokenizer = Tokenizer.from_file(args.tokenizer)

    print(f"Vokabulargroesse: {len(vokab)}")
    for ziel in ziele:
        befund = pruefe_ziel(ziel, vokab, tokenizer.encode(ziel).tokens)
        print(f"\n{ziel!r}")
        print(f"  eigenes Token:              {befund.eigenes_token}")
        print(f"  mit fuehrendem Leerzeichen: {befund.eigenes_token_mit_leerzeichen}")
        print(f"  Zerlegung:                  {list(befund.zerlegung)}")
        print(f"  Teilketten-Treffer:         {list(befund.teilketten_treffer)}")
        print(f"  Urteil:                     {befund.urteil()}")

    rein = reine_zielbuchstaben_tokens(vokab, args.buchstaben)
    anteil = 100 * len(rein) / len(vokab)
    print(
        f"\nTokens ausschliesslich aus {args.buchstaben}: {len(rein)} "
        f"({anteil:.2f} Prozent des Vokabulars)"
    )
    haeufigste = sorted(rein, key=lambda eintrag: vokab[eintrag])[:20]
    print(f"  frueheste Merges: {haeufigste}")

    if args.text:
        kodiert = tokenizer.encode(args.text)
        print(f"\nText: {len(kodiert.ids)} Tokens")
        for index, (tid, tok) in enumerate(zip(kodiert.ids, kodiert.tokens)):
            print(f"  {index:4d}  {tid:6d}  {tok!r}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
