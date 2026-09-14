"""Vorregistrierter Nachfolgeversuch: benutzt Pythia die Tastenfolge WASD als Einheit?

Warum dieser Versuch
--------------------
Die bisherigen Laeufe haben Buchstabendichte im Text gemessen. Das beantwortet die
Frage nicht, weil ein Transformer keine Buchstaben sieht. Dieser Versuch stellt sie
dort, wo sie eine Vorhersage macht: an einer Stelle, an der ``WASD`` tatsaechlich als
Tastenkuerzel im Text steht.

Das Entscheidende liefert der Tokenizer selbst. Im GPT-NeoX-Vokabular zerfaellt

    " WASD"  ->  ['ĠWAS', 'D']       (Token-IDs 22250, 37)

in genau zwei Stuecke. Damit gibt es eine **minimale Kontrollfamilie**: alle
Zeichenfolgen ``WAS?`` mit einem anderen Grossbuchstaben am Ende zerfallen ebenso in
``['ĠWAS', '?']``. Erstes Token identisch, zweites Token ein einzelner Grossbuchstabe
wie ``D``. Zwischen Ziel und Kontrolle unterscheidet sich also **genau ein Token**,
und zwar das, dem die Hypothese ihre Bedeutung zuschreibt.

Dazu kommt eine Reihenfolgekontrolle: ``" ASDW"`` zerfaellt in ``['ĠASD', 'W']`` -
dieselben vier Buchstaben, dieselbe Struktur, falsche Reihenfolge.

Ein Nebenbefund, der die Kleinschreibung ausschliesst: ``" wasd"`` zerfaellt in
``['Ġwas', 'd']``, und ``Ġwas`` ist das englische Wort *was* (Merge-ID 369, einer der
frueheste Merges ueberhaupt). Jeder Effekt an der kleingeschriebenen Form waere mit
der Vergangenheitsform von *to be* vermengt. Der Versuch benutzt deshalb
ausschliesslich die Grossschreibung.

Vorregistrierung
----------------
Die Entscheidungsregel steht in ``VORREGISTRIERUNG`` und wird **vor** dem Lauf
festgelegt. Sie verlangt beides: einen Beobachtungsteil (das Zieltoken hebt das
Bewegungsvokabular staerker als jede Kontrolle) und einen kausalen Teil (ein
Eingriff an genau dieser Tokenposition traegt den Effekt, und die Kontrolle traegt
ihn nicht). Ein einzelner positiver Teil zaehlt nicht.

Aufruf
------
Trockenlauf, prueft nur das Design und braucht kein Modell::

    python experiment_traeger.py --tokenizer tokenizer.json --trockenlauf

Messlauf (braucht ``torch`` und ``transformers``, gedacht fuer eine GPU)::

    python experiment_traeger.py --tokenizer tokenizer.json --messen --ausgabe lauf/
"""

from __future__ import annotations

import argparse
import json
import string
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Protocol, Sequence, cast

MODELL = "EleutherAI/pythia-1.4b"
REVISION = "step98000"
PRAEZISION = "float32"
BATCHGROESSE = 1

#: Die Zielzeichenfolge und ihre Zerlegung im GPT-NeoX-Tokenizer.
ZIEL = "WASD"
ZIEL_ZERLEGUNG = ("ĠWAS", "D")

#: Reihenfolgekontrolle: dieselben Buchstaben, dieselbe Struktur, andere Ordnung.
REIHENFOLGE_KONTROLLE = "ASDW"

#: Zusatzkontrollen, die die Buchstabenfamilie allein nicht leistet. Jede schliesst
#: eine andere Luecke; die Zerlegungen sind am GPT-NeoX-Tokenizer geprueft.
#:
#: ``FORD``  zerfaellt in ``['ĠFOR', 'D']`` - **dasselbe Endtoken** wie das Ziel
#:           (ID 37), aber ein anderer Stamm. Traegt allein der Buchstabe ``D`` den
#:           Effekt, muss FORD ihn ebenfalls zeigen. Das ist die schaerfste Kontrolle
#:           des Designs.
#: ``NOTA``  zerfaellt in ``['ĠNOT', 'A']``, ist strukturgleich und im Korpus fast
#:           gleich haeufig wie das Ziel (14 286 gegen 13 870 Vorkommen). Sie ist die
#:           eigentliche Nulllinie: eine Viererabkuerzung ohne jeden Bezug zu
#:           Eingabegeraeten, bei vergleichbarer Haeufigkeit.
#: ``ESDF``  ist eine echte alternative Bewegungstastenbelegung, zerfaellt aber in
#:           **drei** Stuecke (``['ĠE', 'SD', 'F']``) und ist mit 1 124 Vorkommen
#:           deutlich seltener. Sie kann deshalb **nicht** in die Hauptregel eingehen
#:           und wird getrennt berichtet.
#:
#: Die Rolle ``ROLLE_GETRENNT`` markiert genau die Kontrolle, die aus der Hauptregel
#: herausfaellt. Sie steht als Konstante da, damit Design, Entscheidungsregel und
#: Messlauf denselben Namen benutzen und die Ausnahme nicht an drei Stellen getippt
#: werden muss.
ROLLE_GETRENNT = "tastenkontrolle_unabgeglichen"

ZUSATZKONTROLLEN: tuple[tuple[str, str], ...] = (
    ("FORD", "endtokenkontrolle"),
    ("NOTA", "frequenzkontrolle"),
    ("ESDF", ROLLE_GETRENNT),
)

#: Alle Rollen der Zusatzkontrollen - abgeleitet, nicht noch einmal aufgeschrieben.
ZUSATZROLLEN: frozenset[str] = frozenset(rolle for _, rolle in ZUSATZKONTROLLEN)

#: Woerter, deren Wahrscheinlichkeit gemessen wird. Bewegung gegen zwei Gegenfelder.
#:
#: Jedes Wort muss ein **einzelnes** Token sein. Sonst misst die Sonde nicht die
#: Wahrscheinlichkeit des Wortes, sondern die seines ersten Stuecks - und dieses
#: Stueck teilt es mit anderen Woertern. Zwei urspruenglich vorgesehene Woerter
#: erfuellen das nicht und wurden **vor** jeder Messung ersetzt: ``" strafe"``
#: zerfaellt in ``['Ġstra', 'fe']`` (ersetzt durch ``" sprint"``) und ``" senate"``
#: in ``['Ġsen', 'ate']`` (ersetzt durch ``" parliament"``). Das Kriterium stand
#: vorher fest, die Ersetzung folgt ihm; sie ist keine Anpassung an ein Ergebnis.
MESSVOKABULAR: dict[str, tuple[str, ...]] = {
    "bewegung": (" move", " walk", " sprint", " forward", " backward"),
    "tastatur": (" keys", " key", " keyboard", " bind", " controls"),
    "gegenfeld": (" weather", " orbit", " proof", " recipe", " parliament"),
}

#: Die am Tokenizer von ``EleutherAI/pythia-1.4b`` abgelesenen Token-IDs, in der
#: Reihenfolge von ``MESSVOKABULAR``. Der Messlauf vergleicht die Tokenisierung
#: gegen diese Tabelle und bricht bei Abweichung ab: eine stillschweigend andere
#: Tokenisierung - falsches Modell, falsche Revision - wuerde alle Zahlen
#: entwerten, ohne dass es an ihnen zu sehen waere.
MESSVOKABULAR_IDS: dict[str, tuple[int, ...]] = {
    "bewegung": (2118, 2940, 29644, 3579, 19265),
    "tastatur": (10149, 2234, 15487, 8980, 5760),
    "gegenfeld": (8588, 12801, 4737, 13612, 16005),
}

#: Satzschablonen. ``{ziel}`` wird durch Ziel oder Kontrolle ersetzt; gemessen wird
#: die Verteilung des naechsten Tokens direkt nach dem Platzhalter.
SCHABLONEN: tuple[str, ...] = (
    "In the options menu you can rebind the movement keys. By default the game uses {ziel}",
    "Controls: the player character is moved with {ziel}",
    "Press the standard PC layout keys {ziel}",
    "The tutorial explains that you should hold {ziel}",
    "Most first person shooters map movement to {ziel}",
    "He kept his left hand on {ziel}",
    "The config file remaps the arrow keys to {ziel}",
    "Movement is bound to the four keys {ziel}",
    "You walk around the level using {ziel}",
    "The default binding for movement is {ziel}",
    "Setting: movement keys {ziel}",
    "For strafing and walking forward, use {ziel}",
    "The manual says the character responds to {ziel}",
    "Remap the directional input from the arrows to {ziel}",
    "Her fingers rested on {ziel}",
    "Beginners are told to learn {ziel}",
    "The keybind menu lists movement under {ziel}",
    "Standard PC controls put movement on {ziel}",
    "To move the avatar, press {ziel}",
    "The readme documents the movement cluster {ziel}",
    "Navigation in the editor is bound to {ziel}",
    "Players consistently prefer {ziel}",
    "The input handler reads the keys {ziel}",
    "Forward, back, left and right are mapped to {ziel}",
)


class Kodierung(Protocol):
    @property
    def ids(self) -> list[int]: ...

    @property
    def tokens(self) -> list[str]: ...


class TokenizerLike(Protocol):
    def encode(self, sequence: str, /) -> Kodierung: ...


@dataclass(frozen=True)
class Vorregistrierung:
    """Die Entscheidungsregel, vor dem Lauf festgelegt.

    ``mindest_vorsprung_nats`` ist der Abstand, den das Zieltoken im mittleren
    Log-Wahrscheinlichkeitsgewinn des Bewegungsvokabulars gegenueber der **besten**
    Kontrolle haben muss - nicht gegenueber dem Kontrollmittel. Das verhindert, dass
    ein einzelner Ausreisser unter den 24 Kontrollen als Bestaetigung durchgeht.

    ``mindest_kausalanteil`` ist der Anteil des Beobachtungseffekts, den ein Eingriff
    an der Position des letzten Tokens wiederherstellen muss.

    ``mindest_schichten`` verlangt, dass der kausale Effekt an mindestens so vielen
    **benachbarten** Schichten auftritt; ein einzelner Ausschlag zaehlt nicht.
    """

    mindest_vorsprung_nats: float = 0.5
    #: Zweitfassung: statt gegen das Maximum gegen dieses Quantil der Kontrollen.
    #: Vor dem Lauf festgelegt und unabhaengig vom Ausgang mitberichtet. Die
    #: Simulation (200 000 Ziehungen) ergab fuer sigma = 0.5 nats und 24 Schablonen
    #: eine Trennschaerfe von 0.832 gegen das Maximum und 0.950 gegen dieses Quantil,
    #: bei in beiden Faellen verschwindender Fehlalarmrate.
    kontroll_quantil: float = 0.90
    mindest_kausalanteil: float = 0.5
    mindest_schichten: int = 2
    hoechstes_kontroll_leck: float = 0.2
    #: Wie viele Satzkontexte je Variante gemessen werden. Das ist keine frei
    #: waehlbare Zahl, sondern die Zahl der Schablonen - sie steht hier nur, damit
    #: die Vorregistrierung vollstaendig ist, und wird deshalb abgeleitet statt
    #: getippt. Aus der Trennschaerfesimulation stammt die Untergrenze: unter zwoelf
    #: Kontexten faellt die Trennschaerfe bei sigma = 0.5 nats unter 0.5.
    n_kontexte_je_variante: int = len(SCHABLONEN)


VORREGISTRIERUNG = Vorregistrierung()


@dataclass(frozen=True)
class Variante:
    """Eine Zeichenfolge im Versuch, mit ihrer Rolle und ihrer Zerlegung."""

    text: str
    rolle: str  # "ziel" | "buchstabenkontrolle" | "reihenfolgekontrolle"
    zerlegung: tuple[str, ...]
    token_ids: tuple[int, ...]

    @property
    def letztes_token(self) -> str:
        return self.zerlegung[-1]


#: So viele strukturgleiche Buchstabenkontrollen muss das Design mindestens behalten.
MINDEST_KONTROLLEN = 15


@dataclass(frozen=True)
class Designpruefung:
    """Ergebnis der Strukturpruefung - der Trockenlauf gibt genau das aus."""

    n_varianten: int
    n_buchstabenkontrollen: int
    verworfen: tuple[str, ...]
    erstes_token_identisch: bool
    letzte_token_eindeutig: bool
    reihenfolgekontrolle_gleich_lang: bool
    n_schablonen: int
    n_messpunkte: int
    maengel: tuple[str, ...]

    @property
    def bestanden(self) -> bool:
        return not self.maengel


def baue_varianten(tokenizer: TokenizerLike) -> list[Variante]:
    """Ziel, 24 Buchstabenkontrollen und eine Reihenfolgekontrolle.

    Ausgeschlossen wird ``WASS``: dort verschmilzt das Vokabular anders, weil das
    doppelte S eigene Merges hat. Welche Kandidaten die Struktur verletzen, entscheidet
    nicht eine Annahme, sondern der Tokenizer - deshalb wird jede Variante geprueft.
    """
    varianten: list[Variante] = []

    def erfasse(text: str, rolle: str) -> None:
        kodiert = tokenizer.encode(" " + text)
        varianten.append(
            Variante(
                text=text,
                rolle=rolle,
                zerlegung=tuple(kodiert.tokens),
                token_ids=tuple(kodiert.ids),
            )
        )

    erfasse(ZIEL, "ziel")
    stamm = ZIEL[:-1]
    for buchstabe in string.ascii_uppercase:
        if buchstabe == ZIEL[-1]:
            continue
        erfasse(stamm + buchstabe, "buchstabenkontrolle")
    erfasse(REIHENFOLGE_KONTROLLE, "reihenfolgekontrolle")
    for text, rolle in ZUSATZKONTROLLEN:
        erfasse(text, rolle)
    return varianten


def strukturgleiche_kontrollen(varianten: Sequence[Variante]) -> list[Variante]:
    """Behaelt nur die Kontrollen, die exakt wie das Ziel zerfallen.

    Fuenf der 25 Kandidaten tun das nicht: ``WASE``, ``WASH``, ``WASK``, ``WASS`` und
    ``WAST`` zerfallen in ``['ĠW', 'ASE']`` und so weiter, weil ``ASH``, ``ASK``,
    ``ASS``, ``AST`` und ``ASE`` eigene Vokabeleintraege sind. Sie gehoeren nicht ins
    Design: bei ihnen unterscheidet sich **beides** vom Ziel, nicht nur das letzte
    Token. Welche das sind, entscheidet der Tokenizer, nicht eine Annahme.
    """
    return [
        variante
        for variante in varianten
        if variante.rolle == "buchstabenkontrolle"
        and len(variante.zerlegung) == len(ZIEL_ZERLEGUNG)
        and variante.zerlegung[0] == ZIEL_ZERLEGUNG[0]
    ]


def pruefe_design(varianten: Sequence[Variante]) -> Designpruefung:
    """Prueft die Strukturgleichheit, auf der die ganze Aussagekraft beruht."""
    maengel: list[str] = []

    ziele = [v for v in varianten if v.rolle == "ziel"]
    if len(ziele) != 1:
        maengel.append(f"genau eine Zielvariante erwartet, gefunden {len(ziele)}")
    ziel = ziele[0] if ziele else None

    if ziel is not None and ziel.zerlegung != ZIEL_ZERLEGUNG:
        maengel.append(
            f"Ziel zerfaellt in {list(ziel.zerlegung)}, erwartet {list(ZIEL_ZERLEGUNG)}"
        )

    kontrollen = [v for v in varianten if v.rolle == "buchstabenkontrolle"]
    brauchbar = strukturgleiche_kontrollen(varianten)
    verworfen = sorted({v.text for v in kontrollen} - {v.text for v in brauchbar})
    if len(brauchbar) < MINDEST_KONTROLLEN:
        maengel.append(
            f"nur {len(brauchbar)} strukturgleiche Kontrollen, mindestens "
            f"{MINDEST_KONTROLLEN} noetig"
        )

    erstes_identisch = all(v.zerlegung[0] == ZIEL_ZERLEGUNG[0] for v in brauchbar)

    letzte = [v.letztes_token for v in brauchbar] + ([ziel.letztes_token] if ziel else [])
    eindeutig = len(set(letzte)) == len(letzte)
    if not eindeutig:
        maengel.append("letzte Tokens sind nicht paarweise verschieden")

    reihenfolge = [v for v in varianten if v.rolle == "reihenfolgekontrolle"]
    gleich_lang = bool(reihenfolge) and all(
        len(v.zerlegung) == len(ZIEL_ZERLEGUNG) for v in reihenfolge
    )
    if not gleich_lang:
        maengel.append("Reihenfolgekontrolle hat eine andere Tokenzahl als das Ziel")

    zusatz = [v for v in varianten if v.rolle in ZUSATZROLLEN]
    # Die strukturgleichen Zusatzkontrollen muessen wie das Ziel zerfallen; die
    # ausdruecklich unabgeglichene darf das nicht und wird deshalb nicht geprueft.
    for variante in zusatz:
        if variante.rolle == ROLLE_GETRENNT:
            continue
        if len(variante.zerlegung) != len(ZIEL_ZERLEGUNG):
            maengel.append(
                f"Zusatzkontrolle {variante.text} zerfaellt in "
                f"{list(variante.zerlegung)}, erwartet {len(ZIEL_ZERLEGUNG)} Stuecke"
            )

    nutzbar = len(brauchbar) + len(ziele) + len(reihenfolge) + len(zusatz)
    return Designpruefung(
        n_varianten=len(varianten),
        n_buchstabenkontrollen=len(brauchbar),
        verworfen=tuple(verworfen),
        erstes_token_identisch=erstes_identisch,
        letzte_token_eindeutig=eindeutig,
        reihenfolgekontrolle_gleich_lang=gleich_lang,
        n_schablonen=len(SCHABLONEN),
        n_messpunkte=nutzbar * len(SCHABLONEN),
        maengel=tuple(maengel),
    )


def baue_prompts(varianten: Iterable[Variante]) -> list[tuple[Variante, str, str]]:
    """Kreuzt jede Variante mit jeder Satzschablone."""
    prompts: list[tuple[Variante, str, str]] = []
    for variante in varianten:
        for schablone in SCHABLONEN:
            prompts.append((variante, schablone, schablone.format(ziel=variante.text)))
    return prompts


@dataclass(frozen=True)
class Beobachtung:
    """Was je Variante aus dem Beobachtungsteil herauskommt."""

    text: str
    rolle: str
    #: mittlerer Log-Wahrscheinlichkeitsgewinn des Bewegungsvokabulars, in nats
    bewegung_nats: float
    tastatur_nats: float
    gegenfeld_nats: float


@dataclass(frozen=True)
class Kausalmessung:
    """Was je Schicht aus dem Eingriffsteil herauskommt."""

    schicht: int
    #: Anteil des Beobachtungseffekts, den der Eingriff wiederherstellt
    wiederherstellung_ziel: float
    #: derselbe Eingriff mit einer Kontrollvariante als Quelle
    wiederherstellung_kontrolle: float


def urteile(
    beobachtungen: Sequence[Beobachtung],
    kausal: Sequence[Kausalmessung],
    regel: Vorregistrierung = VORREGISTRIERUNG,
) -> dict[str, object]:
    """Wendet die vorregistrierte Entscheidungsregel an.

    Gibt beide Teilurteile getrennt zurueck und das Gesamturteil nur dann positiv,
    wenn beide Teile halten. Die Regel ist absichtlich streng gegen den Ausreisser:
    verglichen wird gegen die **beste** Kontrolle, nicht gegen ihren Mittelwert.

    Eine Kontrolle geht ausdruecklich **nicht** in die Hauptregel ein: die
    unabgeglichene Tastenkontrolle zerfaellt anders als das Ziel und ist im Korpus
    deutlich seltener. Sie wird getrennt berichtet, damit sie das Urteil weder
    stuetzen noch verwaessern kann.
    """
    ziele = [b for b in beobachtungen if b.rolle == "ziel"]
    kontrollen = [b for b in beobachtungen if b.rolle not in ("ziel", ROLLE_GETRENNT)]
    getrennt = [b for b in beobachtungen if b.rolle == ROLLE_GETRENNT]
    if len(ziele) != 1 or not kontrollen:
        raise ValueError("Beobachtungsteil braucht genau ein Ziel und Kontrollen")
    ziel = ziele[0]

    beste_kontrolle = max(kontrollen, key=lambda b: b.bewegung_nats)
    vorsprung = ziel.bewegung_nats - beste_kontrolle.bewegung_nats
    beobachtung_haelt = vorsprung >= regel.mindest_vorsprung_nats

    # Vorab benannte Zweitfassung: gegen das Quantil statt gegen das Maximum.
    sortiert = sorted(b.bewegung_nats for b in kontrollen)
    stelle = regel.kontroll_quantil * (len(sortiert) - 1)
    unten = int(stelle)
    oben = min(unten + 1, len(sortiert) - 1)
    quantil = sortiert[unten] + (stelle - unten) * (sortiert[oben] - sortiert[unten])
    vorsprung_quantil = ziel.bewegung_nats - quantil
    beobachtung_haelt_quantil = vorsprung_quantil >= regel.mindest_vorsprung_nats

    laeufe: list[int] = []
    aktuell = 0
    for messung in sorted(kausal, key=lambda m: m.schicht):
        traegt = (
            messung.wiederherstellung_ziel >= regel.mindest_kausalanteil
            and messung.wiederherstellung_kontrolle <= regel.hoechstes_kontroll_leck
        )
        aktuell = aktuell + 1 if traegt else 0
        laeufe.append(aktuell)
    laengster_lauf = max(laeufe) if laeufe else 0
    kausal_haelt = laengster_lauf >= regel.mindest_schichten

    return {
        "vorsprung_nats": vorsprung,
        "beste_kontrolle": beste_kontrolle.text,
        "n_kontrollen_in_der_regel": len(kontrollen),
        "vorsprung_gegen_quantil_nats": vorsprung_quantil,
        "beobachtungsteil_quantilsfassung": beobachtung_haelt_quantil,
        "getrennt_berichtet": {b.text: b.bewegung_nats for b in getrennt},
        "beobachtungsteil": beobachtung_haelt,
        "laengster_kausallauf": laengster_lauf,
        "kausalteil": kausal_haelt,
        "wasd_traeger_bestaetigt": beobachtung_haelt and kausal_haelt,
        "regel": asdict(regel),
    }


def _lade_tokenizer(pfad: str) -> TokenizerLike:
    import importlib

    modul = importlib.import_module("tokenizers")
    lader: Callable[[str], object] = getattr(modul, "Tokenizer").from_file
    return cast(TokenizerLike, lader(pfad))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vorregistrierter WASD-Traegerversuch")
    parser.add_argument("--tokenizer", required=True, help="Pfad zu tokenizer.json")
    parser.add_argument(
        "--trockenlauf",
        action="store_true",
        help="nur das Design pruefen und ausgeben, kein Modell laden",
    )
    parser.add_argument("--ausgabe", default=None, help="Verzeichnis fuer den Messlauf")
    args = parser.parse_args(argv)

    tokenizer = _lade_tokenizer(cast(str, args.tokenizer))
    varianten = baue_varianten(tokenizer)
    pruefung = pruefe_design(varianten)

    print("Design")
    print(f"  Modell:            {MODELL} @ {REVISION}, {PRAEZISION}, Batch {BATCHGROESSE}")
    print(f"  Varianten:         {pruefung.n_varianten}")
    print(f"  Buchstabenkontrollen mit identischer Struktur: {pruefung.n_buchstabenkontrollen}")
    print(f"  verworfen (andere Zerlegung): {list(pruefung.verworfen)}")
    print(f"  Schablonen:        {pruefung.n_schablonen}")
    print(f"  Messpunkte:        {pruefung.n_messpunkte}")
    print(f"  Strukturpruefung:  {'bestanden' if pruefung.bestanden else 'FEHLER'}")
    for mangel in pruefung.maengel:
        print(f"    - {mangel}")

    print("\nZerlegungen")
    for variante in varianten:
        marke = {
            "ziel": "ZIEL",
            "reihenfolgekontrolle": "ORDN",
            "endtokenkontrolle": "ENDT",
            "frequenzkontrolle": "FREQ",
            ROLLE_GETRENNT: "SEP ",
        }.get(variante.rolle, "    ")
        print(f"  {marke} {variante.text:6s} {list(variante.zerlegung)}  {list(variante.token_ids)}")

    print("\nVorregistrierte Entscheidungsregel")
    for schluessel, wert in asdict(VORREGISTRIERUNG).items():
        print(f"  {schluessel}: {wert}")

    if args.trockenlauf:
        return 0 if pruefung.bestanden else 1

    if not pruefung.bestanden:
        print("\nStrukturpruefung nicht bestanden - kein Messlauf.")
        return 1

    print(
        "\nDer Messteil ist bewusst nicht Teil dieser Datei: er braucht eine GPU und "
        "gehoert in den Laufordner. Diese Datei liefert das Design, die Pruefung und "
        "die Entscheidungsregel; `urteile` nimmt die Messwerte entgegen."
    )
    if args.ausgabe:
        ziel = cast(str, args.ausgabe)
        with open(f"{ziel.rstrip('/')}/design.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "modell": MODELL,
                    "revision": REVISION,
                    "praezision": PRAEZISION,
                    "batchgroesse": BATCHGROESSE,
                    "varianten": [asdict(v) for v in varianten],
                    "schablonen": list(SCHABLONEN),
                    "messvokabular": {k: list(v) for k, v in MESSVOKABULAR.items()},
                    "vorregistrierung": asdict(VORREGISTRIERUNG),
                    "designpruefung": asdict(pruefung),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Design geschrieben nach {ziel.rstrip('/')}/design.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
