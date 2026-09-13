"""Buchstabenmengen-Statistik mit frequenzangepasstem Nullmodell.

Hintergrund
-----------
Ein Textfenster wird daraufhin geprueft, ob eine *vorher benannte* Buchstabenmenge
(z. B. die Tastaturtasten ``WASD`` plus Nachbarn ``QERF``) darin ungewoehnlich dicht
vorkommt.

Der naheliegende Test - "vergleiche die Dichte der Zielmenge mit der Dichte
zufaellig gezogener Buchstabenmengen gleicher Groesse" - ist **verzerrt**, sobald die
Zielmenge aus haeufigen Buchstaben besteht. ``WASDQERF`` enthaelt E, A, R, S: vier der
sechs haeufigsten Buchstaben des Englischen. Seine erwartete Dichte in gewoehnlichem
englischem Text liegt bei rund 0.413, waehrend eine zufaellig gezogene Achtermenge im
Mittel nur rund 0.308 erreicht. Ein positives z gegen dieses Nullmodell misst damit
die Buchstabenhaeufigkeit, nicht die Besonderheit des Textes.

Dieses Modul stellt drei Nullmodelle nebeneinander:

``unconditional``
    Alle ``C(26, k)`` Buchstabenmengen. Das ist das verzerrte Modell; es wird hier
    nur mitgerechnet, damit sich publizierte Zahlen reproduzieren lassen.
``frequency_matched``
    Nur Mengen, deren *erwartete* Dichte in englischem Text der erwarteten Dichte der
    Zielmenge entspricht (Toleranzband). Damit faellt der Haeufigkeitseffekt heraus und
    uebrig bleibt die Frage, ob dieser Text fuer diese Buchstaben auffaellig ist.
``ratio``
    Verhaeltnis beobachtete zu erwarteter Dichte, ueber alle Mengen oberhalb einer
    Mindesterwartung. Frequenzbereinigt ohne Toleranzband.

Alles hier ist Text- und Tokenizer-Statistik. Kein Vorwaertslauf eines Modells, also
auch keine Aussage darueber, ob ein Netz diese Buchstaben benutzt.
"""

from __future__ import annotations

import itertools
import math
import string
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

__all__ = [
    "ENGLISH_LETTER_FREQUENCY",
    "NullResult",
    "SetStatistics",
    "letter_density",
    "expected_density",
    "analyse_letter_set",
    "enrichment_ranking",
]

#: Relative Buchstabenhaeufigkeiten im Englischen (Norvig, Google-Books-Korpus).
#: Summe 1.0 bis auf Rundung; nur zur Erwartungsbildung, nicht als Messwert.
ENGLISH_LETTER_FREQUENCY: Mapping[str, float] = {
    "e": 0.1249, "t": 0.0928, "a": 0.0804, "o": 0.0764, "i": 0.0757,
    "n": 0.0723, "s": 0.0651, "r": 0.0628, "h": 0.0505, "l": 0.0407,
    "d": 0.0382, "c": 0.0334, "u": 0.0273, "m": 0.0251, "f": 0.0240,
    "p": 0.0214, "g": 0.0187, "w": 0.0168, "y": 0.0166, "b": 0.0148,
    "v": 0.0105, "k": 0.0054, "x": 0.0023, "j": 0.0016, "q": 0.0012,
    "z": 0.0009,
}


def _normalise(letters: Iterable[str]) -> tuple[str, ...]:
    """Kleinschreiben, deduplizieren, sortieren - und nur Buchstaben zulassen."""
    seen: list[str] = []
    for raw in letters:
        for char in raw:
            low = char.lower()
            if low not in string.ascii_lowercase:
                raise ValueError(f"kein ASCII-Buchstabe: {char!r}")
            if low not in seen:
                seen.append(low)
    if not seen:
        raise ValueError("leere Buchstabenmenge")
    return tuple(sorted(seen))


def letter_counts(text: str) -> dict[str, int]:
    """Zaehlt Buchstaben case-insensitiv; alles Nicht-Alphabetische faellt weg."""
    counts = {ch: 0 for ch in string.ascii_lowercase}
    for char in text:
        low = char.lower()
        if low in counts:
            counts[low] += 1
    return counts


def letter_density(text: str, letters: Iterable[str]) -> float:
    """Anteil der Zielbuchstaben an allen alphabetischen Zeichen des Textes."""
    target = _normalise(letters)
    counts = letter_counts(text)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Text enthaelt keine alphabetischen Zeichen")
    return sum(counts[ch] for ch in target) / total


def expected_density(
    letters: Iterable[str],
    frequencies: Mapping[str, float] = ENGLISH_LETTER_FREQUENCY,
) -> float:
    """Erwartete Dichte der Buchstabenmenge in gewoehnlichem Text."""
    target = _normalise(letters)
    return sum(frequencies[ch] for ch in target)


@dataclass(frozen=True)
class NullResult:
    """Ergebnis eines Nullmodells."""

    name: str
    n_reference_sets: int
    reference_mean: float
    reference_sd: float
    observed: float
    z: float
    p_upper: float
    #: Wie viele Referenzmengen mindestens so extrem sind (fuer p_upper).
    n_at_least_as_extreme: int


@dataclass(frozen=True)
class SetStatistics:
    """Vollstaendiges Ergebnis fuer eine Zielmenge in einem Text."""

    letters: tuple[str, ...]
    n_alphabetic: int
    observed_density: float
    expected_density: float
    observed_over_expected: float
    nulls: dict[str, NullResult] = field(default_factory=dict)

    def verdict(self, alpha: float = 0.05) -> str:
        """Urteil auf Basis des frequenzangepassten Nullmodells."""
        matched = self.nulls.get("frequency_matched")
        if matched is None:
            return "UNBESTIMMT"
        if matched.p_upper <= alpha:
            return "ANGEREICHERT"
        if matched.p_upper >= 1.0 - alpha:
            return "ABGEREICHERT"
        return "UNAUFFAELLIG"


def _summarise(
    name: str,
    observed: float,
    reference: Sequence[float],
) -> NullResult:
    n = len(reference)
    if n == 0:
        raise ValueError(f"Nullmodell {name!r} hat keine Referenzmengen")
    mean = math.fsum(reference) / n
    var = math.fsum((value - mean) ** 2 for value in reference) / n
    sd = math.sqrt(var)
    at_least = sum(1 for value in reference if value >= observed)
    return NullResult(
        name=name,
        n_reference_sets=n,
        reference_mean=mean,
        reference_sd=sd,
        observed=observed,
        z=(observed - mean) / sd if sd > 0 else 0.0,
        p_upper=at_least / n,
        n_at_least_as_extreme=at_least,
    )


def analyse_letter_set(
    text: str,
    letters: Iterable[str],
    *,
    frequencies: Mapping[str, float] = ENGLISH_LETTER_FREQUENCY,
    match_band: float = 0.01,
    min_expected_for_ratio: float = 0.05,
) -> SetStatistics:
    """Vergleicht eine Buchstabenmenge gegen drei Nullmodelle.

    ``match_band`` ist die Toleranz des frequenzangepassten Modells: nur Mengen,
    deren erwartete Dichte innerhalb dieses Bandes um die Erwartung der Zielmenge
    liegt, zaehlen als Referenz. Ein zu enges Band laesst zu wenige Referenzmengen
    uebrig; ``ValueError`` weist darauf hin, statt eine leere Statistik zu liefern.
    """
    target = _normalise(letters)
    counts = letter_counts(text)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Text enthaelt keine alphabetischen Zeichen")

    observed = sum(counts[ch] for ch in target) / total
    expected = sum(frequencies[ch] for ch in target)

    all_sets = list(itertools.combinations(string.ascii_lowercase, len(target)))
    obs_per_set = [sum(counts[ch] for ch in s) / total for s in all_sets]
    exp_per_set = [sum(frequencies[ch] for ch in s) for s in all_sets]

    nulls: dict[str, NullResult] = {}
    nulls["unconditional"] = _summarise("unconditional", observed, obs_per_set)

    matched = [
        obs_per_set[i]
        for i, exp_i in enumerate(exp_per_set)
        if abs(exp_i - expected) <= match_band
    ]
    if len(matched) < 30:
        raise ValueError(
            f"frequenzangepasstes Nullmodell hat nur {len(matched)} Referenzmengen; "
            f"match_band={match_band} zu eng"
        )
    nulls["frequency_matched"] = _summarise("frequency_matched", observed, matched)

    ratio_ref = [
        obs_per_set[i] / exp_i
        for i, exp_i in enumerate(exp_per_set)
        if exp_i >= min_expected_for_ratio
    ]
    ratio_obs = observed / expected if expected > 0 else float("nan")
    if expected >= min_expected_for_ratio and ratio_ref:
        nulls["ratio"] = _summarise("ratio", ratio_obs, ratio_ref)

    return SetStatistics(
        letters=target,
        n_alphabetic=total,
        observed_density=observed,
        expected_density=expected,
        observed_over_expected=ratio_obs,
        nulls=nulls,
    )


def enrichment_ranking(
    text: str,
    set_size: int,
    *,
    frequencies: Mapping[str, float] = ENGLISH_LETTER_FREQUENCY,
    min_expected: float = 0.05,
    top: int = 10,
) -> list[tuple[str, float]]:
    """Welche Buchstabenmengen sind in diesem Text wirklich angereichert?

    Sortiert nach dem Quotienten beobachtet zu erwartet, absteigend. Dient als
    Gegenprobe: steht die vorab benannte Zielmenge nicht in dieser Liste, war die
    Auswahl der Menge nicht datengetrieben - was gut ist, solange man sie auch nicht
    nachtraeglich als Fund verkauft.
    """
    counts = letter_counts(text)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Text enthaelt keine alphabetischen Zeichen")
    scored: list[tuple[str, float]] = []
    for combo in itertools.combinations(string.ascii_lowercase, set_size):
        exp = sum(frequencies[ch] for ch in combo)
        if exp < min_expected:
            continue
        obs = sum(counts[ch] for ch in combo) / total
        scored.append(("".join(combo), obs / exp))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top]
