# Projektstand — die Untersuchung im Überblick

Diese Seite ordnet die Mappe 05 in die Gesamtuntersuchung ein. Grundlage ist ein
vollständiges Inventar des Drive-Ordners `Colab_Pythia_Results`: **1 757 Einträge,
210 Ordner, 1 547 Dateien, rund 2.11 GiB**, aufgenommen am 13.09.2026.

Untersuchungsgegenstand ist durchgehend **EleutherAI/pythia-1.4b** in den
Checkpoints 97 000 bis 100 000, gemessen mit einer Suszeptibilitätsmatrix nach der
Methodik des Forschungsinstituts Timaeus. Das im Projekt gebräuchliche Wort „Atlas"
ist eine Eigenprägung; in den zugrunde liegenden Arbeiten heißt die Größe
Suszeptibilitäts- oder Antwortmatrix.

---

## Die sieben Abschnitte

**A — FP32- und Mikrogeometrie-Serie (25. bis 26.08.).** Nummerierte Läufe 1090 bis
1110: FP32-Zustandsgitter gegen Mikrogeometrie, Interaction-Conditioning-Readout,
FrozenQ gegen LocalQ, eine Rettungsschleife für die Kurven-Qualitätssicherung. Am
26.08. Konsolidierung in einer Master-Dokumentation, deren Chronologie-Abbildung bis
Lauf 860 zurückreicht.

Verdikt des Laufs 1090, wörtlich: `upstream_FP32_state_map_is_locally_first_order_
smooth_but_downstream_causal_response_remains_nonlinear_at_majority_of_centers`
(5 von 6 Zentren). Der Bericht warnt ausdrücklich, die ULP-Statistiken seien
„descriptive diagnostics of the executed FP32 tensors, not a claim that the original
model parameters themselves form a new discrete physical variable".

**B — Umschwung auf Timaeus (27.08.).** Die Serie 1121 bis 1130 wechselt von
Geometrie-Audits auf gezielte Suszeptibilitäts-Spektroskopie. Neun Observablen, je 48
Sequenzen, Fenster 256 Token, vier Verlust-Token am Ende; Restriktionen auf die
Blöcke 19 und 23, jeweils MLP und Attention, bei den Checkpoints 98 000 und 99 000.

**C — Atlas- und ClusterMap-Kalibrierung (28. bis 31.08.).** Eine Versionskette V2
bis V6 (DirectPCS, PosteriorFidelity, GeneralizedTikhonov, PublishedProtocolMatchGate)
und daran anschließend acht Themenordner von schwacher Introspektion bis
Repräsentationsspektrum.

**D — Kausale Eingriffe (01.09.).** Löschen und Wiederherstellen im Atlas, eine
kausale Brücke auf Dimension 375 und Kopf L23H07, Störungsbehebung nach Eingriff,
ein Zweigwähler, Renormierungs-Ticks.

**E — Zweiter Umschwung (02. bis 03.09.).** Der Timaeus-Präfix verschwindet, die
Ordnernamen werden thematisch: KV-Symmetrie, Rasterfingerabdruck, Architektur-
abhängigkeit, orthogonale Unsicherheitsachse, hierarchischer Regler, nützliches
Rauschen. Gleichzeitig wechselt die Infrastruktur von einer A100 auf
Acht-B200-Orchestrierung.

**F — Die State-60482-Linie (03. bis 06.09.).** Ab hier läuft alles auf ein einziges
Textfenster zu: Tokenisierungs-Routing, Belohnungslandschaft, Phonologie und
Alphabet, bitweise Tokenizer-Primitive (neun Läufe an einem Tag), endliche
Nachfolgezustände, Namensübergang und Nachahmung, Konsensattraktor, Selektionslinie.

**G — Pause und Neustart (09. und 13.09.).** Eine reine Aufsetz-Umgebung am 09.09.,
dann vier Tage Pause, dann die drei Läufe vom 13.09., um die es in Mappe 05 geht.

---

## Was am 13.09. anders ist

Die drei jüngsten Läufe unterscheiden sich in jeder Hinsicht von den vorherigen:

| | Läufe A bis F | Läufe vom 13.09. |
|---|---|---|
| Rechenaufwand | GPU-Cluster, Telemetrie-Sidecars, Stunden | Minuten, keine GPU |
| Größe | Gigabyte | Kilobyte |
| Modell-Vorwärtslauf | ja | **nein** (nur die `ological`-Linie greift ein) |
| Nullmodell | oft implizit | explizit, mit Permutation |
| Ergebnis | überwiegend positiv berichtet | **durchgehend negativ oder null** |

Das ist der eigentliche Befund über den Projektstand: Sobald ein Lauf ein explizites
Nullmodell mitführt und billig genug ist, um an neunzehn Textfenstern statt an einem
gerechnet zu werden, fällt das Ergebnis negativ aus. Die teuren Läufe davor hatten
diese Eigenschaft überwiegend nicht.

---

## Die drei Arbeitslinien am selben Dokument

State 60482 ist ein Nachrichtenschnipsel von 893 Zeichen und 207 Token. Drei Linien
messen darin:

| Linie | Ort | Stand |
|---|---|---|
| McQuarrie | Token 124 bis 127, `ĠMc` `Qu` `ar` `rie` | verlassen; Tokenzahl der Varianten unkontrolliert |
| *Meteorological* | Token 165 bis 168, `ĠMet` `eor` `ological` | Effekt sitzt am Endtoken, nicht am beanspruchten Anker |
| WASD/QERF | über das ganze Fenster verteilt | kein Träger im Tokenraum, Dichte unterdurchschnittlich |

Dass alle drei an einem einzigen Dokument hängen, ist die größte gemeinsame
Schwachstelle. Ein Effekt, der nur an einem Text auftritt, ist von einer Eigenheit
dieses Textes nicht zu unterscheiden — und dieser Text wurde ausgewählt, weil dort
etwas auffiel.

---

## Was das für den nächsten Schritt heißt

1. **Mehr als ein Dokument.** Jede Kennzahl, die an State 60482 gemessen wird,
   braucht eine Verteilung über unabhängige Fenster. Die WASD-Läufe tun das bereits
   mit neunzehn — deshalb sind sie aussagekräftig, auch wenn sie negativ ausfallen.
2. **Unsicherheit an jede Zahl.** Ein Bootstrap über die 48 Sequenzen der
   Spektroskopie und eine Etikettenpermutation kosten Minuten und würden entscheiden,
   ob ein Zuwachs von +0.0467 zwischen zwei Checkpoints etwas bedeutet.
3. **Anker, die nicht tautologisch sind.** Ein Transfer am letzten Token ändert die
   nächste Vorhersage per Konstruktion. Aussagekräftig ist ein Anker davor.
4. **Die Vorfrage vor dem GPU-Lauf.** Ob eine Zeichenfolge das Netz überhaupt als
   Einheit erreicht, klärt `tokenizer_sonde.py` in Sekunden. Für `WASD` lautete die
   Antwort nein — das hätte drei Läufe erspart.
5. **Die eigene Methode nachlesen.** Der Schätzer der Suszeptibilitäten liefert
   laut seiner eigenen Beschreibung ein komponentenabhängiges Vielfaches, nicht die
   Größe selbst. Rohwerte zwischen Schichten zu vergleichen — „Schicht 14 reagiert
   stärker" — ist deshalb nicht interpretierbar. Einzelheiten in
   [`METHODEN_BRIEFING.md`](METHODEN_BRIEFING.md).

---

## Der nächste Lauf ist gebaut

Aus Punkt 4 folgt der Nachfolgeversuch, und der ist inzwischen vollständig
ausformuliert: das Design in [`experiment_traeger.py`](experiment_traeger.py), der
Messlauf in [`wasd_traeger_messlauf_colab.ipynb`](wasd_traeger_messlauf_colab.ipynb).

Er stellt die Frage dort, wo sie eine Vorhersage macht — an Sätzen, in denen `WASD`
als Tastenkürzel steht. Der Tokenizer liefert dafür ein ungewöhnlich sauberes
Design: `" WASD"` zerfällt in genau zwei Stücke, also unterscheidet sich jede
`WAS?`-Kontrolle vom Ziel in **genau einem Token**. Dazu drei Kontrollen, die die
Buchstabenfamilie nicht leistet — `FORD` mit demselben Endtoken, `NOTA` mit fast
gleicher Korpushäufigkeit, `ESDF` als getrennt berichtete Alternativbelegung.

| | |
|---|---|
| Varianten | 30 gebaut, 25 in der Messung, 5 vom Tokenizer verworfen |
| Messpunkte | 600 (25 × 24 Satzschablonen) |
| Transferzeilen | 2 880 (5 Paare × 24 Schablonen × 24 Blöcke), je zwei Richtungen |
| Vorwärtsläufe | 18 024 insgesamt — bei 60 ms je Lauf rund 18 Minuten |
| Entscheidungsregel | vorregistriert, verlangt Beobachtungs- **und** Kausalteil |

Das Notebook ist gegen ein winziges Zufallsmodell derselben Architektur
durchgelaufen, bevor es eine GPU sieht. Dieser Probelauf hat zwei Fehler gefunden:
zwei Messwörter, die keine Einzeltoken sind, und ein Qualitätstor, das auf `== 0.0`
stand, obwohl es eine Rekonstruktion prüft und nicht eine Identität. Beides ist
korrigiert; Einzelheiten in Abschnitt 5 von [`LIES_MICH.md`](LIES_MICH.md).

Der Punkt gilt über diesen Lauf hinaus und ist derselbe wie Punkt 4 oben: Was vor dem
GPU-Lauf in Sekunden zu klären ist, sollte vor dem GPU-Lauf geklärt werden.

---

Die ausführliche Begründung steht in [`LIES_MICH.md`](LIES_MICH.md), die
Literatur- und Methodenlage in [`METHODEN_BRIEFING.md`](METHODEN_BRIEFING.md).
