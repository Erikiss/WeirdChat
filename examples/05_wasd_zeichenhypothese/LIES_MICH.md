# Mappe 05 — Die WASD-Zeichenhypothese, nachgerechnet

Diese Mappe prüft **eine** Frage: Spielen die Tastenbuchstaben `W A S D` (und die
Nachbartasten `Q E R F`) im Pythia-Modell eine Sonderrolle — lösen sie in den
mittleren Schichten Effekte wie Sprachwechsel oder Fehlerkorrektur aus?

Die Antwort auf dem heutigen Stand der Messungen ist **nein**, und zwar deutlicher,
als die Läufe selbst berichten. Was hier steht, ist kein Gegenargument gegen die
Forschungslinie, sondern ihre Bilanz: drei Läufe vom 13. September 2026, exakt
nachgerechnet, plus zwei Nullmodelle, die in den Läufen fehlen und die Richtung des
Ergebnisses umdrehen, und eine Vorfrage, die vor jedem GPU-Lauf zu klären gewesen
wäre: ob die Zeichenfolge das Netz überhaupt als Einheit erreicht.

Der Gegenstand ist das Textfenster **State 60482** — 893 Zeichen, 698 Buchstaben,
207 Token im GPT-NeoX-Tokenizer, ein Nachrichtenschnipsel aus dem Pile über
Filmfestspiele, `Mission: Impossible – Fallout` und einen indischen Wetterbericht.
An den Token 124 bis 127 steht `ĠMc`, `Qu`, `ar`, `rie` — der Ort des früheren
**McQuarrie-Phänomens**, um das sich die vorherige Arbeitslinie drehte.

---

## 1. Was die drei Läufe gemessen haben

Alle drei sind **Text- und Tokenizer-Statistik ohne Vorwärtslauf**. Kein Gewicht
des Modells wird angefasst; gemessen wird, wie Buchstaben im Text verteilt sind.
Das ist eine legitime Vorstufe, aber es ist wichtig, sie nicht als Modellaussage zu
lesen — die Läufe schreiben das selbst in ihre Berichte.

| Lauf | Frage | Verdikt des Laufs |
|---|---|---|
| `20260913_164737`, `20260913_170159` | Ist die WASDQERF-Dichte in State 60482 ungewöhnlich hoch? | `NOT_UNUSUALLY_HIGH` |
| `20260913_171744` | Folgen die Zielbuchstaben der Tastaturreihenfolge, und sind ihre Token-Positionen periodisch? | `transition_position_cyclicity_supported = False` |

Die beiden Dichteläufe sind bis auf den Zeitstempel identisch.

### Dichte

- beobachtete Dichte unter allen alphabetischen Zeichen: **0.386819**
- Rang unter 19 Textfenstern: **14 von 19**, Perzentil 31.58
- Verhältnis zum Mittel der anderen achtzehn: **0.9705**
- z gegen die anderen achtzehn Fenster: **−0.2136**
- z gegen zufällig gezogene Achterbuchstabenmengen: **+1.0840**, einseitiges
  empirisches p = 0.1467

### Reihenfolge und Periodizität

- kanonischer Nachfolgeranteil W→A→S→D→Q→E→R→F→W: **32 von 269 Übergängen =
  0.118959**
- Perzentil des kanonischen Zyklus unter allen 5 040 gerichteten Zyklen: **47.798**,
  p = 0.570323
- beste Token-Positions-Periode: **32**, NMI 0.208927, Permutations-p = **0.911089**
- von sieben Entscheidungsflaggen ist **eine** wahr

---

## 2. Was in den Läufen fehlt

### 2.1 Das Nullmodell ist verzerrt — und zwar genau in die Richtung des erhofften Ergebnisses

Der einzige positive Wert im ganzen Dichtelauf ist das `z = +1.084` gegen zufällig
gezogene Achterbuchstabenmengen. Dieses Nullmodell fragt aber nicht, ob *dieser
Text* auffällig ist, sondern ob *diese Buchstaben* häufig sind.

`WASDQERF` enthält E, A, R und S — vier der sechs häufigsten Buchstaben des
Englischen. Die erwartete Dichte dieser Achtermenge in gewöhnlichem englischem Text
beträgt **0.4134**; eine zufällig gezogene Achtermenge kommt im Mittel auf 0.3077.
Ein positives z ist damit die Voreinstellung, nicht der Befund.

Drei Gegenproben, alle in `tests/` festgehalten:

**Erstens, über die Geschwistertexte.** Dieselbe Kennzahl, berechnet für alle 19
Fenster desselben Laufs:

| Kennzahl | Wert |
|---|---|
| Fenster mit z > 0 | 18 von 19 |
| Mittelwert z über alle 19 | +1.2064 |
| Median z | +1.3785 |
| Rang von State 60482 | 14 von 19 |
| kleinstes p über alle 19 | 0.0239 (Bonferroni-Schwelle wäre 0.0026) |

Das Nullmodell meldet also fast jedem gewöhnlichen englischen Nachrichtentext einen
Überschuss. State 60482 liegt dabei **unter** dem Durchschnitt seiner eigenen
Vergleichsgruppe.

**Zweitens, frequenzangepasst.** Vergleicht man nicht gegen beliebige Achtermengen,
sondern nur gegen solche mit derselben *erwarteten* Dichte, dreht sich das
Vorzeichen (`zeichensatz_statistik.py`, Toleranzband 0.01, 64 787 Referenzmengen):

| Nullmodell | z | einseitiges p |
|---|---|---|
| unbedingt, alle 1 562 275 Achtermengen (so im Lauf) | +1.0518 | 0.1542 |
| frequenzangepasst | **−0.9367** | 0.8264 |
| Quotient beobachtet zu erwartet | **−1.0454** | 0.8521 |

Nach dem Quotientenmodell steht `WASDQERF` auf Rang **1 331 231 von 1 562 275**.
Die beobachtete Dichte liegt bei 93.6 Prozent der englischen Erwartung. Das
Textfenster ist für diese Buchstabenmenge nicht angereichert, sondern leicht
abgereichert.

Die Buchstaben einzeln betrachtet zerfällt die Gruppe ohnehin: W, S und D liegen
über dem Gruppenmittel, **A fällt auf 64 Prozent** (0.0544 gegen 0.0853) und F auf
66 Prozent. Und die Variante, die `WASD` tatsächlich als Tastenkürzel lesen würde —
Großschreibung — kommt im Zieltext **halb so oft** vor wie im Gruppenmittel
(Verhältnis 0.5088).

**Drittens, gegen das Trainingskorpus selbst.** Die beiden Nullmodelle der Läufe
vergleichen mit achtzehn Geschwisterfenstern und mit Buchstabenmengen. Die
naheliegendste Bezugsgröße fehlt: gewöhnliche Dokumente aus dem Korpus, auf dem
Pythia trainiert wurde. `pile_nullmodell.py` holt sie nach — 2 863 Dokumente aus
The Pile, 20 899 761 Zeichen:

| Kennzahl | Wert |
|---|---|
| mittlere WASDQERF-Dichte über Dokumente | 0.408650 |
| Standardabweichung | 0.029075 |
| Median | 0.408949 |
| State 60482 | 0.386819 |
| **z** | **−0.7508** |
| **Perzentil** | **18.83** |

Gegen das echte Trainingskorpus liegt das Zielfenster im unteren Fünftel. Der
Korpusmittelwert 0.4087 bestätigt nebenbei die Standardtabelle (0.4134) als
brauchbare Näherung — die Wahl der Häufigkeitsquelle ändert am Ergebnis nichts.

### 2.2 Die Zeichenfolge erreicht das Netz nicht als Einheit

Vor jedem GPU-Lauf lässt sich eine Vorfrage klären: **Wie kommt `WASD` überhaupt im
Modell an?** Ein Transformer sieht keine Buchstaben, sondern Token-IDs.

Der GPT-NeoX-Tokenizer von Pythia (50 254 Vokabeleinträge, 50 277 inklusive
Sondertoken) enthält:

```
'WASD'      -> nicht im Vokabular      'WASD'  zerfällt in  ['W', 'AS', 'D']
' WASD'     -> nicht im Vokabular      ' WASD' zerfällt in  ['ĠWAS', 'D']
'wasd'      -> nicht im Vokabular      'wasd'  zerfällt in  ['was', 'd']
'QERF'      -> nicht im Vokabular      'QERF'  zerfällt in  ['Q', 'ER', 'F']
```

Kein einziger Vokabeleintrag enthält `wasd` als Teilkette, in keiner Schreibweise.
Da BPE-Merges nach Häufigkeit entstehen, ist das ein Hinweis auf die Basisrate — und
die lässt sich exakt nachzählen. Der Volltextindex **infini-gram** liefert über den
gesamten Pile-Trainingssatz (`v4_piletrain_llama`, rund 380 Milliarden Token):

| Zeichenfolge | Vorkommen im ganzen Pile |
|---|---|
| ` the` | 9 229 350 572 |
| ` keyboard` | 2 791 750 |
| ` Meteorological` | 115 452 |
| ` McQuarrie` | 17 994 |
| ` strafe` | 17 853 |
| **` WASD`** | **13 870** |
| ` WASD keys` | 2 776 |
| ` use WASD` | 574 |
| ` QERF` | 10 |

Damit ist die Lage präziser als „selten": `WASD` kommt vor, rund zweihundertmal
seltener als `keyboard`, und **jedes fünfte Vorkommen** wird direkt von `keys`
gefolgt. Das Modell hat die Wendung also gesehen. Für einen eigenen BPE-Merge ist
13 870 in 380 Milliarden Token trotzdem viel zu wenig — die relative Häufigkeit liegt
bei etwa 3.6 · 10⁻⁸.

Entscheidend bleibt der andere Punkt: **im Textfenster State 60482 steht `WASD`
kein einziges Mal.** Was dort gezählt wurde, sind acht über den ganzen Text
verstreute Buchstaben, nicht das Tastenkürzel.

> **Zwei Warnungen in eigener Sache.** Erstens: Die erste Fassung der Zählung suchte
> `wasd` als Teilkette in einer Stichprobe und meldete 22 Treffer — sämtlich der
> englische Ortsname *Wasdale* und die Domain *wasdaleweb.com* aus einem einzigen
> Reiseführer-Dokument. Zweitens: Nach der Korrektur auf Wortgrenzen meldete dieselbe
> Stichprobe null Treffer, woraus hier zwischenzeitlich „kommt im Trainingskorpus
> überhaupt nicht vor" wurde. Auch das war falsch. Die Stichprobe umfasst 20.9
> Millionen Zeichen, also etwa ein Vierzigtausendstel des Pile; bei 13 870 Vorkommen
> im Ganzen sind dort rund **0.35 Treffer zu erwarten**. Null zu finden war das
> Erwartbare, nicht der Befund. Erst der Volltextindex beantwortet die Frage.
>
> Beide Fehler sind derselbe, den diese Mappe an den Läufen kritisiert: eine Zahl für
> eine Antwort halten, ohne zu prüfen, was sie zählen kann. Sie stehen hier, weil ein
> Dossier, das Nullmodelle einfordert, seine eigenen aushalten muss.

Umgekehrt bestehen **931 Vokabeleinträge (1.85 Prozent)** ausschließlich aus
WASDQERF-Buchstaben — darunter die frühesten und häufigsten Merges des Englischen:
`re`, `er`, `as`, `ed`, `es`, `ar`, `was`, `are`. Die Zielmenge ist nicht exotisch,
sie ist das Rückgrat der englischen Orthographie.

Damit hat die Hypothese in ihrer jetzigen Form kein Trägersignal: Es gibt keine
Embedding-Zeile, keine Tokenposition und keine Aktivierung, an der ein Eingriff
ansetzen könnte, ohne gleichzeitig halb Englisch mitzunehmen.

---

## 3. Was der Zykluslauf zusätzlich zeigt

Der Lauf vom 17:17 Uhr ist methodisch der sauberste der drei: zwei unabhängige
Familien von Kennzahlen, eine Permutationskorrektur innerhalb des Periodenscans,
und eine Entscheidungsregel, die beide Familien verlangt. Er kommt korrekt zu
`False`. Drei Punkte verdienen es, festgehalten zu werden:

**Der kanonische Anteil liegt unter dem Zufallsniveau.** Acht Buchstaben, acht
kanonische Nachfolger — Zufall wäre 1/8 = 0.125. Gemessen: 0.118959. Über alle 19
Fenster streut der Wert eng um das Zufallsniveau.

**Der Zyklus wird von einem einzigen Paar getragen.** Von den 32 kanonischen
Treffern entfallen **15 auf E→R**. Zwei Glieder der Kette kommen überhaupt nicht
vor: D→Q = 0 und Q→E = 0. E→R ist keine Tastaturbewegung, sondern gewöhnliche
englische Orthographie; E allein hat 81 der 269 Ausgänge, Q genau einen.

**Die einzige wahre Flagge ist ein Gleichstand.** `top3_longest_cycle8_run` ist
wahr, weil State 60482 den Wert 3 erreicht — den **dreizehn von neunzehn** Fenstern
ebenfalls erreichen. Dasselbe bei `best_phase_match_L4`: sechzehn Fenster liegen auf
0.75. Rang 3 bedeutet hier geteilter Rang 3, nicht Vorsprung.

Ein methodischer Hinweis, der über diesen Lauf hinausreicht: Die beste Periode liegt
in **zehn von neunzehn** Fällen bei genau 32 — dem oberen Rand des Suchbereichs. Der
NMI wächst monoton mit der Periodenlänge, weil mehr Klassen mehr geteilte Information
erlauben. Die Permutationskorrektur fängt das ab; der rohe NMI-Wert täte es nicht.

---

## 4. Was die publizierte Methode selbst sagt

Die Recherche in [`METHODEN_BRIEFING.md`](METHODEN_BRIEFING.md) hat einen Punkt
zutage gefördert, der über die WASD-Frage hinausgeht und die gesamte Messlinie
betrifft. Er steht in der Methodenbeschreibung des Verfahrens, das dieses Projekt
benutzt.

**Der Schätzer liefert nicht die Suszeptibilität, sondern ein Vielfaches davon.**
Die Methodenseite zur Skalierung der Suszeptibilitäten schreibt es wörtlich — an der
Quelle nachgeprüft:

> „…is actually an estimator for the renormalized susceptibility
> `Z_full/Z_C · χ^C_xy` **not for the population susceptibility itself**. This
> `C`-dependent (but `xy`-independent) prefactor is absorbed by the column z-scoring
> step…"

Daraus folgt unmittelbar: **Ein Vergleich von Rohwerten zwischen Komponenten — also
zwischen Schichten oder Köpfen — ist nicht interpretierbar.** Ein Satz wie „Schicht
14 und 20 reagieren stärker" setzt genau das voraus, was der Schätzer nicht liefert.
Erst nach der spaltenweisen Standardisierung sind Komponenten vergleichbar, und dann
ist die Aussage eine über das *Profil*, nicht über die Höhe.

Die Einschränkung gilt genau in eine Richtung: Der Vorfaktor ist ausdrücklich
*`xy`-unabhängig*. **Innerhalb einer Komponente** dürfen Tokens also sehr wohl auf
Rohwerten verglichen werden — dort kürzt sich der unbekannte Faktor heraus. Wer die
Frage stellt „welches Token reagiert in Schicht 14 am stärksten", ist auf sicherem
Boden; wer fragt „welche Schicht reagiert am stärksten", nicht.

Zwei weitere Punkte derselben Art:

- Die Zeilenzentrierung „deletes exactly the **1** direction", also den uniformen
  Modus. Eine Aussage der Form „X ist global suszeptibler" ist damit **per
  Konstruktion** aus den standardisierten Daten entfernt — sie kann dort weder
  bestätigt noch widerlegt werden. (Die neuere Fassung ersetzt diesen Schritt durch
  PCA-Whitening, mit demselben Effekt auf die uniforme Richtung.)
- Die Zahl der gefundenen Cluster hängt stark an der Auswertungskette, nicht am
  Modell; und die Autoren berichten selbst, dass schon eine **Gauß-Grundlinie ohne
  jeden Beitrag der Verlustlandschaft interpretierbare Cluster liefert**. Ein
  interpretierbarer Cluster ist also für sich genommen kein Befund.

Dazu eine Nomenklatur-Notiz: Das Wort „Atlas" ist in den zugrunde liegenden Arbeiten
kein Fachbegriff; dort heißt es Suszeptibilitätsmatrix, Antwortmatrix oder Cluster
Map. Als Eigenprägung ist das in Ordnung, sollte aber als solche gekennzeichnet sein,
damit ein externer Leser nicht nach einer Methode sucht, die unter diesem Namen nicht
existiert.

Das Briefing kennzeichnet jede Aussage danach, ob sie an der Primärquelle geprüft
wurde. Vor dem Zitieren lohnt ein Blick auf die Kennzeichnung.

## 5. Was jetzt zu tun wäre

Die Hypothese ist nicht widerlegt — sie wurde bisher nur an einer Stelle geprüft, an
der sie gar keine Vorhersage macht. Eine Buchstabendichte im Text sagt nichts
darüber, ob ein Netz diese Zeichen benutzt. Damit die Frage prüfbar wird, muss sie
vor dem nächsten GPU-Lauf drei Dinge festlegen.

**Erstens: einen Träger.** Entweder man testet Text, in dem `WASD` wirklich als
Tastenkürzel vorkommt (Spielanleitungen, Foren, Konfigurationsdateien) — dann gibt
es eine Tokenfolge `['ĠWAS', 'D']`, an der man ansetzen kann. Oder man testet die
schwächere, aber immer noch interessante Frage, ob Pythia Zeichenidentität über
Tokengrenzen hinweg repräsentiert. Beides ist messbar; die Mischung aus beidem
nicht.

Für den ersten Weg liefert der Tokenizer ein ungewöhnlich sauberes Design, das in
`experiment_traeger.py` ausformuliert ist. Weil `" WASD"` in genau zwei Stücke
zerfällt, gibt es eine **minimale Kontrollfamilie**: alle Folgen `WAS?` mit einem
anderen Großbuchstaben am Ende zerfallen ebenso in `['ĠWAS', '?']`. Erstes Token
identisch, zweites Token ein einzelner Großbuchstabe wie `D`. Zwischen Ziel und
Kontrolle unterscheidet sich damit **genau ein Token** — und zwar das, dem die
Hypothese ihre Bedeutung zuschreibt. Fünf Kandidaten fallen heraus, weil `ASE`,
`ASH`, `ASK`, `ASS` und `AST` eigene Vokabeleinträge sind; zwanzig bleiben. Dazu
kommt `ASDW` → `['ĠASD', 'W']` als Reihenfolgekontrolle: dieselben Buchstaben,
dieselbe Struktur, falsche Ordnung.

Die Buchstabenfamilie allein lässt aber drei Lücken offen, die drei weitere
Kontrollen schließen. Alle drei sind am Tokenizer abgelesen, nicht angenommen:

| Kontrolle | Zerlegung | Korpus | Was sie ausschließt |
|---|---|---|---|
| `FORD` | `['ĠFOR', 'D']` = 6651, 37 | 56 114 | **dasselbe Endtoken** wie das Ziel. Trägt allein der Buchstabe `D` den Effekt, muss `FORD` ihn ebenso zeigen |
| `NOTA` | `['ĠNOT', 'A']` = 5803, 34 | 14 286 | fast gleiche Korpushäufigkeit wie `WASD` (13 870). Die eigentliche Nulllinie |
| `ESDF` | `['ĠE', 'SD', 'F']` | 1 124 | eine echte alternative Belegung — aber **drei** Token und zwölfmal seltener |

`FORD` ist die schärfste der drei. Die `WAS?`-Familie hält das erste Token fest und
variiert das zweite; `FORD` macht es genau umgekehrt. Ein Effekt, der bei `WASD`
auftritt und bei `FORD` nicht, kann nicht am Endtoken hängen — und einer, der bei
beiden auftritt, hängt nicht an `WASD`.

`ESDF` geht **nicht** in die Entscheidungsregel ein und wird getrennt berichtet. Bei
drei Token gibt es keine Position, die der des Ziels entspräche; ihr Wert wäre mit
dem des Ziels nicht vergleichbar. Sie steht trotzdem im Design, weil sie die einzige
tatsächlich gebräuchliche Alternativbelegung ist, die der Tokenizer hergibt: `IJKL`
und `ZQSD` zerfallen noch ungünstiger. Dass keine alternative Tastenbelegung
strukturgleich ist, ist selbst ein Befund — die Frage „liegt es an den Tasten oder an
diesen vier Zeichen?" lässt der Tokenizer nicht sauber stellen.

Ein Nebenbefund schließt die Kleinschreibung aus: `" wasd"` zerfällt in
`['Ġwas', 'd']`, und `Ġwas` ist das englische Wort *was* (Merge-ID 369, einer der
frühesten Merges überhaupt). Jeder Effekt an der kleingeschriebenen Form wäre mit
der Vergangenheitsform von *to be* vermengt.

**Zweitens: eine Vorhersage, die schiefgehen kann.** "Interessante Erscheinungen"
ist keine. Die Vorregistrierung in `experiment_traeger.py` verlangt beides: einen
Beobachtungsteil (das Zieltoken hebt das Bewegungsvokabular um mindestens 0.5 nats
gegenüber der **besten** Kontrolle, nicht gegenüber dem Kontrollmittel) und einen
kausalen Teil (ein Eingriff an der Position des letzten Tokens stellt mindestens die
Hälfte des Effekts wieder her, an mindestens zwei benachbarten Schichten, während
derselbe Eingriff mit einer Kontrollquelle höchstens ein Fünftel überträgt). Ein
einzelner positiver Teil zählt nicht. Die Regel ist gegen Ausreißer gebaut: eine
einzige Kontrolle, die fast gleichauf liegt, kippt den Befund.

**Drittens: abgestimmte Kontrollen.** Die Vergleichsbedingung darf sich nicht in der
Buchstabenhäufigkeit unterscheiden, sonst misst man wieder die Häufigkeit. Die
`WAS?`-Familie erfüllt das von selbst, weil alle Varianten dasselbe erste Token
teilen. Der Trockenlauf prüft das vor jedem GPU-Lauf nach und verwirft, was nicht
passt:

```bash
python examples/05_wasd_zeichenhypothese/experiment_traeger.py \
  --tokenizer tokenizer.json --trockenlauf
```

Er meldet 20 strukturgleiche Kontrollen, fünf verworfene (`WASE`, `WASH`, `WASK`,
`WASS`, `WAST`) und 600 Messpunkte: 25 in die Messung eingehende Varianten mal 24
Satzschablonen.

**Viertens: Unsicherheit an den Kennzahlen.** Das gilt über die WASD-Frage hinaus.
Der Bericht der Timaeus-Spektroskopie (`1130_..._EVALUATION_REPORT.md`) nennt für
Schritt 98 000 eine Paar-Kosinusähnlichkeit von 0.9198 und für Schritt 99 000 eine
von 0.9665, daraus einen Zuwachs von +0.0467, und schließt auf eine
„EventB-spezifische Suszeptibilitätssignatur, schon bei 98k vorhanden". Alle diese
Zahlen sind Punktschätzer aus 48 Sequenzen, ohne Streuungsangabe und ohne
Permutationsnull. Ob +0.0467 zwischen zwei benachbarten Checkpoints viel oder nichts
ist, lässt sich daraus nicht entscheiden — und der ausgewiesene „pair-minus-control
gap" von 1.7884 ist im Wesentlichen die Summe zweier Kosinuswerte mit
entgegengesetztem Vorzeichen, also eine Größe, deren Skala ohne Nullverteilung keine
Bedeutung hat. Ein Bootstrap über die 48 Sequenzen und eine Etikettenpermutation
kosten wenige Minuten Rechenzeit und würden diese Berichte tragfähig machen.

Dazu die drei Punkte, die für jeden Lauf dieser Linie gelten:

- **Mehrfachvergleiche.** In den bisherigen Läufen werden 19 Fenster × 12 Kennzahlen
  × 7 Flaggen geprüft, ohne Korrektur. Bei 19 Tests liegt die Bonferroni-Schwelle bei
  0.0026; das kleinste beobachtete p ist 0.0239.
- **Die Textstelle war nicht vorregistriert.** State 60482 wurde ausgewählt, weil
  dort früher etwas auffiel. Jede Statistik, die an derselben Stelle weiterrechnet,
  erbt diese Auswahl. Eine Replikation braucht neue Fenster.
- **Numerik vor Interpretation.** Bevor ein Unterschied in den mittleren Schichten
  als Modellverhalten gilt, muss ausgeschlossen sein, dass er von Batchgröße,
  Reduktionsreihenfolge oder Kernelwahl stammt. Die Läufe tun hier bereits das
  Richtige: Batchgröße eins, FP32, ein Selbstpatch-Test mit
  `max_self_patch_logit_error = 0.0`.

**Fünftens: der Messlauf, vorher trockengelaufen.** Das Design allein misst nichts.
Der Messlauf steht als Notebook in
[`wasd_traeger_messlauf_colab.ipynb`](wasd_traeger_messlauf_colab.ipynb) — ein
Colab-Notebook, das von der Modellprüfung bis zum Schlussbericht durchläuft und die
vorregistrierte Regel am Ende selbst anwendet. Es lädt `pythia-1.4b` bei
`step98000` in FP32 mit Batchgröße eins, misst 720 Grundlinien, prüft zwei
Qualitätstore und führt 2 880 Transferzeilen über alle 24 Blöcke aus — je Zeile in
beide Richtungen, Zielaktivierung in den Kontrollkontext und umgekehrt. Zusammen
18 024 Vorwärtsläufe; bei 60 ms je Lauf rund 18 Minuten reine Rechenzeit.

Bevor es auf eine GPU geht, ist es gegen ein winziges Zufallsmodell mit derselben
Architektur durchgelaufen — vier Blöcke, 32 Dimensionen, dieselben Codepfade. Dieser
Probelauf hat zwei Fehler gefunden, die sonst erst die GPU gefunden hätte:

1. **Zwei Messwörter waren keine Einzeltoken.** `" strafe"` zerfällt in
   `['Ġstra', 'fe']`, `" senate"` in `['Ġsen', 'ate']`. Die Sonde hätte dann nicht
   die Wahrscheinlichkeit von *strafe* gemessen, sondern die von *stra* — einem
   Stück, das es mit *strategy*, *strange* und *straight* teilt. Beide sind vor jeder
   Messung durch `" sprint"` und `" parliament"` ersetzt; das Kriterium stand
   vorher fest, die Ersetzung folgt ihm. Die verifizierten Token-IDs stehen jetzt als
   Tabelle im Code, und der Lauf bricht ab, wenn der geladene Tokenizer sie nicht
   reproduziert.

2. **Ein Qualitätstor war prinzipiell falsch gesetzt.** Das Residualtor prüft, ob
   `resid_out = resid_in + attn + mlp` gilt — die Identität, die
   `use_parallel_residual=True` behauptet. Es stand auf `== 0.0`. Das ist aber keine
   Identität, sondern eine **Rekonstruktion**: die Summe läuft in anderer Reihenfolge
   als im Block selbst, und FP32 rundet dabei anders. Der Probelauf zeigte 7.5e-09.
   Das Tor hätte einen völlig korrekten Lauf abgebrochen. Es misst jetzt relativ zum
   Betrag der Aktivierung, mit Schranke 1e-4 — eng genug, dass eine strukturell
   falsche Zerlegung durchfällt, denn die läge um Größenordnungen darüber.

Der Unterschied zum **Selbstpatch-Tor** ist genau der Punkt: dort wird eine
Aktivierung durch sich selbst ersetzt. Das ist eine echte Identität, derselbe Tensor
an derselben Stelle, und muss Bit für Bit dasselbe ergeben. Der Probelauf bestätigt
es: exakt `0.0` über alle Prüfungen. Ein Tor auf null ist dort richtig und hier
falsch, und die beiden Fälle sehen im Code fast gleich aus.

Auf dem untrainierten Probemodell fällt das Urteil erwartungsgemäß negativ aus
(Vorsprung −0.0415 nats bei Schwelle 0.5, längster Kausallauf 0). Das ist die
Kalibrierungsprobe: die Regel spricht nicht auf Rauschen an.

**Was der Probelauf nicht finden konnte.** Der erste echte GPU-Lauf lief durch und
war trotzdem verloren: Das Notebook schrieb seine neun Ausgabedateien nach
`./wasd_traeger_lauf` — in das Dateisystem der Colab-Laufzeit, nicht nach Drive. Die
in Drive gespeicherte Notebook-Kopie enthielt keine Zellausgaben. Damit war von
18 024 Vorwärtsläufen nichts erhalten.

Ein Probelauf gegen ein kleines Modell prüft, ob die Rechnung stimmt. Er prüft nicht,
ob das Ergebnis irgendwo landet, wo es den Prozess überlebt — dort lief er im selben
Verzeichnis wie das Skript und fand alles vor. Das Notebook sichert deshalb jetzt in
einem Schlussabschnitt nach Drive, mit Zeitstempel, damit ein Lauf den vorigen nicht
überschreibt; außerhalb von Colab läuft der Abschnitt wirkungslos durch.

---

## 6. Rückblick: ein unkontrollierter Faktor in der McQuarrie-Linie

Die vorherige Arbeitslinie verglich Oberflächenvarianten des Namens — Großschreibung,
Homoglyphen, Tippfehler — und maß je Variante einen Nutzen aus der wahren Ziel-NLL
(Lauf `20260906_120822_ba0e3d`). Die Varianten unterscheiden sich aber nicht nur im
Zeichenbild, sondern in der **Tokenisierung**:

| Variante | Tokens | Zerlegung |
|---|---|---|
| `McQuarrie` | 4 | `ĠMc` `Qu` `ar` `rie` |
| `MCQUARRIE` | 5 | `ĠMC` `QU` `AR` `RI` `E` |
| `McOuarrie` | 4 | `ĠMc` `O` `uar` `rie` |
| `Mc0uarrie` | 4 | `ĠMc` `0` `uar` `rie` |
| `McQuarr1e` | 5 | `ĠMc` `Qu` `arr` `1` `e` |

Daraus folgen zwei Dinge, die im Bericht nicht auftauchen — und dazu eine dritte
Auffälligkeit, die keine Folgerung ist, sondern eine Bitte um Nachsehen:

**Der analysierte Übergang existiert nicht überall.** Die Auswertung dreht sich um
das Residuum des Übergangs `Qu → ar`. In den Varianten mit O, Null oder X gibt es
weder `Qu` noch `ar`; dort steht `O` gefolgt von `uar`. Über die Varianten hinweg
wird also nicht dieselbe Größe verglichen.

**Die Tokenzahl schwankt.** Ein Nutzenmaß über eine unterschiedliche Zahl
vorhergesagter Tokens ist zwischen Varianten nicht ohne Weiteres vergleichbar. Der
Verdacht lässt sich an den vorliegenden Zahlen prüfen — und **bestätigt sich nicht**:
Im exakten Permutationstest über alle 210 Aufteilungen schneiden die
Fünf-Token-Varianten bei 98k und 99k zwar besser ab, aber nicht signifikant
(p = 0.181 und p = 0.129), und bei 97k dreht das Vorzeichen. Zehn Varianten sind zu
wenig, um den Faktor auszuschließen oder nachzuweisen. Er bleibt unkontrolliert; ein
Nachfolgelauf sollte tokenzahlgleiche Varianten gegeneinander stellen.

**Eine Zahl, die zweimal dasteht.** In der Residuentabelle desselben Berichts steht
die Spalte `residual_rms_percentile_under_test_names` mit vier Werten:

| Checkpoint | Schicht | `residual_rms` | Perzentil |
|---|---|---|---|
| 98 000 | 14 | 1.247435 | **55.675333** |
| 98 000 | 20 | 1.775950 | 45.466075 |
| 99 000 | 14 | 1.239061 | 55.421687 |
| 99 000 | 20 | 1.824206 | **55.675333** |

Zwei Zeilen mit deutlich verschiedenem Residuum — 1.247435 gegen 1.824206, also fast
fünfzig Prozent Unterschied — tragen ein auf sechs Nachkommastellen identisches
Perzentil. Das kann stimmen, wenn die beiden Bezugsverteilungen entsprechend
auseinanderliegen. Wahrscheinlicher ist ein versehentlich übernommener Wert. Das ist
kein Befund, sondern ein Punkt zum Nachsehen im Auswertungscode.

## 7. Warum die Vorgängerlinie stecken blieb

Zwischen dem McQuarrie-Phänomen und dem WASD-Umschwung liegen zwei Läufe, die
denselben Ort mit Zeichen-Hypothesen angingen. Beide sind negativ ausgefallen, und
ihre Zahlen erklären, warum die Linie nicht weiterkam.

**Phonologie, Alphabet, QR-Code (05.09.).** Der Lauf fragt, ob der Übergang `Qu`→`ar`
eine Aussprache-Umkodierung trägt: klingt es wie *kwor* oder wie *qar*? Von zehn
Hypothesenflaggen ist **eine** wahr. Die gemessenen Verschiebungen liegen zwischen
0.0003 und 0.038 — Größenordnungen unterhalb dessen, was die Grundlinienmarge von
−0.70 bis −0.75 ausmacht. Auffällig nebenbei: die Qualität aller sieben
Konzeptachsen wird mit exakt 1.0 angegeben, für *Phonologie* ebenso wie für
*Froschquaken*. Eine Gütezahl, die für jede Achse denselben Bestwert liefert,
unterscheidet nichts.

**Bitweise Tokenizer-Primitive (06.09.).** Der Lauf sucht eine Zerlegung der
Übergänge in Byteoperationen — Verschiebungen, Rotationen, ASCII-Nachfolger — und
wählt je Checkpoint und Schicht die besten vier. Das Ergebnis, in seinen eigenen
Zahlen:

| | Schicht 14 | Schicht 20 | Schicht 23 |
|---|---|---|---|
| Projektionsenergie 98k | 3.27 % | 1.41 % | 2.98 % |
| Projektionsenergie 99k | 2.73 % | 1.29 % | 2.07 % |
| Residuum (Anteil) | 98.4–98.6 % | 99.3–99.4 % | 98.5–99.0 % |
| Auswahl stabil 98k→99k | 2 von 4 | **1 von 4** | 2 von 4 |

Zwei Dinge stehen damit fest. Erstens erklären die Primitive fast nichts: über 98
Prozent bleiben Residuum. Zweitens hält die Auswahl zwischen zwei **benachbarten**
Checkpoints nicht — von zwölf Plätzen überleben fünf, und einer davon,
`drop_last_byte`, steht in allen sechs Auswahlen und unterscheidet deshalb nichts.
Zehn der vierzehn überhaupt gewählten Primitive kommen genau einmal vor.

Das ist das Muster einer Auswahl, die Rauschen anpasst. Der Lauf zieht daraus selbst
das richtige Fazit (`strong_bitwise_reduction_supported = false`) — und ausgerechnet
Schicht 20, um die sich die frühere Linie drehte, ist an beiden Checkpoints die
schwächste der drei.

Die Linie kam also nicht deshalb nicht weiter, weil die Messungen zu grob waren,
sondern weil die gesuchte Struktur zwischen zwei Checkpoints nicht wiederkehrt. Das
ist ein verwertbares Ergebnis, kein Scheitern — und es spricht dafür, den nächsten
Anlauf nicht an einer feineren Zerlegung desselben Ortes zu versuchen, sondern an
mehr Orten.

### Was die Vorgeschichte immerhin belegt

Ein Punkt spricht klar für die Messkette: der Reproduktionstest des Laufs
`20260906_103133_0e2bc9` vergleicht 144 Kennzahlen zwischen zwei Durchgängen und
findet einen maximalen absoluten Unterschied von **exakt 0.0**. Der Verdacht aus der
frühen Phase, in den mittleren Schichten stecke ein hardwareabhängiger Effekt, ist
für diese Messkette damit erledigt: sie liefert bitgleiche Ergebnisse. Was immer die
Läufe zeigen oder nicht zeigen, es ist kein Rauschen der Ausführung.

Derselbe Lauf zeigt aber auch, wie hartnäckig das Muster ist. Drei voneinander
unabhängige Zerlegungsversuche am selben Ort — Alphabet-Achsen, bitweise Primitive,
Nachfolgerelationen — landen alle im selben Bereich:

| Zerlegung | erklärter Anteil | Residuum |
|---|---|---|
| bitweise Primitive, k = 4 | 1.3 – 3.3 % | 98.4 – 99.4 % |
| Nachfolger, nicht-bitweise | 0.13 – 2.3 % | 97.9 – 98.7 % |

Die einzige positive Aussage des Laufs — ein „source-state-conditioned low-rank
operator" — beruht auf einer Anpassung mit **Rang 32** in einem Raum der Breite
2 048, die den Kosinus um 0.21 bis 0.31 und den normierten Fehler um 6 bis 9 Prozent
verbessert. Ob das mehr ist, als eine Rang-32-Anpassung an Rauschen erreicht, sagt
der Bericht nicht; ein Vergleich gegen zufällige Operatoren gleichen Ranges wäre die
naheliegende Gegenprobe. Der Lauf selbst schreibt an anderer Stelle die passende
Warnung hin: „rank-128/512 success is not evidence that the true name mechanism is
low-rank".

### Zwei Architekturdetails, die man dabei kennen muss

Aus `config.json` von `EleutherAI/pythia-1.4b`, nachgeladen und in
`daten/PYTHIA14B_ARCHITEKTUR.csv` abgelegt:

- **Vierundzwanzig Blöcke, Breite 2 048, sechzehn Köpfe à 128.** Die Rotation wirkt
  nur auf einem Viertel der Kopfdimensionen, und Attention und MLP hängen parallel am
  Residuum (`use_parallel_residual: true`) — was der `ological`-Lauf korrekt prüft.
  Block 23 ist damit der letzte; eine Einschränkung auf „B19 und B23" trifft das
  hintere Drittel des Netzes.
- **Das Vokabular ist gepolstert.** Die Konfiguration nennt 50 304 Zeilen, der
  Tokenizer kennt 50 277 Einträge, das reine Vokabular 50 254. Die letzten
  **27 Embedding-Zeilen gehören zu keinem Token**. Wer über Token-IDs rechnet — und
  eine der ausgewählten Primitiven heißt `id_xor_b14` — arbeitet auf einer Menge, die
  am oberen Rand aus unbenutzten Zeilen besteht. Eine bitweise Operation auf IDs kann
  dort hinzeigen, wo nie etwas trainiert wurde.

## 8. Der parallele Lauf am selben Textfenster

Am selben 13. September lief ein zweiter Versuch, `ological_c677b4881415`, der nicht
Buchstaben zählt, sondern eingreift. Er verdient eine eigene Notiz, weil er
methodisch deutlich weiter ist als die WASD-Läufe — und weil seine eigenen Zahlen
etwas anderes sagen als seine Überschrift.

**Was er macht.** Er stört das Wort *Meteorological* im selben Textfenster (Token 165
bis 168) in acht Varianten: Aufspaltung, Null statt o, Eins statt o, Großbuchstabe,
Tippfehler, zwei Vertauschungen. Dann verpflanzt er Aktivierungen zwischen den
Varianten — Schichten 13, 14 und 15, vier Bauteile je Block, zwei Anker, beide
Richtungen: 672 Transfers, Batchgröße eins, FP32.

**Was gut ist.** Die Kontrollen sitzen. Der Selbstpatch, bei dem eine Aktivierung
durch sich selbst ersetzt wird, hat einen maximalen Logit-Fehler von exakt `0.0`.
Die Rekonstruktion des parallelen Residualpfads von Pythia (`resid_out = resid_in +
attn + mlp`) stimmt ebenfalls auf `0.0`. Und `interpretation.txt` schreibt die
Grenzen selbst hin, einschließlich des Satzes, ein Transfer identifiziere „causal
involvement at a selected anchor, not the origin of a logical concept or a full
circuit".

**Was die Zahlen sagen.** Der eigentliche mechanistische Anspruch hängt am Anker
`department` — dort sollte sich zeigen, dass die Störung über das Wort hinaus wirkt.
Genau dort passiert fast nichts:

| Anker | größter absoluter Effekt | nativ |
|---|---|---|
| `readout` (letztes Token) | 0.1203 nats | 0.1203 |
| `department` | 0.0274 nats | 0.0105 |

Der Effekt sitzt also am Endtoken selbst — dessen Residuum den Readout ohnehin
direkt speist. Das ist beinahe eine Tautologie: wer das letzte Token überschreibt,
ändert die nächste Vorhersage. Dazu kommt, dass die Schichten 13, 14 und 15
praktisch gleich reagieren; eine Sonderrolle von Schicht 14, wie sie die frühere
McQuarrie-Linie vermutete, ist in diesen Zahlen nicht zu sehen.

**Wo es wackelt.** 144 der 672 Zeilen sind als `baseline_gap_too_small` markiert, der
normalisierte Transfer geht in einzelnen Ziffernpaaren bis 1.47 — mehr als
vollständige Übertragung, ein Zeichen dafür, dass der Nenner zu klein ist. Ein
Nullmodell jenseits des trivialen Selbstpatches fehlt, ebenso eine Korrektur über die
672 Vergleiche. Und die Themenmarker sind, wie der Lauf selbst vermerkt, rein
lexikalisch: die Marge zwischen Logik- und Wettermarkern ist durchweg stark negativ,
weil der Text vom Wetter handelt.

**Was die Grundlinien verraten.** Die Verschlechterung der Vorhersage verteilt sich
so: Aufspalten kostet 0.168 nats, die Null statt des o weitere 0.121, und die
Identität der Ziffer — 0 gegen 1 — nur etwa 0.04. Der größte Teil des Effekts kommt
also von der veränderten Tokenisierung, nicht vom eingesetzten Zeichen.

Das passt zur Tokenstruktur, die in `daten/OLOGICAL_TOKENISIERUNG.csv` nachgerechnet
ist: `original_split`, `zero`, `one` und `capital_o` bilden eine saubere Familie
`ĠMet | e | ? | rological`, die sich nur an der dritten Position unterscheidet — für
diese vier gilt die Behauptung des Laufs, es unterscheide sich genau ein Token. Der
Tippfehler `x_typo` fällt heraus (drei Tokens statt vier), und die Vertauschungen
stellen das Token `ological` wieder her, das in der Nullvariante gar nicht vorkommt:
zwischen `zero` und `zero_transpose` stimmt von vier Positionen nur die erste.

**Das dritte Werkzeug.** Das Notebook `State60482_Pythia_Fragilitaet.ipynb` vom selben
Tag ist ein anderer Versuch: kein Patching, sondern normierte Zufallsstörungen auf
Q, K und V vor der Rotationskodierung, gegen neun textgleiche Varianten mit
verschobenen Tokengrenzen. Zeichenketten aus dem Gaming-Bereich kommen dort nicht
vor. Auch hier sind die Interaktionen winzig — der größte mittlere Absolutwert liegt
bei 0.00199 nats, alle Ausfallraten bei null, nichts überschreitet die
Neutralitätsschwellen. Mit vier Richtungsziehungen meldet der Lauf sein Vertrauensband
selbst als `too_few_paired_direction_seeds`.

**Fazit für diese Linie.** Drei Arbeitslinien messen an einem einzigen Dokument von
207 Token: McQuarrie an 124 bis 127, *Meteorological* an 165 bis 168, WASD über das
ganze Fenster verteilt. Keine davon hat bisher einen Effekt gezeigt, der außerhalb
des Endtokens liegt. Das ist kein Grund aufzuhören — aber es ist ein Grund, den
nächsten Versuch an mehr als einem Text zu führen.

## 9. Was in dieser Mappe liegt

| Datei | Inhalt |
|---|---|
| `zeichensatz_statistik.py` | Buchstabendichte gegen drei Nullmodelle, inklusive des frequenzangepassten |
| `tokenizer_sonde.py` | prüft vor einem GPU-Lauf, ob eine Zeichenfolge das Netz als Einheit erreicht |
| `experiment_traeger.py` | der vorregistrierte Nachfolgeversuch: Design, Strukturprüfung, Entscheidungsregel |
| `wasd_traeger_messlauf_colab.ipynb` | der Messlauf für die GPU: Vorflug, beide Qualitätstore, 720 Grundlinien, 5 760 Transfers, Auswertung nach der Regel |
| `pile_nullmodell.py` | das Dokument-Nullmodell aus dem Trainingskorpus, plus die Basisrate von WASD |
| `PROJEKTSTAND.md` | die Gesamtuntersuchung im Überblick: sieben Abschnitte, drei Arbeitslinien, was am 13.09. anders ist |
| `METHODEN_BRIEFING.md` | die Literatur- und Methodenrecherche: was die publizierte Suszeptibilitätsmethode misst, Patching-Checkliste, Numerik, Basisraten, Quellen |
| `daten/` | die Originalausgaben der drei Läufe, damit die Nachrechnung nicht von Drive abhängt |
| `tests/test_zeichensatz_statistik.py` | Kalibrierung gegen gepflanzte Wahrheiten: neutraler Text darf nicht ausschlagen, gepflanzte An- und Abreicherung muss gefunden werden |
| `tests/test_nullmodell_verzerrung.py` | der Beleg für die Verzerrung, direkt an den Messdaten |
| `tests/test_zyklus_und_periode.py` | was die Zyklus- und Positionsanalyse wirklich zeigt |
| `tests/test_mcq_varianten.py` | der Tokenisierungsfaktor in der früheren McQuarrie-Linie |
| `tests/test_experiment_traeger.py` | Entscheidungsregel gegen gepflanzte Wahrheiten, inklusive der halb positiven Fälle |
| `tests/test_messlauf_notebook.py` | hält Notebook und Designmodul zusammen: jede doppelt geführte Konstante, beide Qualitätstore |
| `tests/test_pile_nullmodell.py` | Rechenlogik plus Konsistenz gegen den gespeicherten Korpuslauf |
| `tests/test_ological_struktur.py` | die Tokenstruktur der Störungen im parallelen Lauf |
| `tests/test_primitivauswahl.py` | warum die Vorgängerlinie stecken blieb, an ihren eigenen Zahlen |

Nachrechnen:

```bash
pip install pytest tokenizers
pytest examples/05_wasd_zeichenhypothese/tests -q

curl -sL -o tokenizer.json \
  https://huggingface.co/EleutherAI/pythia-1.4b/resolve/main/tokenizer.json
python examples/05_wasd_zeichenhypothese/tokenizer_sonde.py \
  --tokenizer tokenizer.json --ziel WASD --ziel wasd --ziel QERF
```

---

## 10. Grenzen dieser Mappe

- Alles hier ist Text- und Tokenizer-Statistik. Es wird **keine** Aussage darüber
  getroffen, was Pythia in den mittleren Schichten tut; dafür braucht es einen
  Vorwärtslauf mit Eingriff.
- Die englischen Buchstabenhäufigkeiten stammen aus einer Standardtabelle, nicht aus
  dem Pile. Für eine Veröffentlichung müssten sie am tatsächlichen Trainingskorpus
  geschätzt werden; für die hier gezogenen Schlüsse genügt die Größenordnung, weil
  das Gruppenmittel der 19 Fenster (0.3980) und die Standardtabelle (0.4134) zum
  selben Ergebnis führen.
- Dass `WASD` kein Token ist, schließt eine zeichenweise Repräsentation nicht aus.
  Es verschiebt nur die Beweislast: Wer sie annimmt, muss sie zeigen.
- Die 19 Textfenster stammen aus demselben Auswahlprozess. Ob sie eine faire
  Vergleichsgruppe sind, ist hier nicht geprüft.
