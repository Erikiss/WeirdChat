# Mappe 05 — Die WASD-Zeichenhypothese, nachgerechnet

Diese Mappe prüft **eine** Frage: Spielen die Tastenbuchstaben `W A S D` (und die
Nachbartasten `Q E R F`) im Pythia-Modell eine Sonderrolle — lösen sie in den
mittleren Schichten Effekte wie Sprachwechsel oder Fehlerkorrektur aus?

Die Antwort auf dem heutigen Stand der Messungen ist **nein**, und zwar deutlicher,
als die Läufe selbst berichten. Was hier steht, ist kein Gegenargument gegen die
Forschungslinie, sondern ihre Bilanz: drei Läufe vom 13. September 2026, exakt
nachgerechnet, plus zwei Befunde, die in den Läufen nicht enthalten sind und die
Richtung des Ergebnisses umdrehen.

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

## 2. Zwei Befunde, die in den Läufen fehlen

### 2.1 Das Nullmodell ist verzerrt — und zwar genau in die Richtung des erhofften Ergebnisses

Der einzige positive Wert im ganzen Dichtelauf ist das `z = +1.084` gegen zufällig
gezogene Achterbuchstabenmengen. Dieses Nullmodell fragt aber nicht, ob *dieser
Text* auffällig ist, sondern ob *diese Buchstaben* häufig sind.

`WASDQERF` enthält E, A, R und S — vier der sechs häufigsten Buchstaben des
Englischen. Die erwartete Dichte dieser Achtermenge in gewöhnlichem englischem Text
beträgt **0.4134**; eine zufällig gezogene Achtermenge kommt im Mittel auf 0.3077.
Ein positives z ist damit die Voreinstellung, nicht der Befund.

Zwei Gegenproben, beide in `tests/` festgehalten:

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
Da BPE-Merges nach Häufigkeit entstehen, ist das zugleich ein Hinweis auf die
Basisrate: wäre `WASD` im Trainingskorpus häufig, gäbe es dafür ein Token.

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

## 4. Was jetzt zu tun wäre

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

**Zweitens: eine Vorhersage, die schiefgehen kann.** "Interessante Erscheinungen"
ist keine. Prüfbar wäre etwa: *Ein Eingriff an der Tokenposition von `D` in `WASD`
verschiebt die Wahrscheinlichkeit von Bewegungsvokabular (`move`, `strafe`,
`keys`) um mindestens x, während derselbe Eingriff an einem frequenzangepassten
Kontrollbigramm dies nicht tut.* Mit Effektgröße und Richtung, vor dem Lauf
aufgeschrieben.

**Drittens: abgestimmte Kontrollen.** Die Vergleichsbedingung darf sich nicht in der
Buchstabenhäufigkeit unterscheiden, sonst misst man wieder die Häufigkeit. Für jede
Zielzeichenfolge gehört eine Kontrollzeichenfolge dazu, die in Tokenzahl,
Tokenhäufigkeit und Position übereinstimmt und sich nur in der Tastaturnachbarschaft
unterscheidet.

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

---

## 5. Rückblick: ein unkontrollierter Faktor in der McQuarrie-Linie

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

Daraus folgen zwei Dinge, die im Bericht nicht auftauchen:

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

## 6. Was in dieser Mappe liegt

| Datei | Inhalt |
|---|---|
| `zeichensatz_statistik.py` | Buchstabendichte gegen drei Nullmodelle, inklusive des frequenzangepassten |
| `tokenizer_sonde.py` | prüft vor einem GPU-Lauf, ob eine Zeichenfolge das Netz als Einheit erreicht |
| `daten/` | die Originalausgaben der drei Läufe, damit die Nachrechnung nicht von Drive abhängt |
| `tests/test_zeichensatz_statistik.py` | Kalibrierung gegen gepflanzte Wahrheiten: neutraler Text darf nicht ausschlagen, gepflanzte An- und Abreicherung muss gefunden werden |
| `tests/test_nullmodell_verzerrung.py` | der Beleg für die Verzerrung, direkt an den Messdaten |
| `tests/test_zyklus_und_periode.py` | was die Zyklus- und Positionsanalyse wirklich zeigt |
| `tests/test_mcq_varianten.py` | der Tokenisierungsfaktor in der frueheren McQuarrie-Linie |

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

## 7. Grenzen dieser Mappe

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
