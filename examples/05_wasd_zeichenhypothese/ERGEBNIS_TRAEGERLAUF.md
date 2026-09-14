# Der Trägerlauf vom 14.09.2026 — was herauskam

`EleutherAI/pythia-1.4b` @ `step98000`, float32, Batchgröße 1, A100.
18 024 Vorwärtsläufe, 25.2 ms je Lauf. Alle Qualitätstore bestanden.

> **Das Ergebnis in drei Sätzen.**
> Die vorregistrierte Regel ist in beiden Teilen nicht erfüllt: im allein als Endpunkt
> benannten Feld `bewegung` liegt `WASD` auf Rang 3 von 30 und 0.13 nats **unter** der
> besten Kontrolle. Der Kausalteil ist dabei nicht negativ, sondern **leer** — weil
> Eingriffsort, einzige Unterschiedsstelle und Messposition zusammenfallen, war sein
> Kriterium vor dem ersten Vorwärtslauf unerfüllbar. Im mitgemessenen, aber nicht als
> Endpunkt benannten Feld `tastatur` liegt `WASD` dagegen auf Rang 1 mit 1.70 nats
> Vorsprung — was eine **Hypothese** ist, kein Befund, und inhaltlich genau das, was
> die Nullerklärung einer gewöhnlichen lexikalischen Assoziation vorhersagt.

---

## 1. Die Qualitätstore

| Tor | Verlangt | Gemessen | |
|---|---|---|---|
| Selbstpatch | exakt `0.0` | **`0.000e+00`** über 2 880 Prüfungen | ✓ |
| Paralleles Residuum | relativ ≤ `1e-4` | `1.526e-05` absolut, **`1.069e-07`** relativ | ✓ |
| Brauchbare Nenner | ≥ 80 % | **100 %** (0.0 % zu klein) | ✓ |
| Architektur | 24 Blöcke, 2048, parallel | stimmt | ✓ |
| Tokenisierung | gegen die vorab abgelesene ID-Tabelle | stimmt | ✓ |
| Streuung | σ ≤ 0.8 nats | **σ = 1.4579** | ✗ |

Der Selbstpatch bei exakt null über 2 880 Prüfungen ist die belastbarste Zahl des
Laufs: der Eingriff sitzt, wo er sitzen soll, und die Numerik ist deterministisch.

Die σ-Flagge ist gefallen — aber sie misst die falsche Größe. σ ist die Streuung
**einer** Variante über die 24 Schablonen, verglichen mit einer Schwelle für
**Mittelwerte**. Alle Varianten laufen durch dieselben Schablonen, der Vergleich ist
also gepaart; der Schablonenanteil fällt weitgehend heraus. Die Flagge bindet
trotzdem: sie stand vorher fest, sie ist gefallen, und sie nachträglich umzudeuten
wäre derselbe Zug wie das Wechseln des Endpunkts.

---

## 2. Der Beobachtungsteil

Die Regel nannte **ausschließlich** `bewegung`.

| Feld | Ziel `WASD` | Rang | beste Kontrolle | Vorsprung | Schwelle |
|---|---|---|---|---|---|
| `bewegung` | −7.9354 | **3** von 30 | WASV −7.8054 | **−0.1300** | 0.5 |
| `tastatur` | −3.3529 | **1** von 30 | WASA −5.0504 | +1.6975 | — |
| `gegenfeld` | −13.7947 | **letzter** | WASW −10.4182 | −3.3765 | — |

Beide vorab benannten Fassungen der Beobachtungsregel scheitern: gegen das Maximum
−0.1300, gegen das 90-Prozent-Quantil +0.0387 — beide weit unter 0.5.

**Warum `bewegung` nicht gewinnen konnte.** Alle 24 Schablonen enden mit dem
Platzhalter, gemessen wird das unmittelbar folgende Token. Nach einem Tastenkürzel
steht dort ein Nomen, eine Konjunktion oder ein Satzzeichen — „Movement is bound to
the four keys WASD" geht mit „,", „." oder „and" weiter, nicht mit „ move". Das
Entscheidungsfeld lag am Messfenster vorbei. Bezeichnend: im Feld `bewegung` führt
`WASV`, eine Zeichenfolge ohne jede Bewegungsbedeutung.

Der Befund lautet deshalb **nicht nachgewiesen**, nicht **widerlegt**.

---

## 3. Der Kausalteil war leer, nicht negativ

Kein einziger der 24 Blöcke erfüllt beide Bedingungen. Das ist kein Ergebnis über
`WASD`, sondern über das Instrument — und es ist beweisbar:

Drei der fünf vorab festgelegten Eingriffspaare sind `WASF`, `WASR`, `WASZ`. Sie
unterscheiden sich vom Ziel **nur im letzten Token**. Bei kausaler Attention sind die
Positionen 0…n−2 beider Prompts bitgleich. Wird das volle Residuum der letzten
Position nach Block L ersetzt, ist der Zustand ab dort identisch — also ist die
Ausgabe identisch. Für diese drei Paare gilt an **jedem** Block:

```
Wiederherstellung = 1     Leck = 1     exakt, per Konstruktion
```

Damit hat das gemittelte Leck die Untergrenze (3·1 + 0 + 0)/5 = **0.60**. Verlangt
war **≤ 0.20**. Das Kriterium war unerfüllbar, bevor der erste Vorwärtslauf startete.

**1 728 der 2 880 Transferzeilen (60.0 %) und 6 912 der 11 520 Vorwärtsläufe haben
eine vorab bekannte Konstante ausgerechnet.**

Rechnet man den informativen Anteil zurück — nur `FORD` und `NOTA`, die sich im
**ersten** Token unterscheiden —, ergibt sich als kleinstes Leck 0.5852 bei Block 6.
Auch das liegt weit über 0.20. Die Blöcke 1–7 fallen dort immerhin sichtbar ab
(0.59–0.77 gegen 1.07–1.17 in den oberen Blöcken), was ein Hinweis wäre — aber nur
ein Hinweis, aus einem Test, der nicht dafür gebaut war.

Was gefehlt hat, ist eine **Positivkontrolle**: irgendein Paar, das nachweislich ein
Leck ≤ 0.20 erreichen kann. Ohne sie war nicht zu erkennen, dass das Instrument nichts
messen kann.

---

## 4. Ein Fehler im Code, der nicht aufgefallen ist

Das Design ruhte auf einer Zusicherung: Ziel und Kontrolle unterscheiden sich in
**genau einem** Token. Fünf Kandidaten erfüllen das nicht — `WASE`, `WASH`, `WASK`,
`WASS`, `WAST` zerfallen als `['ĠW','ASE']` und so weiter, weil `ASE`, `ASH`, `ASK`,
`ASS` und `AST` eigene Vokabeleinträge sind. Die Strukturprüfung fand sie und meldete
sie als verworfen.

**Und die Entscheidungsregel benutzte sie trotzdem.** Es gab zwei Begriffe von
„Kontrolle": die Prüfung rechnete mit 20, die Regel mit 25. Die geprüfte Menge wurde
gedruckt und nie verwendet.

Folgen:

- Der **Befund dreht sich nicht**: die beste Kontrolle ist so oder so `WASV`, der
  Vorsprung bleibt −0.1300. Die fünf lagen alle darunter; ihre Aufnahme war
  konservativ, nicht günstig.
- Aber **jede berichtete Streuung war falsch**. Mit dem designkonformen 23er-Satz
  sinkt die Kontroll-Streuung im Feld `tastatur` von 0.82 auf 0.48, und z steigt von
  +3.35 auf **+5.13**.

Behoben: die Strukturentscheidung fällt jetzt beim Bau der Varianten und wird in die
Rolle geschrieben, sodass keine spätere Rechnung sie übersehen kann. Drei Tests
halten die Invariante fest, ein vierter prüft, dass jede tragende Kontrolle sich vom
Ziel wirklich in genau einem Token trennt.

---

## 5. Was im Tastaturfeld steht — und warum es kein Befund ist

`WASD` führt dort mit 1.6975 nats Abstand vor der besten von 23 Kontrollen. Die
Verteilung dahinter ist eng: Plätze 2 bis 5 liegen innerhalb von 0.35 nats. Der
Abstand des Ziels ist also keine Randschwankung, sondern eine Lücke — fünfmal so
groß wie die Spannweite des gesamten Verfolgerfelds.

Die Spezifitätskontrollen greifen alle vier:

| | `tastatur` | schließt aus |
|---|---|---|
| `WASD` | **−3.3529** | |
| `FORD` | −5.7762 | dasselbe Endtoken (ID 37) — nicht der Buchstabe `D` |
| `NOTA` | −6.3594 | gleiche Korpushäufigkeit — nicht die Frequenz |
| `ASDW` | −5.9847 | gleiche Buchstaben — nicht die Zeichenmenge |
| `ESDF` | −5.8724 | echte Alternativbelegung — nicht der Begriff „Bewegungstasten" |

Der Effekt hängt an der **exakten geordneten Zeichenfolge**. Das ist eine ungewöhnlich
saubere Spezifitätsprüfung.

**Und trotzdem ist es kein Befund.** Die Regel benannte `bewegung` als Endpunkt. Ein
anderes Feld nach Kenntnis der Daten zum Endpunkt zu erheben, ändert nicht nur den
Test, sondern die Behauptung — aus „WASD hebt Bewegungssemantik" wird „WASD hebt
Tastaturtoken". Genau dieser Zug ist es, den diese Mappe den früheren Läufen
vorhält. Die Grenze verläuft nicht zwischen *vorab gemessen* und *nachträglich
erfunden*, sondern zwischen *vorab gemessen* und *vorab als Endpunkt benannt*. Nur
Letzteres trägt eine Entscheidung.

Der Wert ist also der bestmögliche Hypothesengenerator — und nichts darüber hinaus.

### Das Gegenfeld ist kein zweites Argument

`WASD` ist im Kontrastfeld von allen dreißig am stärksten unterdrückt. Naheliegender
Einwand: Das ist bloß die Kehrseite — wenn eine Wahrscheinlichkeitsmasse steigt, muss
andere sinken. Nachgerechnet stimmt das nicht:

- Die Tastaturmasse steigt von 0.23 % (Kontrollmittel) auf 3.5 % (Ziel). Die dadurch
  **erzwungene** Absenkung aller übrigen Token beträgt 0.0333 nats — **1.5 %** der
  beobachteten 2.2351 nats.
- Eine reine Renormierung verschiebt **alle** Felder um denselben Betrag. Beobachtet
  sind +0.56 / +2.73 / −2.24 — eine Spannweite von 4.97 nats, die ein gemeinsamer
  Skalar nicht erzeugen kann.
- Über die 29 Nicht-Ziel-Varianten ist `r(tastatur, gegenfeld) = **+0.233**`. Die
  Störrichtung ist also *positiv*; `WASD` zeigt das Gegenteil und ist damit **schwerer**
  artefaktisch zu erzeugen, nicht leichter.

Der saubere Weg, beides in **einer** Zahl zu führen, ist der Kontrast
`tastatur − gegenfeld`: dort fällt `log Z` per Konstruktion heraus.

| | Kontrast | |
|---|---|---|
| `WASD` | **10.4418** | |
| `FORD` | 7.1937 | beste Kontrolle |
| `ESDF` | 5.4552 | |
| `ASDW` | 5.4333 | |
| `NOTA` | 5.1201 | |

Vorsprung **+3.2481 nats**, z = **+6.82** gegen die 23 designkonformen Kontrollen.
Das Gegenfeld ist damit kein zweiter Befund, sondern die zweite Ablesung desselben
Effekts — und der Kontrast ist die richtige Art, ihn zu berichten.

---

## 6. Was das inhaltlich heißt

Die Ausgangshypothese lautete, `WASD` löse **Language Switching, Error Correction und
andere besondere Erscheinungen** aus. Dazu sagt dieser Lauf: **nichts.** Keine dieser
Größen wurde operationalisiert. Alle drei Felder messen Thematik. Sprachwechsel wäre
Masse auf nicht-englischen Token, Fehlerkorrektur wären Reparaturtoken nach einem
eingebauten Tippfehler — beides hat in diesem Design keine Messgröße. Die Hypothese
ist weder gestützt noch widerlegt; sie wurde nicht geprüft.

Was der Lauf zeigt, ist etwas anderes: nach `WASD` steigt die Wahrscheinlichkeit von
` keys`, ` key`, ` keyboard`, ` bind`, ` controls` stark an — und zwar gebunden an die
exakte Zeichenfolge. Das ist ein **Kollokationseffekt**. `WASD` *bezeichnet* wörtlich
Tastaturtasten; ` keys` und ` keyboard` sind Quasi-Synonyme seiner Denotation, ` bind`
und ` controls` die übliche Nachbarschaft. Für eine 13 870 mal gesehene Abkürzung ist
das genau das, was gewöhnliches lexikalisches Wissen vorhersagt.

Der Effekt ist real, groß und sauber isoliert. Er ist nur nicht das Gesuchte.

---

## 7. Was als Nächstes zu tun ist

**Zuerst den Grundlinienteil neu rechnen — 720 Vorwärtsläufe, rund 18 Sekunden.**
Nicht um etwas zu belegen, sondern um zu sehen, was an dieser Position überhaupt
vorhergesagt wird. Zu protokollieren sind die volle Matrix Variante × Schablone ×
Feld statt bloßer Mittel, dazu `log Z`, Entropie und die Top-50-Fortsetzungen je
Zelle. Das liefert die gepaarte Auswertung, den Permutationstest und die bislang
fehlende Konsistenzzahl — in wie vielen der 24 Schablonen liegt `WASD` vorn? —, und
es zeigt zum ersten Mal, was das Modell dort tatsächlich erwartet. Genau diese
Information fehlte, als `bewegung` zum Endpunkt gemacht wurde. Dieser Schritt ist
ausdrücklich **deskriptiv** und begründet keinen Befund.

**Danach Runde zwei vorregistrieren**, auf **neuen** Schablonen, mit:

- `tastatur − gegenfeld` als einzigem primärem Endpunkt (renormierungsfrei);
- lexikalischen Positivkontrollen (` keybinds`, ` hotkey`, ` arrow keys`) und
  Bedeutungsparaphrasen, die vorab definieren, was „über gewöhnliche Assoziation
  hinaus" überhaupt heißen soll;
- einem Messfenster, das nicht unmittelbar hinter dem Platzhalter liegt;
- den nie gemessenen Zielgrößen Sprachwechsel und Fehlerkorrektur;
- einem Multiplizitätsplan, der vorher sagt, was bei einem positiven Nebenfeld gilt.

**Der Kausalteil wird erst gebaut, wenn eine Positivkontrolle steht.** Eingriffsort
und Messort müssen getrennt sein — Messung k Token hinter dem Platzhalter,
komponenten- oder unterraumaufgelöst, `FORD` als endtokengleiche Quelle. Solange kein
Paar nachweislich ein Leck ≤ 0.20 erreichen kann, misst der Test nichts.

---

## 8. Was von diesem Lauf bleibt

Verloren: die Rohtabellen. Sie lagen nur im flüchtigen Colab-Dateisystem. Erhalten
sind die 30 Variantenmittel je Feld und die 24 Blockmittel, gerettet aus dem
PDF-Export und abgelegt unter [`daten/`](daten/). Gepaarte Auswertung, Permutationstest
und die paarweise Aufschlüsselung des Kausalteils sind damit nicht mehr nachprüfbar,
obwohl 18 024 Vorwärtsläufe bezahlt wurden. Das Notebook sichert seit diesem Lauf
nach Drive.

Geblieben ist mehr, als das negative Gesamturteil vermuten lässt: ein Selbstpatch bei
exakt null, eine Kontrollfamilie, die vier verschiedene Erklärungen einzeln ausschließt,
ein Kollokationseffekt von seltener Schärfe — und drei Designfehler, die jetzt
benannt und teils behoben sind. Ein Lauf, der seine eigenen Fehler sichtbar macht,
ist mehr wert als einer, der bestätigt, was man hören wollte.
