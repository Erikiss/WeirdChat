# Mappe 04 — Bigramm-Repräsentation im Morse-Expertennetz

Eigenständige Mappe für **eine** Frage: Wie ist eine Folge von genau **zwei
Buchstaben** im zeichenweisen Transkodier-Mechanismus von
`qwen/qwen3.6-35b-a3b` repräsentiert — und gilt die Antwort über Alphabete
hinweg?

Aufgesetzt auf den Befund aus
[`../03_deepspec_draft_surprise/RESULTS.md`](../03_deepspec_draft_surprise/RESULTS.md):
**L33/E228** ist der einzige einzeln kausale Experte für Braille- und
Morse-Transkodierung (dreifach repliziert, zuletzt mit Decode-Feuerindex:
E228 in der top-8 an 60–62 % aller Decode-Positionen der konstruierenden
Arme, gegen 1 % im Bezugsarm). Ein separater GPU-Lauf (Voll-Scan über alle
10 240 Experten) sucht derzeit Partner — falls er welche findet, wird aus
dem Einzelbefund ein **Expertennetz**, und diese Mappe misst dessen
Binnenstruktur. Alles hier funktioniert aber schon mit E228 allein;
`EXPERTEN` im Notebook ist eine Liste und wird bei Scan-Rückkehr erweitert.

## 1. Einordnung: das Semantiknetz-Paper und das Methodenpapier

**Das gemeinte Paper** ist Ha & Kim, *„Semantic Networks as Clues: A
Theoretical Foundation and Process Optimization for Semantic Network
Construction"* (Soongsil University). Es behandelt semantische Netzwerke
aus **Text**: Schlüsselphrasen als Knoten, semantische Verwandtschaft als
Kantengewichte, Communities als Themen — konstruiert in drei Stufen
(AKE → Keyness, EW → Interpretierbarkeit, CD → Distinktheit) und als
Gesamtprozess über eine Zielfunktion J optimiert (ClueNetwork).
Epistemisch zentral: solche Netze sind **„clues, not surrogates"** — es
gibt keinen Goldstandard, ihre Legitimität kommt aus der Peirce'schen
**Abduktion** (überraschender Befund → erklärende Hypothese → später
induktiv prüfbar).

**Bestätigen kann ein Morse-Expertennetz dieses Paper nicht** — es macht
keinerlei Aussage über MoE-Experten oder neuronale Mechanik, und Begriffe
wie „Kommunikation" oder „Aneignung" von Semantik kommen darin nicht vor.
Was es liefert, ist etwas Nützlicheres: den **Bauplan und die Epistemik**
für genau das, was diese Mappe und der Voll-Scan tun. Die Übersetzung,
Stufe für Stufe:

| SNC-Stufe (Paper) | Ziel (Paper) | Gegenstück hier |
|---|---|---|
| AKE — welche Einheiten werden Knoten | Keyness | Voll-Scan: welche Experten gehören ins Netz (Kriterium strenger: **kausal**, nicht häufigkeitsbasiert) |
| EW — Kantengewichte aus Ko-Vorkommen (Distributionshypothese) | Interpretierbarkeit | Ko-Feuern der Experten über Decode-Positionen — die `feuerwuerfel_*.npz` dieser Mappe enthalten dafür alle 256 Experten je Zielschicht |
| CD — Communities als Themen | Distinktheit | bildet {E228 + Partner} eine abgegrenzte Community im Ko-Feuer-Graphen? |
| J / ClueNetwork — Kandidatennetze ranken | Wissensrepräsentation | konkurrierende Netz-Definitionen nach Scan-Rückkehr vergleichen |

Auch die **Zweibuchstaben-Frage** hat im Paper-Vokabular eine präzise
Form: ist „ch" ein **eigener Knoten** (wie eine Mehrwort-Schlüsselphrase
in AKE) oder nur eine **Kante** zwischen den Knoten c und h? Genau das
trennen die Urteile `PAARE-EIGEN` und `BUCHSTABEN-ADDITIV` in Stufe A.

Die Abduktions-Epistemik des Papers deckt sich mit der Hausregel dieser
Untersuchung: Feuerkarten (Stufe A/B) sind Hypothesenerzeugung, keine
Verifikation — die leistet erst die Maskierung in Stufe C. Dieselbe
Trennung mahnt das **Methodenpapier** (`erikiss/spectral-probe-circuits`,
§9.2: Selektivität allein ist kein Schaltkreisfinder — in diesem MoE
bereits bestätigt, Phasen 14–15; §7.6: dieselbe Zerlegung ist je Modell
kausal, wirkungslos oder hinderlich). Ein kleines Morse-Expertennetz
wäre eine weitere Instanz von dessen Kernbefund (kleine kausale Gruppen
je Fähigkeit) auf Router-Kanälen statt Attention-Köpfen.

## 2. Die Frage, formalisiert

Für jeden Experten E des Netzes und jedes geordnete Buchstabenpaar (a, b)
misst der Decode-Feuerindex die Rate

    F_E[a,b] = P(E in top-8 | Decode-Position der Antwort auf „schreibe ab in Morse")

Zerlegung F = μ + α_a + β_b + γ_ab. Drei mögliche Welten:

| Befund | Bedeutung | Folge für die Kombinatorik |
|---|---|---|
| `BUCHSTABEN-BLIND` | E feuert aufs Format, nicht auf Buchstaben | Bigramm-Frage ist für E gegenstandslos |
| `BUCHSTABEN-ADDITIV` | γ ≡ 0: zwei Buchstaben = Summe der Einzelbuchstaben | Stufe B skaliert mit **Σ\|Alphabet\|**, nicht Σ\|Alphabet\|² |
| `PAARE-EIGEN` | γ ≠ 0: es gibt Paar-Einheiten | Spitzenzellen werden Stufe-C-Kandidaten |

Dazu orthogonal die **Positionsfrage**: feuert E während der Morse-Gruppe
des ersten oder des zweiten Buchstabens (Token-zu-Buchstaben-Zuordnung
zeichengenau per Präfix-Dekodierung)?

**Vorregistrierung.** Primärkandidat für eine Paar-Einheit ist **„ch"**:
im deutschen Landes-Morse ist ch ein *eigenes* Zeichen (`----`), das
einzige echte Zweibuchstaben-Morsezeichen des Kernbestands. Sekundär
(erklärtermaßen explorativ): en, er, ei, st, ck. Alles andere, was die
Spitzenzellen-Liste hochspült, ist Befund dritter Ordnung und braucht
Stufe C, bevor es behauptet werden darf.

## 3. Warum „alle Alphabete" nicht explodiert

Die Sorge liegt nahe: 26×26 im Deutschen, und dann noch einmal je Schrift?
Die Rechnung entwarnt, aus zwei Gründen.

**Erstens ist Stufe A/B reine Beobachtung** — ein Generate-Aufruf misst
GRUPPE×N_JE Bigramme gleichzeitig, ohne Maskierungs-Fixkosten je
Konfiguration. Bigrammzahlen der geplanten Schriften:

| Schrift | Zeichen | Bigramme | Morse-Gegenstück |
|---|---:|---:|---|
| lateinisch (dt. Rahmen) | 26 | 676 | internationaler Morse |
| kyrillisch | 32 | 1 024 | russischer Landescode (Е=Ё) |
| griechisch | 24 | 576 | griechischer Landescode |
| hebräisch | 22 | 484 | hebräischer Landescode |
| arabisch | 28 | 784 | arabischer Landescode |
| Kana | 48 | 2 304 | Wabun-Code (japanischer Morse) |
| Hangul-Jamo | 24 | 576 | koreanischer Morse (SKATS-Familie) |
| **Summe** | | **6 424** | |

Bei der im Pilot gemessenen Größenordnung (Ziehungen zu ~0,75 s bei
96 neuen Token; hier 24) liegt die **gesamte Stufe B über alle sieben
Schriften bei grob ein bis zwei GPU-Stunden** (N_JE = 12 Runden,
Stapel zu 48 Reihen). Stufe A druckt die präzise
Hochrechnung aus dem eigenen Durchsatz (Abschnitt 3 des Protokolls).

**Zweitens entscheidet Stufe A selbst, ob die Explosion überhaupt
droht:** fällt sie `BUCHSTABEN-ADDITIV` aus, genügen in Stufe B
Buchstabenprofile plus Stichproben-Bigramme je Schrift (≈ Σ\|A\| statt
Σ\|A\|² Stimuli). Nur `PAARE-EIGEN` rechtfertigt volle Bigramm-Raster —
und dann auch nur dort, wo γ lebt. **Trigramme** (26³ = 17 576 allein im
Deutschen) sind erst dann ein Thema, wenn Paare tragen; sie stehen
bewusst in keiner Stufe dieser Mappe.

**Chinesisch ist kein Alphabet** und läuft deshalb nicht als siebte
Schrift mit: der chinesische Telegrafencode bildet ganze *Zeichen* auf
Vierergruppen von *Ziffern* ab, nicht Buchstaben auf Morsegruppen; Pinyin
wäre mit dem lateinischen Arm konfundiert, Bopomofo hat keinen
etablierten Morse. Chinesisch ist dafür der geplante **Kontrastfall** in
Stufe B: transkodiert das Netz auch dort, wo die Einheit kein Buchstabe
ist — oder ist es an alphabetische Einheiten gebunden? **Hangul** ist aus
dem umgekehrten Grund der interessanteste Prüfstein: die 14
Konsonanten-Jamo kodieren Artikulationsorte (Zungenwurzel, Zungenspitze,
Lippen, Zähne, Kehle — mit Strichzusatz für Aspiration), die 10 Vokale
dagegen folgen der Himmel/Erde/Mensch-Symbolik, nicht der Mundstellung.
Clustern die Konsonanten-Effekte α nach Artikulationsklassen — und die
Vokale als eingebaute Negativkontrolle eben nicht —, wäre das eine
merkmalsbasierte Organisation, die sich messen ließe, statt behauptet
zu werden.

Für jede Schrift gilt ein **Lebt-Tor** wie im Pilot: erst zeigen, dass
das Modell den jeweiligen Landescode überhaupt hinreichend oft gültig
produziert, dann interpretieren. Eine Schrift, die das Tor nicht nimmt,
liefert `MESSFELD-TOT` und keine Aussage.

## 4. Stufenplan

- **Stufe A — dieses Notebook**
  ([`stufeA_bigramm_feuer_colab.ipynb`](stufeA_bigramm_feuer_colab.ipynb)):
  lateinisches Alphabet, 676 Bigramme × N_JE Ziehungen, Decode-Feuerindex
  an allen Zielschichten (alle 256 Experten im Mitschnitt — die spätere
  Netzwerk-Suche liest die npz-Dateien, ohne neuen GPU-Lauf). Urteile je
  Experte plus Positionsbefund plus Durchsatz-Hochrechnung für Stufe B.
- **Stufe B — Schriftenvergleich:** dieselbe Messung je Schrift aus der
  Tabelle oben, Rahmen in der jeweiligen Aufgabensprache, Lebt-Tor je
  Schrift; Chinesisch als Kontrastfall (Telegrafencode statt Alphabet).
  Umfang hängt am Stufe-A-Urteil (additiv → Buchstabenprofile plus
  Stichprobe; paar-eigen → volles Raster der betroffenen Schriften).
- **Stufe C — Kausalprobe an Kandidatenzellen:** Maskierung des Netzes
  auf den Spitzenzellen aus A/B gegen ratengleiche Kontrollzellen
  (Apparat aus `phase18_kern`/Pilot unverändert). Erst hier werden aus
  Feuerkarten Aussagen über Mechanik.

Erst nach A–C ist die Ausgangsfrage — „hat das Netz eine eigene,
merkmals- oder paarbasierte Binnenorganisation?" — mit Daten statt
Analogie beantwortbar.

## 5. Dateien

| Datei | Inhalt |
|---|---|
| [`stufeA_bigramm_feuer_colab.ipynb`](stufeA_bigramm_feuer_colab.ipynb) | Stufe-A-Instrument, selbstversorgend (eine Zelle, Colab-A100) |
| [`tests/test_bigramm_logik.py`](tests/test_bigramm_logik.py) | Logiktests gegen Miniaturwelten mit gepflanzter Wahrheit (reines numpy, ohne GPU) |

Konventionen wie in Mappe 03: deterministische Saaten (FNV-1a, eigener
Namensraum `bigramm/<WIEDERHOLUNG>/`), Tore vor Urteilen, Permutation nur
über unabhängige Einheiten, Ergebnisse als Lauf-Ordner in Drive. Der
E228-Hintergrund in voller Tiefe:
[Dossier-Ordner in Drive](https://drive.google.com/drive/folders/1uYdeDjiPjpHDETAPVqY-5jwNXkXUsZX4?usp=sharing)
und [`../03_deepspec_draft_surprise/ANALYSE_AUFTRAG_L33_E228.md`](../03_deepspec_draft_surprise/ANALYSE_AUFTRAG_L33_E228.md).
