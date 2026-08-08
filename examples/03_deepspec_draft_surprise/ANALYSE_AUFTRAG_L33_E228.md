# Analyseauftrag: der Umkodier-Experte L33/E228 in Qwen3.6-35B-A3B

## Gegenstand

Im MoE-Modell `Qwen/Qwen3.6-35B-A3B-FP8` (40 Schichten, 256 Experten je Schicht, top-8;
auf A100 zu bf16 dequantisiert) traegt ein einzelner Experte - **Schicht 33, Experte 228** -
das zeichenweise Umkodieren bekannter Zeichenketten in konstruierte Symbolsysteme (Braille,
Morse). Kausal belegt durch Router-Maskierung in zwei unabhaengigen Durchgaengen: allein
gesperrt faellt Braille von 69 auf 23 % bzw. 65 auf 10 % und Morse von 85 auf 27 % bzw.
90 auf 19 %, waehrend die uebrigen 41 Paare einer kausal verifizierten 42er-Menge dort
zusammen wirkungslos bleiben (ueber 4000 gesperrte Router-Plaetze). Kana-Umschaltung laeuft
getrennt ueber die anderen 41; kyrillische Kontrollarme sind ueberall unberuehrt. Der blinde
Einzelscan gegen eine empirische Null aus 42 ratengleichen Fremdexperten fand zweimal genau
einen Treffer, zweimal dasselbe Paar, null von 84 fremden.

## Wo alles liegt

| | |
|---|---|
| Dossier (Daten, rund 4 GB) | https://drive.google.com/drive/folders/1uYdeDjiPjpHDETAPVqY-5jwNXkXUsZX4?usp=sharing |
| Code, der es erzeugt hat | https://github.com/Erikiss/WeirdChat/tree/claude/repo-published-weights-u71yew/examples/03_deepspec_draft_surprise |
| Checkpoint | https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8 |

`LIES_MICH.md` im Dossier beschreibt jede Datei, `INHALT.txt` listet alles mit Groessen.
Der Ordner ist Beleg und nicht Arbeitsverzeichnis: bitte nichts darin aendern, sondern
herunterladen und lokal arbeiten.

## Kernfrage 1 - Was aktiviert ihn?

Die Routergeometrie liegt in `04_gewichte/router_gewichte_L00-L39.safetensors` (Schluessel
`L33.router`, Form [256, 2048]; Zeile 228 ist die Auswahlrichtung). Die Empirie dazu in
`05_aktivierungen/`: je Arm und Beispiel die Eingaenge der MoE-Schicht 33 ([T, 2048], das
ist der Vektor, den der Router liest), die Routerlogits ([T, 256]), die top-8-Auswahl und
die Token-IDs; `e228_feuerindex.csv` sagt fuer jede Position, ob E228 gewaehlt wurde.

Zu klaeren: feuert er auf den Zeichen des Zielsystems, auf einem Umkodier-Zustand davor,
oder auf etwas Drittem? Und die offene Dissoziation: er kam ueber die Routing-Differenz an
der JAPANISCHEN Entscheidungsstelle in die 42er-Menge, feuert bei Kana-Produktion aber kaum
(14 bis 20 Plaetze je 48 Antworten). Fuer genau diese Frage liegen in
`05_aktivierungen/entscheidungsstelle_routerlogits.safetensors` die Routerlogits ALLER 40
Schichten an der Entscheidungsstelle, je Arm.

## Kernfrage 2 - Was schreibt er zurueck?

`04_gewichte/L33_E228_gewichte.safetensors` enthaelt `gate_up` ([1024, 2048]) und `down`
([2048, 512]). Die 512 Spalten von `down` sind seine Ausgaberichtungen. Gegen
`04_gewichte/einbettung_und_ausgabe.safetensors` (Ein- und Ausgabeeinbettung, Endnorm)
laesst sich pruefen, ob diese Richtungen Braille-/Morse-/Satzzeichen-Token direkt
verstaerken - oder ob er eine Zwischenrichtung schreibt, die spaetere Schichten lesen. Die
42er-Menge haeuft sich in den Schichten 34 bis 39 (`01_befund/die_42_paare.json`); ihre
Gewichte liegen in `04_gewichte/die_42_paare_gewichte.safetensors`.

## Kernfrage 3 - Ist er einzigartig?

Im Gewichtsraum: `04_gewichte/L33_alle_experten_gate_up.safetensors` und `_down` enthalten
alle 256 Experten der Schicht - ist E228 dort ein Ausreisser (Norm, Nachbarschaft,
Spektrum)? Im Verhaltensraum: der komplette, zweifach replizierte Einzelscan-Apparat steht
in `06_code/` (`phase18_kern.ipynb`). Der Goldstandard waere derselbe Scan ueber alle
10240 Paare am Braille-Arm - teuer; ein zweistufiger Aufbau (Grobscan mit wenigen
Ziehungen, Feinscan der Auffaelligen gegen die empirische Null) ist der gangbare Weg.

## Methodische Auflagen (Hausregeln dieser Untersuchung)

1. Jede neue Statistik zuerst gegen eine Miniatur mit eingebauter Wahrheit; eine Statistik,
   die eine gepflanzte Wirkung nicht findet, darf ihre Abwesenheit nicht berichten.
   Beispiele unter `06_code/` im Ordner `tests/`.
2. Empirische Nullen statt Formeln: ratengleiche Vergleichsexperten, Label-Shuffle,
   Vorzeichenpermutation auf UNABHAENGIGEN Einheiten (nicht auf autokorrelierten Schritten).
3. Positivkontrolle zuerst; Verdikte in fester Sperrreihenfolge; jeder Nullbefund mit
   ausgewiesener Aufloesung ("kein Effekt" heisst sonst nur "unter dem, was messbar war").
4. Post hoc Gefundenes als solches kennzeichnen und prospektiv bestaetigen. Vorbild: die
   Kettenrechnung, die L33/E228 vorhersagte, stand NICHT im Scan-Notebook - der Scan lief
   blind, und die Vorhersage fiel von selbst heraus.

## Fallstricke

- Die Gewichte im Dossier sind bf16 (dequantisiert) - exakt so hat jede Messung gerechnet.
  Das FP8-Original liegt unter https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8; falls Platz war, liegt eine Kopie unter
  `04_gewichte/vollstaendiger_checkpoint_fp8/`. Ohne sie genuegen fuer alle drei
  Kernfragen die Einzeltensoren in `04_gewichte/`.
- Prefill- und Decode-Routing stimmen nur zu rund 81 % ueberein (Phase 19, H5; mittlere
  Ueberlappung 7.80 von 8). Die Aktivierungsmitschnitte hier sind lehrergefuehrte
  Prefill-Laeufe - fuer Aussagen ueber das Erzeugen selbst neu messen.
- Die Erzeugungskette ist saatgesteuert und deterministisch; gleicher Durchgangswert
  reproduziert bitgleich. Unabhaengige Wiederholungen brauchen einen NEUEN Wert.
- Nicht jede Antwort trifft das Zielmass (Braille: rund 60 %); der Feuerindex traegt das
  Zielmass-Flag je Antwort.

## Erwartete Abgaben

1. Befundbericht je Kernfrage: Effektgroessen, Nullen, und ausdruecklich das, was gegen
   den eigenen Befund spricht.
2. Lauffaehiger Code fuer jede neue Messung (Colab, eine selbstversorgende Zelle - Muster
   in `06_code/` und unter https://github.com/Erikiss/WeirdChat/tree/claude/repo-published-weights-u71yew/examples/03_deepspec_draft_surprise).
3. Eine Liste dessen, was dieses Dossier NICHT hergibt und was ein naechster Lauf erheben
   muesste.
