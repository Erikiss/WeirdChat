<!-- Erzeugt aus einer Literatur- und Methodenrecherche mit acht parallelen
Rechercheagenten und je einer Pruefrunde (Workflow wf_18e220d6-e56, 13.09.2026).
Jede Aussage traegt eine Kennzeichnung: [V] an der Primaerquelle geprueft,
[E] eigene Rechnung aus geprueften Zahlen, [U] unsicher, [X] nicht im
Rechercheset. Vor dem Zitieren bitte die Kennzeichnung beachten. -->

# Methoden- und Literatur-Briefing: Suszeptibilitätsmessung an Pythia-1.4b und die WASD-Hypothese

**Stand:** 13.09.2026 · **Gegenstand:** EleutherAI/pythia-1.4b, Checkpoints step97000–step100000 · **Status der laufenden Linie:** drei negative Vorläufe (reine Text-/Tokenizer-Statistik, kein Forward-Pass)

Kennzeichnung im Text: **[V]** primärquellenverifiziert · **[E]** eigene Rechnung aus verifizierten Zahlen · **[U]** unsicher/unbelegt · **[X]** nicht im geprüften Rechercheset, vor Zitat zu verifizieren

---

## 1. Was der Suszeptibilitätsatlas formal misst — und was er nicht zeigt

### 1.1 Die formale Größe

Die Suszeptibilität ist **keine Aktivierungsgröße und kein Forward-Pass-Maß**. Sie ist eine Posterior-Kovarianz im Gewichtsraum.

**[V]** Baker, Wang, Hoogland, Murfet (arXiv:2504.18274, Def. 2.1):

$$\chi = \frac{1}{n\beta}\cdot\frac{\partial}{\partial h}\langle\varphi\rangle_{\beta,h}\Big|_{h=0}, \qquad p_n^\beta(w|h) = \frac{1}{Z_n^{\beta,h}}\exp\{-n\beta L^h(w)\}\varphi(w)$$

mit der Datenmischung $q_h = (1-h)q + h q'$. Lemma 2.2 liefert die für die Praxis maßgebliche Identität:

$$\chi = -\mathrm{Cov}_\beta[\varphi, \Delta L]$$

Für komponentenlokalisierte Per-Token-Suszeptibilitäten (Def. 2.4/2.5):

$$\varphi_C(w) = \delta(u-u^*)\,[L(w)-L(w^*)], \qquad \chi^C_{(x,y)} = -\mathrm{Cov}_\beta(\varphi_C,\ \ell_{(x,y)} - L)$$

Gesampelt wird aus dem lokalisierten Gibbs-Posterior (Gl. 9/10):
$$p(w; w^*,\beta,\gamma,h) \propto \exp\{-n\beta L^h(w) - \tfrac{\gamma}{2}\|w-w^*\|^2\}$$

Die Perturbation $q'$ ist im Per-Token-Fall eine **Punktmasse auf dem Paar $(x,y)$**; der Mischungsanteil ist bei Baker et al. **[V]** auf $\delta h = 0.1$ fixiert. Für die Pythia-Läufe in Tabelle 5 von arXiv:2601.12703 ist $\delta h$ **[U]** nicht angegeben.

### 1.2 Der Schätzer und sein blinder Fleck

**[V]** „How to Scale Susceptibilities" (Gordon, Adam, Hoogland, Murfet, timaeus.co, 21.04.2026) definiert den **Zwei-Ketten-Schätzer**: eine restringierte Kette, die nur die Komponentengewichte $c$ sampelt ($u$ auf $u^*$ geklemmt), und eine unrestringierte Kette; mit $g(c) = L_n(u^*,c) - L_n(w^*)$.

**Der zentrale Vorbehalt:** Der Schätzer zielt wörtlich auf „the renormalized susceptibility $Z_\text{full}/Z_C \cdot \chi^C_{xy}$ **not for the population susceptibility itself**". Der Faktor ist „C-dependent (but xy-independent)" und wird „absorbed by the column z-scoring step". Unabhängig abgesichert durch Elliott & Murfet (arXiv:2605.07970, Gl. 1 §3.1, Remark 3.5, Thm. 3.21/4.8).

**Daraus folgt hart:** Jede Aussage der Form „Layer 14 und 20 reagieren stärker" ist **auf Rohwerten mathematisch bedeutungslos**, weil der unbekannte Vorfaktor genau zwischen Komponenten variiert. Nur nach spaltenweiser Standardisierung sind Komponenten vergleichbar. Das trifft die alte McQuarrie-Linie („Effekt in mittleren Layern 14 und 20") in ihrem Kern.

**[V]** Zusätzlich entfernt die in „Towards Spectroscopy" dokumentierte **Zeilenzentrierung** („The row standardization removes the uniform mode") per Konstruktion genau die Aussage „WASD ist global suszeptibler". Auswertbar bleibt ausschließlich das **relative Profil über die 410 Komponenten**.

**[V]** Vorzeicheninterpretation gilt erst auf der standardisierten Suszeptibilität $\psi_{xy}$: $\psi<0$ = Expression, $\psi>0$ = Suppression (§3.2).

### 1.3 Was der Atlas nicht ist

| Behauptung | Status |
|---|---|
| „Atlas" ist ein Fachbegriff | **[V] Falsch.** 0 Treffer in arXiv:2601.12703, in „How to Scale Susceptibilities", im Primer arXiv:2605.07980 und auf der Spectroscopy-at-Scale-Seite. Korrekt: *susceptibility matrix*, *response matrix*, *structural susceptibility matrix*, *cluster map*, *clusters → neighborhoods → regions*. Eigenprägung ist zulässig, muss aber als solche gekennzeichnet werden. |
| Cluster sind Modelleigenschaften | **[V] Nein.** 510 (14M) vs. 241 (410M) vs. 249 (1.4B) bei gleicher Pipeline — aber **57.236** (1.4B) nach ~10× mehr Token und PCA-Whitening. Die Clusterzahl ist eine Pipeline-Eigenschaft. |
| Interpretierbare Cluster belegen Loss-Landschaftsstruktur | **[V] Nein.** Die Autoren selbst zur Gauß-Baseline: „despite containing no contribution from the loss landscape geometry, this baseline still yields interpretable clusters (that is, the Gaussian baseline is nontrivial)." |
| Suszeptibilität ist kausal | **[V] Nein.** $\chi$ ist eine Kovarianz. Spectroscopy at Scale nennt fehlende Steering-Validierung auf dieser Modellgröße explizit als offene Limitation. |
| Der Atlas gilt für step97000–100000 | **[U] Unbekannt.** Weder arXiv:2601.12703 noch die Spectroscopy-at-Scale-Seite nennen einen Checkpoint. Vermutlich step143000/main — unbelegt. Die Übertragung auf 97k–100k ist eine unvalidierte Extrapolation über ~30 % des Trainings. |

**[V]** Umfang der publizierten 1.4B-Messung: $H=410$ Komponenten (384 Attention-Heads + 24 MLP-Layer + Embedding + Unembedding; **[E]** $24\times16+24+2=410$, exakt passend zur Architektur), 4,25 Mio. Ground-Truth-Vektoren zu 4800 H200-Stunden, Upsampling per unspezifiziertem Aux-Modell **[U]** auf 46 Mio. Vektoren, 19 annotierte Regionen.

### 1.4 Konsequenz für den aktuellen Projektstand

Die drei neuesten Läufe enthielten **keinen Forward-Pass**. Sie können daher **prinzipiell keine Aussage über Suszeptibilitäten** treffen — weder positiv noch negativ. Was sie widerlegen, ist eine *textstatistische* Version der Hypothese. Das ist trotzdem relevant: Sie entziehen der Hypothese ihre billigste Motivation. Formulierungen wie „WASD-Suszeptibilität ist nicht erhöht" wären auf dieser Datenbasis falsch; korrekt ist: „Die Buchstabendichte W,A,S,D,Q,E,R,F in den geprüften Textfenstern liegt im Nullbereich."

---

## 2. Pythia-1.4b und seine Checkpoints: was benachbarte Checkpoints unterscheidet

### 2.1 Verifizierte Architektur (direkt aus `config.json`) **[V]**

| Feld | Wert |
|---|---|
| `num_hidden_layers` | 24 |
| `hidden_size` | 2048 |
| `num_attention_heads` | 16 → $d_\text{head}=128$ |
| `intermediate_size` | 8192 |
| `rotary_pct` | 0.25 → **32 von 128** Head-Dimensionen tragen RoPE, 96 sind positionsfrei |
| `use_parallel_residual` | **true** |
| `tie_word_embeddings` | **false** |
| `torch_dtype` | **float16** |
| `vocab_size` | 50304 |
| Parameter | 1.414.647.808 gesamt / 1.208.602.624 non-embedding |

**[E]** Differenz $1.414.647.808 - 1.208.602.624 = 206.045.184 = 2 \times 50304 \times 2048$ — untied Embedding/Unembedding auch rechnerisch belegt.

**Verwechslungsfalle [V]:** `pythia-1b` hat `hidden_size` 2048 und `intermediate_size` 8192 — **identisch** zu 1.4b — und unterscheidet sich nur in `num_hidden_layers` (16) und `num_attention_heads` (8). Eine Verwechslung liefert identisch dimensionierte Aktivierungen und **schlägt still fehl**.

### 2.2 Checkpoint-Arithmetik für das Projektfenster **[V]/[E]**

Batch = $1024 \times 2048 = 2.097.152$ Token pro Schritt; 143.000 Schritte = 299.892.736.000 Token.

| Revision | Token | Anteil am Training |
|---|---|---|
| step97000 | 203.423.744.000 | 67,83 % |
| step98000 | 205.520.896.000 | 68,53 % |
| step99000 | 207.618.048.000 | 69,23 % |
| step100000 | 209.715.200.000 | 69,93 % |

Ein Intervall = 2,097 Mrd. Token = 1.024.000 Sequenzen = **0,699 %** des Trainings. Das gesamte Fenster umfasst **2,10 %** des Trainings.

### 2.3 Was benachbarte Checkpoints tatsächlich trennt

Zwischen step98000 und step99000 liegen genau vier Dinge:

1. **1000 SGD-Schritte auf 2,097 Mrd. neuen, deterministisch festgelegten Token.** **[V]** Pythia-Datenreihenfolge ist öffentlich und rekonstruierbar (`utils/unshard_memmap.py`, `utils/batch_viewer.py`). Ein „Peak" bei 98k→99k kann schlicht abbilden, **was in diesen Batches lag**.
2. **Ein Stück Cosine-Decay der Lernrate** (max. LR 2.0e-4 **[V]**; Warmup-Anteil und min-LR für 1.4b **[U]** nicht primärbelegt — aus der YAML im `models/`-Verzeichnis zu holen, nicht aus der Model Card).
3. **Bei `-deduped`: die Epochengrenze.** **[E]** $207\cdot10^9 / 2.097.152 \approx$ **Schritt 98.705** — mitten im Fenster. Wer `pythia-1.4b-deduped` verwendet, konfundiert jeden 98k→99k-Unterschied mit dem Beginn des zweiten Datendurchlaufs. *Die Repo-Wahl ist zu benennen und zu begründen.* (Die 207B-Zahl steht in der GitHub-README, **nicht** auf der HF-Model-Card **[V]**.)
4. **Nichts sonst.** Kein publiziertes Entwicklungsereignis.

### 2.4 Die Literaturlage zum Fenster ist ungünstig

- **[V]** Urdshals et al. (arXiv:2510.12077) **schließen genau diesen Bereich aus**: Checkpoints „ranging from 2k to 90k, excluding later checkpoints because of apparent instability in the original training runs". Die Instabilität wird dort **nicht quantifiziert [U]**.
- **[V]** Alle dokumentierten Entwicklungsereignisse in Pythia liegen früher: Induction-Bildung ab ~1000 Schritten, Scheitel ~30k (Lee et al. arXiv:2510.12071, korrigiert gegen die falsche Angabe „~128 Schritte"; konsistent mit Tigges et al. 2024).
- **Aber [V]:** Das Pythia-Paper berichtet „a significant phase change occurs after 65,000 training steps (45% through training)" — für Modelle ab 2.8B. **Späte Übergänge existieren also.** Die Annahme „bei 98k ist nur graduelle Konsolidierung" ist eine Plausibilitätsannahme, kein Befund, und darf nicht als Begründung dienen, Sprunghaftigkeit gar nicht erst zu testen.
- **[V]** Bayazit et al. (arXiv:2509.05291, Crosscoding Through Time) zeigen an Pythia-1B: zwischen 4B und 286B Token entstehen weiterhin Features mit gezielten linguistischen Funktionen, „although overall performance plateaus". Plateau im Loss ≠ Stillstand in der Repräsentation.

### 2.5 Was daraus methodisch folgt

- **Vier Checkpoints sind keine Entwicklungsreihe.** **[V]** Die Stadiendetektion nach Hoogland et al. definiert Grenzen als Nullstellen der Ableitung von $\hat\lambda$ nach $\log t$. Vier linear benachbarte Punkte können weder ein Plateau noch eine Ableitungsnullstelle identifizieren. Die Wörter „Phasenübergang", „Stadium", „Entwicklungsgrenze" sind im SLT-Sinn zu vermeiden, solange nicht ≥20 Punkte im Zielfenster plus log-spaced Punkte über die Trajektorie vorliegen.
- **Rauschboden zuerst.** Dieselbe Messung über ein breiteres Raster (z. B. 90k–110k in 1000er-Schritten, plus einen Kontrollbereich 60k–63k) liefert die Checkpoint-zu-Checkpoint-Referenzvarianz. Erst gegen diese darf 98k→99k als „besonders" bezeichnet werden.
- **Hyperparameter-Sensitivität ist von derselben Größenordnung wie der Effekt.** **[V]** Wang et al. (arXiv:2410.02984): eine Änderung von $n\beta$ 23→30 verschob eine Stadiengrenze von „6.5k–8.5k" auf „5.5k–8k" — **ca. 1000 Trainingsschritte**, also genau der hier untersuchte Checkpoint-Abstand. Die Autoren schreiben selbst, das sei „more likely to be due to changes in the hyperparameters used for LLC estimation".
- **Ein Hyperparametersatz, fixiert über alle Komponenten und Checkpoints [V]** (Sampling Guide). Für 1.4b existieren zwei publizierte, **nicht kompatible** Sätze:

| Zweck | Quelle | $\gamma$ | $n\beta$ | $\varepsilon$ | Batch | Ketten | Draws | Steps b/w |
|---|---|---|---|---|---|---|---|---|
| Suszeptibilitäten | Gordon et al., Tab. 5 | 300 | 10 | 1e-5 | 16 | 4 | 100 | 160 |
| LLC | Urdshals et al. | 300 | 30 | 3e-5 | 32 | 4 | 100 | — |

Wer Suszeptibilitäten misst, nimmt den Suszeptibilitätssatz (pSGLD/RMSProp) — und dokumentiert die Wahl, statt zu mischen.
- **`-v0`-Repos ausschließen [V]:** Für 1.4B existieren v0-Varianten, bei denen „step1000 … was actually step 500". Revisionsname ↔ Tokenzahl ist konsistent, Revisionsname ↔ Optimizer-Schritt **nicht**. Revision und Commit-Hash protokollieren.

---

## 3. Tokenizer-Artefakte und Zeichenwissen — und was für WASD daraus folgt

### 3.1 Die verifizierte Tokenisierung **[V]** (BPE-Merge-Algorithmus auf der echten `tokenizer.json` nachgebaut)

| String | Zerlegung | IDs | #Tokens |
|---|---|---|---|
| `WASD` | `['W','AS','D']` | 56, 1719, 37 | 3 |
| `' WASD'` | `['ĠWAS','D']` | 22250, 37 | **2** |
| `wasd` | `['was','d']` | 4238, 69 | 2 |
| `QERF` | `['Q','ER','F']` | 50, 947, 39 | 3 |
| `' QERF'` | `['ĠQ','ER','F']` | 1165, 947, 39 | **3** |
| `QWERTY` | `['QW','ERTY']` | — | 2 |
| `ASDW` | `['AS','DW']` | — | 2 |
| `ESDF` | `['ES','DF']` | — | 2 |

Weder `WASD`, `QERF`, `ASD`, `QE`, `ERF` noch `QWERTY` existieren als Einzeltoken.

Vokabular **[V]**: 50.254 BPE-Einträge (max. ID 50253), 50.009 Merges, ID 0 `<|endoftext|>`, ID 1 `<|padding|>`, 23 Whitespace-Run-Tokens auf 50254–50276. Höchste belegte ID = 50276 → **50.277 nutzbare IDs**, **27 tote Zeilen** (50277–50303) in der 50304-zeiligen Embedding-/Unembedding-Matrix. Achtung: das `added_tokens`-Array enthält 25 Einträge (inkl. IDs 0 und 1); naives `50254 + 25 = 50279` ist falsch.

### 3.2 Der dominante Konfund: `was`

Dies ist der schwerwiegendste Einzelbefund für die Hypothese.

- `' WASD'` zerfällt in `['ĠWAS','D']` — **`ĠWAS` ist das englische Wort ` WAS` in Großschreibung**.
- `wasd` zerfällt in `['was','d']` — **`was` ist ein hochfrequentes englisches Hilfsverb**.

Ein beobachteter „Sondereffekt von WASD in mittleren Layern" ist damit **per Default als Verarbeitung des Wortes *was* zu erklären**, nicht als Tastatureffekt. Das ist kein Randeffekt; es ist die sparsamste konkurrierende Erklärung für die gesamte Hypothese.

Zwingende Gegenmaßnahmen:
1. Vergleich gegen `' WASX'`, `' WASQ'`, `' WASB'`, … — zeigt, ob ausgerechnet `D` etwas Besonderes tut.
2. Vergleich gegen die leerzeichenlose Variante `WASD` → `['W','AS','D']`, die den `WAS`-Token **nicht** enthält.
3. **Vorab festlegen:** Die Hypothese gilt als widerlegt, wenn der Effekt bei `' WASX'` in gleicher Höhe auftritt.

### 3.3 QERF ist als Kontrolle ungeeignet **[V]**

Ohne Leerzeichen sind beide 3-Token-Strings; **mit** Leerzeichen wird `' WASD'` zu 2 Tokens, `' QERF'` bleibt bei 3. Unterschiedliche Tokenzahl verschiebt sämtliche Residual-Stream-Positionen — man vergleicht dann Position gegen Position, nicht Layer gegen Layer. Kontrollen müssen matchen in: **Tokenanzahl, Zerlegungsmuster** (Einzelbuchstabe + Bigramm + Einzelbuchstabe), Groß-/Kleinschreibung, Byte-Länge, An-/Abwesenheit des führenden Leerzeichens.

### 3.4 Detokenisierung: der Effekt, den die Hypothese vorhersagt, ist der Normalfall

**[V]** Die Literatur zu Zeichenwissen (arXiv:2506.10641) dokumentiert „a distinct breakthrough in their spelling behavior" in **mittleren bis oberen Layern**; die Embedding-Schicht kodiert Zeichen jenseits des ersten kaum. Jede mehrfach zerlegte Großbuchstabenkette aktiviert diese Rekonstruktionsmaschinerie **in genau den mittleren Layern — unabhängig von der Tastaturbelegung**.

Die alte McQuarrie-Linie (Effekt in Layern 14 und 20 bei `Mc|Qu|ar|rie`, also ebenfalls einer mehrfach zerlegten Eigennamenkette) ist damit als **Detokenisierungssignatur** erklärbar, ohne dass ein WASD- oder McQuarrie-spezifischer Mechanismus nötig wäre. Wer diesen Konfund nicht kontrolliert, misst Detokenisierung und nennt es WASD.

### 3.5 Glitch-/Under-trained-Token-Screening ist vorzuschalten

**[V]** GlitchProber (arXiv:2408.04905) beschreibt bei Glitch-Token „significant deviations in the distributions of attention patterns and dynamic information from intermediate model layers" — also **exakt die Signatur, die die Hypothese erwartet**. Mechanismus (arXiv:2404.09894, arXiv:2410.15052): Mismatch zwischen Tokenizer-Trainingskorpus und LM-Pretraining-Korpus → geometrisch isolierte, untertrainierte Embeddings.

Pflichtdiagnostik vor jeder SLT-Deutung, für alle IDs 56, 1719, 37, 50, 947, 39, 22250, 1165, 4238, 69:
- Korpusfrequenz in der Pile (bzw. im deduped-Split)
- Embedding-Norm relativ zur Vokabularverteilung
- Isolation im Embedding-Graph (Leiden-Clustering)
- Maximale Prädiktionsentropie (GlitchMiner-Kriterium)
- Indikatoren nach Land & Bartolo (arXiv:2405.05417) **[U]** — Pythia-1.4b-Ergebnisse dort nicht in verifizierbarer Form gelistet

### 3.6 Literatursuch-Falle **[V]**

arXiv:2603.18474 „WASD: Locating Critical Neurons as Sufficient Conditions for Explaining and Controlling LLM Behavior" ist ein **Akronym** („unWeaving Actionable Sufficient Directives", Gemma-2-2B, SST-2/CounterFact) und hat mit Tastaturbuchstaben nichts zu tun. **Nicht als Vorarbeit zitieren.**

### 3.7 Zwischenfazit

Es existiert **[V]** keine publizierte Arbeit zu WASD, QERF oder Tastaturlayout-Zeichenketten in Sprachmodellen — weder in der SLT-/devinterp-Literatur noch in der Glitch-Token-Literatur. Die Hypothese ist unbelegt und unwiderlegt; sie hat keinen Prior. Das macht sie nicht falsch, erhöht aber die Beweislast: Mindestens drei etablierte Alternativerklärungen (`was`-Token, Detokenisierung, Untertrainiertheit) müssen **vor** jeder mechanistischen Deutung ausgeschlossen sein.

---

## 4. Sprachwechsel und Selbstreparatur: zwingende Kontrollen

### 4.1 „Sprachwechsel in mittleren Layern" ist die Baseline, nicht der Befund

**[V]** Die einschlägige, SLT-unabhängige Literatur (arXiv:2402.16438 *Language-Specific Neurons*, ACL 2024; arXiv:2509.17030 *Transfer Neurons Hypothesis*; arXiv:2505.05111) ist sich einig: sprachspezifische Neuronen sitzen überwiegend in den **obersten und untersten** Layern, während **mittlere Layer generell einen englischzentrierten Pivot-/Interlingua-Raum bilden** (vgl. auch arXiv:2402.10588).

Ein Mittellayer-Sprachwechseleffekt ist damit für **beliebige** Token zu erwarten. Er ist kein WASD-Spezifikum.

**Zusätzliches Problem:** Pythia ist auf dem Pile trainiert, der englischdominiert ist. Ein „Wechsel" setzt eine zweite, im Modell überhaupt repräsentierte Sprache voraus.

**Zwingende Kontrollen:**

| # | Kontrolle | Zweck |
|---|---|---|
| S1 | **Vorregistrierte quantitative Definition** von „Sprachwechsel": z. B. Verschiebung der Sprach-ID-Massenverteilung über das Vokabular unter Tuned Lens, mit fixem Schwellwert | ohne Metrik ist jede UMAP-Karte nur Illustration |
| S2 | **Positivkontroll-Stimulus**, bei dem ein Sprachwechsel unstrittig auftritt (echtes Code-Switching) | kalibriert die Metrik; ohne sie ist sie bedeutungslos |
| S3 | **Frequenzgematchte Negativkontrollen** mit identischem Zerlegungsmuster | trennt WASD von „seltene Großbuchstabenfolge" |
| S4 | **Tuned Lens statt Logit Lens** | `tie_word_embeddings=false` **[V]** — die naive Logit-Lens-Annahme ist hier nicht garantiert gültig |
| S5 | Messung über **alle 24 Layer**, nicht nur die vermuteten | verhindert Post-hoc-Layerwahl |
| S6 | Vergleich gegen die **Aktivierung sprachspezifischer Neuronen** nach arXiv:2402.16438 | prüft, ob überhaupt der etablierte Mechanismus beteiligt ist |

**Falsifikationsschwelle:** Der WASD-Effekt muss die Baseline, die frequenzgematchte Kontrolltoken in denselben Layern erzeugen, **quantitativ übersteigen**. Gleichauf = widerlegt.

### 4.2 Zwei verschiedene Dinge heißen „Fehlerkorrektur"

Das Projekt verwendet den Begriff mehrdeutig. Er ist zu trennen:

**(a) Behaviorale Typo-Korrektur.** Vorab messbar zu operationalisieren, z. B.:
- Recovery-Rate nach eingefügtem Tippfehler
- Logit-Differenz korrigiertes vs. korrumpiertes Token
- Perplexitätsdelta auf gestörten Sequenzen

Der interne Mittellayer-Befund muss diese Größe **vorhersagen**. Eine geometrische Auffälligkeit ohne behaviorale Entsprechung ist keine Fehlerkorrektur, sondern eine unbenannte Aktivierungsanomalie.

**(b) Mechanistische Selbstreparatur (self-repair / Hydra-Effekt / backup heads).** **[X]** — diese Literatur ist **nicht Teil des geprüften Rechercheresultats** und vor Zitat im Volltext zu verifizieren: McGrath et al., *The Hydra Effect* (arXiv:2307.15771); Rushing & Nanda, *Explorations of Self-Repair in Language Models* (arXiv:2402.15390); Wang et al., IOI / backup name mover heads (arXiv:2211.00593).

Der Kern für dieses Projekt: **Ablation einer Komponente wird teilweise von anderen Komponenten kompensiert**, und ein substantieller Teil der scheinbaren Kompensation ist ein reiner LayerNorm-Effekt — die Ablation senkt die Residual-Norm, LN reskaliert, die Logit-Beiträge der übrigen Komponenten wachsen. **[U]** Größenordnung des LN-Anteils in der Literatur uneinheitlich; vor Zitat nachzuschlagen.

**Konsequenz:** Ablationsbasierte Kausaltests **unterschätzen** Komponentenwichtigkeit systematisch. Ein Nulleffekt bei Ablation von Layer 14/20 widerlegt deren Beteiligung **nicht**. Erforderlich sind daher:
- **Mehrfach-Ablation** (Komponente + ihre plausiblen Backups gemeinsam)
- **LN-Einfrieren** in einer Kontrollbedingung (LN-Skalen auf den Clean-Werten festhalten), um den LN-Anteil der Kompensation abzuziehen
- Getrennter Bericht von **Denoising** (Clean→Corrupt patchen) und **Noising** (Corrupt→Clean patchen); die beiden Richtungen können auseinanderfallen

---

## 5. Patching-Methodik: Checkliste

Grundlage: Zhang & Nanda (arXiv:2309.16042, ICLR 2024) — bei gleichem Modell und gleicher Aufgabe führen unterschiedliche Metrik- und Korruptionswahlen zu **unterschiedlichen Schlüssen darüber, welche Komponenten wichtig sind**.

**Vor dem ersten Forward-Pass**

1. **Modellidentität assertieren.** Nach dem Laden hart prüfen: `n_layers == 24`, `n_heads == 16`, `hidden_size == 2048`, `rotary_pct == 0.25`, `use_parallel_residual == True`, `tie_word_embeddings == False`. Grund **[V]**: `pythia-1b` unterscheidet sich nur in zwei Feldern und liefert identisch dimensionierte Aktivierungen — Verwechslung schlägt still fehl.
2. **Revision, Commit-Hash und Repo-Variante loggen.** Explizit protokollieren: kein `-v0`-Repo; `-deduped` ja/nein (Epochengrenze bei **[E]** ≈ Schritt 98.705 liegt im Fenster).
3. **dtype auf fp32 oder bf16 setzen.** Default ist `float16` **[V]**. In fp16 fallen Mittellayer-Unterschiede potenziell in die Rundung (siehe §6).
4. **Determinismus erzwingen und dokumentieren.** `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.backends.cuda.matmul.allow_tf32=False`, `torch.backends.cudnn.allow_tf32=False`, feste Seeds, feste Batchgröße, feste Sequenzlänge, SDPA-Backend explizit wählen.
5. **Tokenisierung aller Stimuli und Kontrollen protokollieren** (String → Tokenliste → IDs → Positionen). Ein Experiment, das dies nicht berichtet, ist nicht bewertbar.
6. **Präregistrierung.** Metrik, Effektgröße, Schwellwert, Layer-/Komponentenmenge, Falsifikationskriterium und Analysepfad **schriftlich fixieren, bevor Daten angesehen werden**. „Mittlere Layer" ist als Indexmenge zu definieren (z. B. Layer 8–15), nicht post hoc zu wählen.

**Design**

7. **Matched Controls, nicht Einzelstring.** Mindestens vier vorab definierte Klassen: (a) frequenzgematchte Zufallstoken; (b) Permutationen derselben Buchstaben (`ASDW`, `SDAW`, `DWSA`) — **bei identischem Zerlegungsmuster**; (c) tastaturfreie Großbuchstabenketten (`XKPN`, `MRLV`); (d) andere Tastatur-Strings (`HJKL`/vim, `ZQSD`/AZERTY, `IJKL`, `ESDF`). Plus die `' WASX'`-Serie aus §3.2.
8. **$N > 1$ auf beiden Seiten.** WASD gegen QERF ist 1-gegen-1 und erlaubt keine Statistik. Nötig ist eine Nullverteilung aus mehreren Dutzend gezogenen Kontrollketten mit identischem Zerlegungsmuster; der WASD-Effekt ist als **Effektstärke plus Perzentilrang** gegen diese Verteilung auszuweisen.
9. **Mehrere Korruptionsbaselines nebeneinander berichten** (Zero-Ablation, Mean-Ablation über den Datensatz, Resample-Ablation aus Kontrollprompts). Einigkeit über Baselines hinweg ist Teil des Befunds; Uneinigkeit ist berichtspflichtig.
10. **Denoising und Noising getrennt berichten.** Die Richtungen können divergieren; nur eine zu berichten ist selektiv.
11. **Positionen und Trägerkontexte variieren.** RoPE wirkt nur auf **32 von 128** Head-Dimensionen **[V]**; 96 sind positionsfrei. Derselbe Stimulus ist an mehreren Sequenzpositionen und in mehreren Trägerkontexten zu testen. Ein Effekt, der nur an einer Position auftritt (vgl. die alte Position-124..127-Linie), ist ein Positionsartefakt, bis das Gegenteil gezeigt ist.
12. **Parallele Residualstruktur respektieren.** `use_parallel_residual=true` **[V]**: Attention und MLP lesen beide $\mathrm{LN}(x)$ **desselben** Residualzustands und schreiben additiv zurück. Es gibt innerhalb eines Layers **kein „MLP nach Attention"**. Jede Aussage der Form „Layer $L$ Attention bereitet vor, was Layer $L$ MLP verarbeitet" ist für Pythia architektonisch falsch. Patching-Rezepte aus GPT-2-Arbeiten übertragen sich nicht 1:1.
13. **Komponentenauflösung statt Layer-Blob.** Die 410er-Partition nutzen (384 Heads + 24 MLPs + Embed + Unembed), in devinterp über `create_param_masks` mit Mustern wie `l12h3`, `l12 attn`, `l12 mlp`; `weight_restrictions` muss zwingend den Eintrag `'full'` enthalten **[V]**. Für rLLC-artige Größen gilt die Selbsteinschränkung: außerhalb von Deep Linear Networks ist nur **Ordinalität** beanspruchbar, kein Absolutwert.
14. **Selbstreparatur adressieren** (§4.2): Mehrfach-Ablation plus LN-eingefrorene Kontrollbedingung. Ein Nulleffekt bei Einzel-Ablation ist kein Gegenbeweis.
15. **Die 27 toten Embedding-Zeilen maskieren.** Zeilen 50277–50303 sind untrainiert **[V]**. Jede Analyse, die über die volle 50304er-Matrix sweept — Nachbarschaftssuche, Logit-Lens-Top-$k$, Cluster-Statistik — muss sie ausschließen, sonst fließt Initialisierungsrauschen als Signal ein.

**Auswertung**

16. **Rauschboden vor Effekt.** Identische Wiederholungsläufe und Checkpoint-Referenzvarianz (§2.5, §6) sind zu messen und der Effekt daran zu normieren.
17. **Multiplizität korrigieren.** Der Suchraum ist 410 Komponenten × 24 Layer × 4 Checkpoints × $N$ Stimuli (× ggf. Cluster). FDR oder Bonferroni über die **tatsächlich durchsuchte** Menge, mit Angabe dieser Menge.
18. **Exploration von Bestätigung trennen.** Hypothese auf einem Explorationssplit finden, einfrieren, auf einem **disjunkten Halteset** bestätigen. Explorativ gefundene Effekte als solche kennzeichnen.
19. **Konfidenzintervalle über drei Quellen:** unabhängige SGLD-Ketten, unabhängige Datenbatches, unabhängige Seeds (`init_seed`, `obs_seed`). Ein Peak kleiner als die Ketten-zu-Ketten-Streuung existiert nicht.
20. **Negativergebnisse vollständig berichten**, inkl. aller negativen $\hat\lambda$-Schätzungen (nicht filtern) und aller nicht bestandenen Kontrollen.
21. **Keine Kausalaussage ohne Intervention.** Suszeptibilität ist Kovarianz. „Löst Sprachwechsel aus", „löst Fehlerkorrektur aus" sind Verhaltensbehauptungen und brauchen Ablation/Activation-Patching genau der identifizierten Komponenten mit Messung der **vorab definierten** Verhaltensgröße.

---

## 6. Numerik: hardwareabhängiges Rauschen von echtem Verhalten trennen

Die McQuarrie-Linie vermutete einen „hardwareabhängigen Effekt in den mittleren Layern 14 und 20". Das ist die klassische Signatur numerischen Rauschens. Hier die Größenordnungen, um beides zu trennen.

### 6.1 Relative Maschinengenauigkeit

| Format | Mantissenbits | $\epsilon = 2^{-(p)}$ | Größenordnung |
|---|---|---|---|
| fp16 | 10 (+1 implizit) | $2^{-11}$ | **4,88 · 10⁻⁴** |
| bf16 | 7 (+1) | $2^{-8}$ | **3,91 · 10⁻³** |
| TF32 (Ampere+) | 10 (+1) | $2^{-11}$ | **4,88 · 10⁻⁴** |
| fp32 | 23 (+1) | $2^{-24}$ | **5,96 · 10⁻⁸** |
| fp64 | 52 (+1) | $2^{-53}$ | **1,11 · 10⁻¹⁶** |

Pythia-1.4b lädt per Default in **fp16** **[V]**.

### 6.2 Fehlerfortpflanzung im Modell **[E], Größenordnungen**

- **Reduktionslänge.** Ein Attention-/MLP-GEMM summiert über $d_\text{model}=2048$ bzw. $d_\text{ff}=8192$. Bei Baum-/paarweiser Summation wächst die Fehlerschranke wie $O(\log_2 n \cdot \epsilon)$: $\log_2 2048 = 11$, $\log_2 8192 = 13$. Mit fp32-Akkumulation (cuBLAS-Standard bei fp16-Inputs) dominiert der **Eingangsrundungsfehler**, also ~$5\cdot10^{-4}$ relativ pro GEMM.
- **Tiefe.** 24 Layer × 2 Teilblöcke ≈ 48 additive Beiträge in den Residual-Stream. Der Residual-Stream wächst in der Norm mit der Tiefe; **absolute** Abweichungen wachsen mit, **relative** bleiben näherungsweise flach.
- **Praktisch beobachtete Logit-Abweichungen** zwischen zwei Kernel-/Batchgrößen-/GPU-Varianten bei fp16-Inferenz: **10⁻³ bis 10⁻²** in Logit-Einheiten. **[U] Erfahrungswert, im Projekt selbst zu messen, nicht anzunehmen.**
- **Argmax-Kipppunkt.** Bei 50.277 realen Vokabeleinträgen ist der Top-2-Logit-Abstand häufig $<10^{-2}$. Ein Rauschen dieser Größe kippt dann die Vorhersage — **ohne dass sich am Modell irgendetwas geändert hätte**.

### 6.3 Die drei Hauptquellen scheinbarer Hardwareabhängigkeit

1. **Batchgröße ändert die Reduktionsreihenfolge.** cuBLAS wählt Kernel und Split-k-Zerlegung abhängig von den Matrixdimensionen. Derselbe Prompt in Batch 1 vs. Batch 32 liefert **numerisch verschiedene Logits**. Dies ist die häufigste Ursache angeblicher „Hardwareeffekte".
2. **TF32 auf Ampere/Hopper.** Standardmäßig aktiv für fp32-Matmuls; senkt die effektive Mantisse von 24 auf 11 Bit — Faktor ~8000 mehr Rundungsfehler bei nominell „fp32"-Rechnung.
3. **Nicht-deterministische Kernels** (Atomics in Scatter-/Backward-Ops, Flash-Attention-Blockgrößen, cuDNN-Autotuning).

### 6.4 Protokoll zur Trennung

**Schritt 1 — Rauschboden bestimmen.** Denselben Stimulus $R \ge 20$ mal messen, dabei **nur** variieren: (a) Batchgröße (1, 8, 32), (b) Position des Stimulus innerhalb des Batches, (c) Padding-Menge, (d) falls verfügbar, GPU-Modell. Modellgewichte, Seed und Eingabetoken identisch halten. Berichten:
$$\eta_\ell = \max_r \big| a_\ell^{(r)} - \bar a_\ell \big| \Big/ \|\bar a_\ell\|_2 \quad \text{pro Layer } \ell$$
also die **relative** Abweichung, normiert auf die Residual-Norm **desselben** Layers. Ohne diese Normierung sehen mittlere/späte Layer automatisch „stärker betroffen" aus, weil die Residual-Norm mit der Tiefe wächst — genau das erzeugt scheinbare „Layer-14-und-20-Effekte".

**Schritt 2 — Akzeptanzschwelle.** Ein Effekt gilt nur dann als real, wenn er den Rauschboden um **mindestens den Faktor 10** übersteigt: $|\Delta_\ell| \ge 10\,\eta_\ell$. Ein Faktor 2–3 ist zu wenig, weil $\eta$ selbst über Kernelwahlen streut.

**Schritt 3 — Erwartungswerte.** In fp32 mit ausgeschaltetem TF32 und deterministischen Kernels sollte $\eta_\ell$ bei **10⁻⁶ bis 10⁻⁵** liegen. Liegt es bei **10⁻³ oder höher**, läuft die Messung faktisch in fp16/TF32 oder mit nicht-deterministischen Kernels — dann ist die Konfiguration zu reparieren, nicht das Ergebnis zu interpretieren.

### 6.5 Größenordnungsvergleich: Was hier eigentlich dominiert

| Rauschquelle | Relative Größenordnung | Betrifft |
|---|---|---|
| fp32 deterministisch | 10⁻⁶ … 10⁻⁵ | Patching, Logit-Lens |
| fp16 / TF32 / Batch-Reordering | 10⁻³ … 10⁻² | Patching, Logit-Lens, Argmax |
| **SGLD-Ketten-zu-Ketten-Streuung** (4 Ketten × 100 Draws, autokorreliert) | **~5 · 10⁻² … 1,5 · 10⁻¹** **[U] Schätzung** | **Suszeptibilitäten, LLC** |
| Checkpoint-zu-Checkpoint-Jitter 97k→100k | empirisch zu bestimmen | alles |

**Die entscheidende Schlussfolgerung:** Für **Suszeptibilitäten** ist Gleitkommarauschen um 2–3 Größenordnungen irrelevant — dort dominiert das MCMC-Rauschen, und die richtige Kontrolle sind Ketten, Seeds und Datenbatches. Für **Patching und Logit-Lens** ist Gleitkommarauschen dagegen der limitierende Faktor und kann einen Mittellayer-„Effekt" vollständig erklären. Die alte McQuarrie-Vermutung eines „hardwareabhängigen Effekts" gehört in die zweite Kategorie und hätte sich mit Schritt 1 oben in einer Stunde entscheiden lassen.

### 6.6 Trace-Diagnostik als Vorbedingung **[V]** (Sampling Guide)

Akzeptable Loss-Traces müssen „increase monotonically above the loss at $w_0$, level off quickly, don't have large spikes, have signal larger than local variation, and different chains level off at approximately the same value". Zusätzlich zu berichten: laufende Stichprobenmittel, Autokorrelation der Loss-Werte (zur Begründung von `num_steps_bw_draws`), Gelman-Rubin $\hat R \approx 1$, Between-/Within-Chain-Varianz. Failure Modes: Loss steigt über den stabilen Wert → $\gamma$ zu klein oder $n\beta$ zu groß; `inf`/`nan` → $\varepsilon$, $n\beta$ oder $\gamma$ zu groß; niedriges SNR → $n\beta$ oder Batchgröße erhöhen. Empfohlener Sweep: $\varepsilon \in \{10^{-6},10^{-5},10^{-4}\}$, $n\beta \in \{1,10,100\}$, $\gamma \in \{10,100,1000\}$.

**Die $n\beta=0$-Negativkontrolle ist der entscheidende Test [V].** Bei $n\beta=0$ verschwindet der Loss-Gradient aus dem Sampler; übrig bleibt reine Gewichts-/Embedding-Geometrie. Gordon et al. führen genau diesen Vergleich durch (Fig. 12/13, Tag- und Anführungszeichen-Cluster). **Tritt ein „WASD-Effekt" auch bei $n\beta=0$ auf, ist er ein Tokenizer-/Embedding-Artefakt und kein Suszeptibilitätsphänomen.** Dieser Test muss vor jeder mechanistischen Interpretation bestanden sein.

---

## 7. Basisraten: was über WASD in Korpora bekannt ist, und welche Vorhersage die Hypothese machen müsste

### 7.1 Der Kenntnisstand ist dünn

**[V]** Unbekannt und bisher nicht gemessen: die tatsächliche Pile-Frequenz von `WASD`, `' WASD'`, `wasd`, `QERF` und der beteiligten Einzeltoken (IDs 56, 1719, 37, 50, 947, 39, 22250, 1165, 4238, 69).

Diese Zahlen sind **beschaffbar** — der Dataloader ist deterministisch rekonstruierbar: prätokenisierte `.bin`/`.idx` von HF laden, mit `utils/unshard_memmap.py` zusammenführen, `utils/batch_viewer.py` über die relevanten Iterationen laufen lassen. **Geschätzte Frequenzen genügen nicht.** Solange sie fehlen, ist jede Aussage über „Sondereffekte" nicht von einem reinen Frequenzeffekt trennbar.

**Zur Einordnung [U], grobe Erwartung, ausdrücklich als Schätzung:** Der Pile enthält keine dedizierte Gaming-Quelle; `WASD` dürfte überwiegend aus Pile-CC, HackerNews und StackExchange stammen. Eine plausible Größenordnung für die Häufigkeit des Strings in 299,89 Mrd. Token liegt bei **10⁴ bis 10⁵ Vorkommen**, also einer relativen Frequenz von **10⁻⁷ bis 10⁻⁶** (etwa 1 pro 1–10 Mio. Token). **Diese Zahl ist nicht gemessen und darf nicht zitiert werden** — sie dient nur dazu, den Erwartungsbereich für die Messung abzustecken.

### 7.2 Warum Frequenz der stärkste Konkurrent ist

**[V]** Razeghi et al. (arXiv:2202.07206), im Pythia-Paper repliziert: Modelle sind „in some cases above 70% (absolute) more accurate on the top 10% frequent terms in comparison to the bottom 10%". Ein Effekt dieser Größe allein aus Pretraining-Termfrequenz macht jeden unkontrollierten Frequenzunterschied zum dominanten Erklärungskandidaten.

**Präzisierung [V]:** Die Formulierung „Modelle unter 1B zeigen vernachlässigbare Frequenzabhängigkeit" ist irreführend. Das Pythia-Paper sagt, kleine Modelle „rarely produce accurate results on the task despite being given up to 16 few-shot examples" — ein **Floor-Effekt**, keine nachgewiesene Frequenzunabhängigkeit. Für pythia-1.4b liefert die Term-Frequenz-Case-Study daher **kein** positives Ergebnis, auf das man sich stützen könnte.

### 7.3 Die vorliegenden negativen Läufe, quantitativ eingeordnet

**Lauf A — WASDQERF-Buchstabendichte**

| Kennzahl | Wert |
|---|---|
| Dichte | 0,386819 |
| Rang | 14 von 19 Textfenstern |
| $z$ vs. Random-8-Buchstaben-Null | 1,084 |
| $p$ | 0,1467 |
| Verdikt | NOT_UNUSUALLY_HIGH |

Drei Anmerkungen:

1. **[E] Das Design ist durch seine Stichprobengröße unfalsifizierbar.** Ein Rangtest über 19 Fenster kann bestenfalls $p = 1/19 = 0{,}0526$ erreichen — also **selbst bei Rang 1 nicht unter 0,05**. Der Rang 14/19 ist damit nicht nur negativ; die Kennzahl hätte auch im besten Fall nichts zeigen können. Künftige Rangtests brauchen $\ge 20$, realistisch $\ge 100$ Fenster.
2. **[E] Die Nullhypothese ist die falsche.** Ein Random-8-Buchstaben-Null zieht 8 beliebige Buchstaben; dass W,A,S,D,Q,E,R,F über diesem Mittel liegen ($z = 1{,}084$), folgt trivial daraus, dass E, A, S, R zu den häufigsten englischen Buchstaben gehören. Die **richtige** Null ist die Unigramm-Erwartung genau dieser acht Buchstaben. Mit Standard-Buchstabenfrequenzen des Englischen (Lewand) ergibt sich: w 2,36 + a 8,17 + s 6,33 + d 4,25 + q 0,10 + e 12,70 + r 5,99 + f 2,23 = **42,13 %**. Die gemessene Dichte von **38,68 %** liegt damit **3,45 Prozentpunkte unter** der Erwartung (Verhältnis 0,918). Gegen die korrekte Null zeigt das Fenster also nicht einmal die Richtung, die die Hypothese verlangt. **[U]** Die Lewand-Frequenzen gelten für Standardenglisch; für das konkrete Fenster ist die korpuseigene Unigrammverteilung zu verwenden.
3. **[E]** $\Phi(1{,}084) \approx 0{,}860$, einseitiges $p \approx 0{,}140$ unter Normalität; berichtet sind 0,1467. Die kleine Differenz ist mit einer empirischen Null konsistent und unkritisch.

**Lauf B — Transition/Position-Zyklizität**

| Kennzahl | Wert |
|---|---|
| Perzentil des kanonischen 8er-Zyklus | 47,798 von 5040 gerichteten Zyklen |
| $p$ | 0,570 |
| Beste Token-Position-Periode | 32 |
| NMI | 0,2089 |
| Permutations-$p$ | 0,911 |
| `transition_position_cyclicity_supported` | **False** |

**[E]** 5040 = $7!$ = Anzahl gerichteter Hamilton-Zyklen auf 8 markierten Knoten (bis auf Rotation) — der Suchraum ist korrekt abgezählt. Das Perzentil 47,798 liegt praktisch exakt am Median; der kanonische WASD→QERF-Zyklus ist in dieser Statistik **vom Zufall ununterscheidbar**. Die beste Periode 32 ist eine Zweierpotenz und daher als Format-/Zeilenperiodizität verdächtig; das Permutations-$p$ von 0,911 bestätigt, dass NMI 0,2089 bei einer Best-of-many-Suche der erwartete Wert ist.

**Fazit zu §7.3:** Beide Läufe sind sauber negativ, und Lauf A wäre bei korrekter Null sogar leicht gegenläufig. Sie betreffen jedoch ausschließlich Oberflächenstatistik.

### 7.4 Welche Vorhersage die Hypothese machen müsste, um prüfbar zu sein

Damit „W, A, S, D spielen eine Sonderrolle" eine wissenschaftliche Hypothese ist und nicht eine Suchanweisung, muss sie **vor** der nächsten Messung eine Aussage dieses Typs festlegen:

> **H1 (vorzuregistrieren):** Auf der spaltenweise standardisierten und zeilenzentrierten Suszeptibilitätsmatrix, gemessen mit $\gamma = 300$, $n\beta = 10$, $\varepsilon = 10^{-5}$, Batch 16, 4 Ketten, 100 Draws, 160 Steps zwischen Draws (pSGLD), fixiertem $\delta h$, an den Revisionen step97000/98000/99000/100000 von `EleutherAI/pythia-1.4b` (nicht -deduped, nicht -v0), liegt das über die vorab definierte Indexmenge *mittlere Layer* (L8–L15) gemittelte $|\psi|$ für die Stimulusvariante `WASD` → `['W','AS','D']` **oberhalb des 95. Perzentils** der Verteilung, die von $\ge 40$ frequenzgematchten Kontrollketten mit identischem Zerlegungsmuster (Einzelbuchstabe + Bigramm + Einzelbuchstabe) erzeugt wird — **und zwar in mindestens 3 der 4 Checkpoints**.

Und, zwingend, das komplementäre Kriterium:

> **F1 (Falsifikation):** Die Hypothese gilt als **widerlegt**, wenn eine der folgenden Bedingungen eintritt:
> (a) `WASD` liegt in $\ge 2$ der 4 Checkpoints unter dem 95. Perzentil der Kontrollverteilung;
> (b) die `' WASX'`-Serie (X ≠ D) erzeugt einen Effekt gleicher Größe — dann misst man `WAS`;
> (c) der Effekt tritt bei $n\beta = 0$ in gleicher Höhe auf — dann misst man Embedding-Geometrie;
> (d) die beteiligten Token bestehen das Under-trained-Screening nicht;
> (e) tastaturfreie Kontrollketten mit gleichem Zerlegungsmuster (`XKPN`, `MRLV`) zeigen denselben Mittellayer-Ausschlag — dann misst man Detokenisierung;
> (f) der Effekt überschreitet den in §6.4 bestimmten Rauschboden nicht um Faktor 10.

Ohne **F1** ist die Hypothese in einem Suchraum von 410 Komponenten × 24 Layern × 4 Checkpoints × $N$ Stimuli (× ggf. 57.236 Clustern) **unfalsifizierbar**, und jeder gefundene Peak ist erwartbares Rauschen.

### 7.5 Ehrliche Einschätzung des Prior

- Es gibt **[V]** keine Literatur, keinen Prior und keine Baseline für WASD/QERF.
- Die drei nächstgelegenen publizierten Befunde sind Cluster zu `<`/`</` (HTML-Tags) und Anführungszeichen in arXiv:2601.12703 sowie die „spacing fin" (Leerzeichen-Zählung) im 3M-Modell — also **Formatzeichen**, nicht Buchstabenmengen.
- Drei starke Alternativerklärungen stehen bereit (`was`-Token, Detokenisierung, Untertrainiertheit), von denen keine bisher ausgeschlossen ist.
- Zwei gut kontrollierte Vorläufe sind negativ ausgefallen.

**Empfehlung:** Vor jedem weiteren Forward-Pass-Experiment sind die **Pile-Frequenzen** (§7.1) und das **Under-trained-Screening** (§3.5) zu erheben. Beides ist billig, beides ist entscheidungsrelevant, und beides kann die Linie mit hoher Wahrscheinlichkeit abschließen, bevor GPU-Stunden fließen. Die publizierte Ground-Truth-Messung für Pythia-1.4b kostete **[V]** 4800 H200-Stunden für 4,25 Mio. Vektoren; ein kleineres Budget erlaubt keine clusterartige Struktur, sondern höchstens gezielte Hypothesentests auf einer vorab definierten Token- und Komponentenmenge — was mit den Kontrollen oben aber auch ausreicht.

---

## 8. Quellenliste

### 8.1 Suszeptibilitäten: Theorie und Schätzer

- **arXiv:2504.18274** — Baker, Wang, Hoogland, Murfet, *Structural Inference: Interpreting Small Language Models with Susceptibilities* (v1 25.04.2025, v3 06.03.2026; **ICLR 2026**, nicht 2025). Def. 2.1/2.4/2.5, Lemma 2.2, Gl. 9/10 und 13/14, Vorzeichenkonvention §3.2, Hyperparameter Anh. C.3/C.4/C.6 ($\gamma$=300, $n\beta$=30, $\varepsilon$=0.001, Batch 64, 4 Ketten, 200 Draws; per-Token 100 Draws/Batch 16; **kein** RMSProp; $\delta h$=0.1; Checkpoint 49900; 10 bzw. 16 A100). → https://arxiv.org/abs/2504.18274
- **How to Scale Susceptibilities** — Gordon, Adam, Hoogland, Murfet, 21.04.2026 (Timaeus-Research-Seite, **kein** arXiv-Preprint). Einzige Quelle für den Zwei-Ketten-Schätzer, $g(c) = L_n(u^*,c) - L_n(w^*)$, die $Z_\text{full}/Z_C$-Renormierung und deren Aufhebung durch spaltenweises z-Scoring. → https://timaeus.co/research/2026-04-21-spectroscopy-definitions
- **arXiv:2605.07970** — Elliott & Murfet, *Linear Response Estimators for Singular Statistical Models*. Gl. 1 §3.1, Remark 3.5, Thm. 3.21/4.8; Def. 3.6 verallgemeinert den Zwei-Ketten-Fall auf „separate SGLD chains on each submanifold $S_i$". → https://arxiv.org/abs/2605.07970
- **arXiv:2605.07980** — Elliott & Murfet, *Susceptibilities and Patterning: A Primer on Linear Response in Bayesian Learning*. Maßgebliche Terminologie *response matrix* / *structural susceptibility matrix*. → https://arxiv.org/abs/2605.07980

### 8.2 Cluster-Map / „Atlas"

- **arXiv:2601.12703** — Gordon, Baker, Wang, Snell, van Wingerden, Murfet, *Towards Spectroscopy: Susceptibility Clusters in Language Models*, 19.01.2026. Pythia-**14M**, 510 Cluster, 259 (50,8 %) SAE-Match. Tab. 4 (Clustering: $k$=45, $\alpha$=0.001, PPR-Toleranz 1e-7, Main-Body-Schwelle 0.99, Min-Clustergröße 20, Abbruch <0,1 %); Tab. 5 (SGLD pro Modellgröße); Tab. 7 (Komponenten/Cluster); Anh. A.1 (Clustering-Details), A.2, A.3 (Compute), D.1 (UMAP $n\_neighbors$=45), G (SAE, Layer **2–4**, $d$=512, $D$=32768). Preprocessing: spaltenweise Standardisierung **plus Zeilenzentrierung**, **kein** PCA-Whitening. $n\beta$=0-Kontrollen Fig. 12/13. → https://arxiv.org/abs/2601.12703
- **Spectroscopy at Scale: Finding Interpretable Structure in Pythia-1.4B** — Murfet, Gordon, Adam, Wang, Hoogland, Baker, Snell, van Wingerden, Newgas, Snikkers, Hitchcock, 21.04.2026. $H$=410, 4,25 Mio. Ground-Truth-Vektoren / 4800 H200-Stunden, 46 Mio. upgesampelte Vektoren, 57.236 Cluster, 19 Regionen, PCA-Whitening, Gauß-Baseline-Warnung, fehlende Steering-Validierung. → https://timaeus.co/research/2026-04-21-spectroscopy-main/
- **Guide for Sampling Hyperparameter Selection** — Hitchcock, Hoogland, Wang, Gordon, 21.04.2026. Primär-/Sekundär-/Sammelparameter, Sweep-Bereiche, Trace- und Konvergenzdiagnostik, Failure Modes, Forderung nach einem einzigen Hyperparametersatz. → https://timaeus.co/research/2026-04-21-sampling-guide

### 8.3 SLT-Grundlagen und Entwicklungsstadien

- **arXiv:2402.02364** — Hoogland, Wang, Farrugia-Roberts, Carroll, Wei, Murfet, *Loss Landscape Degeneracy and Stagewise Development in Transformers* (TMLR; v1 04.02.2024, v3 01.08.2025; v1-Titel war *The Developmental Landscape of In-Context Learning*). Gl. 3 ($\hat\lambda$), Gl. 4 ($F_n$), LM1–LM5, LR1–LR5. **Korrekte Anhangsnummern (v3):** A.1 ($\lambda=3/2$, „does not measure curvature"), A.5 (kein echter Log-Likelihood), A.6 („far from any local minima"), B.5 (Hessian-Vergleich). → https://arxiv.org/abs/2402.02364 · maschinenlesbar: https://ar5iv.labs.arxiv.org/html/2402.02364
- **arXiv:2308.12108** — Lau, Murfet, Wei et al. LLC-Definition und SGLD-Schätzer. Maßgebliche Quelle, in die arXiv:2402.03698 eingearbeitet wurde. Anh. H (Schätzer-Pathologien) **[U]** nicht maschinell auslesbar, im PDF nachzuschlagen. → https://arxiv.org/abs/2308.12108
- ⚠️ **arXiv:2402.03698** — Furman & Lau, *Estimating the Local Learning Coefficient at Scale*: **ZURÜCKGEZOGEN** („This paper has been withdrawn by Edmund Lau"). **Nicht zitieren.** Alle darauf gestützten Aussagen auf 2308.12108 umhängen und dort neu belegen.
- **arXiv:2310.06301** — Chen, Lau, Mendel, Wei, Murfet, *Dynamical versus Bayesian Phase Transitions in a Toy Model of Superposition*. „no necessary relation" zwischen beiden Übergangstypen; Bayesian Antecedent Hypothesis als **Hypothese**; nie beobachteter 5→6-Übergang; „opposing staircases". → https://arxiv.org/abs/2310.06301
- **arXiv:2410.02984** — Wang, Hoogland, van Wingerden, Furman, Murfet, *Differentiation and Specialization of Attention Heads via the Refined Local Learning Coefficient* (ICLR 2025 **[U]**, Venue auf arXiv nicht ausgewiesen). wrLLC/drLLC/wdrLLC; Anh. F.2 ($\varepsilon$=1e-3, $\beta$=30/n, $\gamma$=200, 4 Ketten, Burn-in 0, 200 Draws); Sensitivitätsbeleg $n\beta$ 23→30 ≈ 1000 Schritte Grenzverschiebung. → https://arxiv.org/abs/2410.02984
- **arXiv:2510.12077** — Urdshals, Lau, Hoogland, van Wingerden, Murfet, *Compressibility Measures Complexity: Minimum Description Length Meets Singular Learning Theory*, 14.10.2025. Einzige Quelle mit Pythia-1.4b-LLC-Hyperparametern **und** dem für dieses Projekt kritischen Ausschluss von Checkpoints über step90000. Lernraten: 14M 1e-3; 31M/70M 3e-4; 160M/410M 1e-4; 1B/1.4B **3e-5**; 2.8B 1e-5; 6.9B 3e-6. → https://arxiv.org/abs/2510.12077
- **arXiv:2510.12071** — Lee, Smith, Adam, Hoogland, *Influence Dynamics and Stagewise Data Attribution*. BIF $=-\mathrm{Cov}(\ell_i,\varphi)$; Law-of-Total-Covariance-Peak bei $\pi\approx0{,}5$; Vorzeichenwechsel an Stadiengrenzen; LOO-Retraining; zeitfensterspezifische Ablation (§C.5); korrigierte Pythia-Zeitachse (**~1000** Schritte, Scheitel ~30k). → https://arxiv.org/abs/2510.12071
- **arXiv:2508.00331** — Wang, Baker, Gordon, Murfet, *Embryology of a Language Model*, 01.08.2025. **Negativbeleg:** analysiert **nicht** Pythia, sondern ein 3M-Parameter-Attention-only-Modell (2 Layer, 8 Heads/Layer, 4 Seeds, bis step 49900). Die Autoren nennen die Modellgröße selbst als Hauptlimitation. Nicht auf Pythia-1.4b übertragbar. → https://arxiv.org/abs/2508.00331
- **arXiv:2501.17745** — Carroll, Hoogland, Farrugia-Roberts, Murfet, *Dynamics of Transient Structure in In-Context Linear Regression Transformers*. Verwendet *joint trajectory PCA*; die „essential dynamics"-Differentialgeometrie (Cusp-Singularitäten, Evolute) ist dort **nicht** enthalten — sie stammt aus v1/v2 von 2402.02364 und ist nur dort zitierbar. → https://arxiv.org/abs/2501.17745
- **Joar Skalse, *My Criticism of Singular Learning Theory*** (LessWrong/AlignmentForum, 19.11.2023) mit Murfets und Carrolls Antworten in den Kommentaren. Pflichtlektüre für die Limitations-Sektion: RLCT ≠ Kolmogorov-Komplexität; SLT ist stärker bei Phasenübergängen als bei Generalisierung. → https://www.lesswrong.com/posts/ALJYj4PpkqyseL7kZ/my-criticism-of-singular-learning-theory

### 8.4 Pythia: Modell, Daten, Werkzeuge

- **HF Model Card pythia-1.4b** — 143.000 Schritte, 154 Checkpoints, Batch 2.097.152 Token, 299.892.736.000 Token, 24 Layer, 16 Heads, $d_\text{model}$ 2048, GPT-NeoX-20B-Tokenizer, 1.414.647.808 / 1.208.602.624 Parameter. → https://huggingface.co/EleutherAI/pythia-1.4b
- **EleutherAI/pythia (GitHub)** — v0-Warnung (step1000 = real Schritt 500), Umbenennung 20.01.2023, Sequenzlänge 2049 mit EOD-Konkatenation, $d_\text{head}$-Tabelle, 207B Token/Epoche des deduplizierten Pile (**hier**, nicht auf der Model Card), `utils/unshard_memmap.py`, `utils/batch_viewer.py`. → https://github.com/EleutherAI/pythia
- **arXiv:2304.01373** — Biderman et al., *Pythia*. Fünf Abweichungen vom GPT-3-Standard; Poisson-Punktprozess-Modell für Memorisierung; **Phasenwechsel nach 65.000 Schritten (45 % des Trainings)** für Modelle ab 2.8B. → https://arxiv.org/abs/2304.01373
- **arXiv:2304.11158** — *Emergent and Predictable Memorization in Large Language Models*. $k$=32-extractible; Recall 70M→0,197, 1,0B→0,512, 6,9B→0,795; 0,513 bei 23M Sequenzen, 0,918 bei 126M (**[E]** = 86,05 % des Trainings). → https://arxiv.org/abs/2304.11158
- **arXiv:2204.06745** — GPT-NeoX-20B. Begründung für `use_parallel_residual` (ein statt zwei All-Reduces, „a 15% throughput increase") und für `rotary_pct`=0.25. → https://arxiv.org/abs/2204.06745
- **arXiv:2509.05291** — Bayazit, Mueller, Bosselut, *Crosscoding Through Time*. Sparse Crosscoders über Pythia-1B-Checkpoints (128M/1B/4B/286B Token), RelIE-Metrik; Feature-Entstehung auch nach dem Loss-Plateau. → https://arxiv.org/abs/2509.05291
- **devinterp** — Version 2.0.1 (23.04.2026), MIT, Python ≥ 3.10. `sample()`-Defaults (4 Ketten, 200 Draws, Batch 32, Burn-in 0, `num_steps_bw_draws`=1, …); `lr` und `n_beta` sind **keyword-only ohne Default**; `rmsprop_eps`/`rmsprop_alpha` sind im Code **`None`** (0.99 ist eine Empfehlung des Sampling Guide, **kein** Default); `weight_restrictions` muss `'full'` enthalten; Muster `'l0'`, `'l0h1'`, `'l0g0'`, `'l0 attn'`, `'l0 mlp'`, `'embed'`, `'unembed'`. → https://github.com/timaeus-research/devinterp · https://pypi.org/project/devinterp/

### 8.5 Tokenizer-Artefakte, Glitch-Token, Zeichenwissen

- **arXiv:2404.09894** — *Glitch Tokens in Large Language Models*. Mechanismus: Mismatch Tokenizer-Korpus ↔ Pretraining-Korpus. → https://arxiv.org/abs/2404.09894
- **arXiv:2408.04905** — *GlitchProber*. „significant deviations in the distributions of attention patterns and dynamic information from **intermediate model layers**" — die Signatur, die die WASD-Hypothese erwartet. → https://arxiv.org/abs/2408.04905
- **arXiv:2410.15052** — *GlitchMiner*. Detektion über maximale Prädiktionsentropie. → https://arxiv.org/abs/2410.15052
- **arXiv:2405.05417** — Land & Bartolo. Indikatoren für under-trained Token; Pythia-1.4b-Ergebnisse **[U]** nicht in verifizierbarer Form gelistet. → https://arxiv.org/abs/2405.05417
- **arXiv:2506.10641** — Zeichenwissen/Detokenisierung: „a distinct breakthrough in their spelling behavior" in mittleren bis oberen Layern. → https://arxiv.org/abs/2506.10641
- ⚠️ **arXiv:2603.18474** — *WASD: Locating Critical Neurons…* — **Akronym** („unWeaving Actionable Sufficient Directives"), Gemma-2-2B, SST-2/CounterFact. **Keine Vorarbeit zu dieser Hypothese. Nicht zitieren.**

### 8.6 Mehrsprachigkeit / Pivot-Baseline

- **arXiv:2402.16438** — *Language-Specific Neurons* (ACL 2024). Sprachspezifische Neuronen überwiegend in obersten/untersten Layern. → https://arxiv.org/abs/2402.16438
- **arXiv:2509.17030** — *The Transfer Neurons Hypothesis*. → https://arxiv.org/pdf/2509.17030
- **arXiv:2402.10588** — Englischzentrierter Pivot-Raum in mittleren Layern. → https://arxiv.org/abs/2402.10588
- **arXiv:2505.05111** — SAE-basierte Analyse sprachspezifischer Repräsentationen. → https://arxiv.org/abs/2505.05111

### 8.7 Patching- und Ablationsmethodik

- **arXiv:2309.16042** — Zhang & Nanda, *Towards Best Practices of Activation Patching* (ICLR 2024). Metrik- und Korruptionswahl ändern die Schlussfolgerung darüber, welche Komponenten wichtig sind; Interpretability-Illusionen bei Subspace-Patching. → https://arxiv.org/abs/2309.16042
- **[X] arXiv:2307.15771** — McGrath et al., *The Hydra Effect*. **Nicht im geprüften Rechercheset**; vor Zitat im Volltext verifizieren. → https://arxiv.org/abs/2307.15771
- **[X] arXiv:2402.15390** — Rushing & Nanda, *Explorations of Self-Repair in Language Models*. LayerNorm-vermittelte Selbstreparatur. **Nicht im geprüften Rechercheset**; vor Zitat verifizieren. → https://arxiv.org/abs/2402.15390
- **[X] arXiv:2211.00593** — Wang et al., IOI / backup name mover heads. **Nicht im geprüften Rechercheset**; vor Zitat verifizieren. → https://arxiv.org/abs/2211.00593
- **arXiv:2202.07206** — Razeghi et al., *Impact of Pretraining Term Frequencies*. „above 70% (absolute)" Genauigkeitsunterschied allein aus Termfrequenz. → https://arxiv.org/abs/2202.07206

---

## Anhang: Sieben offene Punkte, die vor der nächsten Messung zu klären sind

1. **Checkpoint des publizierten Cluster-Artefakts** — in keiner Quelle genannt. Bei den Autoren erfragen oder als explizite Annahme mit Sensitivitätsanalyse ausweisen. **[U]**
2. **Trainingsstabilität bei 97k–100k** — Trainings-Loss und LLC-Kurve für Pythia-1.4b über mindestens step80000–step110000 plotten, plus Kontrollbereich 60k–63k. Solange nicht gezeigt ist, dass das Fenster glatt ist, ist jede „Entwicklung" nicht von Trainingsrauschen unterscheidbar. **[V, Urdshals et al.]**
3. **Repo-Entscheidung** `pythia-1.4b` vs. `-deduped` — entscheidbar, entscheidungspflichtig, wegen der Epochengrenze bei **[E]** ≈ Schritt 98.705.
4. **Pile-Frequenzen** der zehn beteiligten Token-IDs — beschaffbar, bisher nicht beschafft.
5. **$\delta h$ für die Pythia-Läufe** — in Tab. 5 von arXiv:2601.12703 nicht gelistet. **[U]**
6. **LR-Schedule-Details für 1.4b** (Warmup-Anteil, min-LR) — aus der YAML im `models/`-Verzeichnis, nicht aus der Model Card. **[U]**
7. **Terminologie im eigenen Text bereinigen** — „Suszeptibilitätsatlas" ist keine Literaturvokabel. Entweder die publizierten Begriffe verwenden oder die Eigenprägung als solche kennzeichnen.
