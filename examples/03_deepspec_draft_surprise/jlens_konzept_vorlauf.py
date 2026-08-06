# === VORLAUF: WELCHES KONZEPT STEHT AM KOEDER-TOKEN? =======================
# Kostet KEINEN Vorwaertspass. Nimmt die schon transportierten Vektoren aus der
# J-Linsen-Zelle und misst sie gegen DREI DISJUNKTE Token-Mengen:
#   FREMD  - fremde Schrift (die bisherige Maske)
#   SPRACHE- Sprachnamen und Uebersetzungs-Vokabular, lateinisch geschrieben
#   I18N   - Lokalisierungs-BEZEICHNER aus Code (Locale, localized, i18n, ...)
# Damit ist die Lokalisierungs-Lesart auf den VORHANDENEN Daten pruefbar:
# spikt I18N an Q=43 nicht, ist sie erledigt - ohne eine neue Rechnung.
# Schreibt alles nach Drive, druckt nur eine Quittung.
BEREITS_GETEILT = False
import os, sys, json, time, re
import numpy as np, torch
if not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive; drive.mount("/content/drive")
GL = globals()
OUT = GL.get("RUN_OUT") or ("/content/drive/MyDrive/WeirdChat_Runs/jlens_konzepte_"
                            + time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(OUT, exist_ok=True)
LINES = []
def P(s=""): LINES.append(str(s))
def schreibe(n, t):
    with open(os.path.join(OUT, n), "w", encoding="utf-8") as f: f.write(t)
# ---------------- reine Logik (offline geprueft) ---------------------------
RX_I18N = re.compile(r"(locale|localiz|localis|i18n|l10n|gettext|resourcebundle"
                     r"|nslocalized|getstring|\.properties|strings\.xml)", re.I)
RX_LANG = re.compile(r"(language|translat|untranslated|linguis|multiling)", re.I)
SPRACHNAMEN = set("""english arabic chinese japanese korean hindi russian french german
spanish portuguese italian thai vietnamese turkish hebrew greek persian urdu bengali
tamil polish dutch swedish czech romanian hungarian indonesian malay swahili""".split())
def klassifiziere(tok, ist_fremd):
    """genau EINE Klasse je Token - Reihenfolge legt die Disjunktheit fest"""
    if ist_fremd: return "FREMD"
    s = tok.strip()
    if s.lower().lstrip("/-_.") in SPRACHNAMEN: return "SPRACHE"
    if RX_LANG.search(s): return "SPRACHE"
    if RX_I18N.search(s): return "I18N"
    return None
def rang(vals, idx):
    v = list(vals); t = v[idx]; return 1 + sum(1 for x in v if x > t)
def urteil_i18n(rang_i18n_spaet, rang_fremd_spaet, n_pos):
    """rang_*_spaet: Listen der Raenge von Q an den spaeten Schichten"""
    schwelle = max(2, n_pos // 10)
    i18n_spikt = sum(1 for r in rang_i18n_spaet if r <= schwelle) >= 2
    fremd_spikt = sum(1 for r in rang_fremd_spaet if r <= schwelle) >= 2
    if i18n_spikt and fremd_spikt: return "BEIDE-KONZEPTE"
    if i18n_spikt: return "LOKALISIERUNG"
    if fremd_spikt: return "LOKALISIERUNG-ERLEDIGT"
    return "KEINS"
# ---------------- Ausfuehrung ----------------------------------------------
FEHLER = None
try:
    for _n in ("ACC", "NOK", "H", "POS", "LAYERS", "jQ", "IDS", "Q", "unembed",
               "tokenizer", "model", "MASK_NPZ"):
        assert _n in GL, "%s fehlt - lief die J-Linsen-Zelle in dieser Laufzeit?" % _n
    ACC, NOK, H = GL["ACC"], int(GL["NOK"]), GL["H"]
    POS, LAYERS, jQ, Q = list(GL["POS"]), list(GL["LAYERS"]), int(GL["jQ"]), int(GL["Q"])
    IDS, tok, unembed, dev = GL["IDS"], GL["tokenizer"], GL["unembed"], GL["model"].device
    MEAN = {l: (ACC[l] if BEREITS_GETEILT else ACC[l] / NOK) for l in LAYERS}
    P("KONZEPT-VORLAUF AUF DER J-LINSE - %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    P("=" * 72)
    P("Q=%d %r | %d Positionen | Schichten %s | %d gemittelte Prompts"
      % (Q, tok.decode([IDS[Q]]), len(POS), LAYERS, NOK))
    # ---- Token-Mengen aus dem echten Vokabular bauen -----------------------
    fremd = torch.tensor(np.load(GL["MASK_NPZ"])["script"]).numpy().astype(bool)
    VOC = int(GL["model"].get_output_embeddings().weight.shape[0])
    if fremd.shape[0] < VOC:
        fremd = np.concatenate([fremd, np.zeros(VOC - fremd.shape[0], bool)])
    fremd = fremd[:VOC]
    toks = []
    for i in range(0, VOC, 8192):
        toks += tok.batch_decode([[j] for j in range(i, min(i + 8192, VOC))])
    MASKEN = {k: np.zeros(VOC, bool) for k in ("FREMD", "SPRACHE", "I18N")}
    for i, t in enumerate(toks):
        k = klassifiziere(t, bool(fremd[i]))
        if k: MASKEN[k][i] = True
    P("")
    P("Token-Mengen (disjunkt, aus %d Vokabular-Eintraegen):" % VOC)
    for k in ("FREMD", "SPRACHE", "I18N"):
        bsp = [repr(toks[i]) for i in np.where(MASKEN[k])[0][:10]]
        P("  %-8s %6d Tokens | Beispiele: %s" % (k, MASKEN[k].sum(), " ".join(bsp)))
    ueb = sum(MASKEN["FREMD"] & MASKEN["SPRACHE"]) + sum(MASKEN["FREMD"] & MASKEN["I18N"]) \
        + sum(MASKEN["SPRACHE"] & MASKEN["I18N"])
    P("  Ueberschneidung: %d Tokens (muss 0 sein)" % ueb)
    assert ueb == 0, "Mengen nicht disjunkt"
    assert MASKEN["I18N"].sum() >= 5, "I18N-Menge zu klein - der Test haette keine Kraft"
    # ---- Massen je Konzept, Schicht, Position -----------------------------
    MT = {k: torch.tensor(MASKEN[k]) for k in MASKEN}
    def massen(lg):
        p = torch.softmax(lg.float(), -1)
        return {k: p[..., MT[k].to(p.device)].sum(-1).cpu().numpy() for k in MT}
    M = {k: np.zeros((len(LAYERS), len(POS))) for k in MASKEN}
    ML = {k: np.zeros((len(LAYERS), len(POS))) for k in MASKEN}
    with torch.no_grad():
        for i, l in enumerate(LAYERS):
            a = massen(unembed(MEAN[l].to(dev)))
            b = massen(unembed(H[l].to(dev)))
            for k in MASKEN: M[k][i] = a[k]; ML[k][i] = b[k]
    # ---- Raenge von Q ------------------------------------------------------
    P("")
    P("RANG VON Q=%d unter %d Positionen, je Konzept und Schicht (1 = hoechste Masse)"
      % (Q, len(POS)))
    P("  %-6s %-26s %-26s %s" % ("", "FREMD (Jacobi)", "SPRACHE (Jacobi)", "I18N (Jacobi)"))
    R = {k: [] for k in MASKEN}
    for i, l in enumerate(LAYERS):
        z = []
        for k in ("FREMD", "SPRACHE", "I18N"):
            r = rang(M[k][i], jQ); R[k].append(r)
            z.append("Rang %2d  Q/Med %6.2f" % (r, M[k][i][jQ] / max(np.median(M[k][i]), 1e-12)))
        P("  L%-5d %-26s %-26s %s" % (l, z[0], z[1], z[2]))
    P("")
    P("Dasselbe fuer die LOGIT-Linse (Kontrolle - dort steckte 'izedName'):")
    RL = {k: [] for k in MASKEN}
    for i, l in enumerate(LAYERS):
        z = []
        for k in ("FREMD", "SPRACHE", "I18N"):
            r = rang(ML[k][i], jQ); RL[k].append(r)
            z.append("Rang %2d  Q/Med %6.2f" % (r, ML[k][i][jQ] / max(np.median(ML[k][i]), 1e-12)))
        P("  L%-5d %-26s %-26s %s" % (l, z[0], z[1], z[2]))
    # ---- Verdikt -----------------------------------------------------------
    spaet = [i for i, l in enumerate(LAYERS) if l >= 27]
    code = urteil_i18n([R["I18N"][i] for i in spaet], [R["FREMD"][i] for i in spaet], len(POS))
    P("")
    P("VERDIKT (Jacobi, Schichten %s): %s" % ([LAYERS[i] for i in spaet], code))
    if code == "LOKALISIERUNG-ERLEDIGT":
        P("  Die Fremdschrift-Menge spikt an Q, die Lokalisierungs-Bezeichner NICHT.")
        P("  Die i18n-Lesart ist auf den vorhandenen Daten erledigt: was der")
        P("  Transport an Q aufmacht, ist Schrift und Sprache, nicht Code.")
    elif code == "LOKALISIERUNG":
        P("  Umgekehrt: die Lokalisierungs-Bezeichner spiken, die fremde Schrift")
        P("  nicht. Dann war die Koeder-Lesart die falsche Spur.")
    elif code == "BEIDE-KONZEPTE":
        P("  Beide Mengen spiken an Q. Der Zustand traegt beides; die Mengen")
        P("  trennen die Hypothesen dann nicht - es braucht den Schrift-Tausch.")
    else:
        P("  Keine der beiden Mengen ragt an Q heraus. Die urspruengliche Aussage")
        P("  stuetzt sich damit allein auf die breite Fremdschrift-Maske.")
    P("")
    P("SPRACHE dient als Bruecke: sie ist lateinisch geschrieben wie I18N, aber")
    P("inhaltlich Sprache wie FREMD. Spikt SPRACHE mit FREMD und nicht mit I18N,")
    P("liegt es am Konzept und nicht an der Schrift der Token.")
    P("")
    P("(Kein neuer Vorwaertspass. Gleiche transportierte Vektoren, nur andere")
    P(" Zielmengen. Widerlegt die i18n-Lesart oder nicht - bestaetigen kann es")
    P(" die Koeder-Lesart nicht, dafuer braucht es den Schrift-Tausch.)")
    KONZEPT_RESULTS = dict(verdict=code, layers=LAYERS, positions=POS, Q=Q, n_corpus=NOK,
                           groessen={k: int(MASKEN[k].sum()) for k in MASKEN},
                           jacobi={k: M[k].tolist() for k in M},
                           logit={k: ML[k].tolist() for k in ML},
                           rang_jacobi=R, rang_logit=RL)
    GL["KONZEPT_RESULTS"] = KONZEPT_RESULTS
    schreibe("KONZEPT_RESULTS.json", json.dumps(KONZEPT_RESULTS, ensure_ascii=False, indent=1))
    np.savez_compressed(os.path.join(OUT, "konzept_massen.npz"),
                        pos=np.array(POS), layers=np.array(LAYERS),
                        **{("jac_" + k): M[k] for k in M},
                        **{("log_" + k): ML[k] for k in ML})
except Exception as e:
    import traceback
    FEHLER = traceback.format_exc(); P(""); P("ABBRUCH: %s" % e); P(FEHLER)
finally:
    schreibe("bericht_konzepte.txt", "\n".join(LINES))
    print("GESCHRIEBEN NACH:", OUT)
    print("\n".join(LINES[-40:]) if not FEHLER else FEHLER.splitlines()[-1])
