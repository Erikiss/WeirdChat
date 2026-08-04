# === J-LINSE: AUSWERTUNG UND DIREKT-EXPORT NACH DRIVE =======================
# Eine Zelle, die alles macht: mittelt, liest aus, zeichnet, urteilt - und JEDE
# Zeile nach Drive schreibt, bevor sie irgendwas druckt. Was Colab anzeigt, ist
# nur eine Quittung; die Wahrheit liegt in der Datei.
# Laeuft auch, wenn die Abschlusszelle schon lief (ACC wird nicht veraendert).
BEREITS_GETEILT = False   # nur True, wenn die HAUPTZELLE regulaer durchlief
import os, sys, json, time
import numpy as np, torch
if not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive; drive.mount("/content/drive")
GL = globals()
OUT = GL.get("RUN_OUT") or ("/content/drive/MyDrive/WeirdChat_Runs/jlens_"
                            + time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(OUT, exist_ok=True)
LINES = []
def P(s=""):
    LINES.append(str(s))
def schreibe(name, text):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as f: f.write(text)
    return p
def _enc(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)
FEHLER = None
try:
    for _n in ("ACC", "NOK", "H", "POS", "LAYERS", "jQ", "IDS", "Q", "K"):
        assert _n in GL, "%s fehlt im Namensraum - lief die Hauptzelle?" % _n
    assert GL["NOK"] > 0, "kein einziger Korpus-Prompt fertig"
    ACC, NOK, H = GL["ACC"], int(GL["NOK"]), GL["H"]
    POS, LAYERS, jQ = list(GL["POS"]), list(GL["LAYERS"]), int(GL["jQ"])
    IDS, Q, K = GL["IDS"], int(GL["Q"]), int(GL["K"])
    TOPK = int(GL.get("TOPK", 6)); MODE = GL.get("MODE", "?")
    NPLAN = len(GL.get("corp", [])) or NOK
    MEAN = {l: (ACC[l] if BEREITS_GETEILT else ACC[l] / NOK) for l in LAYERS}
    P("J-LINSE AN POSITION %d - BERICHT %s" % (Q, time.strftime("%Y-%m-%d %H:%M:%S")))
    P("=" * 72)
    P("Transport: %s | gemittelt ueber %d von %d Korpus-Prompts%s"
      % ({"jvp": "Doppel-Rueckwaerts (exakt)", "fd": "zentrale Differenz"}.get(MODE, MODE),
         NOK, NPLAN, "" if NOK == NPLAN else "  (TEILLAUF)"))
    P("Koeder: K=%d %r | Q=%d %r | %d Lesepositionen | Schichten %s"
      % (K, GL["tokenizer"].decode([IDS[K]]), Q, GL["tokenizer"].decode([IDS[Q]]),
         len(POS), LAYERS))
    if GL.get("LIN"): P("Linearitaetsprobe: %.3f" % max(GL["LIN"]))
    # ---------- Auslesen ----------------------------------------------------
    M_script = torch.tensor(np.load(GL["MASK_NPZ"])["script"])
    def fmass(lg):
        p = torch.softmax(lg.float(), -1); V = p.shape[-1]
        m = M_script.to(p.device)
        if m.shape[0] < V:
            m = torch.cat([m, torch.zeros(V - m.shape[0], dtype=torch.bool, device=m.device)])
        return p[..., m[:V]].sum(-1)
    unembed = GL["unembed"]; tok = GL["tokenizer"]; dev = GL["model"].device
    JAC = np.zeros((len(LAYERS), len(POS))); LOG = np.zeros_like(JAC); TOPS = {}
    with torch.no_grad():
        for i, l in enumerate(LAYERS):
            lj = unembed(MEAN[l].to(dev)); ll = unembed(H[l].to(dev))
            JAC[i] = fmass(lj).cpu().numpy(); LOG[i] = fmass(ll).cpu().numpy()
            TOPS["jac_L%d" % l] = [tok.decode([t]) for t in lj[jQ].topk(TOPK).indices.tolist()]
            TOPS["log_L%d" % l] = [tok.decode([t]) for t in ll[jQ].topk(TOPK).indices.tolist()]
            del lj, ll
    P("")
    P("JACOBI-LINSE am Koeder-Token Q=%d (Top-%d je Schicht):" % (Q, TOPK))
    for l in LAYERS: P("  L%-2d  %s" % (l, " | ".join(repr(t) for t in TOPS["jac_L%d" % l])))
    P("")
    P("LOGIT-LINSE an derselben Stelle:")
    for l in LAYERS: P("  L%-2d  %s" % (l, " | ".join(repr(t) for t in TOPS["log_L%d" % l])))
    # ---------- Statistik ---------------------------------------------------
    def rang(vals, idx):
        v = list(vals); t = v[idx]; return 1 + sum(1 for x in v if x > t)
    BEST = int(np.argmax(JAC[:, jQ])); rq = rang(JAC[BEST], jQ); rl = rang(LOG[BEST], jQ)
    P("")
    P("FREMDSCHRIFT-MASSE (staerkste Jacobi-Schicht L%d von %s):" % (LAYERS[BEST], LAYERS))
    P("  Jacobi  Q=%.6f | Median ueber %d Positionen %.6f | Rang von Q: %d"
      % (JAC[BEST, jQ], len(POS), float(np.median(JAC[BEST])), rq))
    P("  Logit   Q=%.6f | Median %.6f | Rang von Q: %d"
      % (LOG[BEST, jQ], float(np.median(LOG[BEST])), rl))
    P("")
    P("VOLLE TABELLE - Zeilen Schichten, Spalten Positionen (Jacobi):")
    P("  %-6s %s" % ("", " ".join("%9d" % p for p in POS)))
    for i, l in enumerate(LAYERS):
        P("  L%-5d %s" % (l, " ".join("%9.2e" % v for v in JAC[i])))
    P("VOLLE TABELLE (Logit):")
    for i, l in enumerate(LAYERS):
        P("  L%-5d %s" % (l, " ".join("%9.2e" % v for v in LOG[i])))
    P("Positions-Tokens: %s" % ", ".join("%d=%r" % (p, tok.decode([IDS[p]])) for p in POS))
    # ---------- Karten ------------------------------------------------------
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.4))
    for ax, (M, name) in zip(axs, [(JAC, "Jacobi-Linse"), (LOG, "Logit-Linse")]):
        im = ax.imshow(np.log10(np.maximum(M, 1e-12)), aspect="auto", cmap="magma", origin="lower")
        ax.set_yticks(range(len(LAYERS))); ax.set_yticklabels(["L%d" % l for l in LAYERS], fontsize=8)
        ax.set_xticks(range(len(POS)))
        ax.set_xticklabels(["%d %s" % (p, tok.decode([IDS[p]]).strip()[:7]) for p in POS],
                           rotation=90, fontsize=7)
        ax.axvline(jQ, color="#22D3EE", lw=1.6)
        ax.set_title("%s: log10 Fremdschrift-Masse (%d Prompts)\n(tuerkis = Koeder Q=%d)"
                     % (name, NOK, Q), fontsize=10)
        plt.colorbar(im, ax=ax, fraction=.03, pad=.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "jlens_karten.png"), dpi=150, bbox_inches="tight")
    # ---------- Verdikt -----------------------------------------------------
    schwelle = max(2, len(POS) // 10)
    code = ("KEINE" if rq > schwelle else ("BEIDE" if rl <= schwelle else "RICHTUNG"))
    P("")
    P("VERDIKT: %s   (Schwelle: Rang <= %d von %d)" % (code, schwelle, len(POS)))
    if code == "RICHTUNG":
        P("  Die Jacobi-Linse liest am Koeder-Token eine Fremdschrift-Disposition")
        P("  (Rang %d), die Logit-Linse dort nicht (Rang %d). Der Zustand ist" % (rq, rl))
        P("  darauf gerichtet, etwas zu SAGEN, was im Residuum noch nicht steht.")
    elif code == "BEIDE":
        P("  Die Disposition steht schon im Residuum (Logit-Rang %d), der Transport" % rl)
        P("  bestaetigt sie (Rang %d), fuegt aber nichts hinzu." % rq)
    else:
        P("  Q ragt in keiner Linse heraus (Jacobi %d, Logit %d von %d)." % (rq, rl, len(POS)))
        P("  An dieser Position ist die Disposition nicht als Vokabular lesbar.")
    P("")
    P("(Deskriptiv, ein Zielprompt: die Raenge vergleichen Positionen INNERHALB")
    P(" desselben Prompts, kein Test ueber Prompts hinweg.%s)"
      % ("" if NOK == NPLAN else " Teillauf: %d/%d Prompts." % (NOK, NPLAN)))
    # ---------- Dateien -----------------------------------------------------
    JLENS_RESULTS = dict(verdict=code, mode=MODE, layers=LAYERS, positions=POS,
                         Q=Q, K=K, jac=JAC.tolist(), logit=LOG.tolist(),
                         n_corpus=NOK, n_geplant=NPLAN, rank_jac=rq, rank_logit=rl,
                         best_layer=LAYERS[BEST], teillauf=bool(NOK != NPLAN),
                         tops=TOPS)
    GL["JLENS_RESULTS"] = JLENS_RESULTS
    schreibe("JLENS_RESULTS.json", json.dumps(JLENS_RESULTS, ensure_ascii=False,
                                              indent=1, default=_enc))
    np.savez_compressed(os.path.join(OUT, "jlens_rohdaten.npz"),
                        jac=JAC, logit=LOG, pos=np.array(POS), layers=np.array(LAYERS),
                        **{"mean_L%d" % l: MEAN[l].numpy() for l in LAYERS})
    kurz = ["J-LINSE Q=%d | %s | %d/%d Prompts | VERDIKT %s" % (Q, MODE, NOK, NPLAN, code),
            "Jacobi Q=%.4g Median %.4g Rang %d/%d" % (JAC[BEST, jQ], float(np.median(JAC[BEST])), rq, len(POS)),
            "Logit  Q=%.4g Median %.4g Rang %d/%d" % (LOG[BEST, jQ], float(np.median(LOG[BEST])), rl, len(POS)),
            "staerkste Schicht L%d" % LAYERS[BEST], "",
            "Jacobi-Top-Tokens an Q:"]
    kurz += ["  L%-2d %s" % (l, " ".join(repr(t) for t in TOPS["jac_L%d" % l])) for l in LAYERS]
    kurz += ["", "Logit-Top-Tokens an Q:"]
    kurz += ["  L%-2d %s" % (l, " ".join(repr(t) for t in TOPS["log_L%d" % l])) for l in LAYERS]
    schreibe("zusammenfassung.txt", "\n".join(kurz))
except Exception as e:
    import traceback
    FEHLER = traceback.format_exc()
    P(""); P("ABBRUCH: %s" % e); P(FEHLER)
finally:
    pfad = schreibe("bericht_jlens.txt", "\n".join(LINES))
    print("=" * 60)
    print("GESCHRIEBEN NACH DRIVE:", OUT)
    for f in sorted(os.listdir(OUT)):
        try: print("  %-28s %8.1f kB" % (f, os.path.getsize(os.path.join(OUT, f)) / 1024))
        except Exception: print("  %s" % f)
    print("=" * 60)
    if FEHLER:
        print("FEHLER - siehe bericht_jlens.txt"); print(FEHLER.splitlines()[-1])
    else:
        print(open(os.path.join(OUT, "zusammenfassung.txt"), encoding="utf-8").read())
