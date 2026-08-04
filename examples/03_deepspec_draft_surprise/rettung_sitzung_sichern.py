# === RETTUNG: alles aus der laufenden Sitzung nach Drive sichern ===========
# In die NOCH LEBENDE Laufzeit einfuegen und ausfuehren. Greift auf die
# Variablen zu, die die vorige Zelle hinterlassen hat, schreibt Zahlen,
# Abbildungen und einen Textbericht nach Drive - unabhaengig vom PDF-Export.
import os, sys, json, time, glob
import numpy as np
if not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive; drive.mount("/content/drive")
OUT="/content/drive/MyDrive/WeirdChat_Runs/rettung_"+time.strftime("%Y%m%d-%H%M%S")
os.makedirs(OUT,exist_ok=True)
GL=globals()   # NICHT 'G' nennen: so heisst in der RDX-Zelle der Graph
def _enc(o):
    if isinstance(o,np.ndarray): return o.tolist()
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)): return float(o)
    if isinstance(o,(np.bool_,)): return bool(o)
    return str(o)
# ---- 1. Ergebnis-Objekte ---------------------------------------------------
saved=[]
for name in [k for k in list(GL) if k.endswith("_RESULTS")]:
    with open(os.path.join(OUT,name+".json"),"w",encoding="utf-8") as f:
        json.dump(GL[name],f,ensure_ascii=False,indent=1,default=_enc)
    saved.append(name+".json")
# ---- 2. Rohdaten, soweit vorhanden -----------------------------------------
arr={}
for k in ("Y","YM","YR","E","L","lab10","lab01","FAMS","ZIEL","LENS","KOE","emb",
          "JAC","LOG","POS","P","base_prof","blind_prof"):
    v=GL.get(k)
    if v is None: continue
    try: arr[k]=np.asarray(v)
    except Exception: pass
if "G" in GL and isinstance(GL["G"],dict):
    for k in ("am_10","am_01","diff_10","r0_rank","r1_rank"):
        if k in GL["G"]:
            try: arr["graph_"+k]=np.asarray(GL["G"][k])
            except Exception: pass
if arr:
    np.savez_compressed(os.path.join(OUT,"rohdaten.npz"),**arr)
    saved.append("rohdaten.npz (%s)"%", ".join(sorted(arr)))
if "sel" in GL:
    with open(os.path.join(OUT,"prompts.json"),"w",encoding="utf-8") as f:
        json.dump([{"id":p,"text":GL["PROMPTS"][p]} for p in GL["sel"]],f,
                  ensure_ascii=False,indent=1)
    saved.append("prompts.json")
# ---- 3. Abbildungen: noch offene Figuren, sonst neu zeichnen ---------------
import matplotlib.pyplot as plt
n=0
for num in plt.get_fignums():
    n+=1; plt.figure(num).savefig(os.path.join(OUT,"abb_%02d.png"%n),dpi=150,
                                  bbox_inches="tight")
if n==0 and {"emb","Y","lab10","STAT"} <= set(GL):
    emb,Y,lab10,STAT=GL["emb"],GL["Y"],GL["lab10"],GL["STAT"]
    tab=GL.get("tab",[]); nms=list(STAT)
    fig=plt.figure(figsize=(15,4.6)); gs=fig.add_gridspec(1,3,wspace=.3)
    ax=fig.add_subplot(gs[0,0])
    sc=ax.scatter(emb[:,0],emb[:,1],c=Y,cmap="viridis",s=42,edgecolor="#222",linewidth=.4)
    plt.colorbar(sc,ax=ax,fraction=.046,label="log10 Fremdschrift-Druck")
    if "ZIEL" in GL and np.sum(GL["ZIEL"])>0:
        z=np.asarray(GL["ZIEL"])>0
        ax.scatter(emb[z,0],emb[z,1],s=150,facecolors="none",edgecolors="#DC2626",
                   linewidth=1.6,label="Zielverhalten"); ax.legend(frameon=False,fontsize=8)
    ax.set_title("RDX-Differenzgraph (Richtung 10)",fontsize=10)
    ax2=fig.add_subplot(gs[0,1])
    if tab:
        cls=[r["label"] for r in tab]; rts=[r["rate"] for r in tab]
        ax2.bar(range(len(cls)),rts,color=["#DC2626" if r>np.mean(Y) else "#2563EB" for r in rts])
        ax2.axhline(np.mean(Y),ls="--",c="#666",lw=1)
        ax2.set_xticks(range(len(cls))); ax2.set_xticklabels(["C%d"%c for c in cls],fontsize=8)
    ax2.set_ylabel("log10 Druck"); ax2.set_title("Druck je RDX-Cluster",fontsize=10)
    ax3=fig.add_subplot(gs[0,2])
    es=[STAT[x]["eta"] for x in nms]
    ax3.barh(range(len(nms))[::-1],es,color=["#DC2626","#F59E0B","#9CA3AF","#9CA3AF"][:len(nms)])
    for i,x in enumerate(nms):
        ax3.text(es[i]+.004,len(nms)-1-i,"p=%.3f"%STAT[x]["p"],va="center",fontsize=8)
    ax3.set_yticks(range(len(nms))[::-1]); ax3.set_yticklabels(nms,fontsize=8)
    ax3.set_xlabel("eta²"); ax3.set_title("Differenz gegen Einzelschichten",fontsize=10)
    fig.savefig(os.path.join(OUT,"abb_01.png"),dpi=150,bbox_inches="tight"); n=1
    plt.show()
if n: saved.append("%d Abbildung(en)"%n)
# ---- 4. Textbericht: die Zahlen noch einmal, druckbar ----------------------
rep=[]
def P(s=""):
    rep.append(s); print(s)
P("BERICHT %s"%time.strftime("%Y-%m-%d %H:%M:%S"))
if "STAT" in GL:
    P("\nErklaerte Varianz (eta^2), Median ueber Saaten, schlechtestes p:")
    for k,v in GL["STAT"].items():
        P("  %-28s eta^2=%.4f [%.4f..%.4f]  p=%.4f | innerhalb %.4f"
          %(k,v["eta"],v["eta_lo"],v["eta_hi"],v["p"],v.get("p_within",float("nan"))))
if "tab" in GL and GL["tab"]:
    P("\nCluster (Richtung 10):")
    cols=[c for c in ("label","n","rate","druck","kipp","laenge","koeder","ziel")
          if c in GL["tab"][0]]
    P("  "+" ".join("%10s"%c for c in cols))
    for r in GL["tab"]:
        P("  "+" ".join(("%10.4g"%r[c] if isinstance(r[c],float) else "%10s"%r[c]) for c in cols))
for k in ("YM","Y","YR"):
    if k in GL:
        v=np.asarray(GL[k]).astype(float)
        P("\n%s: n=%d Median %.4g | Spanne %.4g .. %.4g | Streuung %.4g"
          %(k,v.size,np.median(v),v.min(),v.max(),v.std()))
for name in [k for k in list(GL) if k.endswith("_RESULTS")]:
    P("\n%s:"%name)
    for k,v in GL[name].items():
        s=json.dumps(v,ensure_ascii=False,default=_enc)
        P("  %-16s %s"%(k,s[:160]+("…" if len(s)>160 else "")))
saved.append("bericht.txt")
P("\nGESICHERT in %s"%OUT)
for _s in saved: P("  - "+_s)
with open(os.path.join(OUT,"bericht.txt"),"w",encoding="utf-8") as f:
    f.write("\n".join(rep))
