# === ABSCHLUSS: Auswertung mit dem, was bis zum Abbruch gemittelt wurde =====
# Nach dem Unterbrechen der Hauptzelle einfuegen und ausfuehren. ACC haelt die
# SUMME ueber die durchgelaufenen Korpus-Prompts (NOK Stueck) - geteilt wird
# erst am Schleifenende, das beim Abbruch nicht mehr erreicht wurde.
BEREITS_GETEILT=False      # nur auf True setzen, wenn die Zelle REGULAER durchlief
import numpy as np, torch, os
assert "ACC" in globals() and "NOK" in globals(), "ACC/NOK fehlen - lief die Hauptzelle?"
assert NOK>0, "kein einziger Korpus-Prompt fertig"
MEAN={l:(ACC[l] if BEREITS_GETEILT else ACC[l]/NOK) for l in LAYERS}
print("Auswertung ueber %d von %d Korpus-Prompts (Weg: %s, Stapel %d)"
      %(NOK,len(corp),MODE,BS))
M_script=torch.tensor(np.load(MASK_NPZ)["script"])
def fmass(lg):
    p=torch.softmax(lg.float(),-1); V=p.shape[-1]
    m=M_script.to(p.device)
    if m.shape[0]<V: m=torch.cat([m,torch.zeros(V-m.shape[0],dtype=torch.bool,device=m.device)])
    return p[...,m[:V]].sum(-1)
JAC=np.zeros((len(LAYERS),len(POS))); LOG=np.zeros_like(JAC); TOPS={}
with torch.no_grad():
    for i,l in enumerate(LAYERS):
        lj=unembed(MEAN[l].to(model.device)); ll=unembed(H[l].to(model.device))
        JAC[i]=fmass(lj).cpu().numpy(); LOG[i]=fmass(ll).cpu().numpy()
        TOPS[("jac",l)]=[tokenizer.decode([t]) for t in lj[jQ].topk(TOPK).indices.tolist()]
        TOPS[("log",l)]=[tokenizer.decode([t]) for t in ll[jQ].topk(TOPK).indices.tolist()]
        del lj,ll
print("\nWas die Jacobi-Linse am Koeder-Token Q=%d liest (Top-%d):"%(Q,TOPK))
for l in LAYERS: print("  L%-2d  %s"%(l," | ".join(repr(t) for t in TOPS[("jac",l)])))
print("\nLogit-Linse an derselben Stelle:")
for l in LAYERS: print("  L%-2d  %s"%(l," | ".join(repr(t) for t in TOPS[("log",l)])))
BEST=int(np.argmax(JAC[:,jQ])); rq=rank_of(JAC[BEST],jQ); rl=rank_of(LOG[BEST],jQ)
print("\nFremdschrift-Masse (staerkste Jacobi-Schicht L%d):"%LAYERS[BEST])
print("  Jacobi: Q=%.4f | Median ueber %d Positionen %.4f | Rang von Q: %d"
      %(JAC[BEST,jQ],len(POS),float(np.median(JAC[BEST])),rq))
print("  Logit : Q=%.4f | Median %.4f | Rang von Q: %d"
      %(LOG[BEST,jQ],float(np.median(LOG[BEST])),rl))
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(15,4.4))
for ax,(M,name) in zip(axs,[(JAC,"Jacobi-Linse"),(LOG,"Logit-Linse")]):
    im=ax.imshow(np.log10(np.maximum(M,1e-12)),aspect="auto",cmap="magma",origin="lower")
    ax.set_yticks(range(len(LAYERS))); ax.set_yticklabels(["L%d"%l for l in LAYERS],fontsize=8)
    ax.set_xticks(range(len(POS)))
    ax.set_xticklabels(["%d %s"%(p,tokenizer.decode([IDS[p]]).strip()[:7]) for p in POS],
                       rotation=90,fontsize=7)
    ax.axvline(jQ,color="#22D3EE",lw=1.6)
    ax.set_title("%s: log10 Fremdschrift-Masse (%d Prompts)\n(tuerkis = Koeder-Token Q=%d)"
                 %(name,NOK,Q),fontsize=10)
    plt.colorbar(im,ax=ax,fraction=.03,pad=.02)
plt.tight_layout(); plt.show()
code=verdict_jlens(rq,len(POS),rl)
print("\nVERDIKT:",end=" ")
if code=="RICHTUNG":
    print("VERBALISIERBAR: die Jacobi-Linse liest am Koeder-Token eine Fremdschrift-")
    print("  Disposition (Rang %d von %d bei L%d), die Logit-Linse dort nicht"%(rq,len(POS),LAYERS[BEST]))
    print("  (Rang %d). Der Zustand ist darauf gerichtet, etwas zu SAGEN, was im"%rl)
    print("  Residuum selbst noch nicht steht.")
elif code=="BEIDE":
    print("BEIDE LINSEN: die Disposition steht schon im Residuum (Logit-Rang %d),"%rl)
    print("  der Transport bestaetigt sie (Rang %d), fuegt aber nichts hinzu."%rq)
else:
    print("KEIN AUSSCHLAG AM KOEDER: Q ragt in keiner Linse heraus (Jacobi-Rang %d,"%rq)
    print("  Logit-Rang %d von %d)."%(rl,len(POS)))
print("(Gemittelt ueber %d Korpus-Prompts statt der geplanten %d - der Transport"%(NOK,len(corp)))
print(" ist dadurch verrauschter, die Reduktion selbst bleibt unveraendert.)")
JLENS_RESULTS=dict(verdict=code,mode=MODE,layers=LAYERS,positions=POS,Q=Q,K=K,
                   jac=JAC.tolist(),logit=LOG.tolist(),n_corpus=int(NOK),
                   n_geplant=len(corp),rank_jac=rq,rank_logit=rl,
                   best_layer=LAYERS[BEST],teillauf=True,
                   tops={"%s_L%d"%(a,b):v for (a,b),v in TOPS.items()})
np.savez_compressed(os.path.join(RUN_OUT,"jlens_rohdaten.npz"),
                    jac=JAC,logit=LOG,pos=np.array(POS),layers=np.array(LAYERS),
                    **{"mean_L%d"%l:MEAN[l].numpy() for l in LAYERS})
wc_save_all()
