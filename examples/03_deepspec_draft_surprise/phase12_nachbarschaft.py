"""Warum sagt Leave-one-out ueber 64 Woerter vorher (rho 0.26-0.47, p<=0.0125),
waehrend dasselbe Modell auf 24 NEUEN Woertern bei rho ~0 landet?
Verdacht: LOO sagt einen Punkt aus seinen NACHBARN vorher. Sind die Woerter in
Familien geclustert (precise/specific/particular, printed/written/published...),
ist das Interpolation in der Nachbarschaft und keine Verallgemeinerung.
Test ohne neue Daten: LOO, aber zusaetzlich die m naechsten Nachbarn aus dem
Training entfernen. Bricht rho mit wachsendem m ein, war es die Nachbarschaft."""
import math, random
import numpy as np
d=np.load("darstellungen.npz",allow_pickle=True)
W=[str(x) for x in d["woerter"]]
HIST={"precise":0,"particular":0,"specific":2,"designated":5,"exact":3,"applicable":6,
 "actual":15,"correct":13,"given":16,"individual":27,"relevant":14,"respective":28,
 "proper":18,"corresponding":20,"equivalent":0,"canonical":2,"customary":12,
 "matching":15,"standard":9,"associated":15,"verbatim":12,"printed":13,"written":22,
 "literal":20,"own":19,"native":38,"usual":5,"common":0,"typical":4,"normal":6,
 "regular":24,"ordinary":3,"general":8,"main":10,"primary":13,"principal":13,
 "current":27,"existing":3,"established":4,"recognized":14,"accepted":5,"preferred":1,
 "assigned":3,"listed":15,"stated":11,"displayed":11,"published":17,"registered":4,
 "formal":9,"popular":0,"familiar":0,"everyday":0,"traditional":13,"modern":0,
 "alternative":2,"secondary":8,"short":0,"abbreviated":0,"complete":0,"true":14,
 "real":11,"genuine":12,"authentic":7,"appropriate":14}
y=np.array([math.log(((HIST[w]+0.5)/49.0)/(1-(HIST[w]+0.5)/49.0)) for w in W])
def rg(v):
    idx=sorted(range(len(v)),key=lambda i:v[i]); r=[0.0]*len(v); i=0
    while i<len(idx):
        j=i
        while j+1<len(idx) and v[idx[j+1]]==v[idx[i]]: j+=1
        m=(i+j)/2.0+1.0
        for k in range(i,j+1): r[idx[k]]=m
        i=j+1
    return r
def pe(x,z):
    n=len(x); mx=sum(x)/n; mz=sum(z)/n
    s=sum((a-mx)*(b-mz) for a,b in zip(x,z))
    sx=math.sqrt(sum((a-mx)**2 for a in x)); sz=math.sqrt(sum((b-mz)**2 for b in z))
    return s/(sx*sz) if sx*sz else float("nan")
def sp(x,z): return pe(rg(list(x)),rg(list(z)))
def fit(X,yy,al):
    mx=X.mean(0); my=float(yy.mean()); Xc=X-mx
    return mx,my,Xc,np.linalg.solve(Xc@Xc.T+al*np.eye(len(yy)),yy-my)
def pred(m,Xn):
    mx,my,Xc,a=m; return (Xn-mx)@Xc.T@a+my
for rep in ("emb","L11","L23","L35"):
    X=np.stack([d["%s_%s"%(rep,w)] for w in W]).astype(np.float64)
    D=((X[:,None,:]-X[None,:,:])**2).sum(-1)
    zeile=[]
    for m_aus in (0,1,2,4,8):
        yh=np.zeros(len(W))
        for i in range(len(W)):
            nah=list(np.argsort(D[i]))            # [0] ist i selbst
            raus=set(nah[:1+m_aus])
            tr=[j for j in range(len(W)) if j not in raus]
            yh[i]=pred(fit(X[tr],y[tr],1.0),X[i:i+1])[0]
        zeile.append(sp(yh,y))
    print("  %-4s LOO rho:  m=0 %+.3f | m=1 %+.3f | m=2 %+.3f | m=4 %+.3f | m=8 %+.3f"
          %(rep,*zeile))
print()
print("Naechster-Nachbar-Vorhersage (kein Modell, nur die Rate des aehnlichsten Wortes):")
for rep in ("emb","L23"):
    X=np.stack([d["%s_%s"%(rep,w)] for w in W]).astype(np.float64)
    D=((X[:,None,:]-X[None,:,:])**2).sum(-1); np.fill_diagonal(D,np.inf)
    yh=np.array([y[int(np.argmin(D[i]))] for i in range(len(W))])
    print("  %-4s rho %+.3f"%(rep,sp(yh,y)))
print()
print("Naechste Nachbarn einiger Woerter in L23 (zeigt, ob es Familien gibt):")
X=np.stack([d["L23_%s"%w] for w in W]).astype(np.float64)
D=((X[:,None,:]-X[None,:,:])**2).sum(-1); np.fill_diagonal(D,np.inf)
for w in ("precise","printed","native","corresponding","modern"):
    i=W.index(w); nb=[W[j] for j in np.argsort(D[i])[:3]]
    print("  %-14s -> %s"%(w,", ".join("%s (%d%%)"%(x,round(100*HIST[x]/48)) for x in nb)))
