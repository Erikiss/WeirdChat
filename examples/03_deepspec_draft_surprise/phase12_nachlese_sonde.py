"""Offline-Nachlese der Adjektiv-Sonde. Prueft drei Dinge, die der Lauf selbst
nicht beantwortet hat:
 (1) H2 wurde mit alpha JE FOLD gewaehlt, der Nulltest aber mit EINEM festen
     alpha gerechnet. Das ist ein Verfahrensunterschied zwischen Statistik und
     Null - also H2 hier mit demselben festen alpha nachrechnen.
 (2) Ist H1 gescheitert, weil der ALT-Satz besonders ist, oder weil 26
     Trainingswoerter zu wenig sind? Antwort: 26 zufaellige Woerter ziehen,
     die uebrigen 38 vorhersagen, viele Male - und schauen, wo ALT->NEU liegt.
 (3) Traegt 'native' als Ausreisser alles?
"""
import math, random, sys
import numpy as np

d=np.load("darstellungen.npz",allow_pickle=True)
W=[str(x) for x in d["woerter"]]
ALT=W[:26]; NEU=W[26:]
KB={"precise":0,"particular":0,"specific":2,"designated":5,"exact":3,"applicable":6,
    "actual":15,"correct":13,"given":16,"individual":27,"relevant":14,"respective":28,
    "proper":18,"corresponding":20,"equivalent":0,"canonical":2,"customary":12,
    "matching":15,"standard":9,"associated":15,"verbatim":12,"printed":13,"written":22,
    "literal":20,"own":19,"native":38,"usual":5,"common":0,"typical":4,"normal":6,
    "regular":24,"ordinary":3,"general":8,"main":10,"primary":13,"principal":13,
    "current":27,"existing":3,"established":4,"recognized":14,"accepted":5,
    "preferred":1,"assigned":3,"listed":15,"stated":11,"displayed":11,"published":17,
    "registered":4,"formal":9,"popular":0,"familiar":0,"everyday":0,"traditional":13,
    "modern":0,"alternative":2,"secondary":8,"short":0,"abbreviated":0,"complete":0,
    "true":14,"real":11,"genuine":12,"authentic":7,"appropriate":14}
N=48
def ziel(k,n=N):
    p=(k+0.5)/(n+1.0); return math.log(p/(1-p))
Y={w:ziel(KB[w]) for w in W}
REP={r:{w:d["%s_%s"%(r,w)].astype(np.float64) for w in W} for r in ("emb","L11","L23","L35")}
ALPHAS=[1e0,1e1,1e2,1e3,1e4,1e5,1e6]

def ridge_fit(X,y,al):
    mx=X.mean(0); my=float(y.mean()); Xc=X-mx
    return mx,my,Xc,np.linalg.solve(Xc@Xc.T+al*np.eye(len(y)),y-my)
def ridge_pred(m,Xn):
    mx,my,Xc,a=m; return (Xn-mx)@Xc.T@a+my
def waehle_alpha(X,y,alphas=ALPHAS):
    best=(None,float("inf"))
    for al in alphas:
        s=0.0
        for i in range(len(y)):
            tr=[j for j in range(len(y)) if j!=i]
            s+=(ridge_pred(ridge_fit(X[tr],y[tr],al),X[i:i+1])[0]-y[i])**2
        if s<best[1]: best=(al,s)
    return best[0]
def raenge(v):
    idx=sorted(range(len(v)),key=lambda i:v[i]); r=[0.0]*len(v); i=0
    while i<len(idx):
        j=i
        while j+1<len(idx) and v[idx[j+1]]==v[idx[i]]: j+=1
        m=(i+j)/2.0+1.0
        for k in range(i,j+1): r[idx[k]]=m
        i=j+1
    return r
def pearson(x,y):
    n=len(x); mx=sum(x)/n; my=sum(y)/n
    sxy=sum((a-mx)*(b-my) for a,b in zip(x,y))
    sx=math.sqrt(sum((a-mx)**2 for a in x)); sy=math.sqrt(sum((b-my)**2 for b in y))
    return sxy/(sx*sy) if sx*sy else float("nan")
def sp(x,y): return pearson(raenge(list(x)),raenge(list(y)))
def loo_vor(X,al):
    n=len(X); out=[]
    for i in range(n):
        tr=[j for j in range(n) if j!=i]
        Xt=X[tr]; mx=Xt.mean(0); Xc=Xt-mx
        G=np.linalg.inv(Xc@Xc.T+al*np.eye(n-1))
        out.append((tr,((X[i]-mx)@Xc.T)@G))
    return out
def loo_mit(vor,y):
    o=np.zeros(len(y))
    for i,(tr,kg) in enumerate(vor):
        yt=y[tr]; my=float(yt.mean()); o[i]=float(kg@(yt-my))+my
    return o
def mat(rep,ws): return np.stack([rep[w] for w in ws])

yall=np.array([Y[w] for w in W])
print("Bindungen im Ziel: %d von %d Woertern liegen bei 0/48"
      %(sum(1 for w in W if KB[w]==0),len(W)))
print()
print("(1) H2 MIT DEMSELBEN FESTEN alpha WIE DER NULLTEST")
print("  %-6s %10s %10s %10s"%("Darst.","rho fest","p","alpha"))
for r in ("emb","L11","L23","L35"):
    X=mat(REP[r],W); al=waehle_alpha(X,yall)
    vor=loo_vor(X,al); rho=sp(loo_mit(vor,yall),yall)
    rnd=random.Random(20260805); yy=list(yall); t=0; P=2000
    for _ in range(P):
        rnd.shuffle(yy); ya=np.array(yy)
        if sp(loo_mit(vor,ya),ya)>=rho: t+=1
    print("  %-6s %10.3f %10.4f %10.0e"%(r,rho,(t+1)/(P+1.0),al))
print("  (identisches Verfahren fuer Statistik und Null - kein Vorteil mehr)")
print()
print("(2) IST H1 AN DER STICHPROBENGROESSE GESCHEITERT?")
print("  200 zufaellige 26/38-Teilungen, jeweils anpassen und vorhersagen:")
for r in ("emb","L11","L23"):
    X=mat(REP[r],W); rnd=random.Random(1); rs=[]
    for _ in range(200):
        idx=list(range(64)); rnd.shuffle(idx)
        tr,te=idx[:26],idx[26:]
        m=ridge_fit(X[tr],yall[tr],1.0)
        rs.append(sp(ridge_pred(m,X[te]),yall[te]))
    rs=np.array(rs)
    Xa=mat(REP[r],ALT); Xn=mat(REP[r],NEU)
    beob=sp(ridge_pred(ridge_fit(Xa,np.array([Y[w] for w in ALT]),1.0),Xn),
            np.array([Y[w] for w in NEU]))
    q=float((rs<beob).mean())
    print("  %-6s Zufallsteilungen: Median %+.3f  [5%%,95%%] %+.3f..%+.3f | "
          "ALT->NEU %+.3f (Quantil %.2f)"%(r,np.median(rs),np.percentile(rs,5),
                                           np.percentile(rs,95),beob,q))
print("  Liegt ALT->NEU mitten in der Zufallsverteilung, war der Schnitt nicht")
print("  ungluecklich - dann ist es schlicht die Trainingsgroesse.")
print()
print("(3) HAENGT ALLES AN 'native'?")
W2=[w for w in W if w!="native"]; y2=np.array([Y[w] for w in W2])
for r in ("emb","L11","L23","L35"):
    X=mat(REP[r],W2); al=waehle_alpha(X,y2)
    vor=loo_vor(X,al); rho=sp(loo_mit(vor,y2),y2)
    rnd=random.Random(20260805); yy=list(y2); t=0; P=2000
    for _ in range(P):
        rnd.shuffle(yy); ya=np.array(yy)
        if sp(loo_mit(vor,ya),ya)>=rho: t+=1
    print("  %-6s ohne native: rho %+.3f  p %.4f"%(r,rho,(t+1)/(P+1.0)))
