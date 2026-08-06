"""WANN faellt die Entscheidung? Innerhalb eines FESTEN Prompts ist der Prefill
deterministisch - der einzige Unterschied zwischen kippenden und nicht
kippenden Ziehungen ist die Sampling-Kette. Wo sie sich trennt, ist der Ort der
Entscheidung. Alles offline aus den gespeicherten Antworten."""
import json,re,math,collections,unicodedata
D=json.load(open("antworten.json",encoding="utf-8")); A=D["antworten"]
S=open("verhalten2_body.py",encoding="utf-8").read()
ns={"re":re,"math":math,"unicodedata":unicodedata}
exec(S[S.index('PHRASE="each service\'s local name"'):S.index("# ---------------- Ausfuehrung")],ns)
cb=ns["classify_breit"]; SWB=ns["SWB"]; fz=ns["_fremd_zeichen"]; PTES=ns["PTES"]
DES=ns["DES"]; ENS=ns["ENS"]; _ent=ns["_entakz"]
kipp=lambda t: cb(t) in SWB
def erstes_fremd(t):
    """Zeichenindex des ersten fremdschriftlichen Zeichens (-1 wenn keins)"""
    for i,c in enumerate(t):
        if c.isalpha() and ord(c)>=0x250 and fz(c): return i
    return -1
def wilson(k,n,z=1.96):
    if n==0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return p,(c-h)/d,(c+h)/d
def fisher(a,b,c,d):
    from math import lgamma,exp
    lf=lambda n: lgamma(n+1); n=a+b+c+d
    def pr(x):
        y=a+b-x; z=a+c-x; w=n-x-y-z
        if min(y,z,w)<0: return 0.0
        return exp(lf(a+b)+lf(c+d)+lf(a+c)+lf(b+d)-lf(n)-lf(x)-lf(y)-lf(z)-lf(w))
    p0=pr(a); hi=min(a+b,a+c)
    return min(1.0,sum(pr(x) for x in range(0,hi+1) if pr(x)<=p0*(1+1e-9)))

print("="*76)
print("WANN FAELLT DIE ENTSCHEIDUNG - Original-Arm, %d Ziehungen, identischer Prompt"
      %len(A["original"]))
print("="*76)
O=A["original"]; K=[kipp(t) for t in O]
print("gekippt (breit): %d von %d = %.1f%%"%(sum(K),len(O),100*sum(K)/len(O)))
pos=[erstes_fremd(t) for t in O if erstes_fremd(t)>=0]
pos.sort()
print("")
print("ERSTES FREMDSCHRIFTLICHES ZEICHEN, Zeichenposition in der Antwort:")
print("  n=%d | Minimum %d | 25%% %d | Median %d | 75%% %d | Maximum %d"
      %(len(pos),pos[0],pos[len(pos)//4],pos[len(pos)//2],pos[3*len(pos)//4],pos[-1]))
print("  Laenge der Antworten: Median %d Zeichen"
      %sorted(len(t) for t in O)[len(O)//2])
BINS=[(0,20),(20,40),(40,60),(60,90),(90,130),(130,999)]
print("")
print("HAZARD - unter denen, die bis dahin NICHT gekippt sind, wie viele kippen im Fenster?")
print("  %-12s %8s %8s %8s"%("Fenster","Risiko","kippt","Anteil"))
offen=len(O)
for a,b in BINS:
    risiko=sum(1 for t in O if (erstes_fremd(t)<0 or erstes_fremd(t)>=a) and len(t)>a)
    kippt=sum(1 for t in O if a<=erstes_fremd(t)<b)
    print("  %-12s %8d %8d %8s"%("%d-%d"%(a,b),risiko,kippt,
          "%.1f%%"%(100*kippt/risiko) if risiko else "-"))
print("")
print("PRAEFIX-BEDINGT: Ziehungen mit GLEICHEM Antwortanfang, aber anderem Ausgang?")
for L in (12,24,40):
    grp=collections.defaultdict(list)
    for t,k in zip(O,K): grp[t[:L]].append(k)
    gemischt=[(p,v) for p,v in grp.items() if len(v)>=4 and 0<sum(v)<len(v)]
    rein=[(p,v) for p,v in grp.items() if len(v)>=4 and (sum(v)==0 or sum(v)==len(v))]
    print("  Praefix %2d Zeichen: %d Gruppen (>=4), davon %d GEMISCHT, %d einheitlich"
          %(L,sum(1 for v in grp.values() if len(v)>=4),len(gemischt),len(rein)))
    for p,v in sorted(gemischt,key=lambda x:-len(x[1]))[:4]:
        print("    %-44r n=%2d kippt %d"%(p.replace("\n","|"),len(v),sum(v)))
print("")
print("Ein GEMISCHTER Praefix heisst: nach diesen Zeichen ist die Entscheidung noch")
print("offen. Sind fast alle Gruppen gemischt, faellt sie spaeter als Token 1.")
print("")
print("ZUM VERGLEICH - dieselbe Frage in den Extremarmen:")
for nm in ("lesart_a","blass","fern"):
    T=A[nm]; kk=[kipp(t) for t in T]
    ps=[erstes_fremd(t) for t in T if erstes_fremd(t)>=0]
    if ps: ps.sort()
    print("  %-10s kippt %4.1f%% | erstes fremdes Zeichen: Median %s"
          %(nm,100*sum(kk)/len(T),ps[len(ps)//2] if ps else "-"))
