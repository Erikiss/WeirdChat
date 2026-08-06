"""Im Original-Arm faellt auf: Ziehungen, deren Kopf 'Service (Local Name)'
lautet, kippen 14/17 - solche mit 'Service Local Name' nur 1/10. Beides ist
englisch und beides steht VOR der Entscheidung. Verdacht: die Klammer oeffnet
im Kopf ein zweites Feld, und in dieses Feld wird dann lokalisiert.
Das ist POST HOC am Original gefunden. Hier gegen ALLE anderen Arme geprueft."""
import json,re,math,collections,unicodedata
D=json.load(open("antworten.json",encoding="utf-8")); A=D["antworten"]
S=open("verhalten2_body.py",encoding="utf-8").read()
ns={"re":re,"math":math,"unicodedata":unicodedata}
exec(S[S.index('PHRASE="each service\'s local name"'):S.index("# ---------------- Ausfuehrung")],ns)
cb=ns["classify_breit"]; SWB=ns["SWB"]; kz=ns["kopfzellen"]
kipp=lambda t: cb(t) in SWB
def klammer(t):
    z=kz(t)
    if not z: return None
    return "(" in z[0] or (len(z)>1 and "(" in z[1])
def fisher(a,b,c,d):
    from math import lgamma,exp
    lf=lambda n: lgamma(n+1); n=a+b+c+d
    def pr(x):
        y=a+b-x; z=a+c-x; w=n-x-y-z
        if min(y,z,w)<0: return 0.0
        return exp(lf(a+b)+lf(c+d)+lf(a+c)+lf(b+d)-lf(n)-lf(x)-lf(y)-lf(z)-lf(w))
    p0=pr(a); hi=min(a+b,a+c)
    return min(1.0,sum(pr(x) for x in range(0,hi+1) if pr(x)<=p0*(1+1e-9)))
def wil(k,n,z=1.96):
    if n==0: return (0.,0.,0.)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return p,(c-h)/d,(c+h)/d
print("KLAMMER IM KOPF vs KIPPEN, je Arm (nur Antworten mit Tabelle)")
print("  %-11s %14s %14s %10s"%("Arm","mit Klammer","ohne Klammer","Fisher p"))
GK=GN=GKk=GNk=0
for nm in A:
    mk=[t for t in A[nm] if klammer(t) is True]
    on=[t for t in A[nm] if klammer(t) is False]
    if len(mk)<3 or len(on)<3:
        print("  %-11s %14s %14s %10s"%(nm,"%d"%len(mk),"%d"%len(on),"zu wenig")); continue
    a=sum(kipp(t) for t in mk); b=sum(kipp(t) for t in on)
    GK+=len(mk); GKk+=a; GN+=len(on); GNk+=b
    print("  %-11s %6d/%-3d %4.0f%% %6d/%-3d %4.0f%% %10.4f"
          %(nm,a,len(mk),100*a/len(mk),b,len(on),100*b/len(on),
            fisher(a,len(mk)-a,b,len(on)-b)))
print("")
p1,l1,h1=wil(GKk,GK); p0,l0,h0=wil(GNk,GN)
print("GEPOOLT ueber alle Arme:")
print("  mit Klammer  %4d/%-4d = %5.1f%% [%4.1f,%4.1f]"%(GKk,GK,100*p1,100*l1,100*h1))
print("  ohne Klammer %4d/%-4d = %5.1f%% [%4.1f,%4.1f]"%(GNk,GN,100*p0,100*l0,100*h0))
print("  Fisher p = %.3e"%fisher(GKk,GK-GKk,GNk,GN-GNk))
print("")
print("(Beobachtend, nicht kausal: die Klammer wird vom Modell selbst gesetzt,")
print(" nicht von uns. Sie koennte Ursache sein oder nur Anzeige derselben")
print(" schon getroffenen Wahl. Trennen liesse sich das nur, indem man den Kopf")
print(" VORGIBT - dieselbe Tabelle einmal mit und einmal ohne Klammer erzwingen.)")
