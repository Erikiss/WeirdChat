"""Offline-Nachlese des Verhalten-v2-Laufs. Kein Modell, keine GPU - alle 1248
Antworten liegen vor. Zwei Fragen, die der Lauf selbst nicht beantwortet hat:
 (1) Der Kopf-Klassifikator war SCHRIFTBASIERT und uebersieht damit genau das,
     was den Klassifikator schon einmal in die Irre gefuehrt hat: portugiesische
     Ueberschriften. Hier ein schriftunabhaengiger Ersatz.
 (2) Der entscheidende Test ist INNERHALB des Original-Arms: 96 Ziehungen,
     identischer Prompt, nur das Sampling unterscheidet sich. Kippt genau dann,
     wenn das Modell Lesart (a) gewaehlt hat? Dann ist das Verhalten vollstaendig
     durch EINE latente Entscheidung erklaert."""
import json, re, math, collections, unicodedata, sys

D=json.load(open("antworten.json",encoding="utf-8"))
A=D["antworten"]; ARME=list(A)

SRC=open("verhalten2_body.py",encoding="utf-8").read()
ns={"re":re,"math":math}
exec(SRC[SRC.index('PHRASE="each service\'s local name"'):
         SRC.index("# ---------------- Ausfuehrung")],ns)
cs=ns["classify_answer"]; cb=ns["classify_breit"]; SW=ns["SW"]; SWB=ns["SWB"]
kopfzellen=ns["kopfzellen"]; _fremd=ns["_fremd_zeichen"]; _entakz=ns["_entakz"]
PTES=ns["PTES"]; DES=ns["DES"]; ENS=ns["ENS"]

# ---- schriftunabhaengiger Kopf-Klassifikator -------------------------------
def kopf_lesart(t):
    """(b) = die Ueberschrift ist der TEXT 'Local Name'.
       (a) = die Ueberschrift ist LOKALISIERT - fremde Schrift ODER lateinisch
             mit Akzenten/pt-es-de-Markern. Genau diese zweite Haelfte fehlte."""
    z=kopfzellen(t)
    if not z: return "keine-tabelle"
    j=" ".join(z)
    if re.search(r"local\s*nam",j,re.I): return "b_woertlich"
    if _fremd(j): return "a_lokalisiert"
    w=re.findall(r"[a-zA-ZÀ-ſ']+",_entakz(j).lower())
    mark=sum(1 for x in w if x in PTES or x in DES)
    akz=sum(1 for ch in j if ch.isalpha() and 0xC0<=ord(ch)<=0x17F)
    if mark>=1 or akz>=1: return "a_lokalisiert"
    return "sonst"
def kipp(t): return cb(t) in SWB
def verweigert(t):
    """kein Tabellenzeichen und laenger als eine Zeile -> Meta-Kommentar"""
    return "|" not in t and len(t.strip())>40

def fisher2x2(a,b,c,d):
    """zweiseitiger exakter Test fuer [[a,b],[c,d]]"""
    from math import lgamma
    def lf(n): return lgamma(n+1)
    n=a+b+c+d
    def p(a_):
        b_=a+b-a_; c_=a+c-a_; d_=n-a_-b_-c_
        if min(b_,c_,d_)<0: return 0.0
        return math.exp(lf(a+b)+lf(c+d)+lf(a+c)+lf(b+d)-lf(n)-lf(a_)-lf(b_)-lf(c_)-lf(d_))
    p0=p(a); s=0.0
    for a_ in range(0,min(a+b,a+c)+1):
        q=p(a_)
        if q<=p0*(1+1e-9): s+=q
    return min(1.0,s)

print("="*78)
print("NACHLESE - schriftunabhaengige Lesart, %d Antworten"%sum(len(v) for v in A.values()))
print("="*78)
print("")
print("KOPF-LESART je Arm (a = lokalisiert, b = Text 'Local Name')")
print("  %-11s %6s %6s %6s %6s %7s %7s"
      %("Arm","a","b","sonst","keine","Verweig","Kipp%"))
FRAK={}
for nm in ARME:
    L=collections.Counter(kopf_lesart(t) for t in A[nm])
    k=sum(kipp(t) for t in A[nm]); n=len(A[nm]); v=sum(verweigert(t) for t in A[nm])
    FRAK[nm]=(L.get("a_lokalisiert",0)/n,k/n)
    print("  %-11s %6d %6d %6d %6d %7d %6.1f%%"
          %(nm,L.get("a_lokalisiert",0),L.get("b_woertlich",0),L.get("sonst",0),
            L.get("keine-tabelle",0),v,100*k/n))

print("")
print("ZUSAMMENHANG UEBER DIE ARME: Anteil Lesart (a) vs Kipprate")
xs=[FRAK[n][0] for n in ARME]; ys=[FRAK[n][1] for n in ARME]
mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
sxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
sx=math.sqrt(sum((x-mx)**2 for x in xs)); sy=math.sqrt(sum((y-my)**2 for y in ys))
print("  Pearson r = %.3f ueber %d Arme"%(sxy/(sx*sy) if sx*sy else float("nan"),len(ARME)))

print("")
print("DER ENTSCHEIDENDE TEST - INNERHALB eines Arms (gleicher Prompt, nur Sampling):")
print("  %-11s %-26s %-26s %9s"%("Arm","Lesart (a): kippt/n","Lesart (b): kippt/n","Fisher p"))
for nm in ARME:
    ka=kb_=na=nb=0
    for t in A[nm]:
        L=kopf_lesart(t)
        if L=="a_lokalisiert": na+=1; ka+=kipp(t)
        elif L=="b_woertlich": nb+=1; kb_+=kipp(t)
    if na<3 or nb<3:
        print("  %-11s %-26s %-26s %9s"%(nm,"%d/%d"%(ka,na),"%d/%d"%(kb_,nb),"zu wenig"))
        continue
    p=fisher2x2(ka,na-ka,kb_,nb-kb_)
    print("  %-11s %-26s %-26s %9.2e"
          %(nm,"%d/%d = %5.1f%%"%(ka,na,100*ka/na),"%d/%d = %5.1f%%"%(kb_,nb,100*kb_/nb),p))

print("")
print("ORIGINAL-ARM IM DETAIL (96 Ziehungen, identischer Prompt):")
L=[kopf_lesart(t) for t in A["original"]]; Kp=[kipp(t) for t in A["original"]]
tab=collections.Counter(zip(L,Kp))
for l in ("a_lokalisiert","b_woertlich","sonst","keine-tabelle"):
    j=tab.get((l,True),0); n_=tab.get((l,False),0)
    print("  %-16s kippt %2d  kippt nicht %2d   (%d)"%(l,j,n_,j+n_))
print("")
print("  Beispiele 'sonst' die KIPPEN (was ist dort die Ueberschrift?):")
c=0
for t in A["original"]:
    if kopf_lesart(t)=="sonst" and kipp(t):
        print("    %r"%(" | ".join(kopfzellen(t))[:88])); c+=1
        if c>=6: break
print("")
print("  Beispiele 'sonst' die NICHT kippen:")
c=0
for t in A["original"]:
    if kopf_lesart(t)=="sonst" and not kipp(t):
        print("    %r"%(" | ".join(kopfzellen(t))[:88])); c+=1
        if c>=4: break

print("")
print("LATEIN-ARM - was der strenge Klassifikator uebersehen hat:")
for t in A["latein"][:3]:
    print("    %r"%t[:110])
print("  streng gekippt: %d/96 | breit gekippt: %d/96"
      %(sum(cs(t) in SW for t in A["latein"]),sum(cb(t) in SWB for t in A["latein"])))

print("")
print("VERWEIGERUNGEN / META-KOMMENTARE je Arm (kein '|' im Text):")
for nm in ARME:
    v=sum(verweigert(t) for t in A[nm])
    if v: print("  %-11s %2d/96"%(nm,v))
