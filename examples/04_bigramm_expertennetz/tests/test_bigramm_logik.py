"""Logiktests fuer stufeA_bigramm_feuer_colab.ipynb.

Die reine Logik wird aus dem Notebook selbst extrahiert (zwischen den Marken
"reine Logik: ANFANG/ENDE") und gegen Miniaturwelten mit gepflanzter Wahrheit
geprueft: additive Welt, Paar-Welt, buchstabenblinde Welt, Positionswelten,
Luecken. Kein torch, kein Modell - alles numpy.

Lauf:  python -m pytest examples/04_bigramm_expertennetz/tests/ -q
"""
import json,os,random
import numpy as np
import pytest

HIER=os.path.dirname(os.path.abspath(__file__))
NOTEBOOK=os.path.join(HIER,"..","stufeA_bigramm_feuer_colab.ipynb")

def _lade_logik():
    nb=json.load(open(NOTEBOOK,encoding="utf-8"))
    quelle="".join("".join(z["source"]) for z in nb["cells"] if z["cell_type"]=="code")
    a=quelle.index("# ---- reine Logik: ANFANG")
    e=quelle.index("# ---- reine Logik: ENDE")
    raum={"np":np}
    exec(quelle[a:e],raum)
    return raum

L=_lade_logik()

# ---------------------------------------------------------------- Stimuli ---
def test_bigramme_vollstaendig():
    bs=L["ALPHABETE"]["lat26"]
    paare=L["bigramme"](bs)
    assert len(paare)==676
    assert len(set(paare))==676
    assert ("c","h") in paare and ("a","a") in paare

def test_morse_tabelle_vollstaendig():
    bs=L["ALPHABETE"]["lat26"]
    assert set(L["MORSE_INT"])==set(bs)
    assert all(set(v)<=set(".-") and v for v in L["MORSE_INT"].values())
    # bekannte Anker
    assert L["MORSE_INT"]["e"]=="." and L["MORSE_INT"]["t"]=="-"
    assert L["MORSE_INT"]["s"]=="..." and L["MORSE_INT"]["o"]=="---"

def test_kandidaten_sind_bigramme():
    bs=set(L["ALPHABETE"]["lat26"])
    for a,b in L["KANDIDATEN_PRIMAER"]+L["KANDIDATEN_SEKUNDAER"]:
        assert a in bs and b in bs

# ---------------------------------------------------------- Antwort lesen ---
def test_zerlege_zwei_gruppen():
    g=L["zerlege_antwort"](" .-  -... ")
    assert [x[0] for x in g]==[".-","-..."]
    assert g[0][1:] == (1,3) and g[1][1:] == (5,9)

def test_zerlege_varianten_normalisiert():
    g=L["zerlege_antwort"]("·− / −···")
    assert [x[0] for x in g]==[".-","-..."]

def test_gueltig_nur_genau_zwei():
    z=L["zerlege_antwort"]
    assert L["gueltige_zwei"](z(".- -..."))
    assert not L["gueltige_zwei"](z(".-"))
    assert not L["gueltige_zwei"](z(".- -... --."))
    assert not L["gueltige_zwei"](z("Hallo Welt"))
    assert not L["gueltige_zwei"](z(""))

def test_korrekt_gegen_tabelle():
    z=L["zerlege_antwort"]; T=L["MORSE_INT"]
    assert L["antwort_korrekt"](z(".- -..."),"a","b",T)
    assert not L["antwort_korrekt"](z(".- -.."),"a","b",T)   # d statt b
    assert not L["antwort_korrekt"](z("-... .-"),"a","b",T)  # vertauscht

def test_prosa_mit_zwei_gruppen_zaehlt_als_gueltig():
    # "A: .-  B: -..." - Prosa drumherum stoert nicht, die Gruppen zaehlen.
    g=L["zerlege_antwort"]("A: .-  B: -...")
    assert L["gueltige_zwei"](g)

# ------------------------------------------------------------- Tokenkarte ---
def test_token_spannen_praefixe():
    stuecke=[".",".-",".- ",".- -",".- -..."]
    sp=L["token_spannen"](stuecke)
    assert sp==[(0,1),(1,2),(2,3),(3,4),(4,7)]

def test_token_spannen_schrumpfender_praefix_geklemmt():
    # Byte-BPE: Ersatzzeichen verschwindet, Praefix wird kuerzer.
    sp=L["token_spannen"](["a","ab�","ab","abc"])
    assert all(s<=e for s,e in sp)
    assert sp[2]==(2,2)   # leer geklemmt, nicht negativ

def test_token_klassen_zuordnung():
    text=".- -..."
    gruppen=L["zerlege_antwort"](text)      # (0,2) und (3,8)
    spannen=[(0,1),(1,2),(2,3),(3,5),(5,8)] # Token 3+4 im zweiten Zeichen
    kl=L["token_klassen"](spannen,gruppen)
    assert kl==[0,0,-1,1,1]

def test_token_klassen_grenzueberspanner_und_leer():
    gruppen=L["zerlege_antwort"](".- -...")
    kl=L["token_klassen"]([(1,4),(2,2)],gruppen)  # ueberspannt Trenner; leer
    assert kl==[-1,-1]

def test_token_klassen_ungueltig_alles_minus1():
    kl=L["token_klassen"]([(0,1),(1,2)],L["zerlege_antwort"](".-"))
    assert kl==[-1,-1]

# ------------------------------------------------------- Feuern je Klasse ---
def test_feuer_je_klasse_zaehlt_und_stoppt_am_eos():
    top8=np.array([[[5,1,2,3,4,6,7,8],[9,1,2,3,4,6,7,8],
                    [5,1,2,3,4,6,7,8],[5,1,2,3,4,6,7,8]]],dtype=np.int16)
    tokens=np.array([[11,12,99,13]],dtype=np.int32)   # EOS=99 an Position 2
    kl=[[0,1,0,0]]
    je=L["feuer_je_klasse"](top8,tokens,kl,eos_id=99,ziel_e=5)
    assert je==[[[1,1],[0,1],[0,0]]]   # nur Positionen 0+1 zaehlen

def test_feuer_je_klasse_pad_reihen_uebersprungen():
    top8=np.array([[[-1]*8,[5,1,2,3,4,6,7,8]]],dtype=np.int16)
    tokens=np.array([[11,12]],dtype=np.int32)
    je=L["feuer_je_klasse"](top8,tokens,[[0,0]],eos_id=99,ziel_e=5)
    assert je==[[[1,1],[0,0],[0,0]]]

# ------------------------------------------------------------- Statistik ---
def _welt(A,B,n_je,rnd,alpha=None,beta=None,gamma=None,rauschen=0.05,mu=0.3):
    R=np.zeros((A,B,n_je))
    for i in range(A):
        for j in range(B):
            m=mu+(alpha[i] if alpha is not None else 0.0) \
                +(beta[j] if beta is not None else 0.0) \
                +(gamma.get((i,j),0.0) if gamma else 0.0)
            for k in range(n_je):
                R[i,j,k]=m+rnd.gauss(0.0,rauschen)
    return R

def test_passe_additiv_findet_gepflanzte_effekte():
    rnd=random.Random(1)
    a=np.linspace(-0.1,0.1,8); b=np.linspace(0.08,-0.08,8)
    R=_welt(8,8,6,rnd,alpha=a,beta=b,rauschen=0.01)
    M=np.nanmean(R,axis=2)
    fit=L["passe_additiv"](M)
    assert float(np.max(np.abs(M-fit)))<0.02   # additiv erklaert die Welt

def test_additiv_parameter_nan_fuer_leere_zeilen():
    # Eine 0 fuer datenlose Zeilen saehe in beiden Haelften gleich aus und
    # wuerde Spiegelkorrelation vortaeuschen - darum muss dort NaN stehen.
    M=np.array([[1.0,2.0],[np.nan,np.nan]])
    _,_,_,a_aus,b_aus=L["additiv_parameter"](M)
    assert np.isnan(a_aus[1]) and np.isfinite(a_aus[0])
    assert np.isfinite(b_aus).all()

def test_stufe1_erkennt_buchstaben_und_blindheit():
    rnd=random.Random(2)
    a=np.linspace(-0.15,0.15,8)
    R_buchst=_welt(8,8,6,rnd,alpha=a,rauschen=0.03)
    p1,_=L["p_spiegel"](R_buchst,1,300,random.Random(3))
    assert p1<0.05
    R_blind=_welt(8,8,6,rnd,rauschen=0.03)
    p0,_=L["p_spiegel"](R_blind,1,300,random.Random(4))
    assert p0>0.05

def test_stufe2_paarwelt_schlaegt_an_additive_nicht():
    rnd=random.Random(5)
    a=np.linspace(-0.1,0.1,8); b=np.linspace(-0.1,0.1,8)
    gamma={(2,7):0.35,(5,1):0.3,(0,0):0.3}
    R_paar=_welt(8,8,6,rnd,alpha=a,beta=b,gamma=gamma,rauschen=0.03)
    p2,_=L["p_spiegel"](R_paar,2,300,random.Random(6))
    assert p2<0.05
    R_add=_welt(8,8,6,random.Random(500),alpha=a,beta=b,rauschen=0.03)
    p2a,_=L["p_spiegel"](R_add,2,300,random.Random(7))
    assert p2a>0.05   # Einzelwelt; die RATE prueft test_stufe2_fehlalarmrate

def test_stufe2_einzelzelle_auf_voller_groesse():
    """Der ch-Fall: EINE gepflanzte Zelle in 26x26 muss gefunden werden."""
    rnd=random.Random(55)
    R=_welt(26,26,6,rnd,alpha=np.linspace(-0.1,0.1,26),
            beta=np.linspace(-0.1,0.1,26),gamma={(2,7):0.3},rauschen=0.05)
    p,_=L["p_spiegel"](R,2,300,random.Random(56))
    assert p<0.05

def test_stufe2_fehlalarmrate_kalibriert():
    """Additive Welten mit STARKEN Buchstabeneffekten duerfen Stufe 2 nur
       mit ~alpha ausloesen (genau daran ist der zuvor erwogene
       Aus-der-Stichprobe-Gewinn mit 13.5 %% gescheitert)."""
    treffer=0; sims=80
    for s in range(sims):
        rnd=random.Random(100+s)
        a=np.linspace(-0.1,0.1,6); b=np.linspace(-0.08,0.08,6)
        R=_welt(6,6,6,rnd,alpha=a,beta=b,rauschen=0.04)
        p,_=L["p_spiegel"](R,2,150,random.Random(200+s))
        treffer+=int(p<0.05)
    assert treffer<=9, "Fehlalarmrate %d/80"%treffer

def test_stufe1_fehlalarmrate_kalibriert():
    treffer=0; sims=80
    for s in range(sims):
        rnd=random.Random(300+s)
        R=_welt(6,6,6,rnd,rauschen=0.04)
        p,_=L["p_spiegel"](R,1,150,random.Random(400+s))
        treffer+=int(p<0.05)
    assert treffer<=9, "Fehlalarmrate %d/80"%treffer

def test_stufe2_ist_einseitig():
    """Anti-replizierende Welt: die beiden Haelften sehen ENTGEGENGESETZTE
       Interaktionen. Das ist keine Paarstruktur - einseitig muss p gross
       bleiben; ein zweiseitiger Test wuerde hier faelschlich anschlagen."""
    ia,ib=L["teile_haelften"](6,random.Random(21))   # Teilung vorhersagen:
    R=_welt(8,8,6,random.Random(20),rauschen=0.02)   # p_spiegel zieht sie
    for (i,j),v in {(2,7):0.4,(5,1):0.35,(1,3):0.3}.items():  # zuerst
        for k in ia: R[i,j,k]+=v
        for k in ib: R[i,j,k]-=v
    p,r=L["p_spiegel"](R,2,300,random.Random(21))
    assert r<0 and p>0.5

def test_spitzen_zellen_finden_die_gepflanzten():
    rnd=random.Random(8)
    gamma={(2,7):0.4,(5,1):0.35}
    R=_welt(26,26,6,rnd,gamma=gamma,rauschen=0.02)
    g=L["paar_reste"](R)
    bs=L["ALPHABETE"]["lat26"]
    oben=L["spitzen_zellen"](g,bs,hoechstens=2)
    assert {z["paar"] for z in oben}=={bs[2]+bs[7],bs[5]+bs[1]}

def test_spitzen_zellen_markieren_kandidaten():
    bs=L["ALPHABETE"]["lat26"]
    g=np.zeros((26,26)); g[bs.index("c"),bs.index("h")]=0.5
    oben=L["spitzen_zellen"](g,bs,hoechstens=1)
    assert oben[0]["paar"]=="ch" and oben[0]["kandidat"]

def test_luecken_ueberleben_die_kette():
    rnd=random.Random(9)
    R=_welt(8,8,6,rnd,alpha=np.linspace(-0.1,0.1,8),rauschen=0.03)
    R[0,0,:]=np.nan; R[3,4,2:]=np.nan; R[5,:,:]=np.nan
    p1,_=L["p_spiegel"](R,1,200,random.Random(10))
    p2,_=L["p_spiegel"](R,2,200,random.Random(11))
    assert np.isfinite(p1) and p1<0.05
    assert np.isfinite(p2) and p2>0.05

# -------------------------------------------------------------- Position ----
def _positionswelt(n,r0,r1,g=20,rnd=None):
    je=[]
    for _ in range(n):
        t0=sum(1 for _ in range(g) if rnd.random()<r0)
        t1=sum(1 for _ in range(g) if rnd.random()<r1)
        je.append([[t0,g],[t1,g],[0,0]])
    return je

def test_position_erstlastig_und_gleich():
    rnd=random.Random(11)
    je=_positionswelt(40,0.7,0.2,rnd=rnd)
    p,d,n=L["positions_vergleich"](je,400,random.Random(12))
    assert L["urteil_position"](p,d,n)=="ERSTBUCHSTABE-LASTIG"
    je2=_positionswelt(40,0.45,0.45,rnd=rnd)
    p2,d2,n2=L["positions_vergleich"](je2,400,random.Random(13))
    assert L["urteil_position"](p2,d2,n2)=="GLEICHVERTEILT"

def test_position_zweitlastig():
    rnd=random.Random(16)
    je=_positionswelt(40,0.2,0.7,rnd=rnd)
    p,d,n=L["positions_vergleich"](je,400,random.Random(17))
    assert L["urteil_position"](p,d,n)=="ZWEITBUCHSTABE-LASTIG"

def test_position_zu_wenig_daten():
    je=_positionswelt(5,0.8,0.1,rnd=random.Random(14))
    p,d,n=L["positions_vergleich"](je,200,random.Random(15))
    assert L["urteil_position"](p,d,n)=="POSITION-UNGEMESSEN"

# ---------------------------------------------------------------- Urteil ----
def test_urteil_tore_in_reihenfolge():
    u=L["urteil_bigramm"]
    assert u(0.3,0.9,0.5,0.001,0.001)=="MESSFELD-TOT"
    assert u(0.9,0.5,0.5,0.001,0.001)=="MESSFELD-LUECKIG"
    assert u(0.9,0.9,0.01,0.001,0.001)=="RAHMEN-ZU-DUENN"
    assert u(0.9,0.9,None,0.001,0.001)=="RAHMEN-ZU-DUENN"
    assert u(0.9,0.9,0.5,0.5,0.001)=="BUCHSTABEN-BLIND"
    assert u(0.9,0.9,0.5,0.001,0.001)=="PAARE-EIGEN"
    assert u(0.9,0.9,0.5,0.001,0.5)=="BUCHSTABEN-ADDITIV"

def test_raten_wuerfel_baut_und_laesst_luecken():
    z={(0,0,0):[3,10],(0,0,1):[0,10],(1,2,0):[5,5]}
    R=L["raten_wuerfel"](z,3,3,2)
    assert R[0,0,0]==0.3 and R[0,0,1]==0.0 and R[1,2,0]==1.0
    assert np.isnan(R[2,2,0]) and np.isnan(R[1,2,1])

# ------------------------------------------------------- Saat im Notebook ---
def test_saat_namensraum_getrennt_und_deterministisch():
    nb=json.load(open(NOTEBOOK,encoding="utf-8"))
    quelle="".join("".join(z["source"]) for z in nb["cells"] if z["cell_type"]=="code")
    assert '"bigramm/"' in quelle or "'bigramm/'" in quelle, \
        "saat muss im Namensraum bigramm/ haengen (nicht pilot/, nicht phase18)"
    assert "WIEDERHOLUNG" in quelle, "Durchgangszaehler fehlt (Lehre aus Phase 18)"
    assert "MAPPE_SEED" in quelle
