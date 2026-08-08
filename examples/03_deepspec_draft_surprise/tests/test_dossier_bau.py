"""Der Dossier-Bau von phase20_dossier_L33_E228, zweifach geprueft.

Erst die reine Ordnungslogik (Namen, Ausschluesse, Manifest, Auftragstext) -
direkt, weil eine Miniaturwelt eine vertauschte Namenstafel nicht bemerken
kann. Dann ein Rauchtest der GANZEN Zelle gegen die Miniatur aus dem
Kern-Test und einen nachgebauten Drive-Baum: die Dateisystem-Abschnitte
muessen durchlaufen, die GPU-Abschnitte duerfen als AUSGEWIESENE Luecken
enden - aber nie stumm.
"""
import collections
import csv as csv_modul
import gc
import json
import math
import os
import random
import re
import shutil
import sys
import time
import unicodedata

import numpy as np
import pytest

import test_kern_lauf as KERNWELT
from mini_torch import mach_torch

HIER = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HIER, "..", "phase20_dossier_L33_E228.ipynb")


def _quelle():
    with open(NB, encoding="utf-8") as f:
        nb = json.load(f)
    return "".join("".join(c.get("source", [])) for c in nb["cells"]
                   if c.get("cell_type") == "code")


@pytest.fixture(scope="module")
def L():
    q = _quelle()
    a = q.index("# ---------------- Dossier-Logik")
    b = q.index("# ---------------- Ausfuehrung")
    ns = dict(math=math, random=random, np=np)
    exec(compile(q[a:b], "dossier_logik", "exec"), ns)
    return ns


def test_sprechnamen(L):
    sn = L["sprechname"]
    # exakte Tabelle: die vier disambiguierten Laeufe
    assert sn("phase18_kern_20260807-040218") == \
        "phase18_einzelscan_durchgang0__20260807-040218"
    assert sn("phase18_kern_20260808-015932") == \
        "phase18_einzelscan_durchgang1_replikation__20260808-015932"
    # Praefix: Zeitstempel bleibt IMMER erhalten - sonst verliert das
    # Dossier die Herkunft
    assert sn("phase16_kurve_20260806-215752") == \
        "phase16_dosis_wirkungs_kurve__20260806-215752"
    assert sn("phase13_rechnen_20260806-120000") == \
        "phase13_negativkontrolle_rechnen__20260806-120000"
    # Unbekanntes geht unveraendert durch - Vollstaendigkeit vor Schoenheit
    assert sn("phase12_irgendwas_neues_20260901-000000") == \
        "phase12_irgendwas_neues_20260901-000000"


def test_ausschluss_hat_immer_einen_grund(L):
    assert L["ist_ausgeschlossen"]("phase18_kern_20260807-152535")
    assert L["ist_ausgeschlossen"]("phase16_kurve_20260806-215752") is None
    # Praefix-Ausschluss: fremde Untersuchungen und der Bau-Lauf selbst -
    # deren Zeitstempel sind nicht vorab bekannt
    assert L["ist_ausgeschlossen"]("sudoku_sprache_pilot_20260807-033337")
    assert L["ist_ausgeschlossen"]("phase20_dossier_20260808-070430")
    assert L["ist_ausgeschlossen"]("phase18_kern_20260806-233216")
    assert L["ist_ausgeschlossen"]("phase18_kern_20260808-015507")
    for name, grund in L["AUSSCHLUSS"].items():
        assert len(grund) > 40, (name, "Ausschluss ohne echte Begruendung")
    # Kein ausgeschlossener Lauf darf zugleich einen Sprechnamen haben -
    # das hiesse, er wuerde doch kopiert
    for name in L["AUSSCHLUSS"]:
        assert name not in L["SPRECH_EXAKT"], name


def test_zu_entfernen_raeumt_altkopien(L):
    """Der erste Bau hat mitgenommen, was damals nicht auf der Tafel stand.
       Ein Neuaufbau muss diese Kopien entfernen - sonst wird das Dossier nie
       wieder konsistent, egal wie oft man die Zelle laufen laesst."""
    ze = L["zu_entfernen"]
    da = ["phase16_dosis_wirkungs_kurve__20260806-215752",
          "sudoku_sprache_pilot_20260807-033337",
          "phase20_dossier_20260808-070430",
          "phase18_kern_20260806-233216",
          "phase18_kern_20260808-015907",
          "AUSGESCHLOSSEN.md",
          "phase12_fruehe_maskenlaeufe_rohdaten"]
    weg = ze(da)
    assert "sudoku_sprache_pilot_20260807-033337" in weg
    assert "phase20_dossier_20260808-070430" in weg
    assert "phase18_kern_20260806-233216" in weg
    assert "phase18_kern_20260808-015907" in weg
    # aber niemals die gueltigen Kopien oder die Doku selbst
    assert "phase16_dosis_wirkungs_kurve__20260806-215752" not in weg
    assert "AUSGESCHLOSSEN.md" not in weg
    assert "phase12_fruehe_maskenlaeufe_rohdaten" not in weg


def test_manifest_und_groessen(L):
    assert L["mensch_groesse"](512.0) == "512.0 B"
    assert L["mensch_groesse"](2048.0) == "2.0 KB"
    assert L["mensch_groesse"](3.5 * 2**30) == "3.5 GB"
    z = L["manifest_zeilen"]([("b/zwei.json", 100), ("a/eins.txt", 2048)])
    assert z[0].startswith("a/eins.txt") and "2.0 KB" in z[0]
    assert z[1].startswith("b/zwei.json")


def test_auftrag_traegt_die_kernfragen(L):
    t = L["auftrag_text"]()
    for muss in ("Kernfrage 1", "Kernfrage 2", "Kernfrage 3",
                 "Schicht 33, Experte 228", "Fallstricke",
                 "Methodische Auflagen", "empirische Null",
                 "81 %", "bf16", "Erwartete Abgaben"):
        assert muss in t, muss
    # Artefakt-Hygiene: kein Assistenten-Modellname in ausgelieferten Texten
    assert "fable" not in t.lower() and "opus" not in t.lower()
    # Ein externer Leser muss OHNE Rueckfrage an Daten, Code und Checkpoint
    # kommen - sonst ist das Dossier nur fuer den brauchbar, der es gebaut hat
    for url in (L["DRIVE_FREIGABE"], L["CODE_URL"], L["MODELL_URL"]):
        assert url in t, url
    assert L["DRIVE_FREIGABE"].startswith("https://drive.google.com/drive/folders/")


def test_auftrag_liegt_auch_im_repo(L):
    """Der Auftrag existiert zweimal: im Drive-Dossier (wo die Daten liegen)
       und im Repo (wo man ihn verlinken kann). Zwei Fassungen, eine Quelle -
       weicht die Repo-Datei vom Notebook ab, ist eine von beiden veraltet,
       und niemand wuesste welche."""
    repo = os.path.join(HIER, "..", "ANALYSE_AUFTRAG_L33_E228.md")
    assert os.path.isfile(repo), "ANALYSE_AUFTRAG_L33_E228.md fehlt im Repo"
    assert open(repo, encoding="utf-8").read() == L["auftrag_text"]()


def test_lies_mich_nennt_jeden_ordner(L):
    t = L["lies_mich_text"](12, 4, "liegt bei (37.5 GB, FP8-Original)")
    assert L["DRIVE_FREIGABE"] in t and L["CODE_URL"] in t
    for o in L["ORDNER"]:
        assert o.rstrip("/") in t, o
    assert "12" in t and "4" in t and "37.5 GB" in t


@pytest.fixture(scope="module")
def gebaut(tmp_path_factory):
    """Die ganze Zelle gegen die Miniatur und einen nachgebauten Drive-Baum."""
    wurzel = tmp_path_factory.mktemp("drive")
    runs = wurzel / "WeirdChat_Runs"
    runs.mkdir()
    # ein gueltiger Lauf, ein ausgeschlossener, beide Einzelscan-Durchgaenge
    (runs / "phase16_kurve_20260806-215752").mkdir()
    (runs / "phase16_kurve_20260806-215752" / "KURVE_RESULTS.json").write_text("{}")
    (runs / "phase18_kern_20260807-152535").mkdir()
    (runs / "phase18_kern_20260807-152535" / "KERN_RESULTS.json").write_text("{}")
    for ordner, wdh in (("phase18_kern_20260807-040218", 0),
                        ("phase18_kern_20260808-015932", 1)):
        (runs / ordner).mkdir()
        (runs / ordner / "KERN_RESULTS.json").write_text(json.dumps(
            dict(verdict="EINZELNER-TRAEGT", kern=[[33, 228]], wiederholung=wdh,
                 scan_basis=11, scan_voll=4, aussen_treffer=[])))
    (wurzel / "weird_transcripts.jsonl").write_text('{"id":"p1"}\n')
    # Ordner, die der Bau ausschliessen bzw. beim Neuaufbau ENTFERNEN muss:
    # fremder Pilot, der Bau-Lauf selbst, der bitidentische phase18-Zwilling,
    # ein abgebrochener Start
    for extra in ("sudoku_sprache_pilot_20260807-033337",
                  "phase20_dossier_20260808-070430",
                  "phase18_kern_20260806-233216",
                  "phase18_kern_20260808-015507"):
        (runs / extra).mkdir()
        (runs / extra / "protokoll.txt").write_text("x")
    # Altkopien aus einem frueheren Bau, die jetzt verschwinden muessen
    alt = wurzel / "WeirdChat_Dossier_L33_E228" / "03_laeufe"
    alt.mkdir(parents=True)
    for stale in ("sudoku_sprache_pilot_20260807-033337",
                  "phase18_kern_20260806-233216"):
        (alt / stale).mkdir()
        (alt / stale / "protokoll.txt").write_text("alt")
    ausgabe = tmp_path_factory.mktemp("run_out")

    w = KERNWELT.Welt()

    class Tok:
        pad_token_id = 0
        pad_token = None
        eos_token = 1
        padding_side = "right"

        def __call__(self, *a, **k):
            return w.tok(*a, **k)

        def decode(self, *a, **k):
            return w.decode(*a, **k)

        def save_pretrained(self, pfad):
            os.makedirs(pfad, exist_ok=True)
            open(os.path.join(pfad, "tokenizer.json"), "w").write("{}")

    q = _quelle()
    marke = "# ---------------- reine Logik"
    ns = dict(os=os, re=re, math=math, torch=mach_torch(), collections=collections,
              unicodedata=unicodedata, random=random, np=np, glob=__import__("glob"),
              json=json, gc=gc, sys=sys, time=time, shutil=shutil, csv=csv_modul,
              model=w, tokenizer=Tok(), PROMPTS={"p1": KERNWELT.PROMPT}, ZIEL_ID="p1",
              N_BSP=8, N_AKT=3, MAX_NEW=8, CHUNK=8, TEMP=1.0, SEED=5, MIN_BSP=2,
              N_PRUEF=2, WIEDERHOLUNG=0,
              DRIVE_WURZEL=str(wurzel), HOLE_REPO=False, VOLLE_GEWICHTE=False,
              wc_save=lambda name, obj: None, wc_save_all=lambda: None,
              RUN_OUT=str(ausgabe))
    np.random.seed(7)
    try:
        exec(compile(q[q.index(marke):], "phase20_dossier", "exec"), ns)
    except SystemExit:
        pass
    return str(wurzel), ns


def test_dossier_entsteht(gebaut):
    wurzel, ns = gebaut
    R = ns["DOSSIER_RESULTS"]
    ziel = os.path.join(wurzel, "WeirdChat_Dossier_L33_E228")
    assert os.path.isdir(ziel)
    for o in ns["ORDNER"]:
        assert os.path.isdir(os.path.join(ziel, o)), o
    assert R["verdict"] in ("DOSSIER-VOLLSTAENDIG", "DOSSIER-MIT-LUECKEN")


def test_laeufe_kopiert_und_ausgeschlossen(gebaut):
    """Der gueltige Lauf liegt unter sprechendem Namen im Dossier, der
       ausgeschlossene NICHT - aber sein Ausschluss steht mit Begruendung in
       AUSGESCHLOSSEN.md. Stilles Filtern waere selbst eine Inkonsistenz."""
    wurzel, ns = gebaut
    L = os.path.join(wurzel, "WeirdChat_Dossier_L33_E228", "03_laeufe")
    assert os.path.isdir(os.path.join(
        L, "phase16_dosis_wirkungs_kurve__20260806-215752"))
    da = os.listdir(L)
    assert not any("152535" in d for d in da), da
    # Praefix-Ausschluesse: nichts Fremdes, nicht der Bau-Lauf selbst
    assert not any(d.startswith("sudoku_") for d in da), da
    assert not any(d.startswith("phase20_dossier") for d in da), da
    # und die Altkopien aus dem ersten Bau sind ENTFERNT, nicht nur nicht
    # neu kopiert
    assert "phase18_kern_20260806-233216" not in da, da
    doku = open(os.path.join(L, "AUSGESCHLOSSEN.md"), encoding="utf-8").read()
    assert "phase18_kern_20260807-152535" in doku
    assert "Determinismusnachweis" in doku


def test_befund_und_daten_liegen_bei(gebaut):
    wurzel, ns = gebaut
    B = os.path.join(wurzel, "WeirdChat_Dossier_L33_E228", "01_befund")
    kurz = json.load(open(os.path.join(B, "einzelscan_kurzfassung.json")))
    assert len(kurz) == 2
    assert all(v["kern"] == [[33, 228]] for v in kurz.values())
    D = os.path.join(wurzel, "WeirdChat_Dossier_L33_E228", "02_daten")
    assert os.path.isfile(os.path.join(D, "weird_transcripts.jsonl"))
    prompt = open(os.path.join(D, "prompt_der_untersuchung.md"), encoding="utf-8").read()
    for arm in ("NEU", "JA", "BR1", "MORSE", "SR", "RU"):
        assert "## Arm %s" % arm in prompt, arm


def test_auftrag_und_manifest_geschrieben(gebaut):
    wurzel, ns = gebaut
    Z = os.path.join(wurzel, "WeirdChat_Dossier_L33_E228")
    assert "Kernfrage 3" in open(os.path.join(Z, "00_auftrag",
                                              "analyse_auftrag.md"),
                                 encoding="utf-8").read()
    assert os.path.isfile(os.path.join(Z, "LIES_MICH.md"))
    inhalt = open(os.path.join(Z, "INHALT.txt"), encoding="utf-8").read()
    assert "analyse_auftrag.md" in inhalt and "weird_transcripts" in inhalt


def test_luecken_sind_ausgewiesen_nicht_stumm(gebaut):
    """Die Miniatur hat keine Schicht 33 und keine Router-Module - die
       GPU-Abschnitte MUESSEN als Luecken enden, mit Namen. Ein Dossier-Bau,
       der solche Abschnitte still ueberspringt, wuerde auf der A100 genauso
       still Unsinn liefern."""
    _, ns = gebaut
    R = ns["DOSSIER_RESULTS"]
    assert R["verdict"] == "DOSSIER-MIT-LUECKEN"
    namen = [n for n, _ in R["probleme"]]
    assert "router" in namen, namen
    assert "aktivierungen" in namen, namen
    # aber die Dateisystem-Abschnitte duerfen NICHT darunter sein
    for gut in ("laeufe", "ausschluss-doku", "datensatz", "prompt",
                "einzelscan-ergebnisse", "auftrag", "lies-mich", "manifest"):
        assert gut not in namen, (gut, namen)
