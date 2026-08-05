"""Statischer Undefined-Name-Check fuer die Colab-Zellen.

Die Colab-Zellen sind selbstversorgend und werden als EIN Block ausgefuehrt.
Steht ein Import hinter der Stelle, die den Namen braucht, faellt das erst in
der Laufzeit auf - nach dem Modell-Download, also Minuten spaeter. Genau so ist
`phase12_tokenisierung` beim ersten Lauf gestorben: `import glob` stand hinter
dem Praeambel-Block, der glob benutzt.

Der Check laeuft ueber pyflakes und meldet nur die beiden Kategorien, die
wirklich toedlich sind: undefinierte Namen und Benutzung vor Zuweisung.

Bekannter Fehlalarm: ein Name, der in einem sofort aufgerufenen Lambda benutzt
und spaeter im selben Gueltigkeitsbereich mit `del` entfernt wird. pyflakes
rechnet dort konservativ. Solche Faelle stehen in ERLAUBT.

Geprueft werden nur die EINZELLIGEN, selbstversorgenden Notebooks. Bei den
mehrzelligen aelteren Notebooks ist ein Name aus Zelle 1 in Zelle 5 voellig in
Ordnung - dort haette der Check nur Fehlalarme. Genau in der Einzelzelle ist der
Fehler dagegen toedlich, weil es keine zweite Ausfuehrungsreihenfolge gibt.

Ueberspringt sich selbst, wenn pyflakes fehlt (pip install pyflakes).
"""
import glob
import io
import json
import os

import pytest

pyflakes_api = pytest.importorskip("pyflakes.api")
from pyflakes.reporter import Reporter  # noqa: E402

HIER = os.path.dirname(os.path.abspath(__file__))

# (Notebook-Basisname, Name) -> in einem Lambda benutzt und spaeter geloescht
ERLAUBT = {("phase12_jacobian_lens", "_t"),
           ("phase12_jacobian_lens_v2", "_t")}


def ohne_magie(quelle):
    """IPython-Zeilen (!pip, %cd) sind kein Python und muessen fuer die Analyse
       raus. Achtung: eine Fortsetzungszeile eines Formatausdrucks beginnt
       ebenfalls mit '%'. Deshalb wird zuerst der ROHE Text versucht und nur
       im Fehlerfall geputzt - sonst zerschlaegt das Putzen gueltigen Code."""
    import ast
    try:
        ast.parse(quelle)
        return quelle
    except SyntaxError:
        pass
    return "\n".join("" if l.lstrip().startswith(("!", "%")) else l
                     for l in quelle.splitlines())


def zellen(pfad):
    with open(pfad, encoding="utf-8") as f:
        nb = json.load(f)
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") == "code":
            yield i, ohne_magie("".join(c.get("source", [])))


def einzellig(pfad):
    return sum(1 for _ in zellen(pfad)) == 1


NOTEBOOKS = sorted(p for p in glob.glob(os.path.join(HIER, "..", "*.ipynb"))
                   if einzellig(p))
assert NOTEBOOKS, "keine einzelligen Notebooks gefunden - Testumfang waere leer"


def befunde(quelle, name):
    out, err = io.StringIO(), io.StringIO()
    pyflakes_api.check(quelle, name, Reporter(out, err))
    treffer = []
    for zeile in out.getvalue().splitlines():
        if "undefined name" in zeile or "before assignment" in zeile:
            treffer.append(zeile)
    return treffer


@pytest.mark.parametrize("pfad", NOTEBOOKS, ids=[os.path.basename(p) for p in NOTEBOOKS])
def test_keine_undefinierten_namen(pfad):
    basis = os.path.splitext(os.path.basename(pfad))[0]
    echte = []
    for i, quelle in zellen(pfad):
        for zeile in befunde(quelle, "%s#%d" % (basis, i)):
            if any(n in zeile for b, n in ERLAUBT if b == basis):
                continue
            echte.append(zeile)
    assert not echte, "undefinierte Namen in %s:\n  %s" % (basis, "\n  ".join(echte))


@pytest.mark.parametrize("pfad", NOTEBOOKS, ids=[os.path.basename(p) for p in NOTEBOOKS])
def test_zellen_sind_gueltiges_python(pfad):
    import ast

    for i, quelle in zellen(pfad):
        try:
            ast.parse(quelle)
        except SyntaxError as e:
            raise AssertionError("Zelle %d in %s ist kein gueltiges Python: %s"
                                 % (i, os.path.basename(pfad), e))


# Notebooks, die als CPU-Zellen angekuendigt sind. Ein Modell-Lader darin ist
# kein Schoenheitsfehler: er zieht 37 GB und stirbt auf einer CPU-Laufzeit beim
# Disk-Offload - nach zwanzig Minuten Download. Genau so ist
# phase12_tokenisierung beim zweiten Anlauf gestorben, weil der geerbte
# Praeambel-Block den Lader mitbrachte.
NUR_CPU = ["phase12_tokenisierung", "phase12_moltbook_basisrate"]
VERBOTEN_AUF_CPU = ("AutoModelForCausalLM", "device_map=", "torch.cuda",
                    "cuda.mem_get_info")


@pytest.mark.parametrize("basis", NUR_CPU)
def test_cpu_zellen_laden_kein_modell(basis):
    pfad = os.path.join(HIER, "..", basis + ".ipynb")
    if not os.path.exists(pfad):
        pytest.skip("%s fehlt" % basis)
    quelle = "".join(q for _, q in zellen(pfad))
    gefunden = [w for w in VERBOTEN_AUF_CPU if w in quelle]
    assert not gefunden, ("%s ist als CPU-Zelle angekuendigt, enthaelt aber %s"
                          % (basis, ", ".join(gefunden)))


@pytest.mark.parametrize("basis", NUR_CPU)
def test_cpu_zellen_sagen_es_auch(basis):
    """Die Ankuendigung gehoert ins Notebook, nicht nur in den Chat."""
    pfad = os.path.join(HIER, "..", basis + ".ipynb")
    if not os.path.exists(pfad):
        pytest.skip("%s fehlt" % basis)
    with open(pfad, encoding="utf-8") as f:
        nb = json.load(f)
    text = " ".join("".join(c.get("source", [])) for c in nb.get("cells", [])
                    if c.get("cell_type") == "markdown").lower()
    assert "keine gpu" in text or "ohne gpu" in text, \
        "%s sagt nirgends, dass es ohne GPU laeuft" % basis
