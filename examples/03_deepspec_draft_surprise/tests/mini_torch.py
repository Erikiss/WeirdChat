"""Ein Torch-Ersatz aus numpy, gerade gross genug fuer die Colab-Zellen.

Die Zellen werden gegen ein Miniaturmodell laufen gelassen, damit ihr
Ausfuehrungsteil wirklich ausgefuehrt wird und nicht nur ihre reine Logik.
Die erste FFN-Zelle hat die Architektur richtig erkannt und ist drei Schritte
spaeter mit StopIteration gestorben - nach zwanzig Minuten Modell-Download.

T ist eine Unterklasse von ndarray, damit Ausschnitte SICHTEN sind: nur so
wirkt zero_() auf die Elternmatrix, genau wie bei Torch. Wer hier mit Kopien
arbeitet, testet ein Nullsetzen, das im echten Modell nichts nullt.
"""
import types

import numpy as np


class T(np.ndarray):
    device = "cpu"

    def detach(self):
        return self

    def clone(self):
        return np.array(self).view(T)

    def zero_(self):
        self[...] = 0
        return self

    def copy_(self, v):
        self[...] = np.asarray(v)
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def to(self, *a, **k):
        return self

    def abs(self):
        return np.abs(np.asarray(self)).view(T)

    def masked_fill(self, maske, wert):
        return np.where(np.asarray(maske), wert, np.asarray(self)).view(T)

    def chunk(self, n, dim=-1):
        return tuple(x.view(T) for x in np.split(np.asarray(self), n, axis=dim))


def t(a):
    return np.asarray(a, dtype=np.float64).view(T)


def silu(x):
    return np.asarray(x) / (1.0 + np.exp(-np.asarray(x)))


class _FN:
    @staticmethod
    def linear(x, W):
        return t(np.asarray(x) @ np.asarray(W).T)


class _CUDA:
    @staticmethod
    def empty_cache():
        pass

    @staticmethod
    def synchronize():
        pass

    @staticmethod
    def mem_get_info():
        return (80e9, 80e9)


class _NG:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


LETZTE_SAAT = [0]


def _saat_merken(s):
    LETZTE_SAAT[0] = int(s)
    np.random.seed(int(s) % 2 ** 31)


def mach_torch():
    """frischer Torch-Ersatz je Testlauf - kein geteilter Zustand"""
    return types.SimpleNamespace(
        nn=types.SimpleNamespace(functional=_FN),
        cuda=_CUDA,
        no_grad=lambda: _NG(),
        # Die Saat wird MITGESCHRIEBEN, nicht nur gesetzt. Eine Miniaturwelt
        # zieht deterministisch und wuerde einen Saatfehler sonst nie
        # bemerken: alle Bedingungen bekaemen dieselben Antworten, und ob die
        # Zelle je Bedingung neu saet, waere an keinem Ergebnis zu sehen.
        manual_seed=_saat_merken,
        tensor=lambda a, device=None, dtype=None: t(a),
        isin=lambda a, b: np.isin(np.asarray(a), np.asarray(b)),
        long="long",
    )


def haken_traeger():
    """Mischklasse fuer Miniaturmodule: register_forward_pre_hook mit
       Griff-Objekt und Ketten-Semantik. Gibt ein Haken neue Argumente
       zurueck, sieht der NAECHSTE Haken die geaenderten - genau wie bei
       Torch. Ohne das wuerde eine Maske, die die Argumente ersetzt, im Test
       wirken und im echten Modell nicht."""

    class Traeger:
        def __init__(self):
            self._haken = []

        def register_forward_pre_hook(self, h):
            self._haken.append(h)
            griff = types.SimpleNamespace()
            griff.remove = lambda h=h: (self._haken.remove(h)
                                        if h in self._haken else None)
            return griff

        def feuere(self, *args):
            for h in list(self._haken):
                r = h(self, args)
                if r is not None:
                    args = r
            return args

    return Traeger
