"""Test de regresión de notebooks/utils.py (Entrega 3, S0).

Compara el utils.py del working tree contra el de un commit de referencia:
lo que ya existía debe tener la misma firma, las mismas constantes y regex, y
dar exactamente la misma salida en normalizar_texto y load_corpus.

Uso (desde la raíz del repo, ANTES de hacer commit del utils.py nuevo):

    uv run python tests/test_regresion_utils.py          # compara contra HEAD
    uv run python tests/test_regresion_utils.py HEAD~1   # si ya hiciste commit
"""

import hashlib
import inspect
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_UTILS = "notebooks/utils.py"

EXISTENTES_FUNCIONES = [
    "in_colab",
    "repo_root",
    "_sha256",
    "setup",
    "load_corpus_raw",
    "load_corpus",
    "results_dir",
    "normalizar_texto",
]
EXISTENTES_CONSTANTES = ["CORPUS_URL", "SHA256", "N_DOCS_UNICOS"]
EXISTENTES_REGEX = ["RE_MILES", "RE_DEC_COMA", "RE_DEC_PUNTO", "RE_AMPM", "RE_CTRL"]

# Casos fijos que ejercitan cada regex de normalizar_texto.
CASOS = pd.Series(
    [
        "Costó 1 . 500 . 000 pesos y 3 ' 200 dólares",
        "Creció 4 , 5 % y luego 2 . 75 puntos ; a las 10 . 30 : hora",
        "Llegó a las 8 p . m y salió a las 6 a m",
        "Texto con control \x93 comillas \x94 y \x85",
        "Sin cifras . Solo texto , con puntuación .",
        "",
    ]
)


def cargar_referencia(ref: str) -> types.ModuleType:
    src = subprocess.run(
        ["git", "show", f"{ref}:{RUTA_UTILS}"],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    mod = types.ModuleType("utils_ref")
    mod.__file__ = str(RAIZ / RUTA_UTILS)
    exec(compile(src, f"{ref}:{RUTA_UTILS}", "exec"), mod.__dict__)  # noqa: S102 (código propio del repo)
    return mod


def hash_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


def main(ref: str = "HEAD") -> None:
    sys.path.insert(0, str(RAIZ / "notebooks"))
    import utils as nuevo

    viejo = cargar_referencia(ref)
    print(f"Referencia: {ref}:{RUTA_UTILS}")

    for nombre in EXISTENTES_FUNCIONES:
        a = inspect.signature(getattr(viejo, nombre))
        b = inspect.signature(getattr(nuevo, nombre))
        assert str(a) == str(b), f"firma distinta en {nombre}: {a} vs {b}"
    for nombre in EXISTENTES_CONSTANTES:
        assert getattr(viejo, nombre) == getattr(nuevo, nombre), nombre
    for nombre in EXISTENTES_REGEX:
        a, b = getattr(viejo, nombre), getattr(nuevo, nombre)
        assert (a.pattern, a.flags) == (b.pattern, b.flags), nombre
    print("OK firmas, constantes y regex")

    pd.testing.assert_series_equal(
        viejo.normalizar_texto(CASOS), nuevo.normalizar_texto(CASOS)
    )
    print("OK normalizar_texto en casos fijos")

    df_v = viejo.load_corpus()
    df_n = nuevo.load_corpus()
    pd.testing.assert_frame_equal(df_v, df_n)
    assert hash_df(df_v) == hash_df(df_n)
    print(f"OK load_corpus: {df_n.shape}, hash {hash_df(df_n)[:16]}")

    muestra = df_n["texto"].sample(5000, random_state=42)
    pd.testing.assert_series_equal(
        viejo.normalizar_texto(muestra), nuevo.normalizar_texto(muestra)
    )
    print("OK normalizar_texto en muestra fija (5000 docs, seed 42)")

    pd.testing.assert_series_equal(
        viejo.normalizar_texto(df_n["texto"]), nuevo.normalizar_texto(df_n["texto"])
    )
    print(f"OK normalizar_texto en el corpus completo ({len(df_n):,} docs)")
    print("\nREGRESIÓN SUPERADA: utils.py es retrocompatible.")


if __name__ == "__main__":
    main(*sys.argv[1:2])
