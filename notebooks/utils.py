"""Utilidades compartidas por los notebooks del proyecto."""

import hashlib
import io
import urllib.request
from pathlib import Path

import pandas as pd

CORPUS_URL = "https://github.com/CASDAV/proyecto-pln/releases/download/corpus-v1/noticias_colombia.parquet"
SHA256 = "92b02aa1a74015192d583cede59c3d8bd7794c214028809cc687704255aa2fc7"
N_DOCS_UNICOS = 91585


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def repo_root() -> Path:
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / "pyproject.toml").exists():
            return d
    return Path.cwd()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_corpus_raw(verify: bool = True) -> pd.DataFrame:
    """Carga el corpus crudo, sin deduplicación ni limpieza.

    Colab: descarga a memoria en cada sesión.
    Local: descarga una vez a data/ y reutiliza el archivo.
    """
    if in_colab():
        blob = urllib.request.urlopen(CORPUS_URL).read()
        got = hashlib.sha256(blob).hexdigest()
        if got != SHA256:
            raise RuntimeError(f"checksum no coincide: {got}")
        return pd.read_parquet(io.BytesIO(blob))

    dest = repo_root() / "data" / "noticias_colombia.parquet"

    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(CORPUS_URL, tmp)
        got = _sha256(tmp)
        if got != SHA256:
            tmp.unlink()
            raise RuntimeError(f"checksum no coincide: {got}")
        tmp.rename(dest)
    elif verify and _sha256(dest) != SHA256:
        raise RuntimeError(f"archivo corrupto en {dest}")

    return pd.read_parquet(dest)


def load_corpus(verify: bool = True) -> pd.DataFrame:
    """Corpus deduplicado (criterio de 01-corpus) y con `fecha` parseada.

    Descarta duplicados por texto normalizado: minúsculas, espacios
    colapsados y recortados. Verifica el conteo esperado de documentos.
    """
    df_raw = load_corpus_raw(verify=verify)

    norm = df_raw["texto"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

    df = df_raw[~norm.duplicated()].reset_index(drop=True).copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    if len(df) != N_DOCS_UNICOS:
        raise RuntimeError(f"esperaba {N_DOCS_UNICOS} documentos, hay {len(df)}")

    return df


def results_dir() -> Path:
    """Carpeta de resultados: /content/results en Colab, results/ del repo en local."""
    d = (Path("/content") if in_colab() else repo_root()) / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d
