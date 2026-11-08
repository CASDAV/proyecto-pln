"""Utilidades compartidas por los notebooks del proyecto."""

import hashlib
import io
import re
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


def setup(verbose: bool = True) -> None:
    """Prepara el entorno de ejecución.

    En Colab instala las dependencias que no vienen preinstaladas. En local
    no instala nada (eso lo resuelve uv o pip). En ambos casos descarga el
    corpus de stopwords de NLTK si falta.
    """
    if in_colab():
        import subprocess
        import sys

        paquetes = ["spacy>=3.8", "nltk", "wordcloud", "pyarrow"]
        if verbose:
            print("Instalando dependencias en Colab...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *paquetes], check=True
        )

    import nltk

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=not verbose)


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


RE_MILES = re.compile(r"(?<!\d)(\d{1,3})(?: ['’] | \. )(?=\d{3}(?!\d))")
RE_DEC_COMA = re.compile(r"(?<=\d) , (?=\d{1,2}(?!\d))")
RE_DEC_PUNTO = re.compile(r"(?<=\d) \. (?=\d{1,2}(?!\d)(?! *:))")
RE_AMPM = re.compile(r"(?<=\d)\s+([ap])\s*\.?\s+m\b")
RE_CTRL = re.compile(r"[\x80-\x9f]")


def normalizar_texto(s: pd.Series) -> pd.Series:
    """Reune cifras y horas que la tokenización de origen partió con espacios."""
    return (
        s.str.replace(RE_CTRL, " ", regex=True)
        .str.replace(RE_MILES, r"\1", regex=True)
        .str.replace(RE_DEC_COMA, ",", regex=True)
        .str.replace(RE_DEC_PUNTO, ".", regex=True)
        .str.replace(RE_AMPM, r" \1m", regex=True)
    )


# ---------------------------------------------------------------------------
# Entrega 3 (S0). Solo agregados: nada de lo anterior cambia de firma,
# comportamiento ni valores de retorno. Lo pesado (spaCy, NPMI, parquet)
# vive en prep.py.
# ---------------------------------------------------------------------------

AGENCIAS = ("EFE", "AFP", "AP", "REUTERS")
CORPUS_PROCESADO = "corpus_procesado.parquet"
CORPUS_PROCESADO_FAST = "corpus_procesado_fast.parquet"
STOPWORDS_DOMINIO = "stopwords_dominio.txt"


def data_dir() -> Path:
    """Carpeta data/: /content/data en Colab, data/ del repo en local."""
    d = (Path("/content") if in_colab() else repo_root()) / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Marcadores de maquetación (misma regex del notebook 02).
RE_MAQ = re.compile(
    r"\(\s*VER\s+[A-ZÁÉÍÓÚÑ]+\s*:?[^)]*\)|VER\s+(FOTO|TABLA|GRAFICO|GRÁFICO)",
    re.IGNORECASE,
)
# El corpus trae la puntuación separada por espacios ("www . semana . com").
RE_HTML = re.compile(r"<\s*/?\s*[a-z]+(?:\s[^<>\n]{0,300})?>", re.IGNORECASE)
RE_EMAIL = re.compile(
    r"(?:\w+ ?\. ?)*\w+ ?@ ?(?:[a-z0-9-]+ ?\. ?){1,4}[a-z]{2,4}\b", re.IGNORECASE
)
RE_WEB = re.compile(
    r"(?:https? ?: ?/ ?/ ?)?(?:www ?\. ?)?(?:[a-z0-9-]+ ?\. ?){1,5}"
    r"(?:com|org|net|gov|edu|int|info)\b(?: ?\. ?[a-z]{2}\b)?"
    r"(?: ?/ ?[^\s]*(?: ?[/._?=&-] ?[^\s/]+)*)?"
    r"|\bhttps? ?: ?/ ?/|\bwww\b",
    re.IGNORECASE,
)
# Prefiltros: las dos regex anteriores son lentas, solo se aplican a las
# filas que pueden contener un correo o una dirección web.
_PRE_EMAIL = r"@"
_PRE_WEB = r"(?i)www|http|\. ?(?:com|org|net|gov|edu|int|info)\b"

_AG_TOK = r"(?:EFE|Efe|AFP|Afp|AP|Reuters|REUTERS)"
_AG_SEQ = _AG_TOK + r"(?: *(?:[-–,/]|y|e) *" + _AG_TOK + r")*"

# Créditos y leyendas de foto.
RE_FOTO_CREDITO = re.compile(
    r"\b(?:FOTOS?|Fotos?) */[^\n]*?(?= \. | \.$| - |$)"
    r"|^" + _AG_TOK + r" */[^\n]*$",
    re.MULTILINE,
)
RE_FOTO_LINEA = re.compile(
    r"^ *(?:FOTOLEYENDA|Fotoleyenda|(?i:pie de foto)|FOTOS?|Fotos?)"
    r"(?:(?: *(?:No ?\. ?)?\d+) *:?| *[:.]|(?= *$)) *\.? *"
    r"|^ *(?i:pie de foto) *",
    re.MULTILINE,
)
RE_FOTO_NUM = re.compile(r"\bFOTO +(?:No ?\. ?)?\d+ *:?|\b(?i:fotos?) *: *\.(?= |$)")
# Remisiones "( VER : ... )" que RE_MAQ no cubre (sin letras tras VER).
RE_VER = re.compile(r"\( *VER\b[^)\n]{0,200}\)", re.IGNORECASE)
RE_FOTO_PAREN = re.compile(
    r"\( *(?:en +la +)?(?:fotos?|fotoleyenda) *\)", re.IGNORECASE
)
RE_REF = re.compile(r"\( *Ref *\)", re.IGNORECASE)

# Firmas de agencia (se quitan del texto y se guardan en `agencia`).
RE_AG_PAREN = re.compile(r"\( *(" + _AG_SEQ + r") *\)", re.IGNORECASE)
RE_AG_FIN = re.compile(
    r"(?:(?<=\s)|^)(?<!\bla )(?<!\blas )(?:[-–] *)?"
    r"(?:[Cc]on +(?:información +de +(?:las? +agencias? +)?)?)?"
    r"(" + _AG_SEQ + r")(?: *y)? *[.\-–]? *$",
    re.MULTILINE,
)
RE_AG_INICIO = re.compile(
    r"^((?:[^\s]+ +){0,6}?)(" + _AG_SEQ + r") *[.,\-–]? *(?=[A-ZÁÉÍÓÚÑ¿¡\"]|$)",
    re.MULTILINE,
)
_RE_AG_NOMBRE = re.compile(_AG_TOK)

RE_CARTA = re.compile(r"señor\s+director", re.IGNORECASE)


def _ordenar(s: pd.Series) -> pd.Series:
    """Espacios y saltos de línea sobrantes que deja la limpieza."""
    return (
        s.str.replace(r"[ \t]{2,}", " ", regex=True)
        .str.replace(r"[ \t]+\n", "\n", regex=True)
        .str.replace(r"\n[ \t]+", "\n", regex=True)
        .str.replace(r"\n{3,}", "\n\n", regex=True)
        .str.strip()
    )


def _sub(s: pd.Series, rx: re.Pattern, prefiltro: str | None = None) -> pd.Series:
    if prefiltro is None:
        return s.str.replace(rx, " ", regex=True)
    m = s.str.contains(prefiltro, regex=True)
    s = s.copy()
    s.loc[m] = s.loc[m].str.replace(rx, " ", regex=True)
    return s


def limpiar_artefactos(s: pd.Series) -> pd.Series:
    """Quita artefactos de maquetación y scraping. No toca las agencias.

    Orden: HTML → correos → URLs/dominios → créditos y leyendas de foto →
    marcadores (VER FOTO/TABLA/GRÁFICO) → (Ref). Aplicar después de
    normalizar_texto, y por separado a título y cuerpo.
    """
    s = _sub(s, RE_HTML, "<")
    s = _sub(s, RE_EMAIL, _PRE_EMAIL)
    s = _sub(s, RE_WEB, _PRE_WEB)
    for rx in (
        RE_FOTO_CREDITO,
        RE_FOTO_LINEA,
        RE_FOTO_NUM,
        RE_FOTO_PAREN,
        RE_MAQ,
        RE_VER,
        RE_REF,
    ):
        s = _sub(s, rx)
    return _ordenar(s)


def _agencias_doc(texto: str) -> tuple[str, str | None]:
    encontradas: set[str] = set()

    def quitar(m: re.Match, grupo: int = 1, conservar: str = "") -> str:
        encontradas.update(x.upper() for x in _RE_AG_NOMBRE.findall(m.group(grupo)))
        return conservar

    texto = RE_AG_PAREN.sub(lambda m: quitar(m, 1, " "), texto)
    texto = RE_AG_FIN.sub(lambda m: quitar(m, 1), texto)
    texto = RE_AG_INICIO.sub(lambda m: quitar(m, 2, m.group(1)), texto)
    agencia = "|".join(a for a in AGENCIAS if a in encontradas) or None
    return texto, agencia


def extraer_agencia(s: pd.Series) -> pd.DataFrame:
    """Quita firmas de agencia (EFE, AFP, AP, Reuters) y las guarda en `agencia`.

    Reconoce: "( AFP )" o "( AFP - Efe )", firma al final de línea
    ("... EFE", "Con AFP y Reuters"),
    y data line al inicio de línea ("MONTEVIDEO AFP El presidente...").
    No toca menciones en prosa ("dijo a la AP en el hotel").
    Devuelve columnas `texto` y `agencia` ("AFP|EFE" si hay varias).
    """
    pares = [_agencias_doc(t) for t in s.tolist()]
    out = pd.DataFrame(pares, columns=["texto", "agencia"], index=s.index)
    out["texto"] = _ordenar(out["texto"])
    return out


def es_titulo_capitalizado(titulo: str, umbral: float = 0.8) -> bool:
    """True si >= umbral de las palabras de 4+ letras empiezan en mayúscula
    y el título no está todo en mayúsculas."""
    w = [x for x in titulo.split() if len(x) >= 4 and x[0].isalpha()]
    if len(w) < 2 or titulo.isupper():
        return False
    return sum(x[0].isupper() for x in w) / len(w) >= umbral


def separar_titulo(s: pd.Series, max_palabras: int = 40) -> pd.DataFrame:
    """Separa titular y cuerpo por el primer salto de línea doble.

    Verificado en el 07: el 100 % de los documentos tiene "\\n\\n" y el primer
    segmento tiene mediana de 6 palabras. Si el primer segmento supera
    `max_palabras`, no se considera titular.
    Devuelve `titulo`, `cuerpo`, `sep_titulo` ("salto2" o None) y
    `titulo_capitalizado`.
    """
    partes = s.str.split("\n\n", n=1, regex=False)
    tit = partes.str[0].fillna("").str.strip()
    cue = partes.str[1].fillna("").str.strip()
    ok = (tit.str.split().str.len() <= max_palabras) & (cue.str.len() > 0)
    out = pd.DataFrame(index=s.index)
    out["titulo"] = tit.where(ok, "")
    out["cuerpo"] = cue.where(ok, s.str.strip())
    out["sep_titulo"] = pd.Series("salto2", index=s.index).where(ok, None)
    out["titulo_capitalizado"] = out["titulo"].map(es_titulo_capitalizado)
    return out


def quitar_lead_duplicado(s: pd.Series, min_palabras: int = 8) -> pd.DataFrame:
    """Quita el primer párrafo cuando el siguiente lo repite.

    Artefacto del scraping (El Tiempo 2001-2005, Semana 2000-2003): el
    sumario o la entradilla aparecen dos veces. Se conserva la segunda
    copia. Devuelve `texto` y `lead_duplicado` (bool).
    """
    textos, flags = [], []
    for t in s.tolist():
        lineas = t.split("\n")
        idx = [i for i, ln in enumerate(lineas) if ln.strip()]
        dup = False
        if len(idx) >= 2:
            a, b = lineas[idx[0]].strip(), lineas[idx[1]].strip()
            if len(a.split()) >= min_palabras and b.startswith(a[:120]):
                dup = True
                lineas = lineas[idx[1] :]
        textos.append("\n".join(lineas).strip())
        flags.append(dup)
    return pd.DataFrame({"texto": textos, "lead_duplicado": flags}, index=s.index)


_RE_SECCION = re.compile(r"^#\s*([A-Za-z_]+)\s*$")


def leer_stopwords_dominio(path: Path | None = None) -> dict[str, list[str]]:
    """Lee data/stopwords_dominio.txt como {sección: [palabras]}.

    Formato: una palabra por línea; una línea "# nombre" (una sola palabra)
    abre sección; cualquier otra línea con "#" es comentario. La sección "conservar" lista palabras que NO son
    stopword cuando funcionan como sustantivo (p. ej. "estado", "poder").
    """
    path = path or data_dir() / STOPWORDS_DOMINIO
    secciones: dict[str, list[str]] = {}
    actual = "sin_seccion"
    for linea in path.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        cab = _RE_SECCION.match(linea)
        if cab:
            actual = cab.group(1).lower()
            continue
        palabra = linea.split("#", 1)[0].strip().lower()
        if palabra:
            secciones.setdefault(actual, []).append(palabra)
    return secciones


def cargar_stopwords(
    dominio: bool = True, excluir_secciones: tuple[str, ...] = ()
) -> set[str]:
    """NLTK + spaCy + lista de dominio (sin la sección "conservar")."""
    from nltk.corpus import stopwords
    from spacy.lang.es.stop_words import STOP_WORDS

    sw = set(stopwords.words("spanish")) | set(STOP_WORDS)
    if dominio:
        for sec, palabras in leer_stopwords_dominio().items():
            if sec != "conservar" and sec not in excluir_secciones:
                sw |= set(palabras)
    return sw


def load_corpus_procesado(
    fast: bool = False, columns: list[str] | None = None
) -> pd.DataFrame:
    """Carga el corpus procesado por el 07 desde data/ (no carga spaCy)."""
    path = data_dir() / (CORPUS_PROCESADO_FAST if fast else CORPUS_PROCESADO)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} no existe: ejecuta 07-preprocesamiento-v3.ipynb"
        )
    return pd.read_parquet(path, columns=columns)
