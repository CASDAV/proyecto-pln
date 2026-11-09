"""Preprocesamiento v3 (Entrega 3, S0): la parte pesada y cacheada.

Lo liviano (rutas, regex de limpieza, stopwords, carga del parquet) vive en
utils.py. Aquí van spaCy, la construcción de data/corpus_procesado.parquet con
caché por hash de configuración y las colocaciones NPMI.

Uso típico (notebook 07):

    cfg = prep.PrepConfig(fast=True)
    dfp = prep.construir_corpus_procesado(cfg)

Los notebooks siguientes (08, 09) solo necesitan utils.load_corpus_procesado().
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import utils

# Súbelo a mano cuando cambie la lógica del pipeline (invalida el caché).
# 3.0.2: `conservar` también rescata PROPN + sección conservar_propn (Estados Unidos…)
# 3.0.3: sustantivos de contenido que solo spaCy marca como stopword (acuerdo, uso…),
#        rescatados como NOUN salvo dentro de locuciones ("de acuerdo con")
PIPELINE_VERSION = "3.0.3"

POS_CONTENIDO = {"NOUN": "N", "ADJ": "A", "VERB": "V", "PROPN": "P"}
TIPOS_ENT = ("PER", "ORG", "LOC")

# Rachas de 3+ palabras en mayúscula sostenida (versalitas de Semana, datelines,
# tablas). Se pasan a minúscula SOLO en el texto que entra a spaCy.
RE_VERSALITAS = re.compile(r"\b[A-ZÁÉÍÓÚÑÜ]+(?: +[A-ZÁÉÍÓÚÑÜ]+){2,}\b")

SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("fuente", pa.string()),
        ("fecha", pa.timestamp("us")),
        ("año", pa.int16()),
        ("mes", pa.int8()),
        ("titulo", pa.string()),
        ("cuerpo", pa.string()),
        ("sep_titulo", pa.string()),
        ("titulo_capitalizado", pa.bool_()),
        ("lead_duplicado", pa.bool_()),
        ("agencia", pa.string()),
        ("es_carta", pa.bool_()),
        ("es_breve", pa.bool_()),
        ("breve_motivo", pa.string()),
        ("tokens_forma", pa.list_(pa.string())),
        ("lemas_contenido", pa.list_(pa.string())),
        ("pos_contenido", pa.string()),
        ("cortes_contenido", pa.list_(pa.int32())),
        ("lemas_nom_adj", pa.list_(pa.string())),
        (
            "entidades",
            pa.list_(pa.struct([("texto", pa.string()), ("tipo", pa.string())])),
        ),
        ("n_chars", pa.int32()),
        ("n_palabras_titulo", pa.int32()),
        ("n_palabras_cuerpo", pa.int32()),
        ("n_lemas_contenido", pa.int32()),
        ("n_lemas_nom_adj", pa.int32()),
        ("n_entidades", pa.int32()),
    ]
)


# ---------------------------------------------------------------------------
# Configuración y caché
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrepConfig:
    """Parámetros del pipeline.

    Los campos con metadata hash=False no cambian la salida (solo la
    velocidad) y no invalidan el caché. `umbral_breve` va en un hash aparte:
    cambiarlo recalcula las marcas sin volver a correr spaCy.
    """

    fast: bool = True
    n_fast: int = 6000
    seed: int = 42
    modelo: str = "es_core_news_md"
    min_len_lema: int = 3
    max_hueco: int = 3  # tokens descartados seguidos que cortan un bigrama
    quitar_lead_duplicado: bool = True
    normalizar_versalitas: bool = True
    umbral_capitalizado: float = 0.8
    umbral_breve: int = field(default=80, metadata={"grupo": "generos"})
    n_process: int = field(default=6, metadata={"hash": False})
    batch_size: int = field(default=64, metadata={"hash": False})
    chunk_docs: int = field(default=5000, metadata={"hash": False})


def _sha(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


def config_hash(cfg: PrepConfig) -> dict[str, str]:
    """Dos hashes: `nlp` (todo lo que exige rehacer spaCy) y `generos`."""
    import spacy

    nlp_part, gen_part = {}, {}
    for f in fields(cfg):
        if f.metadata.get("hash", True) is False:
            continue
        destino = gen_part if f.metadata.get("grupo") == "generos" else nlp_part
        destino[f.name] = getattr(cfg, f.name)
    sw_path = utils.data_dir() / utils.STOPWORDS_DOMINIO
    nlp_part |= {
        "pipeline_version": PIPELINE_VERSION,
        "spacy": spacy.__version__,
        "modelo_version": spacy.util.get_package_version(cfg.modelo),
        "stopwords_sha": hashlib.sha256(sw_path.read_bytes()).hexdigest()[:16],
    }
    gen_part["pipeline_version"] = PIPELINE_VERSION
    return {"nlp": _sha(nlp_part), "generos": _sha(gen_part)}


def ruta_salida(cfg: PrepConfig) -> Path:
    nombre = utils.CORPUS_PROCESADO_FAST if cfg.fast else utils.CORPUS_PROCESADO
    return utils.data_dir() / nombre


def _ruta_meta(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def leer_meta(path: Path) -> dict:
    """Hashes y config (en el parquet) + tiempos (en el .meta.json al lado)."""
    if not path.exists():
        return {}
    md = pq.read_schema(path).metadata or {}
    raw = md.get(b"prep_meta")
    meta = json.loads(raw) if raw else {}
    lado = _ruta_meta(path)
    if lado.exists():
        extra = json.loads(lado.read_text(encoding="utf-8"))
        if extra.get("hash_nlp") == meta.get("hash_nlp"):
            meta = extra | meta
    return meta


# ---------------------------------------------------------------------------
# Texto antes de spaCy
# ---------------------------------------------------------------------------


def _combinar_agencias(a: pd.Series, b: pd.Series) -> pd.Series:
    def una(x, y):
        x, y = (v if isinstance(v, str) else "" for v in (x, y))
        s = set(x.split("|")) | set(y.split("|"))
        s.discard("")
        return "|".join(ag for ag in utils.AGENCIAS if ag in s) or None

    return pd.Series([una(x, y) for x, y in zip(a, b)], index=a.index, dtype=object)


def _a_oracion(t: str) -> str:
    return t[:1].upper() + t[1:].lower() if t else t


def _versalitas(t: str) -> str:
    return RE_VERSALITAS.sub(
        lambda m: (
            m.group(0).lower()
            if any(len(w) >= 4 for w in m.group(0).split())
            else m.group(0)
        ),
        t,
    )


def preparar_textos(df: pd.DataFrame, cfg: PrepConfig) -> pd.DataFrame:
    """normalizar_texto → separar titular → limpiar artefactos → agencias →
    lead duplicado. Devuelve las columnas de texto y metadatos, más
    `texto_nlp` y `offset_cuerpo` (lo que entra a spaCy)."""
    txt = utils.normalizar_texto(df["texto"])
    sep = utils.separar_titulo(txt)
    tit = utils.limpiar_artefactos(sep["titulo"])
    cue = utils.limpiar_artefactos(sep["cuerpo"])
    ag_t, ag_c = utils.extraer_agencia(tit), utils.extraer_agencia(cue)
    tit, cue = ag_t["texto"], ag_c["texto"]
    ld = utils.quitar_lead_duplicado(cue)
    if cfg.quitar_lead_duplicado:
        cue = ld["texto"]

    out = pd.DataFrame(
        {
            "id": df["id"].values,
            "fuente": df["fuente"].values,
            "fecha": df["fecha"].values,
            "año": df["fecha"].dt.year.values,
            "mes": df["fecha"].dt.month.values,
            "titulo": tit.values,
            "cuerpo": cue.values,
            "sep_titulo": sep["sep_titulo"].values,
            "titulo_capitalizado": tit.map(
                lambda t: utils.es_titulo_capitalizado(t, cfg.umbral_capitalizado)
            ).values,
            "lead_duplicado": ld["lead_duplicado"].values,
            "agencia": _combinar_agencias(ag_t["agencia"], ag_c["agencia"]).values,
        }
    )

    # Texto para spaCy: titular en mayúscula de oración si venía capitalizado,
    # punto final para que la NER no cruce al cuerpo, versalitas en minúscula.
    tit_nlp = [
        _a_oracion(t) if cap else t
        for t, cap in zip(out["titulo"], out["titulo_capitalizado"])
    ]
    tit_nlp = [t if (not t or t.rstrip()[-1:] in ".:?!") else t + " ." for t in tit_nlp]
    cue_nlp = out["cuerpo"]
    if cfg.normalizar_versalitas:
        cue_nlp = cue_nlp.map(_versalitas)
    out["texto_nlp"] = [f"{t}\n\n{c}" if t else c for t, c in zip(tit_nlp, cue_nlp)]
    out["offset_cuerpo"] = [len(t) + 2 if t else 0 for t in tit_nlp]
    return out


# ---------------------------------------------------------------------------
# spaCy
# ---------------------------------------------------------------------------


def cargar_nlp(cfg: PrepConfig):
    """es_core_news_md sin parser: tok2vec, morphologizer (POS), attribute_ruler,
    lemmatizer y ner."""
    import spacy

    nlp = spacy.load(cfg.modelo, exclude=["parser"])
    nlp.max_length = 2_000_000
    return nlp


def _cortar_entidad(ent) -> str:
    """Recorta la entidad en el primer salto de línea (sin parser no hay
    límites de oración y la NER a veces cruza párrafos)."""
    toks = []
    for t in ent:
        if t.is_space or "\n" in t.text:
            break
        toks.append(t.text)
    return " ".join(toks)


# Locuciones gramaticales por lema, sobre la ventana en minúsculas
# "w-2 w-1 FORMA w+1 w+2" (sin espacios ni saltos). Un NOUN rescatado por
# `conservar` que cae en una de estas locuciones se descarta igual.
LOCUCIONES = {
    "acuerdo": r"\bde acuerdo\b",
    "cuenta": r"\b(?:en|por|según) cuenta\b|\b(?:por|a) (?:su|mi|tu|nuestra) cuenta\b"
    r"|\b(?:da|di|dé|den|dar)\w* cuenta\b",
    "medio": r"\b(?:por|en) medio\b",
    "fin": r"\b(?:a|al|en|por|sin|tal) fin\b|\bcon (?:el |este |ese )?fin\b"
    r"|\bcon fines\b|\bfin de (?:semana|año|mes)\b",
    "embargo": r"\bsin embargo\b",
    "hecho": r"\bde hecho\b|\bel hecho de\b|\bhecho de que\b",
    "bien": r"\b(?:si|más|muy|está) bien\b",
    "parte": r"\b(?:por|de|en) parte\b|\bmayor parte\b|\bparte del?\b",
    "lado": r"\b(?:al|por|del|un|otro|ambos|este|ese) lado\b",
    "tarde": r"\b(?:la|esta|esa|una|en la|por la|de la) tarde\b",
    "total": r"\b(?:un|en|del) total\b",
    "uso": r"\b(?:hacer|hace|hacen|hizo|haga|hicieron) uso\b",
}
_RE_LOCUCIONES = {k: re.compile(v) for k, v in LOCUCIONES.items()}


def _ventana(t, k: int = 2) -> str:
    """Forma del token con k vecinos no-espacio a cada lado, en minúsculas."""
    doc, izq, der = t.doc, [], []
    i = t.i - 1
    while i >= 0 and len(izq) < k:
        if not doc[i].is_space:
            izq.append(doc[i].lower_)
        i -= 1
    i = t.i + 1
    while i < len(doc) and len(der) < k:
        if not doc[i].is_space:
            der.append(doc[i].lower_)
        i += 1
    return " ".join([*izq[::-1], t.lower_, *der])


def _en_locucion(t, lema: str) -> bool:
    rx = _RE_LOCUCIONES.get(lema)
    return bool(rx and rx.search(_ventana(t)))


def _rescatado(t, lema: str, low: str, conservar, conservar_propn) -> bool:
    """Stopwords que se conservan: `conservar` cuando el token es NOUN o PROPN
    ("el Estado", "Poder Judicial", "el acuerdo de paz") y no está dentro de
    una locución gramatical ("de acuerdo con", "a fin de"); `conservar_propn`
    solo cuando es PROPN (primera palabra de nombres compuestos: "Estados
    Unidos", "Nueva York")."""
    if t.pos_ in ("NOUN", "PROPN") and lema in conservar:
        return not (t.pos_ == "NOUN" and _en_locucion(t, lema))
    return t.pos_ == "PROPN" and (lema in conservar_propn or low in conservar_propn)


def extraer_doc(
    doc,
    offset: int,
    sw: set[str],
    conservar: set[str],
    cfg: PrepConfig,
    conservar_propn: frozenset[str] | set[str] = frozenset(),
) -> dict:
    """De un Doc de spaCy a las columnas del parquet (una sola tokenización)."""
    forma: list[str] = []
    lemas: list[str] = []
    pos: list[str] = []
    cortes: list[int] = []
    nom_adj: list[str] = []
    n_tit = 0
    corte, hueco, en_cuerpo = True, 0, offset == 0

    for t in doc:
        if not en_cuerpo and t.idx >= offset:
            en_cuerpo, corte = True, True
        if t.is_space:
            corte = corte or "\n" in t.text
            continue
        low = t.lower_
        if t.is_punct or not any(ch.isalnum() for ch in low):
            corte = True
            continue
        forma.append(low)
        if not en_cuerpo:
            n_tit += 1
        code = POS_CONTENIDO.get(t.pos_)
        # spaCy lematiza los clíticos con espacio ("apretarse" → "apretar él"):
        # se deja solo el verbo para que cada lema sea un único token.
        lema = (t.lemma_.lower().split() or [low])[0]
        es_sw = (lema in sw or low in sw) and not _rescatado(
            t, lema, low, conservar, conservar_propn
        )
        if code and t.is_alpha and len(lema) >= cfg.min_len_lema and not es_sw:
            if lemas and (corte or hueco > cfg.max_hueco):
                cortes.append(len(lemas))
            lemas.append(lema)
            pos.append(code)
            if code in ("N", "A"):
                nom_adj.append(lema)
            corte, hueco = False, 0
        else:
            hueco += 1

    ents = []
    for e in doc.ents:
        if e.label_ in TIPOS_ENT and e.start_char >= offset:
            texto = _cortar_entidad(e)
            if texto:
                ents.append({"texto": texto, "tipo": e.label_})

    return {
        "tokens_forma": forma,
        "lemas_contenido": lemas,
        "pos_contenido": "".join(pos),
        "cortes_contenido": cortes,
        "lemas_nom_adj": nom_adj,
        "entidades": ents,
        "n_palabras_titulo": n_tit,
        "n_palabras_cuerpo": len(forma) - n_tit,
        "n_lemas_contenido": len(lemas),
        "n_lemas_nom_adj": len(nom_adj),
        "n_entidades": len(ents),
    }


# ---------------------------------------------------------------------------
# Géneros (no requieren spaCy: se recalculan desde el parquet)
# ---------------------------------------------------------------------------


def marcar_generos(df: pd.DataFrame, cfg: PrepConfig) -> pd.DataFrame:
    """`es_carta`, `es_breve` y `breve_motivo`. Marca, no borra.

    - es_carta: "señor director" en titular o cuerpo.
    - es_breve: formato de nota breve (titular que termina en ":", columna de
      breves de El Tiempo 1990-2005) o cuerpo con menos de `umbral_breve`
      palabras.
    """
    df = df.copy()
    df["es_carta"] = (df["titulo"] + "\n" + df["cuerpo"]).str.contains(
        utils.RE_CARTA, regex=True
    )
    nota = df["titulo"].str.rstrip().str.endswith(":")
    corto = df["n_palabras_cuerpo"] < cfg.umbral_breve
    df["es_breve"] = nota | corto
    df["breve_motivo"] = pd.Series(
        np.select(
            [nota & corto, nota, corto], ["nota+corto", "nota", "corto"], default=""
        ),
        index=df.index,
    ).replace("", None)
    return df


# ---------------------------------------------------------------------------
# Construcción con caché
# ---------------------------------------------------------------------------


def _a_tabla(df: pd.DataFrame) -> pa.Table:
    df = df.copy()
    df["n_chars"] = df["cuerpo"].str.len()
    df["fecha"] = pd.to_datetime(df["fecha"]).astype("datetime64[us]")
    return pa.Table.from_pandas(df[SCHEMA.names], schema=SCHEMA, preserve_index=False)


def _escribir(path: Path, tablas: Iterable[pa.Table], meta: dict) -> None:
    schema = SCHEMA.with_metadata({"prep_meta": json.dumps(meta, ensure_ascii=False)})
    tmp = path.with_suffix(".parquet.part")
    with pq.ParquetWriter(tmp, schema, compression="zstd") as w:
        for t in tablas:
            w.write_table(t.replace_schema_metadata(schema.metadata))
    tmp.replace(path)


def construir_corpus_procesado(
    cfg: PrepConfig | None = None, force: bool = False, verbose: bool = True
) -> pd.DataFrame:
    """Carga el parquet si el hash coincide; si no, lo construye.

    - Mismo hash `nlp` y distinto `generos`: solo recalcula las marcas.
    - FAST escribe en data/corpus_procesado_fast.parquet (no pisa el completo).
    Tiempo y tamaño quedan en df.attrs["prep_meta"].
    """
    cfg = cfg or PrepConfig()
    path = ruta_salida(cfg)
    h = config_hash(cfg)
    meta = leer_meta(path)

    if not force and meta.get("hash_nlp") == h["nlp"]:
        df = pd.read_parquet(path)
        if meta.get("hash_generos") != h["generos"]:
            if verbose:
                print("Caché válido para spaCy; recalculo marcas de género.")
            df = marcar_generos(df, cfg)
            meta |= {"hash_generos": h["generos"], "config": asdict(cfg)}
            _escribir(path, [_a_tabla(df)], meta)
        elif verbose:
            print(
                f"Caché válido ({path.name}, hash {h['nlp']}): se carga sin recalcular."
            )
        meta["tamano_mb"] = round(path.stat().st_size / 2**20, 1)
        df.attrs["prep_meta"] = meta
        return df

    t0 = time.time()
    df = utils.load_corpus()
    if cfg.fast:
        df = df.sample(n=min(cfg.n_fast, len(df)), random_state=cfg.seed)
        df = df.sort_values("id").reset_index(drop=True)
    base = preparar_textos(df, cfg)
    t_texto = time.time() - t0
    if verbose:
        print(f"{len(base):,} documentos | limpieza y titulares: {t_texto:.0f} s")

    nlp = cargar_nlp(cfg)
    sw = utils.cargar_stopwords()
    secciones = utils.leer_stopwords_dominio()
    conservar = set(secciones.get("conservar", []))
    conservar_propn = set(secciones.get("conservar_propn", []))

    diag = DiagNounSpacy(conservar)

    meta = {
        "hash_nlp": h["nlp"],
        "hash_generos": h["generos"],
        "config": asdict(cfg),
        "pipeline_version": PIPELINE_VERSION,
        "n_docs": len(base),
    }

    def tablas():
        stream = nlp.pipe(
            zip(base["texto_nlp"], base["offset_cuerpo"]),
            as_tuples=True,
            n_process=cfg.n_process,
            batch_size=cfg.batch_size,
        )
        buf, ini, t1 = [], 0, time.time()
        for doc, off in stream:
            buf.append(extraer_doc(doc, off, sw, conservar, cfg, conservar_propn))
            diag.observar(doc, ini + len(buf) - 1)
            if len(buf) == cfg.chunk_docs:
                yield _chunk(ini, buf)
                ini += len(buf)
                buf = []
                if verbose:
                    print(
                        f"  {ini:,}/{len(base):,} docs | {ini / (time.time() - t1):.0f} docs/s"
                    )
        if buf:
            yield _chunk(ini, buf)

    def _chunk(ini, recs):
        sub = base.iloc[ini : ini + len(recs)].drop(
            columns=["texto_nlp", "offset_cuerpo"]
        )
        sub = pd.concat([sub.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
        return _a_tabla(marcar_generos(sub, cfg))

    _escribir(path, tablas(), meta)
    diag.tabla().to_csv(ruta_diag_noun(cfg), index=False)
    meta |= {
        "segundos_total": round(time.time() - t0, 1),
        "segundos_limpieza": round(t_texto, 1),
        "tamano_mb": round(path.stat().st_size / 2**20, 1),
        "n_process": cfg.n_process,
        "batch_size": cfg.batch_size,
    }
    _ruta_meta(path).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if verbose:
        print(
            f"Listo en {meta['segundos_total']:.0f} s → {path} ({meta['tamano_mb']} MB)"
        )

    out = pd.read_parquet(path)
    out.attrs["prep_meta"] = meta
    return out


# ---------------------------------------------------------------------------
# Colocaciones NPMI
# ---------------------------------------------------------------------------


def contar_bigramas(
    secuencias: Sequence,
    cortes: Sequence | None = None,
    pos: Sequence[str] | None = None,
) -> tuple[Counter, Counter, Counter]:
    """Cuenta bigramas adyacentes de cada documento, sin cruzar los `cortes`.

    Devuelve (frecuencia, frecuencia documental, conteo lema→POS).
    """
    bi, dfc, pos_c = Counter(), Counter(), Counter()
    for i, seq in enumerate(secuencias):
        seq = list(seq)
        if len(seq) < 2:
            continue
        cs = set(cortes[i]) if cortes is not None else ()
        pares = [(seq[j - 1], seq[j]) for j in range(1, len(seq)) if j not in cs]
        bi.update(pares)
        dfc.update(set(pares))
        if pos is not None:
            pos_c.update(zip(seq, pos[i]))
    return bi, dfc, pos_c


def colocaciones_npmi(
    secuencias: Sequence,
    cortes: Sequence | None = None,
    pos: Sequence[str] | None = None,
    min_n: int = 20,
    min_df: int = 10,
) -> pd.DataFrame:
    """NPMI sobre un único flujo de tokens.

    p(ab) = n(a,b)/N; p(a) y p(b) son las marginales izquierda y derecha de la
    MISMA tabla de bigramas (Σ_b n(a,b) / N). Como n(a,b) ≤ marginal, NPMI
    queda en [−1, 1] por construcción; igual se verifica con assert sobre todos
    los pares, no solo los filtrados.
    """
    bi, dfc, pos_c = contar_bigramas(secuencias, cortes, pos)
    if not bi:
        return pd.DataFrame(columns=["w1", "w2", "n", "df", "pmi", "npmi"])
    izq, der = Counter(), Counter()
    for (a, b), c in bi.items():
        izq[a] += c
        der[b] += c
    N = sum(bi.values())

    # Verificación del rango sobre TODOS los pares (sin strings, en numpy).
    claves = list(bi.keys())
    n_all = np.fromiter((bi[k] for k in claves), dtype=np.float64, count=len(claves))
    pa_all = np.fromiter(
        (izq[a] for a, _ in claves), dtype=np.float64, count=len(claves)
    )
    pb_all = np.fromiter(
        (der[b] for _, b in claves), dtype=np.float64, count=len(claves)
    )
    pab = n_all / N
    with np.errstate(divide="ignore", invalid="ignore"):
        npmi_all = np.log(pab / ((pa_all / N) * (pb_all / N))) / -np.log(pab)
    finitos = npmi_all[np.isfinite(npmi_all)]
    assert finitos.min() >= -1 - 1e-9 and finitos.max() <= 1 + 1e-9, (
        f"NPMI fuera de rango: [{finitos.min()}, {finitos.max()}]"
    )

    filas = [
        (a, b, c, dfc[(a, b)])
        for (a, b), c in bi.items()
        if c >= min_n and dfc[(a, b)] >= min_df
    ]
    t = pd.DataFrame(filas, columns=["w1", "w2", "n", "df"])
    p_ab = t["n"] / N
    p_a = t["w1"].map(izq) / N
    p_b = t["w2"].map(der) / N
    t["pmi"] = np.log(p_ab / (p_a * p_b))
    t["npmi"] = t["pmi"] / -np.log(p_ab)
    assert t["npmi"].between(-1 - 1e-9, 1 + 1e-9).all()

    if pos_c:
        dom: dict[str, str] = {}
        mejor: dict[str, int] = {}
        for (lema, code), c in pos_c.items():
            if c > mejor.get(lema, 0):
                dom[lema], mejor[lema] = code, c
        t["pos"] = t["w1"].map(dom) + t["w2"].map(dom)
        t["tipo"] = np.select(
            [
                t["pos"].str.fullmatch(r"[NA]{2}"),
                t["pos"].str.contains("V"),
                t["pos"].str.contains("P"),
            ],
            ["nom_adj", "con_verbo", "con_propn"],
            default="otro",
        )
    t.attrs |= {
        "N": N,
        "n_pares": len(bi),
        "rango_todos": (float(finitos.min()), float(finitos.max())),
    }
    return t.sort_values("npmi", ascending=False).reset_index(drop=True)


def exportar_bigramas(
    tabla: pd.DataFrame, umbral: float = 0.35, path: Path | None = None
) -> Path:
    """Exporta los candidatos (npmi ≥ umbral) a data/bigramas_npmi.csv con la
    columna `bigrama` = "w1_w2", lista para usar como feature en S1."""
    path = path or utils.data_dir() / "bigramas_npmi.csv"
    out = tabla[tabla["npmi"] >= umbral].copy()
    out.insert(0, "bigrama", out["w1"] + "_" + out["w2"])
    out.round({"pmi": 4, "npmi": 4}).to_csv(path, index=False)
    return path


def comparar_variantes_npmi(
    dfp: pd.DataFrame,
    sw: set[str],
    min_n: int = 5,
    min_df: int = 3,
    k: int = 300,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compara tres flujos para NPMI: forma literal, lemas sin cortes y lemas
    con cortes. Devuelve (resumen, top-30 lado a lado).

    Métrica de ruido: % del top-k cuyas ocurrencias cruzan mayoritariamente
    una puntuación o un salto de línea (pares que no existen en el texto).
    """
    forma = colocaciones_npmi(dfp["tokens_forma"], min_n=min_n, min_df=min_df)
    ok = lambda w: ~w.isin(sw) & w.str.isalpha() & (w.str.len() >= 3)
    forma = forma[ok(forma["w1"]) & ok(forma["w2"])].reset_index(drop=True)
    sin = colocaciones_npmi(
        dfp["lemas_contenido"], None, dfp["pos_contenido"], min_n=min_n, min_df=min_df
    )
    con = colocaciones_npmi(
        dfp["lemas_contenido"],
        dfp["cortes_contenido"],
        dfp["pos_contenido"],
        min_n=min_n,
        min_df=min_df,
    )
    bi_con, _, _ = contar_bigramas(dfp["lemas_contenido"], dfp["cortes_contenido"])

    def cruza(t: pd.DataFrame) -> float:
        top = t.head(k)
        frac = 1 - np.array(
            [
                bi_con.get((a, b), 0) / n
                for a, b, n in zip(top["w1"], top["w2"], top["n"])
            ]
        )
        return float((frac > 0.5).mean())

    resumen = pd.DataFrame(
        {
            "pares_candidatos": [len(forma), len(sin), len(con)],
            f"top{k}_cruzan_puntuacion": [np.nan, cruza(sin), cruza(con)],
            "espacio": ["forma", "lema", "lema"],
        },
        index=["tokens_forma", "lemas_sin_cortes", "lemas_con_cortes"],
    )
    lado = pd.DataFrame(
        {
            "tokens_forma": (forma["w1"] + " " + forma["w2"]).head(30),
            "lemas_sin_cortes": (sin["w1"] + "_" + sin["w2"]).head(30),
            "lemas_con_cortes": (con["w1"] + "_" + con["w2"]).head(30),
        }
    )
    return resumen, lado


# ---------------------------------------------------------------------------
# Diagnóstico: nombres propios que el filtro de stopwords descarta
# ---------------------------------------------------------------------------


def _origenes_stopwords() -> dict[str, set[str]]:
    from nltk.corpus import stopwords
    from spacy.lang.es.stop_words import STOP_WORDS

    org = {"nltk": set(stopwords.words("spanish")), "spacy": set(STOP_WORDS)}
    for sec, palabras in utils.leer_stopwords_dominio().items():
        if not sec.startswith("conservar"):
            org[f"dominio:{sec}"] = set(palabras)
    return org


def diagnostico_propn_perdidos(
    base: pd.DataFrame,
    cfg: PrepConfig,
    nlp=None,
    top: int = 40,
    ventana: int = 6,
) -> pd.DataFrame:
    """PROPN cuyo lema o forma está en las stopwords (los que la regla de la
    3.0.1 descartaba: `conservar` solo rescataba NOUN).

    Recorre `base` (salida de preparar_textos, típicamente la muestra FAST) con
    spaCy y agrupa por (lema, forma). Para cada grupo: frecuencia, documentos,
    de qué lista viene, la palabra siguiente más común (para ver la entidad que
    se rompe: "Estados" → "Unidos"), un ejemplo de contexto y si la regla
    actual (3.0.2) lo rescata.
    """
    nlp = nlp or cargar_nlp(cfg)
    sw = utils.cargar_stopwords()
    secciones = utils.leer_stopwords_dominio()
    conservar = set(secciones.get("conservar", []))
    conservar_propn = set(secciones.get("conservar_propn", []))
    origenes = _origenes_stopwords()

    n, docs, sig, ejemplo = Counter(), {}, {}, {}
    for i, doc in enumerate(
        nlp.pipe(base["texto_nlp"], n_process=cfg.n_process, batch_size=cfg.batch_size)
    ):
        toks = [t for t in doc if not t.is_space]
        for j, t in enumerate(toks):
            if t.pos_ != "PROPN" or not t.is_alpha:
                continue
            low = t.lower_
            lema = (t.lemma_.lower().split() or [low])[0]
            if len(lema) < cfg.min_len_lema or not (lema in sw or low in sw):
                continue
            k = (lema, low)
            n[k] += 1
            docs.setdefault(k, set()).add(i)
            nxt = toks[j + 1].text if j + 1 < len(toks) else ""
            sig.setdefault(k, Counter())[nxt] += 1
            if k not in ejemplo:
                ini, fin = max(0, j - ventana), j + ventana + 1
                ejemplo[k] = " ".join(x.text for x in toks[ini:fin]).replace("\n", " ")

    filas = []
    for (lema, low), c in n.most_common(top):
        de = [o for o, s in origenes.items() if lema in s or low in s]
        filas.append(
            {
                "lema": lema,
                "forma": low,
                "n": c,
                "docs": len(docs[(lema, low)]),
                "origen": ",".join(de),
                "siguiente_top": ", ".join(
                    f"{w}({k})" for w, k in sig[(lema, low)].most_common(3)
                ),
                "rescatado": lema in conservar
                or lema in conservar_propn
                or low in conservar_propn,
                "ejemplo": ejemplo[(lema, low)],
            }
        )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Diagnóstico: sustantivos que descarta SOLO la lista de stopwords de spaCy
# (se acumula durante la construcción del parquet: no requiere otra pasada)
# ---------------------------------------------------------------------------


def ruta_diag_noun(cfg: PrepConfig) -> Path:
    sufijo = "_fast" if cfg.fast else ""
    return utils.data_dir() / f"diag_noun_spacy{sufijo}.csv"


class DiagNounSpacy:
    """Acumula, documento a documento, los tokens cuyo lema o forma está en
    STOP_WORDS de spaCy y NO en NLTK ni en la lista de dominio (palabras que
    solo spaCy considera vacías). Para cada lema guarda: apariciones como NOUN,
    documentos, POS de todas sus apariciones, cuántas apariciones NOUN caen en
    una locución gramatical (LOCUCIONES), dos contextos fuera de locución, uno
    dentro, y si está en `conservar`. Las apariciones NOUN fuera de locución
    de un lema en `conservar` son las que entran a los lemas (n_fuera_locucion).
    Se escribe en data/diag_noun_spacy[_fast].csv al construir el parquet."""

    def __init__(self, conservar: set[str], ventana: int = 6):
        from nltk.corpus import stopwords
        from spacy.lang.es.stop_words import STOP_WORDS

        nltk_sw = set(stopwords.words("spanish"))
        dominio = set()
        for sec, palabras in utils.leer_stopwords_dominio().items():
            if not sec.startswith("conservar"):
                dominio |= set(palabras)
        self.nltk = nltk_sw | dominio
        self.solo_spacy = set(STOP_WORDS) - self.nltk
        self.conservar = conservar
        self.ventana = ventana
        self.loc = _RE_LOCUCIONES
        self.ej_loc = {}
        self.pos = {}  # lema -> Counter(POS)
        self.n_noun = Counter()
        self.n_loc = Counter()
        self.docs = {}
        self.formas = {}
        self.ejemplos = {}

    def observar(self, doc, i_doc: int) -> None:
        toks = [t for t in doc if not t.is_space]
        for j, t in enumerate(toks):
            if not t.is_alpha:
                continue
            low = t.lower_
            lema = (t.lemma_.lower().split() or [low])[0]
            if not (lema in self.solo_spacy or low in self.solo_spacy):
                continue
            if lema in self.nltk or low in self.nltk:
                continue
            self.pos.setdefault(lema, Counter())[t.pos_] += 1
            if t.pos_ != "NOUN":
                continue
            self.n_noun[lema] += 1
            self.docs.setdefault(lema, set()).add(i_doc)
            self.formas.setdefault(lema, Counter())[low] += 1
            ini = max(0, j - self.ventana)
            texto = " ".join(x.text for x in toks[ini : j + self.ventana + 1])
            if _en_locucion(t, lema):
                self.n_loc[lema] += 1
                self.ej_loc.setdefault(lema, texto.replace("\n", " "))
                continue
            ej = self.ejemplos.setdefault(lema, [])
            if len(ej) < 2:
                ej.append(texto.replace("\n", " "))

    def tabla(self) -> pd.DataFrame:
        filas = []
        for lema, n in self.n_noun.most_common():
            pos = self.pos[lema]
            total = sum(pos.values())
            ej = self.ejemplos.get(lema, []) + ["", ""]
            filas.append(
                {
                    "lema": lema,
                    "n_noun": n,
                    "df": len(self.docs[lema]),
                    "frac_noun": round(n / total, 3),
                    "pos_otros": ", ".join(
                        f"{p}({c})" for p, c in pos.most_common() if p != "NOUN"
                    )[:60],
                    "formas": ", ".join(
                        f"{f}({c})" for f, c in self.formas[lema].most_common(3)
                    ),
                    "frac_locucion": round(self.n_loc[lema] / n, 3)
                    if lema in self.loc
                    else None,
                    "n_fuera_locucion": n - self.n_loc[lema],
                    "rescatado": lema in self.conservar,
                    "ejemplo_1": ej[0],
                    "ejemplo_2": ej[1],
                    "ejemplo_locucion": self.ej_loc.get(lema, ""),
                }
            )
        return pd.DataFrame(filas)
