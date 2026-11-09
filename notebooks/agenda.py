"""Entrega 3, S1: tópicos v2, agenda y puntos de cambio (lo consume el 08).

Todo lo reutilizable vive aquí; el notebook orquesta y explica. No modifica
utils.py ni prep.py: solo los usa (carga del parquet, stopwords de dominio y
colocaciones NPMI).

Bloques:
    Config08 ............................ parámetros declarados antes de ver resultados
    cobertura y formato ................. tabla_cobertura, series_formato, umbrales_mes
    tokens y representación ............. reconstruir_tokens, mapa_plurales, estrategias_bigramas,
                                          tabla_df, vectorizar
    muestras ............................ particion_ajuste_heldout
    NMF ................................. ajustar_nmf, rejilla_nmf, theta, medoide
    métricas ............................ Coherencia (C_NPMI y C_v sobre held-out), diversidad,
                                          estabilidad, error_relativo, regla_1se
    calidad ............................. marcar_residuales, titulares_representativos, tabla_nombres
    agenda .............................. prevalencia, bootstrap_prevalencia, js_div, bh,
                                          js_permutacion, transiciones, js_entre_fuentes, curva_rezago
    cambios ............................. binseg, detectar, detectar_formato, tabla_robustez
    eventos ............................. aciertos, linea_base_eventos
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

ET, SEMANA, DINERO = "www.eltiempo.com", "www.semana.com", "www.dinero.com"
CORTO = {ET: "eltiempo", SEMANA: "semana", DINERO: "dinero"}


# ---------------------------------------------------------------------------
# Configuración (todo lo que decide resultados está aquí y se exporta)
# ---------------------------------------------------------------------------


@dataclass
class Config08:
    fast: bool = False
    seed: int = 42
    año_min: int = 1980
    año_max: int = 2011
    # serie principal
    principal: str = ET
    principal_desde: int = 1990
    # muestras
    cap_ajuste: int = 1200  # docs/año para ajustar (no breves, no cartas)
    n_heldout: int = 300  # docs/año reservados para coherencia
    frac_heldout_max: float = 0.2  # en años chicos, como máximo 20 % al held-out
    # meses comparables: umbral = max(min_docs_mes, frac_mediana_mes * mediana mensual de la fuente)
    min_docs_mes: int = 10
    frac_mediana_mes: float = 0.3
    min_meses_validos: int = 6
    min_n_año: int = 40
    # representación
    min_df: int = 10
    max_df: float = 0.30
    min_df_plural: int = 5
    ratio_plural_min: float = 0.1
    renombres_propn: dict = field(default_factory=lambda: {"unidos": "estados_unidos"})
    no_pluralizar: tuple = ("unidos",)
    npmi_B: float = 0.35
    n_C: int = 50
    npmi_D: float = 0.20
    # NMF
    k_ref: int = 14
    ks: tuple = tuple(range(8, 21))
    semillas: tuple = (0, 1, 2)
    init: str = "random"
    max_iter: int = 1000
    tol: float = 1e-4
    n_jobs: int = 6
    # métricas
    topn_coh: int = 10
    ventana_npmi: int = 10
    ventana_cv: int = 110
    topn_div: int = 25
    estab_min: float = 0.80
    # residuales
    npmi_residual: float = 0.02
    df_generico: float = 0.15
    frac_topicos_generico: float = 0.30
    n_genericos_max: int = 3  # >3 de 10 genéricos -> residual
    ratio_breve: float = 3.0
    rho_longitud: float = 0.30
    estab_topico_min: float = 0.50
    # agenda y cambios
    b_perm: int = 1000
    b_boot: int = 1000
    b_binseg: int = 200
    min_seg: int = 2
    q_binseg: float = 0.95
    cap_peso: int = 3000
    ventana_coincidencia: int = 1
    soporte_min: float = 0.5
    lag_max: int = 15

    @classmethod
    def rapida(cls) -> "Config08":
        """Para el parquet FAST (6.000 docs): menos de todo."""
        return cls(
            fast=True, cap_ajuste=120, n_heldout=40, min_df=3, max_df=0.4,
            min_df_plural=2, ks=(8, 11, 14), k_ref=11, semillas=(0, 1, 2),
            max_iter=200, min_n_año=15, min_docs_mes=3, min_meses_validos=4,
            b_perm=100, b_boot=200, b_binseg=60, n_C=10,
        )

    def a_dict(self) -> dict:
        return asdict(self)


def _cerca(a: int, lista, v: int = 1) -> bool:
    return any(abs(int(a) - int(b)) <= v for b in lista)


# ---------------------------------------------------------------------------
# Cobertura y formato
# ---------------------------------------------------------------------------


def tabla_cobertura(meta: pd.DataFrame) -> pd.DataFrame:
    """Documentos por (fuente, año, mes), formato largo + pivote año×mes por fuente."""
    return (
        meta.groupby(["fuente", "año", "mes"]).size().rename("docs").reset_index()
    )


def series_formato(meta: pd.DataFrame) -> pd.DataFrame:
    """% es_breve, % es_carta, mediana de n_palabras_cuerpo y docs por fuente×año."""
    g = meta.groupby(["fuente", "año"])
    return pd.DataFrame(
        {
            "docs": g.size(),
            "pct_breve": g["es_breve"].mean() * 100,
            "pct_carta": g["es_carta"].mean() * 100,
            "mediana_palabras": g["n_palabras_cuerpo"].median(),
            "mediana_palabras_no_breve": meta[~meta["es_breve"]]
            .groupby(["fuente", "año"])["n_palabras_cuerpo"]
            .median(),
            # marcadores del régimen de archivo/scraping (07): titulares en
            # Mayúscula Inicial y lead duplicado
            "pct_titulo_cap": g["titulo_capitalizado"].mean() * 100,
            "pct_lead_dup": g["lead_duplicado"].mean() * 100,
        }
    ).reset_index()


def umbrales_mes(cob: pd.DataFrame, cfg: Config08) -> pd.Series:
    """Umbral de documentos por mes para cada fuente: max(min_docs_mes, frac·mediana).

    La mediana se toma sobre los meses con al menos un documento de esa fuente.
    Así Semana (semanal) y El Tiempo (diario) se comparan contra su propio nivel.
    """
    med = cob[cob["docs"] > 0].groupby("fuente")["docs"].median()
    return np.maximum(cfg.min_docs_mes, np.ceil(cfg.frac_mediana_mes * med)).astype(int)


def meses_validos(cob: pd.DataFrame, umbral: pd.Series) -> pd.DataFrame:
    """Marca `valido` por (fuente, año, mes); los meses ausentes se agregan con 0 docs."""
    filas = []
    for f, sub in cob.groupby("fuente"):
        años = range(sub["año"].min(), sub["año"].max() + 1)
        idx = pd.MultiIndex.from_product([años, range(1, 13)], names=["año", "mes"])
        s = sub.set_index(["año", "mes"])["docs"].reindex(idx, fill_value=0)
        d = s.reset_index()
        d["fuente"] = f
        d["umbral"] = int(umbral[f])
        d["valido"] = d["docs"] >= d["umbral"]
        filas.append(d)
    return pd.concat(filas, ignore_index=True)[["fuente", "año", "mes", "docs", "umbral", "valido"]]


# ---------------------------------------------------------------------------
# Tokens y representación
# ---------------------------------------------------------------------------

_RE_VOC = re.compile(r"[aeiou](?=[^aeiou]*$)")
_ACENTO = str.maketrans("aeiou", "áéíóú")


def _sin_tildes(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    ).replace("ñ", "ñ")


def _tilde_ultima(s: str) -> str:
    m = list(_RE_VOC.finditer(s))
    if not m:
        return s
    i = m[-1].start()
    return s[:i] + s[i].translate(_ACENTO) + s[i + 1 :]


def _candidatos_singular(w: str) -> list[str]:
    c = [w[:-1]]
    if w.endswith("es"):
        st = w[:-2]
        c += [st, _tilde_ultima(st), _sin_tildes(st)]
        if st.endswith("c"):
            c.append(st[:-1] + "z")  # veces -> vez
    return [x for x in dict.fromkeys(c) if len(x) >= 3 and x != w]


_RE_INVARIABLE = re.compile(r"(?:is|us|[áéíóú]s)$")  # crisis, énfasis, tenis, status, estrés


def mapa_plurales(df_vocab: Counter | dict, min_df: int = 5, excluir=(), ratio_min: float = 0.1) -> pd.DataFrame:
    """Plural -> singular solo si el singular existe en el vocabulario N/A.

    Condiciones: df(singular) ≥ min_df y df(singular) ≥ ratio_min · df(plural).
    La segunda evita mapear una palabra frecuente a un lema raro mal
    lematizado (crisis → "crisi" con df 5, a mediados → "mediado").
    Se excluyen las terminaciones invariables (-is, -us, vocal tildada + s).
    Si hay varios candidatos, gana el de mayor df. Devuelve la tabla auditable.
    """
    filas = []
    for w, dfw in df_vocab.items():
        if not w.endswith("s") or len(w) < 5 or w in excluir or _RE_INVARIABLE.search(w):
            continue
        cands = [(c, df_vocab.get(c, 0)) for c in _candidatos_singular(w)]
        cands = [(c, d) for c, d in cands if d >= min_df and d >= ratio_min * dfw]
        if cands:
            c, d = max(cands, key=lambda x: x[1])
            filas.append((w, c, dfw, d))
    t = pd.DataFrame(filas, columns=["plural", "singular", "df_plural", "df_singular"])
    return t.sort_values("df_plural", ascending=False).reset_index(drop=True)


def _mapear(tok: str, mapa: dict) -> str:
    if "_" in tok:
        return "_".join(mapa.get(p, p) for p in tok.split("_"))
    return mapa.get(tok, tok)


def reconstruir_tokens(
    lemas: Sequence[str],
    pos: str,
    cortes: Sequence[int],
    bigramas: frozenset | set = frozenset(),
    renombres: dict | None = None,
    mapa: dict | None = None,
    completo: bool = False,
) -> list[str]:
    """lemas_contenido + pos_contenido + cortes -> tokens del modelo.

    - Une bigramas de izquierda a derecha, sin cruzar `cortes` (mismo flujo y
      misma convención que prep.contar_bigramas: el par (j-1, j) existe si j
      no está en cortes).
    - Conserva N/A, los bigramas unidos y los PROPN de `renombres`.
    - El mapa de plurales no toca los tokens de `renombres` (`estados_unidos`
      sale igual por el renombre o por el bigrama de la lista blanca).
    - `completo=True` deja "" en las posiciones descartadas (para las
      ventanas deslizantes de la coherencia, que miden distancia en el texto).
    Sin bigramas, renombres ni mapa reproduce `lemas_nom_adj`.
    """
    lemas = list(lemas)
    cs = set(int(c) for c in cortes)
    n, i, out = len(lemas), 0, []
    while i < n:
        if bigramas and i + 1 < n and (i + 1) not in cs:
            b = lemas[i] + "_" + lemas[i + 1]
            if b in bigramas:
                out.append(b)
                i += 2
                continue
        c = pos[i]
        if c == "N" or c == "A":
            out.append(lemas[i])
        elif renombres and c == "P" and lemas[i] in renombres:
            out.append(renombres[lemas[i]])
        elif completo:
            out.append("")
        i += 1
    if mapa:
        fijos = set(renombres.values()) if renombres else set()
        out = [_mapear(t, mapa) if t and t not in fijos else t for t in out]
    return out


def tokens_corpus(dfp: pd.DataFrame, idx, **kw) -> list[list[str]]:
    L, P, C = dfp["lemas_contenido"].values, dfp["pos_contenido"].values, dfp["cortes_contenido"].values
    return [reconstruir_tokens(L[i], P[i], C[i], **kw) for i in idx]


def df_lemas(seqs) -> Counter:
    c = Counter()
    for s in seqs:
        c.update(set(s))
    return c


def tabla_bigramas(dfp: pd.DataFrame, min_n: int = 20, min_df: int = 10) -> pd.DataFrame:
    """Tabla NPMI completa (sin el umbral 0,35 del CSV), mismo cálculo del 07."""
    import prep

    bg = prep.colocaciones_npmi(
        dfp["lemas_contenido"], dfp["cortes_contenido"], dfp["pos_contenido"],
        min_n=min_n, min_df=min_df,
    )
    bg.insert(0, "bigrama", bg["w1"] + "_" + bg["w2"])
    return bg


def leer_lista_blanca(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        w = ln.split("#", 1)[0].strip().lower()
        if w:
            out.append(w)
    return out


def estrategias_bigramas(bg: pd.DataFrame, lista_blanca: Sequence[str], cfg: Config08) -> dict[str, frozenset]:
    """A–E, cada una con y sin lista blanca. D y E usan el mismo N = |C|."""
    na = bg[bg["tipo"] == "nom_adj"]
    B = na[na["npmi"] >= cfg.npmi_B]
    C = B[B["n"] >= cfg.n_C]
    N = len(C)
    D = na[na["npmi"] >= cfg.npmi_D].nlargest(N, "n")
    E = na.nlargest(N, "n")
    # Lista blanca curada a mano: cualquier tipo presente en la tabla (n ≥ 20, df ≥ 10).
    # Sobre todo con_propn (instituciones), más algunos nom_adj temáticos que el
    # NPMI deja fuera por ser pares de palabras frecuentes (acuerdo_paz, npmi 0,32).
    lb = frozenset(set(lista_blanca) & set(bg["bigrama"]))
    base = {
        "A": frozenset(),
        "B": frozenset(B["bigrama"]),
        "C": frozenset(C["bigrama"]),
        "D": frozenset(D["bigrama"]),
        "E": frozenset(E["bigrama"]),
    }
    out = {}
    for k, s in base.items():
        out[k] = s
        out[k + "+LB"] = s | lb
    return out


def tabla_df(seqs, umbrales_min=(2, 3, 5, 10, 20, 50), umbrales_max=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)):
    """Distribución de frecuencias documentales para justificar min_df / max_df.

    Devuelve (por_min_df, por_max_df, top_df): tamaño del vocabulario y % de
    ocurrencias cubiertas según min_df; términos que se pierden según max_df.
    """
    dfc, tf = Counter(), Counter()
    for s in seqs:
        tf.update(s)
        dfc.update(set(s))
    D = len(seqs)
    t = pd.DataFrame({"df": pd.Series(dfc), "tf": pd.Series(tf)})
    tot = t["tf"].sum()
    por_min = pd.DataFrame(
        [
            {"min_df": m, "vocab": int((t.df >= m).sum()), "pct_vocab": (t.df >= m).mean() * 100,
             "pct_ocurrencias": t.loc[t.df >= m, "tf"].sum() / tot * 100}
            for m in umbrales_min
        ]
    )
    por_max = pd.DataFrame(
        [
            {"max_df": m, "terminos_fuera": int((t.df > m * D).sum()),
             "pct_ocurrencias_fuera": t.loc[t.df > m * D, "tf"].sum() / tot * 100,
             "ejemplos": " ".join(t[t.df > m * D].sort_values("df").index[:12])}
            for m in umbrales_max
        ]
    )
    top = t.assign(df_ratio=t.df / D).sort_values("df", ascending=False).head(40)
    return por_min, por_max, top


def _identidad(x):
    return x


def vectorizar(docs_ajuste, cfg: Config08):
    from sklearn.feature_extraction.text import TfidfVectorizer

    v = TfidfVectorizer(
        analyzer=_identidad, lowercase=False, min_df=cfg.min_df, max_df=cfg.max_df,
        sublinear_tf=True, dtype=np.float32,
    )
    X = v.fit_transform(docs_ajuste).tocsr()
    return v, X


# ---------------------------------------------------------------------------
# Muestras
# ---------------------------------------------------------------------------


def particion_ajuste_heldout(meta: pd.DataFrame, cfg: Config08) -> tuple[np.ndarray, np.ndarray]:
    """Población: no breves y no cartas. Por año: primero el held-out, luego el ajuste.

    Dentro de cada año el muestreo es aleatorio, así que la mezcla de fuentes
    del año se conserva en ambas muestras (estratificación implícita).
    Devuelve posiciones (no ids) disjuntas.
    """
    rng = np.random.default_rng(cfg.seed)
    pob = (~meta["es_breve"].values) & (~meta["es_carta"].values)
    años = meta["año"].values
    fit, ho = [], []
    for y in range(cfg.año_min, cfg.año_max + 1):
        idx = np.where(pob & (años == y))[0]
        if not len(idx):
            continue
        idx = rng.permutation(idx)
        h = min(cfg.n_heldout, int(round(cfg.frac_heldout_max * len(idx))))
        ho.append(idx[:h])
        fit.append(idx[h : h + cfg.cap_ajuste])
    return np.sort(np.concatenate(fit)), np.sort(np.concatenate(ho))


# ---------------------------------------------------------------------------
# NMF
# ---------------------------------------------------------------------------


def ajustar_nmf(X, K: int, seed: int, cfg: Config08) -> dict:
    """NMF Frobenius (cd). Un hilo por ajuste: el paralelismo va por (K, semilla)."""
    from sklearn.decomposition import NMF
    from threadpoolctl import threadpool_limits

    import warnings

    from sklearn.exceptions import ConvergenceWarning

    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # se registra en `convergio`
        m = NMF(n_components=K, init=cfg.init, random_state=seed, max_iter=cfg.max_iter,
                tol=cfg.tol, solver="cd")
        W = m.fit_transform(X)
    return {"K": K, "seed": seed, "modelo": m, "H": m.components_, "W": W,
            "err": float(m.reconstruction_err_), "n_iter": int(m.n_iter_),
            "convergio": bool(m.n_iter_ < cfg.max_iter)}


def rejilla_nmf(X, ks, semillas, cfg: Config08) -> dict:
    from joblib import Parallel, delayed

    trabajos = [(K, s) for K in ks for s in semillas]
    res = Parallel(n_jobs=cfg.n_jobs, verbose=0)(delayed(ajustar_nmf)(X, K, s, cfg) for K, s in trabajos)
    return {(r["K"], r["seed"]): r for r in res}


def theta(W: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """θ_dk ∝ W_dk · Σ_v H_kv (masa de términos, no depende de la escala de H)."""
    Wt = W * H.sum(1)[None, :]
    r = Wt.sum(1)
    ok = r > 0
    T = np.zeros_like(Wt, dtype=np.float64)
    T[ok] = Wt[ok] / r[ok, None]
    return T, ok


def _norm_filas(H):
    return H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)


def emparejar(Ha: np.ndarray, Hb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Emparejamiento húngaro por coseno. Devuelve (col de Hb para cada fila de Ha, cosenos)."""
    S = _norm_filas(Ha) @ _norm_filas(Hb).T
    r, c = linear_sum_assignment(-S)
    o = np.argsort(r)
    return c[o], S[r, c][o]


def top_idx(H, n):
    return np.argsort(-H, axis=1)[:, :n]


def estabilidad(Hs: list[np.ndarray], topn: int = 10) -> dict:
    """Media sobre pares de semillas: coseno emparejado y Jaccard del top-n emparejado."""
    cos, jac = [], []
    for i in range(len(Hs)):
        for j in range(i + 1, len(Hs)):
            c, s = emparejar(Hs[i], Hs[j])
            ta, tb = top_idx(Hs[i], topn), top_idx(Hs[j], topn)[c]
            jac.append(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(ta, tb)]))
            cos.append(s.mean())
    return {"estab_cos": float(np.mean(cos)), "estab_jaccard": float(np.mean(jac))}


def estabilidad_por_topico(H: np.ndarray, otras: list[np.ndarray]) -> np.ndarray:
    return np.mean([emparejar(H, o)[1] for o in otras], axis=0)


def medoide(Hs: dict) -> int:
    """Semilla cuyo H más se parece (coseno emparejado medio) a las demás."""
    ks = list(Hs)
    sc = {s: np.mean([emparejar(Hs[s], Hs[t])[1].mean() for t in ks if t != s]) for s in ks}
    return max(sc, key=sc.get)


def residuales_por_emparejamiento(H_ref, feat_ref, resid_ref: set[int], H_v, feat_v, umbral: float = 0.5) -> set[int]:
    """Traslada las marcas de residual del modelo de referencia a una variante
    (otra semilla, otro K u otra representación).

    Alinea H sobre el vocabulario común; cada tópico de la variante hereda la
    marca de su tópico de referencia más parecido si el coseno ≥ umbral.
    """
    comunes = sorted(set(feat_ref) & set(feat_v))
    ia = pd.Index(feat_ref).get_indexer(comunes)
    ib = pd.Index(feat_v).get_indexer(comunes)
    S = _norm_filas(H_ref[:, ia]) @ _norm_filas(H_v[:, ib]).T
    best = S.argmax(0)
    return {j for j in range(H_v.shape[0]) if best[j] in resid_ref and S[best[j], j] >= umbral}


def theta_agenda(T: np.ndarray, ok: np.ndarray, residuales: set[int]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """θ sin los tópicos residuales, renormalizado; docs sin masa restante salen de `ok`."""
    keep = [k for k in range(T.shape[1]) if k not in residuales]
    Tk = T[:, keep]
    r = Tk.sum(1)
    ok2 = ok & (r > 0)
    out = np.zeros_like(Tk)
    out[ok2] = Tk[ok2] / r[ok2, None]
    return out, ok2, keep


def diversidad(H, topn: int = 25) -> float:
    t = top_idx(H, topn)
    return len(np.unique(t)) / t.size


def error_relativo(X, W, H) -> float:
    """||X − WH||_F / ||X||_F sin densificar X."""
    X = sp.csr_matrix(X)
    nx2 = float(X.multiply(X).sum())
    cross = float(np.sum((X @ H.T) * W))
    wh2 = float(np.sum((W.T @ W) * (H @ H.T)))
    return float(np.sqrt(max(nx2 - 2 * cross + wh2, 0.0) / nx2))


# ---------------------------------------------------------------------------
# Coherencia sobre held-out (Röder et al. 2015)
# ---------------------------------------------------------------------------


def matriz_ventanas(docs: list[list[str]], vocab: dict[str, int], w: int) -> sp.csr_matrix:
    """Ventanas deslizantes booleanas (n_ventanas × |vocab|).

    Un documento de n posiciones aporta max(1, n − w + 1) ventanas; la
    posición p cae en las ventanas [p − w + 1, p] recortadas al rango.
    Las posiciones descartadas ("") cuentan para la distancia.
    """
    rows, cols, base = [], [], 0
    for d in docs:
        n = len(d)
        if n == 0:
            continue
        nw = max(1, n - w + 1)
        pos = np.array([p for p, t in enumerate(d) if t in vocab], dtype=np.int64)
        if len(pos):
            ids = np.array([vocab[d[p]] for p in pos], dtype=np.int64)
            lo = np.maximum(0, pos - w + 1)
            hi = np.minimum(pos, nw - 1)
            cnt = hi - lo + 1
            off = np.repeat(lo - np.cumsum(np.r_[0, cnt[:-1]]), cnt)
            rows.append(base + off + np.arange(cnt.sum()))
            cols.append(np.repeat(ids, cnt))
        base += nw
    r = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    c = np.concatenate(cols) if cols else np.zeros(0, np.int64)
    M = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(base, len(vocab)))
    M.sum_duplicates()
    M.data[:] = 1.0  # duplicados (mismo término dos veces en la ventana) cuentan una vez
    return M


class Coherencia:
    """Conteos de ventanas sobre held-out para una unión fija de términos.

    Se construye una vez por representación; luego evalúa cualquier H.
    """

    def __init__(self, docs_completos: list[list[str]], terminos: Sequence[str], ventana: int):
        self.vocab = {t: i for i, t in enumerate(dict.fromkeys(terminos))}
        V = matriz_ventanas(docs_completos, self.vocab, ventana)
        self.N = V.shape[0]
        self.p = np.asarray(V.sum(0)).ravel() / self.N
        self.pij = (V.T @ V).toarray() / self.N

    def _sub(self, palabras):
        ix = np.array([self.vocab[w] for w in palabras])
        return self.p[ix], self.pij[np.ix_(ix, ix)]

    def npmi_matriz(self, palabras, eps: float = 0.0):
        p, pij = self._sub(palabras)
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.log((pij + eps) / np.outer(p, p)) / -np.log(pij + eps)
        if eps == 0:
            m[pij == 0] = -1.0  # nunca co-ocurren: mínimo de NPMI
        m[~np.isfinite(m)] = 0.0
        return m

    def npmi(self, palabras) -> float:
        """C_NPMI: media de NPMI sobre todos los pares (segmentación one-one)."""
        m = self.npmi_matriz(palabras)
        iu = np.triu_indices(len(palabras), 1)
        return float(m[iu].mean())

    def cv(self, palabras) -> float:
        """C_v: vectores de contexto NPMI (ε=1e-12), coseno one-set."""
        m = self.npmi_matriz(palabras, eps=1e-12)
        tot = m.sum(0)
        cos = (m @ tot) / (np.linalg.norm(m, axis=1) * np.linalg.norm(tot) + 1e-12)
        return float(cos.mean())


def palabras_top(H, feat, n):
    return [[feat[j] for j in fila] for fila in top_idx(H, n)]


def metricas_modelo(r: dict, feat, coh: Coherencia, cfg: Config08, coh_cv: Coherencia | None = None,
                    X_fit=None, X_ho=None) -> dict:
    H = r["H"]
    tops = palabras_top(H, feat, cfg.topn_coh)
    np_t = np.array([coh.npmi(t) for t in tops])
    out = {"K": r["K"], "seed": r["seed"], "npmi_media": np_t.mean(), "npmi_min": np_t.min(),
           "npmi_mediana": float(np.median(np_t)), "diversidad": diversidad(H, cfg.topn_div),
           "err_frob": r["err"], "n_iter": r["n_iter"], "convergio": float(r["convergio"])}
    if coh_cv is not None:
        out["cv_media"] = float(np.mean([coh_cv.cv(t) for t in tops]))
    if X_fit is not None:
        out["err_rel_ajuste"] = error_relativo(X_fit, r["W"], H)
    if X_ho is not None:
        out["err_rel_heldout"] = error_relativo(X_ho, r["modelo"].transform(X_ho), H)
    out["_npmi_topicos"] = np_t
    return out


def regla_1se(tab: pd.DataFrame, por: str, metrica: str = "npmi_media") -> pd.DataFrame:
    """Agrega por `por` (K o estrategia): media, EE entre semillas y estabilidad."""
    g = tab.groupby(por)
    agg = g[metrica].agg(["mean", "std", "count"])
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    agg = agg.rename(columns={"mean": metrica})[[metrica, "se"]]
    otras = [c for c in tab.columns if c not in (por, "seed", metrica) and pd.api.types.is_numeric_dtype(tab[c])]
    return agg.join(g[otras].mean())


def elegir_K(agg: pd.DataFrame, cfg: Config08) -> tuple[int, str]:
    """Regla declarada antes de ver resultados (ver sección 0 del 08)."""
    best = agg["npmi_media"].idxmax()
    thr = agg.loc[best, "npmi_media"] - agg.loc[best, "se"]
    cand = agg[(agg["npmi_media"] >= thr) & (agg["estab_cos"] >= cfg.estab_min)]
    if len(cand):
        K = int(cand.index.min())
        return K, f"K más chico con NPMI ≥ {thr:.4f} (mejor K={best} − 1 EE) y estabilidad ≥ {cfg.estab_min}"
    K = int(agg["estab_cos"].idxmax())
    return K, f"ningún K cumple ambas condiciones; se toma el más estable (K={K})"


def elegir_estrategia(agg: pd.DataFrame) -> tuple[str, str]:
    """Mejor NPMI; empates (≤ 1 EE) -> menos residuales -> más estabilidad.
    Si no supera a A por más de 1 EE, se queda A."""
    best = agg["npmi_media"].idxmax()
    thr = agg.loc[best, "npmi_media"] - agg.loc[best, "se"]
    emp = agg[agg["npmi_media"] >= thr].sort_values(["pct_residuales", "estab_cos"], ascending=[True, False])
    elegida = emp.index[0]
    if "A" in agg.index and elegida != "A":
        a = agg.loc["A"]
        if agg.loc[elegida, "npmi_media"] - a["npmi_media"] <= agg.loc[elegida, "se"]:
            return "A", f"{elegida} no supera a A por más de 1 EE: se va sin bigramas"
    return elegida, f"mejor NPMI {agg.loc[best,'npmi_media']:.4f} ({best}); empate ≤1 EE resuelto por residuales y estabilidad"


# ---------------------------------------------------------------------------
# Calidad de tópicos
# ---------------------------------------------------------------------------


def marcar_residuales(H, feat, npmi_t, estab_t, df_ratio: dict, prohibidas: set, cfg: Config08,
                      T: np.ndarray | None = None, es_breve=None, log_long=None) -> pd.DataFrame:
    """Marca automática (editable a mano después). Una fila por tópico con motivo.

    - npmi_bajo: C_NPMI held-out < cfg.npmi_residual
    - generico: > n_genericos_max de los top-10 con df > df_generico o presentes
      en el top-10 de ≥ frac_topicos_generico de los tópicos
    - metatexto: algún top-10 en las secciones dias/meses/nombres_pila/metatexto/scraping
    - inestable: coseno medio con su pareja en otras semillas < estab_topico_min
    - formato (si se pasa θ): prevalencia breve/no breve > ratio_breve (o < 1/ratio),
      o |Spearman(θ_k, log longitud)| > rho_longitud
    """
    K = H.shape[0]
    tops = palabras_top(H, feat, 10)
    cuenta = Counter(w for t in tops for w in t)
    filas = []
    rho = ratio = None
    if T is not None:
        from scipy.stats import rankdata

        rl = rankdata(log_long)
        rT = np.apply_along_axis(rankdata, 0, T)
        rho = np.array([np.corrcoef(rT[:, k], rl)[0, 1] for k in range(K)])
        eb = np.asarray(es_breve, bool)
        ratio = (T[eb].mean(0) + 1e-9) / (T[~eb].mean(0) + 1e-9)
    for k in range(K):
        gen = [w for w in tops[k] if df_ratio.get(w, 0) > cfg.df_generico or cuenta[w] >= cfg.frac_topicos_generico * K]
        meta = [w for w in tops[k] if any(p in prohibidas for p in w.split("_"))]
        m = []
        if npmi_t[k] < cfg.npmi_residual:
            m.append(f"npmi_bajo ({npmi_t[k]:.3f})")
        if len(gen) > cfg.n_genericos_max:
            m.append("generico: " + " ".join(gen))
        if meta:
            m.append("metatexto: " + " ".join(meta))
        if estab_t is not None and estab_t[k] < cfg.estab_topico_min:
            m.append(f"inestable ({estab_t[k]:.2f})")
        if ratio is not None and (ratio[k] > cfg.ratio_breve or ratio[k] < 1 / cfg.ratio_breve):
            m.append(f"formato: breve/no breve = {ratio[k]:.2f}")
        if rho is not None and abs(rho[k]) > cfg.rho_longitud:
            m.append(f"formato: rho longitud = {rho[k]:.2f}")
        filas.append({"topico": f"T{k:02d}", "residual_sugerido": bool(m), "motivo": "; ".join(m),
                      "n_genericos": len(gen),
                      "ratio_breve": None if ratio is None else round(float(ratio[k]), 3),
                      "rho_longitud": None if rho is None else round(float(rho[k]), 3)})
    return pd.DataFrame(filas)


def titulares_representativos(T, titulos, fuentes, fechas, n=5) -> dict[int, list[str]]:
    out = {}
    for k in range(T.shape[1]):
        ix = np.argsort(-T[:, k])[:n]
        out[k] = [f"{T[i,k]:.2f} | {CORTO.get(fuentes[i], fuentes[i])} {pd.Timestamp(fechas[i]):%Y-%m} | {titulos[i]}" for i in ix]
    return out


def tabla_nombres(ruta: Path, base: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """CSV editable: topico, top10, nombre, residual, motivo.

    Si existe y sus top10 coinciden con el modelo actual, se respetan las
    ediciones (nombre, residual). Si no coinciden, se guarda una copia .bak
    y se escribe uno nuevo con las sugerencias automáticas.
    """
    nuevo = base[["topico", "top10", "residual_sugerido", "motivo"]].copy()
    nuevo["nombre"] = nuevo["top10"].str.split().str[:3].str.join(" ")
    nuevo["residual"] = nuevo["residual_sugerido"]
    if ruta.exists():
        viejo = pd.read_csv(ruta, keep_default_na=False)
        if len(viejo) == len(nuevo) and (viejo["top10"].values == nuevo["top10"].values).all():
            viejo["residual"] = viejo["residual"].astype(str).str.lower().isin(["true", "1", "si", "sí", "x"])
            return viejo, "leído (ediciones manuales respetadas)"
        ruta.rename(ruta.with_suffix(".csv.bak"))
        estado = "el modelo cambió: copia anterior en .bak, nuevo CSV con sugerencias"
    else:
        estado = "creado con sugerencias automáticas"
    cols = ["topico", "nombre", "residual", "residual_sugerido", "motivo", "top10"]
    nuevo[cols].to_csv(ruta, index=False)
    return nuevo[cols], estado


# ---------------------------------------------------------------------------
# Agenda: prevalencias, JS, nulos
# ---------------------------------------------------------------------------


def prevalencia(T, años, meses=None, valido_mes: set | None = None, modo: str = "doc",
                años_rango=None, min_meses: int = 6) -> tuple[pd.DataFrame, pd.Series]:
    """Prevalencia anual (filas suman 1 si θ suma 1).

    modo="doc": media simple por documento.
    modo="mes": media de las medias de los meses válidos (igual peso por mes);
                año NaN si tiene menos de `min_meses` meses válidos.
    Devuelve (P años×K, n docs usados por año).
    """
    años = np.asarray(años)
    K = T.shape[1]
    rango = años_rango if años_rango is not None else np.arange(años.min(), años.max() + 1)
    if modo == "doc":
        d = pd.DataFrame(T)
        d["año"] = años
        g = d.groupby("año")
        return g.mean().reindex(rango), g.size().reindex(rango, fill_value=0)
    meses = np.asarray(meses)
    ok = np.fromiter(((int(a), int(m)) in valido_mes for a, m in zip(años, meses)), dtype=bool, count=len(años))
    d = pd.DataFrame(T[ok])
    d["año"], d["mes"] = años[ok], meses[ok]
    pm = d.groupby(["año", "mes"]).mean()
    nm = pm.groupby(level=0).size()
    P = pm.groupby(level=0).mean()
    P.loc[(nm.reindex(P.index) < min_meses).values] = np.nan
    n = pd.Series(años[ok]).value_counts().reindex(rango, fill_value=0)
    n[nm.reindex(rango, fill_value=0) < min_meses] = 0
    P = P.reindex(rango)
    P.columns = range(K)
    return P, n


def bootstrap_prevalencia(T, años, meses=None, valido_mes=None, modo="doc", B=1000, seed=0,
                          años_rango=None, min_meses=6, alfa=0.05) -> pd.DataFrame:
    """IC percentil por remuestreo de documentos dentro del año (o dentro de cada mes válido).

    Multinomial vectorizado: medias bootstrap = conteos @ θ / n.
    Formato largo: año, topico, lo, hi.
    """
    rg = np.random.default_rng(seed)
    años = np.asarray(años)
    rango = años_rango if años_rango is not None else np.unique(años)
    filas = []
    for y in rango:
        if modo == "doc":
            grupos = [np.where(años == y)[0]]
        else:
            meses_a = np.asarray(meses)
            ms = [m for m in range(1, 13) if (int(y), m) in valido_mes]
            if len(ms) < min_meses:
                continue
            grupos = [np.where((años == y) & (meses_a == m))[0] for m in ms]
        grupos = [g for g in grupos if len(g)]
        if not grupos:
            continue
        acc = 0
        for g in grupos:
            cnt = rg.multinomial(len(g), np.full(len(g), 1 / len(g)), size=B)
            acc = acc + cnt @ T[g] / len(g)
        bm = acc / len(grupos)
        lo, hi = np.quantile(bm, [alfa / 2, 1 - alfa / 2], axis=0)
        for k in range(T.shape[1]):
            filas.append({"año": int(y), "topico": k, "lo": lo[k], "hi": hi[k]})
    return pd.DataFrame(filas)


def js_div(p, q):
    p = np.clip(p, 1e-12, None); q = np.clip(q, 1e-12, None)
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)
    return 0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum()


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for rank, i in zip(range(n, 0, -1), o[::-1]):
        prev = min(prev, p[i] * n / rank); q[i] = prev
    return q


def js_permutacion(T, a, b, estratos, B, rg) -> dict:
    """JS entre las medias de θ en a y b, contra el nulo de permutar las
    etiquetas a/b dentro de cada estrato (mismo diseño del 06)."""
    pooled = np.concatenate([a, b])
    Th = T[pooled]
    lab = np.r_[np.zeros(len(a)), np.ones(len(b))]
    est = np.asarray(estratos)[pooled]
    grupos = [np.where(est == e)[0] for e in np.unique(est)]
    obs = js_div(T[a].mean(0), T[b].mean(0))
    tot, n0, n1 = Th.sum(0), len(a), len(b)
    nulls = np.empty(B)
    for r in range(B):
        lp = lab.copy()
        for g in grupos:
            lp[g] = rg.permutation(lab[g])
        s1 = Th.T @ lp
        nulls[r] = js_div((tot - s1) / n0, s1 / n1)
    return {"n1": n0, "n2": n1, "js": obs, "nulo_media": nulls.mean(), "nulo_p95": np.quantile(nulls, 0.95),
            "exceso": obs - nulls.mean(), "z": (obs - nulls.mean()) / (nulls.std() + 1e-12),
            "p": (1 + (nulls >= obs).sum()) / (B + 1)}


def transiciones(T, años, estratos, mask, años_rango, lag=1, B=1000, min_n=40, seed=0) -> pd.DataFrame:
    """JS(año → año+lag) con nulo por permutación dentro de `estratos`
    (p. ej. fuente|es_breve). El reporte usa `exceso`; p_bh es anexo."""
    rg = np.random.default_rng(seed)
    años = np.asarray(años)
    idx = {y: np.where(mask & (años == y))[0] for y in años_rango}
    out = []
    for y in años_rango:
        y2 = y + lag
        if y2 not in idx or len(idx[y]) < min_n or len(idx[y2]) < min_n:
            continue
        out.append({"año": int(y), "año2": int(y2), **js_permutacion(T, idx[y], idx[y2], estratos, B, rg)})
    r = pd.DataFrame(out)
    if len(r):
        r["p_bh"] = bh(r["p"].values)
    return r


def js_entre_fuentes(T, años, fuentes, estratos, mask, fa, fb, años_rango, B=1000, min_n=40, seed=0) -> pd.DataFrame:
    """Referencia de escala: JS entre dos fuentes en el mismo año, con su propio nulo."""
    rg = np.random.default_rng(seed)
    años, fuentes = np.asarray(años), np.asarray(fuentes)
    out = []
    for y in años_rango:
        a = np.where(mask & (años == y) & (fuentes == fa))[0]
        b = np.where(mask & (años == y) & (fuentes == fb))[0]
        if len(a) >= min_n and len(b) >= min_n:
            out.append({"año": int(y), **js_permutacion(T, a, b, estratos, B, rg)})
    return pd.DataFrame(out)


def curva_rezago(P: pd.DataFrame, lag_max=15) -> pd.DataFrame:
    """JS medio entre prevalencias anuales separadas L años (años NaN excluidos)."""
    ys = [y for y in P.index if not P.loc[y].isna().any()]
    out = []
    for L in range(1, lag_max + 1):
        d = [js_div(P.loc[y].values, P.loc[y + L].values) for y in ys if y + L in ys]
        out.append({"rezago": L, "js_medio": np.mean(d) if d else np.nan, "n_pares": len(d)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Puntos de cambio (binseg del 06, parametrizado)
# ---------------------------------------------------------------------------


def _wsse(M, w):
    mu = (M * w[:, None]).sum(0) / w.sum()
    return (w * ((M - mu) ** 2).sum(1)).sum()


def _best_split(M, w, min_seg):
    n = len(M); tot = _wsse(M, w); best = (0.0, None)
    for s in range(min_seg, n - min_seg + 1):
        g = tot - _wsse(M[:s], w[:s]) - _wsse(M[s:], w[s:])
        if g > best[0]:
            best = (g, s)
    return best


def binseg(M, w, min_seg=2, B=200, q=0.95, seed=0):
    """Segmentación binaria; umbral de cada división = cuantil q de la ganancia
    máxima con el orden de los años barajado (idéntico al 06)."""
    rg = np.random.default_rng(seed); cps = []; stack = [(0, len(M))]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2 * min_seg:
            continue
        g, s = _best_split(M[lo:hi], w[lo:hi], min_seg)
        if s is None:
            continue
        gains = [_best_split(M[lo:hi][p], w[lo:hi][p], min_seg)[0]
                 for p in (rg.permutation(hi - lo) for _ in range(B))]
        thr = np.quantile(gains, q)
        if g > thr:
            cps.append((lo + s, g / thr)); stack += [(lo, lo + s), (lo + s, hi)]
    return sorted(cps)


def detectar(P: pd.DataFrame, n: pd.Series, nombre: str, cfg: Config08, min_n=None) -> dict:
    """binseg sobre sqrt(prevalencia) (≈ Hellinger), peso min(n, cap_peso).
    Cada cambio se reporta como el año que INICIA el nuevo régimen."""
    min_n = cfg.min_n_año if min_n is None else min_n
    v = (n.reindex(P.index, fill_value=0) >= min_n) & P.notna().all(1)
    yrs = np.array(P.index)[v.values]
    if len(yrs) < 2 * cfg.min_seg:
        return {"variante": nombre, "años_validos": "insuficientes", "cambios": [], "fuerza": []}
    M = np.sqrt(np.clip(P[v].values.astype(float), 0, None))
    w = np.minimum(n.reindex(P.index)[v].values, cfg.cap_peso).astype(float)
    cps = binseg(M, w, cfg.min_seg, cfg.b_binseg, cfg.q_binseg, seed=cfg.seed)
    return {"variante": nombre, "años_validos": f"{yrs.min()}–{yrs.max()} ({len(yrs)})",
            "cambios": [int(yrs[s]) for s, _ in cps], "fuerza": [round(float(f), 2) for _, f in cps]}


def detectar_formato(fmt: pd.DataFrame, nombre: str, cfg: Config08) -> list[dict]:
    """Mismo detector sobre series de FORMATO de una fuente (índice = año):
    % breves, log mediana de palabras, log docs/año, % titulares en Mayúscula
    Inicial y % lead duplicado; cada una y todas juntas (z-score).
    Las series constantes en la fuente (std = 0) se omiten."""
    f = fmt.sort_index()
    S = pd.DataFrame({"pct_breve": f["pct_breve"], "log_mediana_palabras": np.log(f["mediana_palabras"]),
                      "log_docs": np.log(f["docs"]), "pct_titulo_cap": f["pct_titulo_cap"],
                      "pct_lead_dup": f["pct_lead_dup"]})
    S = S.loc[:, S.std() > 1e-9]
    out = []
    for cols in [[c] for c in S.columns] + [list(S.columns)]:
        Z = S[cols]
        Z = (Z - Z.mean()) / (Z.std() + 1e-12)
        cps = binseg(Z.values, np.ones(len(Z)), cfg.min_seg, cfg.b_binseg, cfg.q_binseg, seed=cfg.seed)
        out.append({"variante": f"{nombre}: " + ("+".join(cols) if len(cols) > 1 else cols[0]),
                    "años_validos": f"{Z.index.min()}–{Z.index.max()} ({len(Z)})",
                    "cambios": [int(Z.index[s]) for s, _ in cps], "fuerza": [round(float(x), 2) for _, x in cps]})
    return out


def tabla_robustez(series: dict[str, list[int]], votantes: dict[str, list[int]],
                   formato: dict[str, list[int]], contraste: dict[str, list[int]],
                   informativas: dict[str, list[int]], cfg: Config08) -> pd.DataFrame:
    """Un renglón por cambio candidato (unión de las 4 series de control, colapsada a ±v).

    `series` debe traer: todos_doc, sin_breves_doc, todos_mes, sin_breves_mes.
    Veredicto:
      robusto            → aparece sin breves, con peso por mes y en ambas a la vez,
                           y en ≥ soporte_min de los votantes
      artefacto probable → coincide con un cambio de formato y NO sobrevive a los controles
      no robusto         → el resto
    Coincidir con formato se reporta siempre, también en los robustos.
    """
    v = cfg.ventana_coincidencia
    todos = sorted(a for c in series.values() for a in c)
    cand = []
    for a in sorted(set(todos), key=lambda x: -sum(_cerca(x, c, v) for c in series.values())):
        if not _cerca(a, cand, v):
            cand.append(a)
    filas = []
    for c in sorted(cand):
        f = {"año": c}
        for k, lst in series.items():
            f[k] = _cerca(c, lst, v)
        f["soporte"] = np.mean([_cerca(c, l, v) for l in votantes.values()]) if votantes else np.nan
        for k, lst in votantes.items():
            f["v:" + k] = _cerca(c, lst, v)
        fmt = [k for k, lst in formato.items() if _cerca(c, lst, v)]
        f["coincide_formato"] = bool(fmt)
        f["series_formato"] = ", ".join(fmt)
        for k, lst in contraste.items():
            f["c:" + k] = _cerca(c, lst, v)
        for k, lst in informativas.items():
            f["i:" + k] = _cerca(c, lst, v)
        sobrevive = f["sin_breves_doc"] and f["todos_mes"] and f["sin_breves_mes"]
        if sobrevive and f["soporte"] >= cfg.soporte_min:
            f["veredicto"] = "robusto" + (" (coincide con formato)" if fmt else "")
        elif fmt and not sobrevive:
            f["veredicto"] = "artefacto probable de formato"
        else:
            f["veredicto"] = "no robusto"
        filas.append(f)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Contraste histórico
# ---------------------------------------------------------------------------


def aciertos(cambios, eventos, ventana=1) -> int:
    return sum(_cerca(c, list(eventos), ventana) for c in cambios)


def linea_base_eventos(cambios, eventos, años_validos, n_sim=5000, seed=42, ventana=1, min_seg=2) -> dict:
    """Aciertos observados vs. los de la misma cantidad de años al azar
    (entre los años en que el detector puede marcar un inicio de régimen)."""
    rg = np.random.default_rng(seed)
    elegibles = list(años_validos)[min_seg:len(años_validos) - min_seg + 1]
    nc = len(cambios)
    obs = aciertos(cambios, eventos, ventana)
    if not nc or len(elegibles) <= nc:
        return {"cambios": list(cambios), "aciertos": obs, "esperado_azar": None, "p_azar_ge_obs": None}
    base = np.array([aciertos(rg.choice(elegibles, nc, replace=False), eventos, ventana) for _ in range(n_sim)])
    return {"cambios": list(map(int, cambios)), "aciertos": int(obs), "esperado_azar": float(base.mean()),
            "p_azar_ge_obs": float((base >= obs).mean())}


def a_json(obj, path: Path):
    def conv(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, frozenset, tuple)):
            return list(o)
        return str(o)

    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=conv), encoding="utf-8")
