"""Entrega 3, S2: cronolecto (lo consume el 09).

El 08 mide QUÉ temas cubre la prensa; este módulo mide CÓMO los nombra.
No modifica utils.py, prep.py ni agenda.py: los usa (binseg, js_div,
reconstruir_tokens, _mapear, leer_lista_blanca, a_json).

Bloques:
    ConfigCrono / congelar ............ parámetros + hash de lo declarado antes de ver resultados
    conteo y rareza ................... matriz_conteos, pmw_zipf, piso_zipf, ic_poisson,
                                        bootstrap_tasa, ancho_relativo, dispersion_dp,
                                        frac_en_entidades, zipf_referencia
    variantes ......................... textos_unidos, compilar, contar_patrones, cuotas,
                                        cuota_estandarizada, año_cruce, cruces, cambios_cuota,
                                        contextos
    codificación (exploratorio) ....... pares_diacriticos, cuotas_formas, tasa_diacriticos
    cambio semántico .................. secuencias_lema, corpus_periodos, muestra_bootstrap,
                                        entrenar_bootstrap, entrenar_w2v,
                                        prep_emb, procrustes, desplazamiento, elegir_umbral_vocab,
                                        control_frecuencia, vecindario_local, vecinos_estables,
                                        reemplazos_candidatos
    neologismos ....................... uso_sostenido, tabla_categorias
    variación dentro del tema ......... cargar_theta, logodds_dirichlet, js_estandarizado
Regla anti-bug del 05: sin datos suficientes se devuelve vacío o NaN explícito,
nunca una lista de relleno.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

import agenda
from agenda import CORTO, DINERO, ET, SEMANA  # noqa: F401 (reexportados para el notebook)

# ---------------------------------------------------------------------------
# Configuración y congelación
# ---------------------------------------------------------------------------


@dataclass
class ConfigCrono:
    fast: bool = False
    seed: int = 42
    # series
    principal: str = ET
    contraste: str = SEMANA
    et_desde: int = 1990
    año_min: int = 1980
    año_max: int = 2011
    periodos: tuple = (("P1", 1990, 1994), ("P2", 1995, 1999), ("P3", 2000, 2005), ("P4", 2006, 2011))
    año_formato: int = 2006  # cambio de archivo de El Tiempo
    # rareza
    bandas: tuple = (3.0, 4.0)  # Zipf < 3 raro, 3–4 medio, ≥ 4 frecuente
    min_tokens_grupo: int = 20_000  # por debajo, el pmw del grupo es NaN
    min_df_formas: int = 2
    min_total_export: int = 20
    b_boot: int = 1000
    hw_rel_max: float = 0.5  # conteo mínimo: semiancho relativo mediano ≤ 0.5
    umbral_propio: float = 0.5
    n_terminos_demo: int = 3000
    # variantes
    min_ocurr_año: int = 20
    cruce_años: int = 3
    ventana_kwic: int = 8
    n_contextos: int = 3
    # binseg (mismos valores que el 08)
    b_binseg: int = 200
    q_binseg: float = 0.95
    min_seg: int = 2
    cap_peso: int = 3000
    # word2vec
    w2v_dim: int = 100
    w2v_ventana: int = 5
    w2v_epocas: int = 10
    w2v_neg: int = 5
    w2v_sample: float = 1e-4
    w2v_min_count: int = 10
    w2v_semillas: tuple = (0, 1, 2, 3, 4)
    w2v_workers: int = 6
    umbrales_vocab: tuple = (20, 50, 100, 200)
    frac_ruido_vocab: float = 0.5
    q_sep: float = 0.05  # cuenta si p05(d_entre) > p95(d_dentro)
    k_vecinos: int = 25
    k_mostrar: int = 10
    min_semillas_vecino: int = 3
    log2_caida: float = -1.0
    log2_subida: float = 1.0
    min_pmw_reemplazo: float = 5.0
    # neologismos
    umbral_neo: float = 5.0  # ≈ 7–10 apariciones/año en El Tiempo (coherente con el conteo mínimo)
    umbrales_neo_sens: tuple = (2.0, 5.0, 10.0)
    ventana_neo: int = 5
    min_en_ventana: int = 4
    frac_previos: float = 0.25
    min_previos: int = 3
    # log-odds
    theta_min: float = 0.4
    a0: float = 500.0
    min_total_logodds: int = 20
    top_logodds: int = 15
    topicos_foco: tuple = ("T02", "T03", "T07")

    @classmethod
    def rapida(cls) -> "ConfigCrono":
        """Para el parquet FAST (6.000 docs): menos de todo."""
        return cls(
            fast=True, min_tokens_grupo=2_000, min_total_export=3, b_boot=200,
            n_terminos_demo=800, min_ocurr_año=5, b_binseg=60, w2v_epocas=5,
            w2v_min_count=3, w2v_semillas=(0, 1, 2), umbrales_vocab=(5, 10, 20, 40),
            min_semillas_vecino=2, min_pmw_reemplazo=20.0, min_total_logodds=5,
        )

    def a_dict(self) -> dict:
        return asdict(self)


_SIN_HASH = {"w2v_workers"}  # solo velocidad


def hash_congelado(cfg: ConfigCrono, *declarados) -> str:
    d = {k: v for k, v in cfg.a_dict().items() if k not in _SIN_HASH}
    obj = json.dumps([d, *declarados], sort_keys=True, ensure_ascii=False, default=list)
    return hashlib.sha256(obj.encode()).hexdigest()[:16]


def congelar(ruta: Path, h: str, contenido: dict) -> str:
    """Primera corrida: guarda el hash. Siguientes: exige que no cambie."""
    if ruta.exists():
        prev = json.loads(ruta.read_text(encoding="utf-8"))
        if prev["hash"] != h:
            raise AssertionError(
                f"Lo congelado cambió ({prev['hash']} → {h}). Los conjuntos, semillas y "
                "reglas no se editan después de ver resultados: lo nuevo va en EXPLORATORIO."
            )
        return f"verificado: hash {h} igual al congelado el {prev['fecha']}"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    agenda.a_json({"hash": h, "fecha": time.strftime("%Y-%m-%d %H:%M"), **contenido}, ruta)
    return f"congelado por primera vez: hash {h}"


@contextmanager
def cronometro(tiempos: dict, nombre: str):
    t0 = time.time()
    yield
    tiempos[nombre] = round(time.time() - t0, 1)
    print(f"[{nombre}] {tiempos[nombre]:.0f} s")


def periodo_de(años, periodos) -> np.ndarray:
    out = np.full(len(años), None, dtype=object)
    años = np.asarray(años)
    for nom, a, b in periodos:
        out[(años >= a) & (años <= b)] = nom
    return out


# ---------------------------------------------------------------------------
# Conteo y rareza
# ---------------------------------------------------------------------------


def _identidad(x):
    return x


def matriz_conteos(seqs, min_df: int = 1):
    """Documento × término (int32, CSR) con el vocabulario como array.
    El denominador NO sale de aquí (min_df recorta términos): usa longitudes."""
    from sklearn.feature_extraction.text import CountVectorizer

    v = CountVectorizer(analyzer=_identidad, lowercase=False, min_df=min_df, dtype=np.int32)
    X = v.fit_transform(seqs).tocsr()
    return X, np.asarray(v.get_feature_names_out())


def indicadora(etiquetas) -> tuple[sp.csr_matrix, list]:
    """Grupos × documentos (1 si el doc pertenece). Etiquetas None quedan fuera."""
    et = pd.Series(list(etiquetas))
    ok = et.notna().values
    codes, uniq = pd.factorize(et[ok], sort=True)
    cols = np.where(ok)[0]
    G = sp.csr_matrix((np.ones(len(cols)), (codes, cols)), shape=(len(uniq), len(et)))
    return G, list(uniq)


def piso_zipf(n_tokens) -> np.ndarray:
    """Zipf de UNA aparición en un grupo de n_tokens: el valor más bajo medible."""
    n = np.asarray(n_tokens, float)
    with np.errstate(divide="ignore"):
        return np.log10(1e6 / n) + 3


def zipf(pmw):
    pmw = np.asarray(pmw, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.log10(pmw) + 3
    return np.where(pmw > 0, z, np.nan)


def banda(z, cortes=(3.0, 4.0)) -> np.ndarray:
    z = np.asarray(z, float)
    return np.select([~np.isfinite(z), z < cortes[0], z < cortes[1]], ["sin_dato", "raro", "medio"], "frecuente")


def pmw_zipf(X, vocab, longitudes, etiquetas, cfg: ConfigCrono, terminos_ok=None) -> pd.DataFrame:
    """pmw, Zipf (log10 pmw + 3) y Zipf de Laplace (van Heuven: (c+1)/(N+V)) por grupo.

    Formato largo, solo celdas con conteo > 0 de los términos en `terminos_ok`.
    Grupos con menos de cfg.min_tokens_grupo tokens: pmw NaN y marca `grupo_chico`.
    """
    G, grupos = indicadora(etiquetas)
    N = G @ np.asarray(longitudes, float)
    C = (G @ X).tocoo()
    V = X.shape[1]
    t = pd.DataFrame({"g": C.row, "j": C.col, "n": C.data.astype(np.int64)})
    if terminos_ok is not None:
        t = t[np.isin(t["j"], np.asarray(terminos_ok))]
    t["termino"] = vocab[t["j"].values]
    t["N_grupo"] = N[t["g"].values]
    t["pmw"] = t["n"] / t["N_grupo"] * 1e6
    t["zipf"] = zipf(t["pmw"])
    t["zipf_laplace"] = np.log10((t["n"] + 1) / (t["N_grupo"] + V) * 1e9)
    t["grupo_chico"] = t["N_grupo"] < cfg.min_tokens_grupo
    t.loc[t["grupo_chico"], ["pmw", "zipf"]] = np.nan
    t["banda"] = banda(t["zipf"], cfg.bandas)
    gr = pd.DataFrame(grupos, columns=["fuente", "año"]) if isinstance(grupos[0], tuple) else pd.DataFrame({"grupo": grupos})
    t = t.join(gr, on="g").drop(columns=["g", "j"])
    return t


def ic_poisson(n, N, alfa: float = 0.05):
    """IC exacto de Garwood para la tasa por millón."""
    n = np.asarray(n, float)
    lo = np.where(n > 0, stats.chi2.ppf(alfa / 2, 2 * n) / 2, 0.0)
    hi = stats.chi2.ppf(1 - alfa / 2, 2 * (n + 1)) / 2
    return lo / N * 1e6, hi / N * 1e6


def bootstrap_tasa(Xg, Lg, B: int, seed: int, alfa: float = 0.05):
    """Bootstrap de Poisson por documento (pesos Poisson(1)): tasa por millón
    = Σ w·conteo / Σ w·longitud. Xg: docs × términos (sparse o denso); Lg: longitudes.
    Devuelve (lo, hi, réplicas B × términos)."""
    rg = np.random.default_rng(seed)
    n = Xg.shape[0]
    if n == 0:
        k = Xg.shape[1]
        return np.full(k, np.nan), np.full(k, np.nan), np.full((0, k), np.nan)
    W = rg.poisson(1.0, size=(B, n)).astype(np.float32)
    num = np.asarray((Xg.T @ W.T).T) if sp.issparse(Xg) else W @ np.asarray(Xg, np.float32)
    den = W @ np.asarray(Lg, np.float32)
    R = num / den[:, None] * 1e6
    lo, hi = np.quantile(R, [alfa / 2, 1 - alfa / 2], axis=0)
    return lo, hi, R


def ancho_relativo(n, N, lo_b, hi_b) -> pd.DataFrame:
    """Semiancho relativo (hi − lo)/2/tasa con Poisson y con bootstrap por documento."""
    tasa = np.asarray(n, float) / N * 1e6
    lo_p, hi_p = ic_poisson(n, N)
    return pd.DataFrame({
        "n": n, "pmw": tasa,
        "hw_rel_poisson": (hi_p - lo_p) / 2 / tasa,
        "hw_rel_boot": (np.asarray(hi_b) - np.asarray(lo_b)) / 2 / tasa,
    }).assign(cociente=lambda d: d["hw_rel_boot"] / d["hw_rel_poisson"])


def conteo_minimo(tab: pd.DataFrame, col="hw_rel_boot", umbral=0.5,
                  bins=(1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 10**9)):
    """Regla declarada: el menor conteo (bin) cuya mediana de semiancho relativo ≤ umbral."""
    t = tab.assign(bin=pd.cut(tab["n"], bins, right=False))
    g = t.groupby("bin", observed=True)[[col, "hw_rel_poisson"]].median()
    g["n_terminos"] = t.groupby("bin", observed=True).size()
    ok = g[g[col] <= umbral]
    return (int(ok.index[0].left) if len(ok) else np.nan), g


def dispersion_dp(X, longitudes) -> pd.DataFrame:
    """DP de Gries (2008) y DP_norm (Lijffijt y Gries 2012) con los documentos como partes.

    DP = ½ Σ_i |v_i − s_i|; v_i = conteo en el doc i / conteo total; s_i = tamaño
    relativo del doc. Los docs sin la palabra aportan s_i: se suman de una vez.
    0 = repartida como el corpus; → 1 = concentrada en pocos docs.
    """
    L = np.asarray(longitudes, float)
    s = L / L.sum()
    Xc = sp.csc_matrix(X, dtype=np.float64)
    tot = np.asarray(Xc.sum(0)).ravel()
    dp = np.full(Xc.shape[1], np.nan)
    df = np.diff(Xc.indptr)
    for j in range(Xc.shape[1]):
        a, b = Xc.indptr[j], Xc.indptr[j + 1]
        if tot[j] == 0:
            continue
        rows = Xc.indices[a:b]
        v = Xc.data[a:b] / tot[j]
        dp[j] = 0.5 * (np.abs(v - s[rows]).sum() + (1 - s[rows].sum()))
    return pd.DataFrame({"n": tot.astype(np.int64), "df": df, "dp": dp, "dp_norm": dp / (1 - s.min())})


def frac_en_entidades(entidades, totales: dict) -> pd.Series:
    """Para cada forma: apariciones dentro de textos de `entidades` / apariciones totales."""
    c = Counter()
    for ents in entidades:
        if ents is None:
            continue
        for e in ents:
            c.update(w for w in re.findall(r"\w+", e["texto"].lower()))
    s = pd.Series({w: min(1.0, k / totales[w]) for w, k in c.items() if totales.get(w, 0) > 0})
    return s


def pos_dominante(lemas_col, pos_col) -> dict:
    c = defaultdict(Counter)
    for L, P in zip(lemas_col, pos_col):
        for l, p in zip(L, P):
            c[l][p] += 1
    return {l: k.most_common(1)[0][0] for l, k in c.items()}


def zipf_referencia(terminos, lang: str = "es") -> pd.Series:
    """Zipf del español general (wordfreq). 0 = ausente de la referencia."""
    from wordfreq import zipf_frequency

    return pd.Series({t: zipf_frequency(t, lang) for t in terminos}, dtype=float)


# ---------------------------------------------------------------------------
# Variantes
# ---------------------------------------------------------------------------


def textos_unidos(tokens_forma) -> list[str]:
    """' w1 w2 … ' por documento: las formas de varias palabras se buscan con espacios."""
    return [" " + " ".join(t) + " " for t in tokens_forma]


def compilar(formas_por_variante: dict[str, list[str]]):
    """Un regex por conjunto: más larga primero y sin solapes ('santafé de bogotá'
    gana sobre 'bogotá'). Devuelve (regex, forma→variante)."""
    mapa = {}
    for var, formas in formas_por_variante.items():
        for f in formas:
            f = f.lower().strip()
            if f in mapa and mapa[f] != var:
                raise ValueError(f"forma '{f}' en dos variantes: {mapa[f]} y {var}")
            mapa[f] = var
    alts = sorted(mapa, key=lambda f: (-len(f.split()), -len(f)))
    rx = re.compile(r"(?<= )(" + "|".join(re.escape(a) for a in alts) + r")(?= )")
    return rx, mapa


def contar_patrones(textos: list[str], formas_por_variante: dict[str, list[str]]) -> pd.DataFrame:
    """Conteos doc × variante, solo filas con algún acierto (índice = posición del doc)."""
    rx, mapa = compilar(formas_por_variante)
    cols = list(formas_por_variante)
    ix = {v: k for k, v in enumerate(cols)}
    filas, pos = [], []
    for i, s in enumerate(textos):
        m = rx.findall(s)
        if m:
            r = np.zeros(len(cols), np.int32)
            for f in m:
                r[ix[mapa[f]]] += 1
            filas.append(r)
            pos.append(i)
    if not filas:
        return pd.DataFrame(columns=cols, dtype=np.int32)
    return pd.DataFrame(np.vstack(filas), index=pos, columns=cols)


def contar_independientes(textos, patrones: dict[str, list[str]]) -> pd.DataFrame:
    """Cada término con su propio regex (se pueden solapar: 'paz' y 'proceso de paz')."""
    partes = [contar_patrones(textos, {k: v}) for k, v in patrones.items()]
    return pd.concat(partes, axis=1).fillna(0).astype(np.int32).sort_index()


def cuotas(C: pd.DataFrame, etiquetas, B: int, seed: int, min_ocurr: int, alfa: float = 0.05):
    """Cuota de cada variante sobre el conjunto (razón de sumas) por grupo, con IC.

    Bootstrap por documento sobre TODOS los docs del grupo, también los que no
    nombran el conjunto: el número de docs con acierto en la réplica sale de una
    Binomial(n, h/n) y, dado ese número, se remuestrean los h docs con acierto
    (equivale al remuestreo multinomial de los n docs, sin construir la matriz B×n).
    Grupos con menos de `min_ocurr` apariciones del conjunto: NaN explícito.
    Devuelve (tabla larga, réplicas {grupo: B × variantes}).
    """
    rg = np.random.default_rng(seed)
    variantes = list(C.columns)
    et = pd.Series(list(etiquetas))
    n_grupo = et.dropna().value_counts()
    M = C.values.astype(np.float64)
    por_grupo = defaultdict(list)
    for k, x in enumerate(et.iloc[C.index.values].values if len(C) else []):
        if x is not None and x == x:
            por_grupo[x].append(k)
    filas, reps = [], {}
    for g, n in n_grupo.items():
        Mh = M[por_grupo.get(g, [])]
        h, tot = len(Mh), float(Mh.sum())
        base = {"grupo": g, "n_docs": int(n), "n_docs_con": int(h), "n_conjunto": int(tot)}
        if tot < min_ocurr:
            reps[g] = None
            for v in variantes:
                filas.append({**base, "variante": v, "n": int(Mh[:, variantes.index(v)].sum()) if h else 0,
                              "cuota": np.nan, "lo": np.nan, "hi": np.nan, "suficiente": False})
            continue
        K = rg.binomial(int(n), h / n, size=B)
        W = rg.multinomial(K, np.full(h, 1.0 / h))
        S = W @ Mh
        with np.errstate(invalid="ignore", divide="ignore"):
            Q = S / S.sum(1, keepdims=True)
        lo, hi = np.nanquantile(Q, [alfa / 2, 1 - alfa / 2], axis=0)
        reps[g] = Q
        p = Mh.sum(0) / tot
        for k, v in enumerate(variantes):
            filas.append({**base, "variante": v, "n": int(Mh[:, k].sum()), "cuota": p[k],
                          "lo": lo[k], "hi": hi[k], "suficiente": True})
    return pd.DataFrame(filas), reps


def cuota_estandarizada(C: pd.DataFrame, etiquetas, topico, min_ocurr: int) -> pd.DataFrame:
    """Control de composición temática: cada doc pesa π_pooled(k)/π_grupo(k), con k su
    tópico dominante, de modo que todos los grupos tengan la mezcla de tópicos del
    conjunto de grupos. Sin bootstrap (es un control de punto)."""
    et = pd.Series(list(etiquetas))
    tp = pd.Series(list(topico))
    ok = et.notna() & tp.notna()
    pooled = tp[ok].value_counts(normalize=True)
    w = np.zeros(len(et))
    for g, sub in tp[ok].groupby(et[ok]):
        pg = sub.value_counts(normalize=True)
        w[sub.index.values] = (pooled.reindex(sub.values).values / pg.reindex(sub.values).values)
    filas = []
    pos = C.index.values
    et_hit = et.values[pos] if len(pos) else np.array([], dtype=object)
    ok_hit = ok.values[pos] if len(pos) else np.array([], dtype=bool)
    Cv = C.values.astype(float)
    for g in et[ok].unique():
        sel = ok_hit & (et_hit == g)
        S = (Cv[sel] * w[pos[sel]][:, None]).sum(0) if sel.any() else np.zeros(C.shape[1])
        n = int(Cv[sel].sum())
        for k, v in enumerate(C.columns):
            filas.append({"grupo": g, "variante": v, "n_conjunto": n,
                          "cuota": S[k] / S.sum() if n >= min_ocurr and S.sum() > 0 else np.nan})
    return pd.DataFrame(filas)


def año_cruce(años, dif, k: int = 3) -> tuple[float, str]:
    """Primer año desde el cual la variante nueva supera a la vieja (dif > 0) en k años
    válidos y consecutivos, habiendo estado debajo (dif < 0) en algún año anterior."""
    años = np.asarray(años)
    dif = np.asarray(dif, float)
    v = np.isfinite(dif)
    if v.sum() < k + 1:
        return np.nan, "datos_insuficientes"
    for i in range(len(años) - k + 1):
        seg, ys = dif[i:i + k], años[i:i + k]
        if np.all(np.isfinite(seg)) and np.all(seg > 0) and ys[-1] - ys[0] == k - 1:
            antes = dif[:i][np.isfinite(dif[:i])]
            if len(antes) == 0:
                return np.nan, "nueva_domina_desde_inicio"
            if (antes < 0).any():
                return float(años[i]), "cruce"
            return np.nan, "nueva_domina_desde_inicio"
    return np.nan, "sin_cruce"


def cruces(tab: pd.DataFrame, reps: dict, par: tuple[str, str], años, variantes, k: int,
           alfa: float = 0.05, año_formato: int = 2006) -> dict:
    """Año de cruce observado + IC percentil sobre las réplicas bootstrap.
    tab: tabla larga de UNA serie (fuente) con columnas año, variante, cuota."""
    vieja, nueva = par
    w = tab.pivot_table(index="año", columns="variante", values="cuota", dropna=False).reindex(años)
    obs, estado = año_cruce(años, (w[nueva] - w[vieja]).values, k)
    iv, inu = variantes.index(vieja), variantes.index(nueva)
    B = next((r.shape[0] for r in reps.values() if r is not None), 0)
    rc = []
    for b in range(B):
        d = np.array([reps[y][b, inu] - reps[y][b, iv] if reps.get(y) is not None else np.nan for y in años])
        rc.append(año_cruce(años, d, k)[0])
    rc = np.array(rc, float)
    fin = rc[np.isfinite(rc)]
    lo, hi = (np.quantile(fin, [alfa / 2, 1 - alfa / 2]) if len(fin) else (np.nan, np.nan))
    if estado != "cruce":
        lo = hi = np.nan  # sin cruce observado no se reporta IC (salía de una fracción ínfima de réplicas)
    return {"vieja": vieja, "nueva": nueva, "cruce": obs, "estado": estado, "lo": lo, "hi": hi,
            "frac_reps_con_cruce": float(len(fin) / B) if B else np.nan,
            "cerca_formato": bool(np.isfinite(obs) and abs(obs - año_formato) <= 1)}


def cambios_cuota(tab: pd.DataFrame, cfg: ConfigCrono) -> dict:
    """binseg de agenda.py sobre √cuota (≈ Hellinger), peso min(n_conjunto, cap)."""
    w = tab.pivot_table(index="año", columns="variante", values="cuota")
    n = tab.groupby("año")["n_conjunto"].first().reindex(w.index)
    w = w.dropna()
    if len(w) < 2 * cfg.min_seg + 1:
        return {"cambios": [], "fuerza": [], "años_validos": len(w)}
    M = np.sqrt(np.clip(w.values, 0, None))
    peso = np.minimum(n.reindex(w.index).values, cfg.cap_peso).astype(float)
    cps = agenda.binseg(M, peso, cfg.min_seg, cfg.b_binseg, cfg.q_binseg, seed=cfg.seed)
    return {"cambios": [int(w.index[s]) for s, _ in cps], "fuerza": [round(float(f), 2) for _, f in cps],
            "años_validos": len(w)}


def contextos(textos, ids, posiciones, rx, mapa: dict, variante: str, n: int, ventana: int,
              seed: int) -> list[str]:
    """n KWIC de la `variante` en los docs de `posiciones` (al azar, uno por doc).
    Usa el regex del conjunto completo, así "bogotá" no se toma de "santafé de bogotá".
    [] explícito si no hay apariciones."""
    rg = np.random.default_rng(seed)
    pos = list(rg.permutation(np.asarray(posiciones, dtype=int)))
    out = []
    for i in pos:
        s = textos[i]
        m = next((m for m in rx.finditer(s) if mapa.get(m.group(0)) == variante), None)
        if m is None:
            continue
        izq = s[: m.start()].split()[-ventana:]
        der = s[m.end():].split()[:ventana]
        out.append(f"{' '.join(izq)} [{m.group(0).upper()}] {' '.join(der)} (id {ids[i]})")
        if len(out) == n:
            break
    return out


# ---------------------------------------------------------------------------
# Cambio semántico
# ---------------------------------------------------------------------------


def secuencias_lema(lemas, pos, cortes, bigramas=frozenset(), renombres=None, mapa=None) -> list[list[str]]:
    """lemas_contenido (N, A, V, P) partidos en `cortes`: cada tramo es una "oración"
    para word2vec. Une los bigramas de la lista blanca, renombra unidos/P y aplica
    el mapa de plurales del 08 solo a N/A."""
    renombres = renombres or {}
    mapa = mapa or {}
    cs = set(int(c) for c in cortes)
    segs, cur, i, n = [], [], 0, len(lemas)
    while i < n:
        if i in cs and cur:
            segs.append(cur)
            cur = []
        if bigramas and i + 1 < n and (i + 1) not in cs:
            b = lemas[i] + "_" + lemas[i + 1]
            if b in bigramas:
                cur.append(agenda._mapear(b, mapa))
                i += 2
                continue
        t, c = lemas[i], pos[i]
        if c == "P" and t in renombres:
            t = renombres[t]
        elif c in ("N", "A"):
            t = mapa.get(t, t)
        cur.append(t)
        i += 1
    if cur:
        segs.append(cur)
    return segs


def corpus_periodos(segs_doc: list, periodo, mascara, seed: int) -> tuple[dict, dict, int]:
    """Agrupa los tramos por período y submuestrea DOCUMENTOS hasta igualar el número
    de tokens del período más chico (el tamaño del corpus afecta los vecinos)."""
    rg = np.random.default_rng(seed)
    per = np.asarray(periodo, dtype=object)
    m = np.asarray(mascara, bool)
    ntok = np.array([sum(len(s) for s in d) for d in segs_doc])
    nombres = [p for p in pd.unique(per[m]) if p is not None]
    totales = {p: int(ntok[m & (per == p)].sum()) for p in nombres}
    objetivo = min(totales.values())
    out, conteos = {}, {}
    for p in sorted(nombres):
        idx = rg.permutation(np.where(m & (per == p))[0])
        acum = np.cumsum(ntok[idx])
        idx = idx[: int(np.searchsorted(acum, objetivo)) + 1]
        out[p] = [s for i in idx for s in segs_doc[i]]
        conteos[p] = Counter(t for s in out[p] for t in s)
    return out, conteos, objetivo


def muestra_bootstrap(segs_doc: list, idx, objetivo: int, rep: int, seed: int, k_periodo: int) -> list:
    """Réplica bootstrap de DOCUMENTOS de un período (con reemplazo) hasta `objetivo`
    tokens: la variación entre réplicas incluye el muestreo del corpus, no solo la
    inicialización (Antoniak y Mimno 2018)."""
    rg = np.random.default_rng([seed, k_periodo, rep])
    idx = np.asarray(idx)
    ntok = np.array([sum(len(s) for s in segs_doc[i]) for i in idx])
    if len(idx) == 0 or ntok.sum() == 0:
        return []
    out, acum = [], 0
    while acum < objetivo:
        j = rg.integers(len(idx))
        out.extend(segs_doc[idx[j]])
        acum += ntok[j]
    return out


def entrenar_bootstrap(segs_doc, idx, objetivo, rep, k_periodo, cfg: ConfigCrono, ruta: Path | None = None):
    """Carga del caché o entrena una réplica bootstrap (semilla de word2vec = rep)."""
    from gensim.models import KeyedVectors

    if ruta is not None and ruta.exists():
        return KeyedVectors.load(str(ruta))
    segs = muestra_bootstrap(segs_doc, idx, objetivo, rep, cfg.seed, k_periodo)
    return entrenar_w2v(segs, rep, cfg, ruta)


def entrenar_w2v(segs, seed: int, cfg: ConfigCrono, ruta: Path | None = None):
    """SGNS. Con workers > 1 gensim no es determinista: esa variación también cuenta
    como ruido entre semillas, que es justo lo que se mide."""
    from gensim.models import KeyedVectors, Word2Vec

    if ruta is not None and ruta.exists():
        return KeyedVectors.load(str(ruta))
    m = Word2Vec(sentences=segs, vector_size=cfg.w2v_dim, window=cfg.w2v_ventana,
                 min_count=cfg.w2v_min_count, sg=1, negative=cfg.w2v_neg, sample=cfg.w2v_sample,
                 epochs=cfg.w2v_epocas, workers=cfg.w2v_workers, seed=seed)
    kv = m.wv
    if ruta is not None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        kv.save(str(ruta))
    return kv


def _norm(E):
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def prep_emb(kv, vocab, mu=None):
    """Normaliza, centra y vuelve a normalizar (Hamilton et al. 2016). Devuelve (E, mu)."""
    E = _norm(kv.vectors[[kv.key_to_index[w] for w in vocab]].astype(np.float64))
    if mu is None:
        mu = E.mean(0)
    return _norm(E - mu), mu


def procrustes(A, B):
    """R ortogonal que minimiza ||A R − B||_F."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt


def _dist_alineada(A, B):
    R = procrustes(A, B)
    return 1 - np.sum((A @ R) * B, axis=1)


def vocab_comun(modelos: dict, conteos: dict, umbral: int) -> list[str]:
    """Palabras con conteo ≥ umbral en TODOS los períodos y presentes en todos los modelos."""
    ps = list(modelos)
    comunes = set.intersection(*[{w for w, c in conteos[p].items() if c >= umbral} for p in ps])
    for p in ps:
        for kv in modelos[p]:
            comunes &= set(kv.key_to_index)
    return sorted(comunes)


def desplazamiento(modelos: dict, vocab: list[str], q: float = 0.05) -> pd.DataFrame:
    """Desplazamiento por transición entre períodos consecutivos, con ruido entre semillas.

    d_entre: coseno tras Procrustes entre todos los pares (semilla de t, semilla de t+1).
    d_dentro: igual, pero entre semillas del MISMO período (t y t+1 juntos).
    efecto = media(d_entre) − media(d_dentro); `cuenta` si p05(d_entre) > p95(d_dentro).
    """
    if len(vocab) < 50:
        return pd.DataFrame(columns=["palabra", "transicion", "d_entre", "d_entre_p05", "d_dentro", "d_dentro_p95",
                                     "efecto", "razon", "cuenta"])
    E = {p: [prep_emb(kv, vocab)[0] for kv in kvs] for p, kvs in modelos.items()}
    dentro = {}
    for p, Es in E.items():
        dentro[p] = np.array([_dist_alineada(Es[i], Es[j]) for i in range(len(Es)) for j in range(i + 1, len(Es))])
    ps = sorted(E)
    filas = []
    for a, b in zip(ps[:-1], ps[1:]):
        ent = np.array([_dist_alineada(Ea, Eb) for Ea in E[a] for Eb in E[b]])
        dd = np.vstack([dentro[a], dentro[b]])
        filas.append(pd.DataFrame({
            "palabra": vocab, "transicion": f"{a}→{b}",
            "d_entre": ent.mean(0), "d_entre_p05": np.quantile(ent, q, axis=0),
            "d_dentro": dd.mean(0), "d_dentro_p95": np.quantile(dd, 1 - q, axis=0),
        }))
    t = pd.concat(filas, ignore_index=True)
    t["efecto"] = t["d_entre"] - t["d_dentro"]
    t["razon"] = t["d_entre"] / (t["d_dentro"] + 1e-12)
    t["cuenta"] = t["d_entre_p05"] > t["d_dentro_p95"]
    return t


def elegir_umbral_vocab(tab: pd.DataFrame, fmin: pd.Series, umbrales, frac: float) -> tuple[float, pd.DataFrame]:
    """Regla declarada: el umbral más bajo cuyo bin de frecuencia [u_i, u_{i+1}) tiene
    mediana de d_dentro < frac · mediana global de d_entre. fmin: conteo mínimo de la
    palabra en los períodos."""
    t = tab.assign(fmin=tab["palabra"].map(fmin))
    ref = t["d_entre"].median()
    cortes = list(umbrales) + [np.inf]
    filas = []
    for lo, hi in zip(cortes[:-1], cortes[1:]):
        s = t[(t["fmin"] >= lo) & (t["fmin"] < hi)]
        filas.append({"umbral": lo, "n_palabras": s["palabra"].nunique(),
                      "mediana_d_dentro": s["d_dentro"].median() if len(s) else np.nan,
                      "mediana_d_entre": s["d_entre"].median() if len(s) else np.nan,
                      "limite": frac * ref})
    r = pd.DataFrame(filas)
    ok = r[r["mediana_d_dentro"] < r["limite"]]
    return (float(ok["umbral"].iloc[0]) if len(ok) else np.nan), r


def control_frecuencia(tab: pd.DataFrame, frec: dict) -> tuple[pd.DataFrame, dict]:
    """¿Explica la frecuencia el ranking? Spearman con log-frecuencia y con el cambio de
    frecuencia; residuo de un MCO efecto ~ log f + |Δ log f| como ranking alternativo."""
    filas, resumen = [], {}
    if tab.empty:
        return tab.assign(log_f=np.nan, dlog_f=np.nan, efecto_resid=np.nan, decil_f=np.nan), resumen
    for tr, s in tab.groupby("transicion"):
        a, b = tr.split("→")
        fa = s["palabra"].map(frec[a]).astype(float).values
        fb = s["palabra"].map(frec[b]).astype(float).values
        s = s.assign(log_f=np.log10((fa + fb) / 2), dlog_f=np.abs(np.log2(fb / fa)))
        Xm = np.column_stack([np.ones(len(s)), s["log_f"], s["dlog_f"]])
        beta, *_ = np.linalg.lstsq(Xm, s["efecto"].values, rcond=None)
        s["efecto_resid"] = s["efecto"].values - Xm @ beta
        s["decil_f"] = pd.qcut(s["log_f"], 10, labels=False, duplicates="drop")
        resumen[tr] = {
            "rho_efecto_logf": float(stats.spearmanr(s["efecto"], s["log_f"]).statistic),
            "rho_efecto_dlogf": float(stats.spearmanr(s["efecto"], s["dlog_f"]).statistic),
            "rho_dentro_logf": float(stats.spearmanr(s["d_dentro"], s["log_f"]).statistic),
            "rho_rank_efecto_vs_resid": float(stats.spearmanr(s["efecto"], s["efecto_resid"]).statistic),
            "cuentan_por_decil": s[s["cuenta"]].groupby("decil_f").size().reindex(range(10), fill_value=0).tolist(),
            "r2_frecuencia": float(1 - s["efecto_resid"].var() / s["efecto"].var()),
        }
        filas.append(s)
    return pd.concat(filas, ignore_index=True), resumen


def vecindario_local(Ea, Eb, k: int = 25, chunk: int = 2000) -> np.ndarray:
    """Hamilton et al. 2016b: para cada palabra, unión de sus k vecinos en a y en b;
    vector de similitudes a esa unión en cada espacio; distancia coseno entre vectores.
    No necesita alinear. Ea y Eb: filas normalizadas sobre el MISMO vocabulario."""
    V = Ea.shape[0]
    out = np.empty(V)
    for i0 in range(0, V, chunk):
        sl = slice(i0, min(V, i0 + chunk))
        Sa, Sb = Ea[sl] @ Ea.T, Eb[sl] @ Eb.T
        r = np.arange(Sa.shape[0])
        Sa[r, r + i0] = -np.inf
        Sb[r, r + i0] = -np.inf
        mask = np.zeros(Sa.shape, bool)
        mask[r[:, None], np.argpartition(-Sa, k, axis=1)[:, :k]] = True
        mask[r[:, None], np.argpartition(-Sb, k, axis=1)[:, :k]] = True
        Sa = np.where(mask, Sa, 0.0)
        Sb = np.where(mask, Sb, 0.0)
        num = (Sa * Sb).sum(1)
        out[sl] = 1 - num / (np.linalg.norm(Sa, axis=1) * np.linalg.norm(Sb, axis=1) + 1e-12)
    return out


def desplazamiento_local(modelos: dict, vocab: list[str], k: int) -> pd.DataFrame:
    """Medida local con ruido: entre = misma semilla i en t y t+1; dentro = semillas
    i e i+1 del mismo período. efecto_local = media(entre) − media(dentro)."""
    if len(vocab) <= k + 1:
        return pd.DataFrame(columns=["palabra", "transicion", "d_local_entre", "d_local_dentro", "efecto_local"])
    E = {p: [_norm(kv.vectors[[kv.key_to_index[w] for w in vocab]].astype(np.float64)) for kv in kvs]
         for p, kvs in modelos.items()}
    ps = sorted(E)
    dentro = {p: np.mean([vecindario_local(Es[i], Es[i + 1], k) for i in range(len(Es) - 1)], axis=0)
              for p, Es in E.items()}
    filas = []
    for a, b in zip(ps[:-1], ps[1:]):
        n = min(len(E[a]), len(E[b]))
        ent = np.mean([vecindario_local(E[a][i], E[b][i], k) for i in range(n)], axis=0)
        dd = (dentro[a] + dentro[b]) / 2
        filas.append(pd.DataFrame({"palabra": vocab, "transicion": f"{a}→{b}", "d_local_entre": ent,
                                   "d_local_dentro": dd, "efecto_local": ent - dd}))
    return pd.concat(filas, ignore_index=True)


def vecinos_estables(kvs, w: str, k: int, min_semillas: int, excluir=frozenset()) -> list[tuple]:
    """Vecinos que aparecen en el top-k de ≥ min_semillas modelos. [] explícito si la
    palabra no está en todos los modelos del período."""
    if not kvs or any(w not in kv.key_to_index for kv in kvs):
        return []
    cnt, sims = Counter(), defaultdict(list)
    for kv in kvs:
        for v, s in kv.most_similar(w, topn=k + len(excluir)):
            if v in excluir:
                continue
            cnt[v] += 1
            sims[v].append(s)
    r = [(v, c, float(np.mean(sims[v]))) for v, c in cnt.items() if c >= min_semillas]
    return sorted(r, key=lambda x: (-x[1], -x[2]))[:k]


def verificar_no_repetidas(listas: dict) -> None:
    """Bug del 05: dos términos distintos devolvían la MISMA lista de relleno.

    La firma del bug es una lista idéntica en vecinos y en similitudes. Dos
    palabras distintas pueden compartir pocos vecinos (con corpus chicos quedan
    listas de 1–2 vecinos estables), pero no las mismas similitudes: por eso la
    clave incluye la similitud redondeada."""
    vistos = {}
    for t, l in listas.items():
        key = tuple((x[0], round(float(x[-1]), 4)) if isinstance(x, tuple) else x for x in l)
        if key and key in vistos:
            raise AssertionError(f"'{t}' y '{vistos[key]}' devuelven la misma lista: {key[:5]}")
        vistos[key] = t


def reemplazos_candidatos(kv_a: list, kv_b: list, vocab_ref: list[str], pmw_a: dict, pmw_b: dict,
                          cfg: ConfigCrono, excluir=frozenset(), top: int = 3,
                          min_pmw: float | None = None) -> pd.DataFrame:
    """EXPLORATORIO. Palabras que caen (log2 pmw_b/pmw_a ≤ log2_caida) → su vecino más
    cercano en el espacio de b, tras alinear a→b con Procrustes sobre `vocab_ref`,
    entre las palabras que suben (≥ log2_subida). Solo candidatos que salen en el
    top-`top` de ≥ min_semillas_vecino semillas."""
    eps = 0.5
    mp = cfg.min_pmw_reemplazo if min_pmw is None else min_pmw
    caen = [w for w, f in pmw_a.items() if f >= mp and w not in excluir
            and np.log2((pmw_b.get(w, 0) + eps) / (f + eps)) <= cfg.log2_caida]
    suben = [w for w, f in pmw_b.items() if f >= mp and w not in excluir
             and np.log2((f + eps) / (pmw_a.get(w, 0) + eps)) >= cfg.log2_subida]
    n = min(len(kv_a), len(kv_b))
    caen = [w for w in caen if all(w in kv.key_to_index for kv in kv_a[:n])]
    suben = [w for w in suben if all(w in kv.key_to_index for kv in kv_b[:n])]
    if not caen or not suben:
        return pd.DataFrame()
    cnt, sims = Counter(), defaultdict(list)
    for i in range(n):
        A, mu_a = prep_emb(kv_a[i], vocab_ref)
        B, mu_b = prep_emb(kv_b[i], vocab_ref)
        R = procrustes(A, B)
        Xc = prep_emb(kv_a[i], caen, mu_a)[0] @ R
        Cb = prep_emb(kv_b[i], suben, mu_b)[0]
        S = Xc @ Cb.T
        for r, w in enumerate(caen):
            for j in np.argsort(-S[r])[:top]:
                if suben[j] != w:
                    cnt[(w, suben[j])] += 1
                    sims[(w, suben[j])].append(S[r, j])
    filas = [{"cae": w, "candidato": v, "n_semillas": c, "sim_media": float(np.mean(sims[(w, v)])),
              "pmw_cae_antes": pmw_a.get(w, 0), "pmw_cae_despues": pmw_b.get(w, 0),
              "pmw_cand_antes": pmw_a.get(v, 0), "pmw_cand_despues": pmw_b.get(v, 0)}
             for (w, v), c in cnt.items() if c >= cfg.min_semillas_vecino]
    if not filas:
        return pd.DataFrame()
    return pd.DataFrame(filas).sort_values(["n_semillas", "sim_media"], ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Diagnóstico de codificación del archivo (exploratorio)
# ---------------------------------------------------------------------------

_SIN = str.maketrans("áéíóúü", "aeiouu")


def pares_diacriticos(con_tilde: list[str], con_dieresis: list[str]) -> dict[str, dict[str, list[str]]]:
    """Para palabras cuya forma sin tilde NO es otra palabra válida (también → tambien),
    variantes 'correcta' / 'sin_tilde'. Para ü: 'ü' / 'u' / 'eliminada' (vergenza)."""
    out = {}
    for w in con_tilde:
        out[w] = {"correcta": [w], "sin_tilde": [w.translate(_SIN)]}
    for w in con_dieresis:
        out[w] = {"ü": [w], "u": [w.replace("ü", "u")], "eliminada": [w.replace("ü", "")]}
    return out


def cuotas_formas(X, ixf: dict, etiquetas, conjuntos: dict) -> pd.DataFrame:
    """Cuota de cada variante (formas sueltas) por grupo, desde la matriz doc × forma."""
    G, grupos = indicadora(etiquetas)
    filas = []
    for c, vs in conjuntos.items():
        cols = {v: [ixf[f] for f in fs if f in ixf] for v, fs in vs.items()}
        n = {v: (np.asarray((G @ X[:, js]).sum(1)).ravel() if js else np.zeros(len(grupos))) for v, js in cols.items()}
        tot = sum(n.values())
        for k, g in enumerate(grupos):
            for v in vs:
                filas.append({"grupo": g, "conjunto": c, "variante": v, "n": int(n[v][k]), "n_conjunto": int(tot[k]),
                              "cuota": n[v][k] / tot[k] if tot[k] > 0 else np.nan})
    return pd.DataFrame(filas)


def tasa_diacriticos(X, vocab, longitudes, etiquetas) -> pd.DataFrame:
    """% de tokens con vocal tildada y con ü por grupo (formas con min_df ≥ 2)."""
    G, grupos = indicadora(etiquetas)
    til = np.array([bool(re.search(r"[áéíóú]", w)) for w in vocab], float)
    die = np.array(["ü" in w for w in vocab], float)
    N = G @ np.asarray(longitudes, float)
    return pd.DataFrame({"grupo": grupos, "pct_tilde": (G @ (X @ til)) / N * 100,
                         "pct_dieresis_por_millon": (G @ (X @ die)) / N * 1e6})


# ---------------------------------------------------------------------------
# Neologismos y obsolescencia
# ---------------------------------------------------------------------------


def uso_sostenido(P: pd.DataFrame, u: float, cfg: ConfigCrono, modo: str = "primero") -> pd.DataFrame:
    """Primer (o último) año de uso sostenido.

    primero: año y con pmw ≥ u, ≥ min_en_ventana de los `ventana_neo` años siguientes
    ≥ u (si quedan menos años, la misma proporción con al menos 3), y mediana de los
    años previos < frac_previos·u con ≥ min_previos años previos (si no, censurado).
    ultimo: lo mismo con el tiempo invertido.
    """
    P = P.sort_index()
    años = P.index.values
    V = P.values
    if modo == "ultimo":
        años, V = años[::-1], V[::-1]
    prop = cfg.min_en_ventana / cfg.ventana_neo
    filas = []
    for j, t in enumerate(P.columns):
        x = V[:, j]
        for i in range(len(x)):
            if not (x[i] >= u):
                continue
            prev = x[:i][np.isfinite(x[:i])]
            if len(prev) < cfg.min_previos or np.median(prev) >= cfg.frac_previos * u:
                continue
            sig = x[i + 1:i + 1 + cfg.ventana_neo]
            sig = sig[np.isfinite(sig)]
            if len(sig) < 3 or (sig >= u).mean() < prop:
                continue
            filas.append({"termino": t, "año": int(años[i]), "pmw_antes": float(np.median(prev)),
                          "pmw_despues": float(np.median(sig)), "pmw_max": float(np.nanmax(x))})
            break
    return pd.DataFrame(filas, columns=["termino", "año", "pmw_antes", "pmw_despues", "pmw_max"])


def tabla_categorias(ruta: Path, nuevos: pd.DataFrame, clave=("termino", "tipo")) -> tuple[pd.DataFrame, str]:
    """CSV editable a mano (columna `categoria`): respeta lo ya escrito, agrega lo nuevo."""
    nuevos = nuevos.copy()
    nuevos["categoria"] = ""
    if ruta.exists():
        viejo = pd.read_csv(ruta, keep_default_na=False)
        m = nuevos.drop(columns="categoria").merge(viejo[[*clave, "categoria"]], on=list(clave), how="left")
        m["categoria"] = m["categoria"].fillna("")
        m.to_csv(ruta, index=False)
        return m, f"leído: {int((m['categoria'] != '').sum())} de {len(m)} categorizados a mano"
    nuevos.to_csv(ruta, index=False)
    return nuevos, "creado: completa la columna `categoria` (tecnologia, conflicto, economia, politica, cultura, otro)"


# ---------------------------------------------------------------------------
# Variación dentro del tema
# ---------------------------------------------------------------------------


def cargar_theta(path: Path) -> pd.DataFrame:
    """id, topico ('Txx') y theta_dom desde results/08/theta_documentos.parquet.
    Detecta las columnas θ (T00…/theta_0…/0…) o una columna de tópico dominante."""
    th = pd.read_parquet(path)
    cols = [c for c in th.columns if re.fullmatch(r"(?:T|theta_?|topico_?)?\d{1,2}", str(c))]
    out = pd.DataFrame({"id": th["id"].values})
    if cols:
        M = th[cols].to_numpy(float)
        k = np.nanargmax(np.nan_to_num(M, nan=-1), axis=1)
        num = [int(re.search(r"\d+", str(c)).group()) for c in cols]
        out["topico"] = [f"T{num[i]:02d}" for i in k]
        out["theta_dom"] = M[np.arange(len(M)), k]
        out.loc[~np.isfinite(M).any(1) | (np.nansum(M, 1) == 0), ["topico", "theta_dom"]] = [None, np.nan]
        return out
    dom = [c for c in th.columns if "domin" in str(c).lower()]
    if not dom:
        raise KeyError(f"no encuentro θ ni tópico dominante en {list(th.columns)}")
    d = th[dom[0]]
    out["topico"] = [None if pd.isna(x) else (x if str(x).startswith("T") else f"T{int(x):02d}") for x in d]
    tc = [c for c in th.columns if "theta" in str(c).lower() and c != dom[0] and pd.api.types.is_numeric_dtype(th[c])]
    out["theta_dom"] = th[tc[0]].values if tc else np.nan
    return out


def logodds_dirichlet(ci: Counter, cj: Counter, prior: Counter, a0: float, min_total: int) -> pd.DataFrame:
    """Monroe et al. 2008: log-odds con prior informativo de Dirichlet. z > 0 → propio de i."""
    vocab = [w for w in set(ci) | set(cj) if ci.get(w, 0) + cj.get(w, 0) >= min_total]
    if not vocab:
        return pd.DataFrame(columns=["termino", "n_i", "n_j", "delta", "z"])
    yi = np.array([ci.get(w, 0) for w in vocab], float)
    yj = np.array([cj.get(w, 0) for w in vocab], float)
    pr = np.array([prior.get(w, 0) for w in vocab], float) + 0.01
    a = pr / pr.sum() * a0
    ni, nj = yi.sum(), yj.sum()
    d = np.log((yi + a) / (ni + a0 - yi - a)) - np.log((yj + a) / (nj + a0 - yj - a))
    z = d / np.sqrt(1 / (yi + a) + 1 / (yj + a))
    return pd.DataFrame({"termino": vocab, "n_i": yi.astype(int), "n_j": yj.astype(int), "delta": d, "z": z}) \
        .sort_values("z", ascending=False).reset_index(drop=True)


def js_estandarizado(cont: dict, pa: str, pb: str, pooled_pi: pd.Series, vocab) -> dict:
    """JS entre los períodos pa y pb (bits) con la mezcla de tópicos real y con la
    mezcla fija (π pooled). cont[(tópico, período)] = Counter. Si la JS estandarizada
    queda cerca de la real, el cambio léxico NO es composición temática."""
    ix = {w: i for i, w in enumerate(vocab)}

    def dist(p, pi=None):
        v = np.zeros(len(vocab))
        for k in pooled_pi.index:
            c = cont.get((k, p))
            if not c:
                continue
            x = np.zeros(len(vocab))
            for w, n in c.items():
                if w in ix:
                    x[ix[w]] += n
            v += x if pi is None else (x / max(x.sum(), 1)) * pi[k]
        return v / v.sum() if v.sum() else v

    real = agenda.js_div(dist(pa), dist(pb))
    est = agenda.js_div(dist(pa, pooled_pi), dist(pb, pooled_pi))
    return {"js_real": real, "js_mezcla_fija": est, "frac_que_queda": est / real if real else np.nan}
