"""Entrega 3, Fase A: estilo único y figuras finales (lo consume el 10).

Solo lee results/08 y results/09 y agrega en liviano: ningún modelo se recalcula.
No modifica utils.py, prep.py, agenda.py ni cronolecto.py: solo los usa.

Excepciones declaradas (tablas que el 09 calculó y no exportó). Se re-derivan
con las MISMAS funciones, semillas y configuración del 09, y se verifican
contra resumen_09.json; si no coinciden, la figura no se exporta y el índice
remite a la figura del 09:
    rederivar_denominador ....... F11b (Δ Zipf por denominador)
    rederivar_ic ................ F12  (IC Poisson vs. bootstrap por documento)

Bloques:
    estilo ....................... PRESETS, estilo, tamaño, colores, marcar_artefactos, marcar_cambios
    índice ....................... FIGURAS, guardar, escribir_indice
    datos ........................ cargar, cambios_agenda, macrocategorias, prevalencia_macro
    figuras 08 ................... f01 … f10
    figuras 09 ................... f11 … f17
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from types import SimpleNamespace

import agenda as ag
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# ---------------------------------------------------------------------------
# Estilo
# ---------------------------------------------------------------------------

CM = 1 / 2.54
PRESETS = {
    # LaTeX sencillo (article, márgenes de ~2,5 cm): ancho útil ≈ 16 cm.
    "informe": {"ancho": 16 * CM, "alto": None, "fuente": 8.0},
    # 16:9 de PowerPoint/Canva (13,33 × 7,5 in). Disponible, no se usa en el 10.
    "diapositiva": {"ancho": 13.33, "alto": 7.5, "fuente": 16.0},
}
_PRESET = "informe"

TINTA = "#222222"
GRIS = "#8c8c8c"
GRIS_CLARO = "#c8c8c8"
GRIS_FONDO = "#f0f0f0"

# Okabe-Ito (apta para daltonismo). Fijos por fuente en TODAS las figuras.
FUENTE = {"eltiempo": "#0072B2", "semana": "#D55E00", "dinero": "#009E73"}
NOMBRE_FUENTE = {"eltiempo": "El Tiempo", "semana": "Semana", "dinero": "Dinero"}
LARGO_A_CORTO = dict(ag.CORTO)  # www.eltiempo.com -> eltiempo

# Paul Tol "muted" sin el arena (poco contraste en líneas). Un color por
# macrocategoría; cada tópico hereda el de su macrocategoría.
PALETA_MACRO = [
    "#332288",
    "#CC6677",
    "#117733",
    "#882255",
    "#44AA99",
    "#999933",
    "#AA4499",
    "#88CCEE",
]
COLOR_RESIDUAL = "#9e9e9e"

# Variantes dentro de un conjunto (Okabe-Ito restantes; no chocan con las fuentes).
PALETA_VARIANTES = ["#000000", "#E69F00", "#56B4E9", "#CC79A7"]
# Neologismo / obsolescencia (Tol bright, par azul/rojo apto para daltonismo).
NEO, OBS = "#4477AA", "#EE6677"

# Marcas de artefacto: el año del cambio de archivo de El Tiempo y el régimen
# de breves/codificación de 2000. Bandas sobre el AÑO (x = año entero).
ARTEFACTOS = [
    {"año": 2006, "alfa": 0.20, "rotulo": "cambio de archivo ET"},
    {"año": 2000, "alfa": 0.08, "rotulo": "breves / codificación"},
]


def estilo(preset: str = "informe") -> None:
    """rcParams del proyecto. Tamaño de letra pensado para el tamaño FINAL."""
    global _PRESET
    _PRESET = preset
    f = PRESETS[preset]["fuente"]
    # Parte de los valores por defecto: el tema del editor (p. ej. VS Code oscuro) puede
    # haber cambiado fondos y colores de texto. style.use no toca el backend.
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.transparent": False,
            "text.color": TINTA,
            "axes.titlecolor": TINTA,
            "legend.labelcolor": TINTA,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": f,
            "axes.titlesize": f,
            "axes.labelsize": f,
            "xtick.labelsize": f - 1,
            "ytick.labelsize": f - 1,
            "legend.fontsize": f - 1,
            "figure.titlesize": f + 1,
            "axes.titlelocation": "left",
            "axes.titlepad": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#555555",
            "axes.labelcolor": TINTA,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "axes.grid": True,
            "grid.color": "#e8e8e8",
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "lines.linewidth": 1.2,
            "lines.markersize": 3,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "figure.constrained_layout.use": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,  # TrueType incrustada: editable en PowerPoint/Canva
            "axes.prop_cycle": mpl.cycler(color=[TINTA, GRIS]),
        }
    )


def tamaño(
    alto_cm: float | None = None, preset: str | None = None, frac_ancho: float = 1.0
) -> tuple[float, float]:
    """figsize en pulgadas. "informe": ancho de página (16 cm) × alto_cm.
    "diapositiva": 16:9 completo (alto_cm se ignora)."""
    p = PRESETS[preset or _PRESET]
    if p["alto"] is not None:
        return (p["ancho"] * frac_ancho, p["alto"])
    return (p["ancho"] * frac_ancho, (alto_cm or 8) * CM)


def marcar_artefactos(ax, rotular: bool = False) -> None:
    """Banda en 2006 (cambio de archivo de El Tiempo) y banda tenue en 2000.
    Va en TODAS las series temporales (x = año). El rótulo va en la leyenda de la
    figura (handles_marcas / leyenda_marcas), no dentro del panel, para no tapar datos;
    `rotular` se conserva por compatibilidad y no dibuja nada."""
    for a in ARTEFACTOS:
        ax.axvspan(
            a["año"] - 0.5,
            a["año"] + 0.5,
            color="#7f7f7f",
            alpha=a["alfa"],
            lw=0,
            zorder=0,
        )


def marcar_cambios(ax, consenso=(), limite=(), offset: float = -0.5) -> None:
    """Cambios de agenda como línea en el INICIO del nuevo régimen (año − 0,5)."""
    for c in consenso:
        ax.axvline(c + offset, color=TINTA, lw=0.8, zorder=1)
    for c in limite:
        ax.axvline(c + offset, color=TINTA, lw=0.7, ls=(0, (3, 2)), zorder=1)


def handles_marcas(con_limite: bool = True) -> list:
    h = [Line2D([], [], color=TINTA, lw=0.8, label="cambio de agenda robusto")]
    if con_limite:
        h.append(
            Line2D(
                [], [], color=TINTA, lw=0.7, ls=(0, (3, 2)), label="cambio en el límite"
            )
        )
    h += [
        Patch(
            color="#7f7f7f",
            alpha=a["alfa"] * 1.6,
            lw=0,
            label=f"{a['año']}: {a['rotulo']}",
        )
        for a in ARTEFACTOS
    ]
    return h


def leyenda_marcas(
    fig, con_limite: bool = True, cambios: bool = True, extra: list | None = None
) -> None:
    """Leyenda de figura (abajo, fuera de los paneles) con las marcas de artefacto."""
    h = handles_marcas(con_limite) if cambios else handles_marcas(False)[1:]
    fig.legend(
        handles=[*(extra or []), *h],
        loc="outside upper center",
        ncol=4,
        fontsize=mpl.rcParams["font.size"] - 1.5,
    )


def nota(fig, texto: str, ancho: int = 150) -> None:
    """Pie dentro de la figura (reglas declaradas, fuentes de los números)."""
    fig.text(
        0.0,
        0.0,
        textwrap.fill(texto, ancho),
        ha="left",
        va="top",
        fontsize=mpl.rcParams["font.size"] - 1.5,
        color="#555555",
    )


def ejes_años(ax, desde=1990, hasta=2011, paso=5) -> None:
    ax.set_xlim(desde - 0.7, hasta + 0.7)
    ax.set_xticks(
        range((desde // paso) * paso + (paso if desde % paso else 0), hasta + 1, paso)
    )


def completar_años(g: pd.DataFrame, desde: int, hasta: int) -> pd.DataFrame:
    """Reindexa por año para que los años sin dato corten la línea (no se interpolan)."""
    return (
        g.drop_duplicates("año")
        .set_index("año")
        .reindex(range(desde, hasta + 1))
        .rename_axis("año")
        .reset_index()
    )


def _corto(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------------------
# Índice
# ---------------------------------------------------------------------------

# id -> (archivo, mensaje en una frase, fuentes de datos, sección del informe)
FIGURAS = {
    "F01": (
        "f01_seleccion_k",
        "K = 18 es el K más chico a 1 EE del mejor NPMI held-out con estabilidad ≥ 0,80.",
        "08/metricas_K.csv; 08/resumen_08.json",
        "Método",
    ),
    "F02": (
        "f02_bigramas",
        "Los bigramas automáticos no suben la coherencia y casi no entran al top-10; solo la lista blanca suma.",
        "08/comparacion_bigramas.csv; 08/resumen_08.json",
        "Apéndice",
    ),
    "F03": (
        "f03_topicos",
        "18 tópicos: 17 interpretables y uno residual (T10, registro de opinión dependiente de la longitud).",
        "08/topicos.csv; 08/nombres_topicos.csv; figuras/macrocategorias.csv",
        "Método",
    ),
    "F04": (
        "f04_macrocategorias",
        "Los 17 tópicos de agenda se agrupan a mano en 7 macrotemas; la co-ocurrencia en θ solo la respalda en parte (ver ARI).",
        "08/theta_documentos.parquet; figuras/macrocategorias.csv",
        "Apéndice",
    ),
    "F05": (
        "f05_prevalencia_topicos",
        "Los cambios de agenda de El Tiempo se concentran en pocos tópicos y en 1995 y 2006.",
        "08/prevalencias_ic.csv; 08/robustez_cambios.csv; 08/resumen_08.json",
        "Agenda",
    ),
    "F06": (
        "f06_prevalencia_macro",
        "La misma lectura en 7 macrotemas: agenda estable con saltos en 1995 y 2006.",
        "08/theta_documentos.parquet; 08/meses_validos.csv; figuras/macrocategorias.csv",
        "Agenda",
    ),
    "F07": (
        "f07_estabilidad",
        "El cambio anual excede al ruido en ≈ 5–6 % de la distancia ET–Semana; el decaimiento se aplana: regímenes, no deriva.",
        "08/transiciones.csv; 08/js_entre_fuentes.csv; 08/decaimiento_rezago.csv",
        "Agenda",
    ),
    "F08": (
        "f08_prueba_artefacto",
        "2000 coincide con cambios de formato y de codificación; 2006 coincide con todos: la ü lo corrobora de forma independiente.",
        "08/cambios_detectados.csv; 08/formato_fuente_año.csv; 09/diagnostico_codificacion.csv",
        "Robustez",
    ),
    "F09": (
        "f09_robustez",
        "1995 es robusto, 1998 queda en el límite, 2000/2002 son artefacto de formato y 2006 está confundido con el archivo.",
        "08/robustez_cambios.csv",
        "Robustez",
    ),
    "F10": (
        "f10_linea_tiempo",
        "Los cambios de agenda no se alinean con eventos más que el azar; los cruces léxicos tienen su propio calendario.",
        "08/robustez_cambios.csv; 08/resumen_08.json; 09/cruces_variantes.csv",
        "Robustez",
    ),
    "F11": (
        "f11_piso_zipf",
        "La banda rara no se puede medir fuera de El Tiempo (piso Zipf ≥ 3) y el denominador desplaza el Zipf ≈ 0,4.",
        "09/pmw_zipf.parquet; data/corpus_procesado.parquet (re-derivación verificada)",
        "Cronolecto",
    ),
    "F12": (
        "f12_ic_poisson_bootstrap",
        "Bajo unas 20 apariciones la tasa no es fiable, y Poisson subestima el error real (ráfagas por documento).",
        "data/corpus_procesado.parquet (re-derivación verificada contra 09/resumen_09.json)",
        "Cronolecto",
    ),
    "F13": (
        "f13_zipf_frecuencia_documental",
        "A igual Zipf, la frecuencia documental varía hasta decenas de veces; la DP se satura.",
        "09/dispersion.csv",
        "Cronolecto",
    ),
    "F14": (
        "f14_cuotas_variantes",
        "Los reemplazos tienen fechas propias; solo tercera edad → adulto mayor cae cerca de 2006.",
        "09/variantes_cuotas.csv; 09/cruces_variantes.csv; 09/tildes_pmw.csv",
        "Cronolecto",
    ),
    "F15": (
        "f15_paz_lexica",
        "El léxico de paz tiene picos distintos: paz 1999–2001, desmovilización 2003–06, acuerdo humanitario 2007.",
        "09/paz_lexica.csv",
        "Cronolecto",
    ),
    "F16": (
        "f16_vecinos_estables",
        "El cambio semántico se ve en los vecinos estables (celular: occel → blackberry; banda: pandilla → bacrim).",
        "09/vecinos_por_periodo_bootstrap.csv",
        "Cronolecto",
    ),
    "F17": (
        "f17_neologismos",
        "Entra la tecnología y el léxico del conflicto; sale la economía de los 90 y la actualidad internacional.",
        "09/categorias_lexico.csv",
        "Cronolecto",
    ),
}
INDICE: dict[str, dict] = {}


def dir_figuras() -> Path:
    d = utils.results_dir() / "figuras"
    d.mkdir(parents=True, exist_ok=True)
    return d


class _Coma(mpl.ticker.ScalarFormatter):
    """ScalarFormatter con coma decimal (las etiquetas del informe van en español)."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def coma(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def _comas(fig) -> None:
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            f = axis.get_major_formatter()
            if type(f) is mpl.ticker.ScalarFormatter:
                axis.set_major_formatter(_Coma())


def guardar(
    fig,
    fid: str,
    mensaje: str | None = None,
    fuentes: str | None = None,
    seccion: str | None = None,
    archivo: str | None = None,
) -> Path:
    """PDF (vectorial) + PNG 300 dpi en results/figuras/ y registro en el índice."""
    a, m, f, s = FIGURAS.get(fid, (fid.lower(), "", "", ""))
    archivo, mensaje, fuentes, seccion = (
        archivo or a,
        mensaje or m,
        fuentes or f,
        seccion or s,
    )
    out = dir_figuras()
    _comas(fig)
    for ext in ("pdf", "png"):
        fig.savefig(
            out / f"{archivo}.{ext}",
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
            transparent=False,
        )
    INDICE[fid] = {
        "id": fid,
        "archivo": f"{archivo}.pdf",
        "mensaje": mensaje,
        "fuentes_datos": fuentes,
        "seccion_informe": seccion,
    }
    return out / f"{archivo}.pdf"


def escribir_indice(extra: dict | None = None) -> pd.DataFrame:
    """Une con el índice existente (la fase B agrega filas): por id, gana lo nuevo."""
    ruta = dir_figuras() / "indice_figuras.csv"
    nuevo = pd.DataFrame([*INDICE.values(), *(extra or {}).values()])
    if ruta.exists():
        viejo = pd.read_csv(ruta, keep_default_na=False)
        nuevo = pd.concat(
            [viejo[~viejo["id"].isin(nuevo["id"])], nuevo], ignore_index=True
        )
    nuevo = nuevo.sort_values("id").reset_index(drop=True)
    nuevo.to_csv(ruta, index=False)
    return nuevo


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


def a_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "sí", "si", "x"])


def años_de(x) -> list[int]:
    return (
        [int(a) for a in re.findall(r"(?<!\d)(\d{4})(?!\d)", str(x))]
        if pd.notna(x)
        else []
    )


def floats_de(x) -> list[float]:
    return (
        [float(a) for a in re.findall(r"-?\d+(?:\.\d+)?", str(x))]
        if pd.notna(x)
        else []
    )


def cargar(res: Path | None = None) -> SimpleNamespace:
    """Todas las tablas de results/08 y results/09 que usan las figuras (sin pmw_zipf)."""
    R = res or utils.results_dir()
    R8, R9 = R / "08", R / "09"
    d = SimpleNamespace(R=R, R8=R8, R9=R9)
    d.r8 = json.loads((R8 / "resumen_08.json").read_text(encoding="utf-8"))
    d.r9 = json.loads((R9 / "resumen_09.json").read_text(encoding="utf-8"))
    d.cfg8 = d.r8["config"]
    rd = pd.read_csv
    d.metricas_K = rd(R8 / "metricas_K.csv")
    d.bigramas = rd(R8 / "comparacion_bigramas.csv")
    d.topicos = rd(R8 / "topicos.csv")
    d.nombres = rd(R8 / "nombres_topicos.csv", keep_default_na=False)
    d.prev = rd(R8 / "prevalencias_ic.csv")
    d.prev["tid"] = d.prev["topico"].str.split().str[0]
    d.trans = rd(R8 / "transiciones.csv")
    d.ref = rd(R8 / "js_entre_fuentes.csv")
    d.lag = rd(R8 / "decaimiento_rezago.csv")
    d.cambios = rd(R8 / "cambios_detectados.csv")
    d.robustez = rd(R8 / "robustez_cambios.csv")
    d.formato = rd(R8 / "formato_fuente_año.csv")
    d.meses_validos = rd(R8 / "meses_validos.csv")
    d.disp = rd(R9 / "dispersion.csv")
    d.cat = rd(R9 / "categorias_lexico.csv", keep_default_na=False)
    d.cruces = rd(R9 / "cruces_variantes.csv")
    d.cuotas = rd(R9 / "variantes_cuotas.csv")
    d.paz = rd(R9 / "paz_lexica.csv")
    d.tildes = rd(R9 / "tildes_pmw.csv")
    d.vecinos = rd(R9 / "vecinos_por_periodo_bootstrap.csv")
    d.codif = rd(R9 / "diagnostico_codificacion.csv")
    d.resid = set(d.nombres.loc[a_bool(d.nombres["residual"]), "topico"])
    d.nombre = dict(zip(d.nombres["topico"], d.nombres["nombre"]))
    d.agenda_ca = cambios_agenda(d)
    return d


def cambios_agenda(d) -> dict:
    """consenso (del 08), en el límite y artefacto, desde robustez_cambios.csv.

    Regla declarada para "en el límite": veredicto no robusto, soporte de votantes
    ≥ soporte_min y presente en ≥ 3 de las 4 series de control (1998 en el 08)."""
    rob = d.robustez.copy()
    ser = ["todos_doc", "sin_breves_doc", "todos_mes", "sin_breves_mes"]
    nser = sum(a_bool(rob[c]).astype(int) for c in ser)
    ver = rob["veredicto"].astype(str)
    limite = rob[
        ver.eq("no robusto") & (rob["soporte"] >= d.cfg8["soporte_min"]) & (nser >= 3)
    ]["año"]
    artef = rob[ver.str.startswith("artefacto")]["año"]
    return {
        "consenso": [int(x) for x in d.r8["consenso"]],
        "limite": [int(x) for x in limite],
        "artefacto": [int(x) for x in artef],
    }


# --- Macrocategorías ---------------------------------------------------------

# Propuesta a mano (editable en results/figuras/macrocategorias.csv). El orden
# de aparición fija el color.
MACRO_PROPUESTA = {
    "T02": "Conflicto y justicia",
    "T09": "Conflicto y justicia",
    "T04": "Política e instituciones",
    "T13": "Política e instituciones",
    "T16": "Política e instituciones",
    "T01": "Economía",
    "T06": "Economía",
    "T14": "Economía",
    "T08": "Internacional",
    "T00": "Sociedad",
    "T07": "Sociedad",
    "T05": "Cultura y deporte",
    "T11": "Cultura y deporte",
    "T15": "Cultura y deporte",
    "T03": "Ciudad, territorio y tecnología",
    "T12": "Ciudad, territorio y tecnología",
    "T17": "Ciudad, territorio y tecnología",
}


def similitud_theta(d) -> tuple[np.ndarray, list[str]]:
    """Coseno entre columnas de θ (renormalizado sin residuales) en los docs de ajuste:
    dos tópicos se parecen si tienden a aparecer en los mismos documentos.
    Sustituye a la similitud entre filas de H, que el 08 no exportó."""
    K = int(d.r8["K"])
    cols = [f"T{k:02d}" for k in range(K)]
    th = pd.read_parquet(
        d.R8 / "theta_documentos.parquet", columns=[*cols, "ok", "en_ajuste"]
    )
    th = th[a_bool(th["en_ajuste"]) & a_bool(th["ok"])]
    resid = {int(t[1:]) for t in d.resid}
    T, ok, keep = ag.theta_agenda(
        th[cols].to_numpy(float), np.ones(len(th), bool), resid
    )
    T = T[ok]
    Tn = T / (np.linalg.norm(T, axis=0, keepdims=True) + 1e-12)
    return Tn.T @ Tn, [cols[k] for k in keep]


def macrocategorias(d, n_grupos: int | None = None) -> tuple[pd.DataFrame, str]:
    """CSV editable: topico, nombre, macro (a mano) y cluster_theta (sugerencia).

    Si existe, se respetan las ediciones de `macro`. El cluster es jerárquico
    (enlace promedio sobre 1 − coseno), cortado en tantos grupos como macros."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import adjusted_rand_score

    S, ids = similitud_theta(d)
    ruta = dir_figuras() / "macrocategorias.csv"
    if ruta.exists():
        tab = pd.read_csv(ruta, keep_default_na=False)
        falta = set(ids) - set(tab.loc[tab["macro"].str.strip() != "", "topico"])
        if falta:
            raise ValueError(f"macrocategorias.csv: faltan macros para {sorted(falta)}")
        estado = "leído (ediciones manuales respetadas)"
    else:
        tab = pd.DataFrame(
            {
                "topico": ids,
                "nombre": [d.nombre.get(t, t) for t in ids],
                "macro": [MACRO_PROPUESTA.get(t, "Otros") for t in ids],
            }
        )
        estado = (
            "creado con la propuesta a mano: edita la columna `macro` y vuelve a correr"
        )
    tab = tab[tab["topico"].isin(ids)].copy()
    n = n_grupos or tab["macro"].nunique()
    D = np.clip(1 - S, 0, None)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    cl = dict(zip(ids, fcluster(Z, n, criterion="maxclust")))
    tab["cluster_theta"] = tab["topico"].map(cl)
    tab[["topico", "nombre", "macro", "cluster_theta"]].to_csv(ruta, index=False)
    ari = adjusted_rand_score(tab["macro"], tab["cluster_theta"])
    if tab["macro"].nunique() > len(PALETA_MACRO):
        raise ValueError(
            f"más de {len(PALETA_MACRO)} macrocategorías: no hay colores aptos para todas"
        )
    d.macro = tab
    d.macro_orden = list(dict.fromkeys(tab["macro"]))
    d.color_macro = {m: PALETA_MACRO[i] for i, m in enumerate(d.macro_orden)}
    d.color_topico = {t: d.color_macro[m] for t, m in zip(tab["topico"], tab["macro"])}
    d.color_topico |= {t: COLOR_RESIDUAL for t in d.resid}
    d.similitud = (S, ids)
    d.ari_macro = float(ari)
    return tab, f"{estado} | ARI(macro a mano, cluster θ) = {ari:.2f}"


def topicos_ordenados(d) -> list[str]:
    """Tópicos de agenda ordenados por macrocategoría (orden del CSV)."""
    m = d.macro.assign(
        o=d.macro["macro"].map({k: i for i, k in enumerate(d.macro_orden)})
    )
    return m.sort_values(["o", "topico"])["topico"].tolist()


def prevalencia_macro(d) -> tuple[pd.DataFrame, float]:
    """Prevalencia anual por macrocategoría, El Tiempo sin breves, igual peso por mes válido,
    con IC bootstrap (mismo protocolo y semilla del 08, sobre θ exportado).
    Verificación: la suma de las prevalencias por tópico del 08 debe coincidir."""
    c = d.cfg8
    K = int(d.r8["K"])
    cols = [f"T{k:02d}" for k in range(K)]
    th = pd.read_parquet(
        d.R8 / "theta_documentos.parquet",
        columns=[*cols, "fuente", "año", "mes", "es_breve", "ok"],
    )
    resid = {int(t[1:]) for t in d.resid}
    T_AG, OK_AG, keep = ag.theta_agenda(
        th[cols].to_numpy(float), a_bool(th["ok"]).values, resid
    )
    años, meses = th["año"].to_numpy(int), th["mes"].to_numpy(int)
    m = (
        OK_AG
        & (th["fuente"].values == ag.ET)
        & ~a_bool(th["es_breve"]).values
        & (años >= c["principal_desde"])
    )
    mv = d.meses_validos
    valido = set(
        zip(
            *[
                mv.loc[(mv["fuente"] == ag.ET) & a_bool(mv["valido"]), k].astype(int)
                for k in ("año", "mes")
            ]
        )
    )
    macro_de = dict(zip(d.macro["topico"], d.macro["macro"]))
    M = np.zeros((len(keep), len(d.macro_orden)))
    for i, k in enumerate(keep):
        M[i, d.macro_orden.index(macro_de[cols[k]])] = 1
    TM = T_AG[m] @ M
    rango = np.arange(c["principal_desde"], c["año_max"] + 1)
    P, n = ag.prevalencia(
        TM, años[m], meses[m], valido, "mes", rango, c["min_meses_validos"]
    )
    ic = ag.bootstrap_prevalencia(
        TM,
        años[m],
        meses[m],
        valido,
        "mes",
        c["b_boot"],
        c["seed"],
        rango,
        c["min_meses_validos"],
    )
    out = (
        P.stack()
        .rename("prev")
        .rename_axis(["año", "j"])
        .reset_index()
        .merge(ic.rename(columns={"topico": "j"}), on=["año", "j"], how="left")
    )
    out["macro"] = out["j"].map(dict(enumerate(d.macro_orden)))
    out["n"] = out["año"].map(n)
    # verificación contra el 08
    ref = d.prev[d.prev["serie"] == "eltiempo|sin_breves|mes"].assign(
        macro=lambda x: x["tid"].map(macro_de)
    )
    ref = ref.groupby(["año", "macro"])["prev"].sum().rename("prev_08").reset_index()
    chk = out.merge(ref, on=["año", "macro"])
    dif = float((chk["prev"] - chk["prev_08"]).abs().max())
    if not dif < 2e-3:
        raise AssertionError(
            f"la prevalencia por macro no reproduce la suma del 08 (máx. |Δ| = {dif:.4f})"
        )
    return out, dif


# ---------------------------------------------------------------------------
# Figuras del 08
# ---------------------------------------------------------------------------


def f01_seleccion_k(d):
    m = d.metricas_K.sort_values("K")
    K = int(d.r8["K"])
    best = m.loc[m["npmi_media"].idxmax()]
    thr = best["npmi_media"] - best["se"]
    fig, ax = plt.subplots(1, 3, figsize=tamaño(5.2))
    ax[0].errorbar(
        m["K"],
        m["npmi_media"],
        yerr=m["se"],
        fmt="o-",
        color=TINTA,
        ms=3,
        capsize=1.5,
        lw=1,
    )
    ax[0].axhline(thr, color=GRIS, ls=":", lw=0.8)
    ax[0].text(m["K"].min(), thr, "mejor − 1 EE", va="bottom", fontsize=6.5, color=GRIS)
    ax[0].set_title("(a) C_NPMI held-out (media ± EE)")
    ax[1].plot(m["K"], m["diversidad"], "o-", color=TINTA, ms=3)
    ax[1].set_title("(b) Diversidad del top-25")
    ax[2].plot(m["K"], m["estab_cos"], "o-", color=TINTA, ms=3, label="coseno")
    ax[2].plot(
        m["K"], m["estab_jaccard"], "s--", color=GRIS, ms=2.5, label="Jaccard top-10"
    )
    ax[2].axhline(d.cfg8["estab_min"], color=GRIS, ls=":", lw=0.8)
    ax[2].set_title("(c) Estabilidad entre semillas")
    ax[2].legend(loc="lower left")
    for a in ax:
        a.axvspan(K - 0.4, K + 0.4, color="#7f7f7f", alpha=0.18, lw=0, zorder=0)
        a.set_xlabel("K")
        a.set_xticks(m["K"].values[::2])
    ax[0].text(
        K + 0.5,
        0.03,
        f"K = {K}",
        transform=ax[0].get_xaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=6.5,
    )
    nota(
        fig,
        f"Regla declarada antes de ver resultados: {d.r8['motivo_K']}. Resultado: K = {K} "
        f"(semilla medoide {d.r8['semilla_medoide']}). 3 semillas por K; EE entre semillas.",
    )
    return fig


def f02_bigramas(d):
    b = d.bigramas.copy()
    b["base"] = b["estrategia"].str.replace("+LB", "", regex=False)
    b["lb"] = b["estrategia"].str.endswith("+LB")
    bases = sorted(b["base"].unique())
    x = {k: i for i, k in enumerate(bases)}
    elegida = d.r8["estrategia_bigramas"]
    fig, ax = plt.subplots(1, 3, figsize=tamaño(5.0))
    for lb, dx, col, mfc, lab in [
        (False, -0.13, GRIS, "white", "sin lista blanca"),
        (True, 0.13, TINTA, TINTA, "con lista blanca"),
    ]:
        s = b[b["lb"] == lb]
        xs = s["base"].map(x) + dx
        ax[0].errorbar(
            xs,
            s["npmi_media"],
            yerr=s["se"],
            fmt="o",
            color=col,
            mfc=mfc,
            ms=3.5,
            capsize=1.5,
            lw=0.8,
            label=lab,
        )
        ax[1].plot(xs, s["estab_cos"], "o", color=col, mfc=mfc, ms=3.5)
        ax[2].bar(
            xs,
            s["bigramas_en_top10"],
            width=0.24,
            color=col if lb else "white",
            edgecolor=col,
            lw=0.8,
        )
    e = b[b["estrategia"] == elegida]
    if len(e):
        xe = x[e["base"].iloc[0]] + (0.13 if e["lb"].iloc[0] else -0.13)
        ax[0].annotate(
            "elegida",
            (xe, e["npmi_media"].iloc[0] + e["se"].iloc[0]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=6.5,
            arrowprops={"arrowstyle": "-", "lw": 0.5},
        )
    ax[0].set_title("(a) C_NPMI held-out (media ± EE)")
    ax[1].set_title("(b) Estabilidad (coseno)")
    ax[2].set_title("(c) Bigramas en el top-10")
    ax[0].legend(loc="lower left")
    for a in ax:
        a.set_xticks(range(len(bases)), bases)
        a.set_xlabel("estrategia")
    nota(
        fig,
        f"K = {int(b['K'].iloc[0])}, 3 semillas. A: sin bigramas; B: NPMI ≥ {d.cfg8['npmi_B']}; "
        f"C: B con n ≥ {d.cfg8['n_C']}; D: los |C| más frecuentes con NPMI ≥ {d.cfg8['npmi_D']}; "
        f"E: los |C| más frecuentes. Regla: {d.r8['motivo_bigramas']}.",
    )
    return fig


def _figura_tabla(figsize):
    """Tabla dibujada a mano sobre un solo eje sin marco (mismo motor de layout que el resto)."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.get_layout_engine().set(w_pad=0, h_pad=0)
    return fig, ax


def f03_topicos(d):
    t = d.topicos.copy()
    t["macro"] = t["topico"].map(dict(zip(d.macro["topico"], d.macro["macro"])))
    orden = topicos_ordenados(d) + sorted(d.resid)
    t = t.set_index("topico").loc[orden].reset_index()
    t["top8"] = t["top10"].str.split().str[:8].str.join(" ").str.replace("_", " ")
    alto_fila = 0.72
    fig, ax = _figura_tabla(tamaño(alto_fila * (len(t) + 2.2) + 0.4))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(len(t) + 2.4, -0.2)
    X = {"sw": 0.0, "id": 0.025, "nom": 0.085, "top": 0.38, "npmi": 0.905, "est": 0.985}
    fs = mpl.rcParams["font.size"] - 0.5
    for k, lab, ha in [
        ("id", "", "left"),
        ("nom", "Nombre", "left"),
        ("top", "Top-8 términos", "left"),
        ("npmi", "NPMI", "right"),
        ("est", "Estab.", "right"),
    ]:
        ax.text(X[k], 0, lab, ha=ha, va="center", fontsize=fs, weight="bold")
    ax.plot([0, 1], [0.5, 0.5], color=TINTA, lw=0.6)
    previo = None
    for i, r in t.iterrows():
        y = i + 1.05
        res = r["topico"] in d.resid
        col = COLOR_RESIDUAL if res else d.color_macro[r["macro"]]
        if previo is not None and r["macro"] != previo:
            ax.plot([0, 1], [y - 0.5, y - 0.5], color=GRIS_CLARO, lw=0.4)
        previo = r["macro"]
        ax.add_patch(
            Rectangle(
                (X["sw"], y - 0.32),
                0.012,
                0.64,
                color=col,
                lw=0,
                hatch="////" if res else None,
                fill=not res,
            )
        )
        kw = {
            "color": GRIS if res else TINTA,
            "style": "italic" if res else "normal",
            "fontsize": fs,
            "va": "center",
        }
        nom = d.nombre.get(r["topico"], r["topico"]) + (" (residual)" if res else "")
        ax.text(X["id"], y, r["topico"], ha="left", **kw)
        ax.text(X["nom"], y, textwrap.fill(nom, 32), ha="left", linespacing=1.0, **kw)
        ax.text(
            X["top"],
            y,
            textwrap.fill(r["top8"], 58),
            ha="left",
            linespacing=1.0,
            **{**kw, "fontsize": fs - 0.5},
        )
        ax.text(X["npmi"], y, coma(r["npmi"], 3), ha="right", **kw)
        ax.text(X["est"], y, coma(r["estabilidad"]), ha="right", **kw)
    ax.plot([0, 1], [len(t) + 0.6, len(t) + 0.6], color=TINTA, lw=0.6)
    hs = [Patch(color=d.color_macro[m], label=m) for m in d.macro_orden]
    ax.legend(
        handles=hs,
        loc="upper left",
        bbox_to_anchor=(0, len(t) + 0.75),
        bbox_transform=ax.transData,
        ncol=4,
        fontsize=fs - 1,
        handlelength=1,
    )
    nota(
        fig,
        "NPMI: C_NPMI held-out (ventana 10) del top-10. Estab.: coseno medio con su pareja en las otras semillas "
        "(emparejamiento húngaro). El color indica la macrocategoría; T10 se excluye de la agenda y se reporta.",
    )
    return fig


def f04_macrocategorias(d):
    S, ids = d.similitud
    orden = topicos_ordenados(d)
    ix = [ids.index(t) for t in orden]
    M = S[np.ix_(ix, ix)].copy()
    np.fill_diagonal(M, np.nan)
    fig, ax = plt.subplots(figsize=tamaño(11.5))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=np.nanmax(M))
    labs = [f"{t} {_corto(d.nombre.get(t, t), 24)}" for t in orden]
    ax.set_xticks(range(len(orden)), [t for t in orden], rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(orden)), labs, fontsize=6.5)
    for tl, t in zip(ax.get_yticklabels(), orden):
        tl.set_color(d.color_topico[t])
    for tl, t in zip(ax.get_xticklabels(), orden):
        tl.set_color(d.color_topico[t])
    ax.grid(False)
    macro_de = dict(zip(d.macro["topico"], d.macro["macro"]))
    ini = 0
    for i in range(1, len(orden) + 1):
        if i == len(orden) or macro_de[orden[i]] != macro_de[orden[ini]]:
            ax.add_patch(
                Rectangle(
                    (ini - 0.5, ini - 0.5),
                    i - ini,
                    i - ini,
                    fill=False,
                    ec=TINTA,
                    lw=0.9,
                )
            )
            ini = i
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02, aspect=30)
    cb.set_label("coseno entre columnas de θ", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    hs = [Patch(color=d.color_macro[m], label=m) for m in d.macro_orden]
    ax.legend(
        handles=hs,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        fontsize=6.3,
        handlelength=1,
    )
    nota(
        fig,
        f"Docs de ajuste del 08, θ renormalizado sin el residual. Recuadros: macrocategorías a mano "
        f"(results/figuras/macrocategorias.csv, editable). Acuerdo con el agrupamiento jerárquico de θ "
        f"(enlace promedio, mismo número de grupos): ARI = {d.ari_macro:.2f}. H no se exportó en el 08.",
    )
    return fig


def _panel_prev(ax, d_main, d_ref, color):
    if len(d_ref):
        ax.plot(d_ref["año"], d_ref["prev"] * 100, color=GRIS, lw=0.6, zorder=2)
    ax.fill_between(
        d_main["año"],
        d_main["lo"] * 100,
        d_main["hi"] * 100,
        color=color,
        alpha=0.25,
        lw=0,
        zorder=2,
    )
    ax.plot(d_main["año"], d_main["prev"] * 100, color=color, lw=1.1, zorder=3)


def f05_prevalencia_topicos(
    d, principal="eltiempo|sin_breves|mes", referencia="eltiempo|todos|doc"
):
    ca = d.agenda_ca
    nmin = d.cfg8["min_n_año"]
    p = d.prev[d.prev["n"] >= nmin]
    orden = topicos_ordenados(d)
    ncol = 3
    nrow = int(np.ceil((len(orden) + 1) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=tamaño(3.3 * nrow + 0.6), sharex=True)
    axs = axs.ravel()
    for a, t in zip(axs, orden):
        dm = p[(p["serie"] == principal) & (p["tid"] == t)]
        dr = p[(p["serie"] == referencia) & (p["tid"] == t)]
        marcar_artefactos(a)
        marcar_cambios(a, ca["consenso"], ca["limite"])
        _panel_prev(a, dm, dr, d.color_topico[t])
        a.set_title(
            f"{t} · {_corto(d.nombre.get(t, t), 27)}",
            color=d.color_topico[t],
            fontsize=7,
        )
        a.tick_params(labelbottom=True)
        a.set_ylim(bottom=0)
        ejes_años(a, 1990, 2011)
    marcar_artefactos(axs[0], rotular=True)
    for a in axs[len(orden) :]:
        a.axis("off")
    hs = [
        Line2D([], [], color=TINTA, lw=1.1, label="sin breves, peso por mes (IC 95 %)"),
        Line2D([], [], color=GRIS, lw=0.6, label="todos, media por documento"),
        *handles_marcas(),
    ]
    axs[len(orden)].legend(handles=hs, loc="center left", fontsize=6.5)
    fig.supylabel("prevalencia (%)", fontsize=7)
    nota(
        fig,
        f"El Tiempo 1990–2011, años con ≥ {nmin} documentos. IC 95 % por bootstrap de documentos dentro de cada mes "
        f"válido (B = {d.cfg8['b_boot']}). Cambios: {', '.join(map(str, ca['consenso']))} robustos; "
        f"{', '.join(map(str, ca['limite'])) or '—'} en el límite. El color indica la macrocategoría (F03).",
    )
    return fig


def f06_prevalencia_macro(d, pm: pd.DataFrame):
    ca = d.agenda_ca
    pm = pm[pm["n"] >= d.cfg8["min_n_año"]]
    ncol = 4
    nrow = int(np.ceil((len(d.macro_orden) + 1) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=tamaño(3.6 * nrow + 0.6), sharex=True)
    axs = axs.ravel()
    for a, m in zip(axs, d.macro_orden):
        s = pm[pm["macro"] == m]
        marcar_artefactos(a)
        marcar_cambios(a, ca["consenso"], ca["limite"])
        _panel_prev(a, s, s.iloc[0:0], d.color_macro[m])
        a.set_title(textwrap.fill(m, 24), color=d.color_macro[m], fontsize=7)
        a.tick_params(labelbottom=True)
        a.set_ylim(bottom=0)
        ejes_años(a, 1990, 2011, 10)
    marcar_artefactos(axs[0], rotular=True)
    for a in axs[len(d.macro_orden) :]:
        a.axis("off")
    axs[len(d.macro_orden)].legend(
        handles=handles_marcas(), loc="center left", fontsize=6
    )
    fig.supylabel("prevalencia (%)", fontsize=7)
    nota(
        fig,
        "El Tiempo sin breves, igual peso por mes válido; IC 95 % bootstrap con la semilla del 08. "
        "Cada macrotema suma las θ de sus tópicos (verificado contra la suma de prevalencias del 08).",
    )
    return fig


TR_PRINCIPAL = "eltiempo sin breves | estrato fuente×mes"
TR_TODOS = "eltiempo | estrato fuente×breve"


def f07_estabilidad(d):
    ca = d.agenda_ca
    tr = d.trans
    tp, tt = tr[tr["serie"] == TR_PRINCIPAL], tr[tr["serie"] == TR_TODOS]
    fig, ax = plt.subplots(
        1, 3, figsize=tamaño(6.4), gridspec_kw={"width_ratios": [1.15, 1.15, 1]}
    )
    a = ax[0]
    marcar_artefactos(a, rotular=True)
    marcar_cambios(a, ca["consenso"], ca["limite"])
    a.fill_between(
        tp["año2"],
        tp["nulo_media"],
        tp["nulo_p95"],
        color=GRIS_CLARO,
        lw=0,
        label="ruido (media–p95)",
    )
    a.plot(
        tp["año2"],
        tp["js"],
        "o-",
        color=FUENTE["eltiempo"],
        ms=2.5,
        label="JS observado",
    )
    a.set_title("(a) JS año a año vs. ruido")
    a.set_ylabel("JS (bits)")
    a.legend(loc="upper left", fontsize=6)
    ejes_años(a, 1991, 2011)
    a = ax[1]
    marcar_artefactos(a)
    marcar_cambios(a, ca["consenso"], ca["limite"])
    a.axhline(0, color=GRIS, lw=0.6)
    a.plot(
        tt["año2"],
        tt["frac_dist_fuentes"] * 100,
        "-",
        color=GRIS,
        lw=0.7,
        label="todos",
    )
    a.plot(
        tp["año2"],
        tp["frac_dist_fuentes"] * 100,
        "o-",
        color=FUENTE["eltiempo"],
        ms=2.5,
        label="sin breves",
    )
    mu = tp["frac_dist_fuentes"].mean() * 100
    a.axhline(mu, color=FUENTE["eltiempo"], ls=":", lw=0.8)
    a.text(
        2011.5,
        mu,
        f"media {coma(mu, 1)} %",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=FUENTE["eltiempo"],
        bbox={"fc": "white", "ec": "none", "pad": 0.5, "alpha": 0.8},
    )
    a.set_title("(b) Exceso sobre el ruido\n(% de la distancia ET–Semana)")
    a.legend(loc="upper left", fontsize=6)
    ejes_años(a, 1991, 2011)
    a = ax[2]
    estilos = {
        "eltiempo sin breves/mes": (FUENTE["eltiempo"], "-", "El Tiempo sin breves"),
        "eltiempo todos/doc": (FUENTE["eltiempo"], ":", "El Tiempo todos"),
        "semana todos/doc": (FUENTE["semana"], "-", "Semana"),
        "dinero todos/doc": (FUENTE["dinero"], "-", "Dinero"),
    }
    for s, (c, ls, lab) in estilos.items():
        g = d.lag[d.lag["serie"] == s].dropna(subset=["js_medio"])
        if len(g) >= 3:
            a.plot(g["rezago"], g["js_medio"], ls, color=c, marker="o", ms=2, label=lab)
    a.axhline(d.r8["js_entre_fuentes"]["js"], color=TINTA, ls="--", lw=0.7)
    a.text(
        1,
        d.r8["js_entre_fuentes"]["js"],
        "JS ET–Semana (mismo año)",
        va="bottom",
        fontsize=6,
    )
    a.set_title("(c) Decaimiento por rezago")
    leyenda_marcas(fig)
    a.set_xlabel("rezago (años)")
    a.legend(loc="lower right", fontsize=6)
    nota(
        fig,
        f"Transiciones en el año de llegada. Nulo: permutación de etiquetas dentro de fuente×mes "
        f"(B = {d.cfg8['b_perm']}). Distancia de referencia: exceso medio ET–Semana en el mismo año "
        f"= {coma(d.r8['js_entre_fuentes']['exceso'], 4)} bits.",
    )
    return fig


def _cambios_variante(d, nombre) -> tuple[list[int], list[float]]:
    r = d.cambios[d.cambios["variante"] == nombre]
    if not len(r):
        return [], []
    return años_de(r["cambios"].iloc[0]), floats_de(r["fuerza"].iloc[0])


def f08_prueba_artefacto(d):
    ca = d.agenda_ca
    fmt = d.formato[
        (d.formato["fuente"] == ag.ET) & (d.formato["año"] >= d.cfg8["principal_desde"])
    ].sort_values("año")
    cod = d.codif[d.codif["fuente"] == "eltiempo"].sort_values("año")
    cod = cod[cod["n_total"] > 0]
    filas = [
        ("pct_breve", "% breves", "formato ET: pct_breve", False),
        (
            "mediana_palabras",
            "mediana de palabras",
            "formato ET: log_mediana_palabras",
            True,
        ),
        (
            "pct_titulo_cap",
            "% titulares en\nMayúscula Inicial",
            "formato ET: pct_titulo_cap",
            False,
        ),
        ("pct_lead_dup", "% lead duplicado", "formato ET: pct_lead_dup", False),
    ]
    fig, axs = plt.subplots(
        2 + len(filas),
        1,
        figsize=tamaño(19),
        sharex=True,
        gridspec_kw={"height_ratios": [1.5] + [1] * len(filas) + [1.1]},
    )
    for a in axs:
        marcar_artefactos(a)
        marcar_cambios(a, ca["consenso"], ca["limite"])
        a.grid(axis="x", visible=False)
    marcar_artefactos(axs[0], rotular=True)
    # (a) cambios temáticos y de formato (todas juntas)
    series = [
        ("ET todos | media doc", "temas: todos · doc"),
        ("ET sin breves | media doc", "temas: sin breves · doc"),
        ("ET todos | peso por mes", "temas: todos · mes"),
        ("ET sin breves | peso por mes", "temas: sin breves · mes"),
        (
            "formato ET: pct_breve+log_mediana_palabras+log_docs+pct_titulo_cap+pct_lead_dup",
            "formato (5 series, z)",
        ),
    ]
    a = axs[0]
    for i, (v, lab) in enumerate(series):
        cs, fs = _cambios_variante(d, v)
        fmt_ = v.startswith("formato")
        a.scatter(
            [c - 0.5 for c in cs],
            [i] * len(cs),
            s=[14 * f for f in fs] if fs else 14,
            marker="v" if fmt_ else "o",
            color=GRIS if fmt_ else TINTA,
            zorder=3,
            lw=0,
        )
    a.set_yticks(range(len(series)), [s[1] for s in series], fontsize=6.5)
    a.set_ylim(len(series) - 0.4, -0.6)
    a.set_title("(a) Cambios detectados (binseg; tamaño = fuerza)", fontsize=7)
    a.grid(axis="y", visible=False)
    # (b–e) series de formato con sus cambios
    for k, (a, (col, lab, var, logy)) in enumerate(zip(axs[1:], filas)):
        a.plot(fmt["año"], fmt[col], "o-", color=FUENTE["eltiempo"], ms=2.2, lw=1)
        if logy:
            a.set_yscale("log")
        cs, _ = _cambios_variante(d, var)
        for c in cs:
            a.plot(
                c - 0.5,
                1.0,
                "v",
                color=GRIS,
                ms=4,
                transform=a.get_xaxis_transform(),
                clip_on=False,
            )
        a.set_ylabel(lab, fontsize=6.5)
        a.set_title(f"({'bcde'[k]})", fontsize=7)
    # (f) régimen de ü
    a = axs[-1]
    for col, lab, c in [
        ("pct_eliminada", "ü eliminada (vergenza)", PALETA_VARIANTES[0]),
        ("pct_u", "ü → u (verguenza)", PALETA_VARIANTES[1]),
        ("pct_ü", "ü correcta", PALETA_VARIANTES[2]),
    ]:
        a.plot(cod["año"], cod[col], "o-", color=c, ms=2.2, lw=1, label=lab)
    a.set_ylabel("palabras con ü\n(% de casos)", fontsize=6.5)
    a.set_title(
        "(f) Codificación del archivo (corroboración independiente, 09)", fontsize=7
    )
    ejes_años(a, 1990, 2011)
    a.legend(
        handles=a.get_legend_handles_labels()[0] + handles_marcas(),
        fontsize=5.8,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0, -0.25),
    )
    nota(
        fig,
        "Mismo detector (binseg, umbral por permutación) sobre temas y sobre FORMATO de El Tiempo. ● cambio "
        "temático, ▼ cambio de formato; ambos en el inicio del nuevo régimen. Un cambio temático que coincide "
        "con formato y no sobrevive a los controles es artefacto probable.",
    )
    return fig


def f09_robustez(d):
    rob = d.robustez.sort_values("año").reset_index(drop=True)
    cols = list(rob.columns)
    bloques = [
        (
            "Series de control",
            ["todos_doc", "sin_breves_doc", "todos_mes", "sin_breves_mes"],
        ),
        ("Votantes", [c for c in cols if c.startswith("v:")]),
        ("Formato", ["coincide_formato"]),
        ("Contraste", [c for c in cols if c.startswith("c:")]),
        ("Informativas", [c for c in cols if c.startswith("i:")]),
    ]
    orden = [c for _, cs in bloques for c in cs]
    M = np.column_stack(
        [
            a_bool(rob[c]).astype(int) * (2 if c == "coincide_formato" else 1)
            for c in orden
        ]
    )
    bonito = lambda c: (
        re.sub(r"^[vci]:", "", c)
        .replace("_", " ")
        .replace(" doc", " · doc")
        .replace(" mes", " · mes")
    )
    fig, ax = plt.subplots(figsize=tamaño(7.5))
    ax.imshow(
        M,
        cmap=ListedColormap([GRIS_FONDO, "#33475b", "#c2185b"]),
        vmin=0,
        vmax=2,
        aspect="auto",
    )
    ax.set_xticks(
        range(len(orden)),
        [bonito(c) for c in orden],
        rotation=60,
        ha="right",
        fontsize=6.3,
    )
    ax.set_yticks(range(len(rob)), rob["año"].astype(int))
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    j = 0
    for nom, cs in bloques:
        if not cs:
            continue
        ax.text(
            j + len(cs) / 2 - 0.5, -0.75, nom, ha="center", va="bottom", fontsize=6.5
        )
        if j:
            ax.axvline(j - 0.5, color="white", lw=2.5)
        j += len(cs)
    ax.set_xlim(-0.5, len(orden) - 0.5)
    lim = set(d.agenda_ca["limite"])
    for i, r in rob.iterrows():
        v = str(r["veredicto"]) + (" · en el límite" if int(r["año"]) in lim else "")
        ax.text(
            len(orden) - 0.2,
            i,
            f"{coma(r['soporte'])}  {v}",
            va="center",
            ha="left",
            fontsize=6.5,
        )
    ax.text(
        len(orden) - 0.2,
        -0.75,
        "soporte · veredicto",
        va="bottom",
        ha="left",
        fontsize=6.5,
    )
    ax.legend(
        handles=[
            Patch(color="#33475b", label="aparece (±1 año)"),
            Patch(color="#c2185b", label="coincide con formato"),
            Patch(color=GRIS_FONDO, label="no aparece"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.0, -0.06),
        ncol=1,
        fontsize=6.3,
    )
    nota(
        fig,
        "Robusto: aparece sin breves, con peso por mes y en ambas a la vez, y en ≥ 50 % de los votantes. "
        "Artefacto probable: coincide con un cambio de formato y no sobrevive a los controles. Contraste e "
        "informativas no votan. «En el límite» (regla de la F05): no robusto con soporte ≥ 50 % y en ≥ 3 de 4 "
        "series de control.",
    )
    return fig


EVENTOS_CORTO = {  # rótulos abreviados para la figura (el texto completo está en EVENTOS)
    1985: "Palacio de Justicia / Armero",
    1989: "Asesinato de Galán",
    1991: "Constitución de 1991",
    1993: "Muerte de Escobar",
    1996: "Proceso 8000",
    1998: "Caguán (Pastrana)",
    2002: "Uribe: seguridad democrática",
    2006: "Reelección / parapolítica",
    2010: "Santos / ola invernal",
}
EVENTOS = {  # LISTA CONGELADA (copiada del 08, que la copió del 06; no editar)
    1985: "Toma del Palacio de Justicia / Armero",
    1989: "Asesinato de Galán; narcoterrorismo",
    1991: "Constitución de 1991 / apertura económica",
    1993: "Muerte de Pablo Escobar",
    1996: "Proceso 8000",
    1998: "Pastrana, zona de distensión del Caguán",
    2002: "Uribe y 'seguridad democrática'",
    2006: "Reelección de Uribe / parapolítica",
    2010: "Santos / ola invernal",
}


def etiqueta_conjunto(d, conj: str, serie: str = "eltiempo_base") -> str:
    r = d.cruces[(d.cruces["conjunto"] == conj) & (d.cruces["serie"] == serie)]
    if len(r) and pd.notna(r["vieja"].iloc[0]) and pd.notna(r["nueva"].iloc[0]):
        return f"{r['vieja'].iloc[0]} → {r['nueva'].iloc[0]}"
    vs = d.cuotas.loc[d.cuotas["conjunto"] == conj, "variante"].unique()
    return " / ".join(vs[:3]) if len(vs) else conj


def f10_linea_tiempo(d, serie="eltiempo_base"):
    ca = d.agenda_ca
    cru = d.cruces[d.cruces["serie"] == serie].copy()
    cru["cambios_l"] = cru["cambios"].map(años_de)
    cru = cru[(cru["estado"] == "cruce") | (cru["cambios_l"].str.len() > 0)].copy()
    cru["orden"] = [
        c if pd.notna(c) else min(l) for c, l in zip(cru["cruce"], cru["cambios_l"])
    ]
    cru = cru.sort_values("orden")
    filas = len(cru)
    fig, ax = plt.subplots(figsize=tamaño(2.2 + 0.5 * filas + 3.0))
    marcar_artefactos(ax, rotular=True)
    # eventos (arriba, rótulos hacia afuera del eje)
    y1 = 0
    for a, txt in EVENTOS.items():
        ax.plot(a, y1, "D", color=GRIS, ms=3.5)
        ax.text(
            a,
            y1 - 0.35,
            EVENTOS_CORTO.get(a, txt),
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=5.6,
            color="#555555",
            clip_on=False,
        )
    # agenda
    y0 = 1
    for c in ca["consenso"]:
        ax.plot(c - 0.5, y0, "o", color=TINTA, ms=6)
    for c in ca["limite"]:
        ax.plot(c - 0.5, y0, "o", mfc="white", mec=TINTA, ms=6)
    for c in ca["artefacto"]:
        ax.plot(c - 0.5, y0, "x", color=GRIS, ms=6, mew=1.2)
    ax.axhline(1.6, color=GRIS_CLARO, lw=0.5)
    # cruces de variantes
    ys = []
    for i, (_, r) in enumerate(cru.iterrows()):
        y = 2.2 + i * 0.5
        ys.append(y)
        cerca = bool(
            a_bool(pd.Series([r["cerca_formato"], r["cambio_cerca_formato"]])).any()
        )
        if r["estado"] == "cruce" and pd.notna(r["cruce"]):
            if pd.notna(r["lo"]) and pd.notna(r["hi"]):
                ax.plot(
                    [r["lo"], r["hi"]],
                    [y, y],
                    color=TINTA,
                    lw=1.4,
                    solid_capstyle="butt",
                )
            ax.plot(
                r["cruce"], y, "o", color=TINTA, mfc="white" if cerca else TINTA, ms=4.5
            )
        for c in r["cambios_l"]:
            ax.plot(c - 0.5, y, "|", color=GRIS, ms=7, mew=1.2)
    ax.set_yticks(
        [y1, y0, *ys],
        [
            "Eventos (lista congelada)",
            "Agenda ET (cambios)",
            *[_corto(etiqueta_conjunto(d, c, serie), 40) for c in cru["conjunto"]],
        ],
        fontsize=6.3,
    )
    ax.set_ylim((ys[-1] if ys else y0) + 0.5, -0.5)
    ejes_años(ax, 1985, 2011)
    ax.grid(axis="y", visible=False)
    hs = [
        Line2D([], [], marker="o", ls="", color=TINTA, ms=5, label="cambio robusto"),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc="white",
            mec=TINTA,
            ms=5,
            label="en el límite",
        ),
        Line2D(
            [], [], marker="x", ls="", color=GRIS, ms=5, label="artefacto de formato"
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="-",
            color=TINTA,
            ms=4,
            label="cruce de variantes (IC 95 %)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc="white",
            mec=TINTA,
            ms=4,
            label="cruce/cambio a ±1 de 2006",
        ),
        Line2D(
            [],
            [],
            marker="|",
            ls="",
            color=GRIS,
            ms=7,
            label="cambio en la cuota (binseg)",
        ),
    ]
    leyenda_marcas(fig, cambios=False, extra=hs)
    e = d.r8["eventos"]
    nota(
        fig,
        f"Eventos: lista del 06, congelada. Cambios de consenso a ±1 año de un evento: {e['aciertos']} de "
        f"{len(e['cambios'])}; esperado con la misma cantidad de años al azar: {coma(e['esperado_azar'])} "
        f"(P(azar ≥ obs.) = {coma(e['p_azar_ge_obs'])}). Variantes: El Tiempo base; cruce = la forma nueva supera "
        f"a la vieja 3 años seguidos tras haber estado debajo.",
    )
    return fig


# ---------------------------------------------------------------------------
# Re-derivaciones verificadas (tablas del 09 no exportadas)
# ---------------------------------------------------------------------------


def rederivar_denominador(d) -> tuple[pd.DataFrame, bool, dict]:
    """Δ Zipf por denominador (sección 2.4 del 09), desde columnas enteras del parquet.

    len(tokens_forma) = n_palabras_titulo + n_palabras_cuerpo y
    len(lemas_contenido) = n_lemas_contenido (prep.extraer_doc). No carga listas."""
    import cronolecto as cr

    cfg = cr.ConfigCrono()
    cols = [
        "fuente",
        "año",
        "es_breve",
        "es_carta",
        "n_palabras_titulo",
        "n_palabras_cuerpo",
        "n_lemas_contenido",
    ]
    df = utils.load_corpus_procesado(columns=cols)
    df = df[(df["año"] >= cfg.año_min) & (df["año"] <= cfg.año_max)]
    t = pd.DataFrame(
        {
            "fuente": df["fuente"].map(ag.CORTO).values,
            "año": df["año"].astype(int).values,
            "tok": (
                df["n_palabras_titulo"].astype(np.int64)
                + df["n_palabras_cuerpo"].astype(np.int64)
            ).values,
            "cont": df["n_lemas_contenido"].astype(np.int64).values,
            "base": (
                ~df["es_breve"].astype(bool) & ~df["es_carta"].astype(bool)
            ).values,
        }
    )
    tb = t[t["base"]].groupby(["fuente", "año"])[["tok", "cont"]].sum()
    tt = t.groupby(["fuente", "año"])[["tok", "cont"]].sum()
    den = pd.DataFrame(
        {
            "frac_contenido_base": tb["cont"] / tb["tok"].clip(lower=1),
            "frac_contenido_todos": tt["cont"] / tt["tok"].clip(lower=1),
            "tokens_base": tb["tok"],
        }
    ).reset_index()
    den["desplazamiento_zipf_base"] = np.log10(1 / den["frac_contenido_base"])
    den["desplazamiento_zipf_todos"] = np.log10(1 / den["frac_contenido_todos"])
    obs = (
        den.groupby("fuente")["desplazamiento_zipf_base"]
        .agg(["min", "max"])
        .round(3)
        .to_dict("index")
    )
    ref = d.r9["rareza"]["desplazamiento_zipf_denominador"]
    ok = all(abs(obs[f][k] - ref[f][k]) <= 0.001 for f in ref for k in ("min", "max"))
    return den, ok, {"obtenido": obs, "resumen_09": ref}


def rederivar_ic(d) -> tuple[pd.DataFrame | None, bool, dict]:
    """Semiancho relativo Poisson vs. bootstrap (sección 2.2 del 09). Copia fiel del 09:
    misma matriz de formas, mismos años de demostración, mismas semillas.
    ⏱ ~2 min (matriz documento × forma de 91k documentos)."""
    import cronolecto as cr
    from scipy import stats

    cfg = cr.ConfigCrono()
    dfp = utils.load_corpus_procesado(
        columns=["fuente", "año", "es_breve", "es_carta", "tokens_forma"]
    )
    dfp = dfp[(dfp["año"] >= cfg.año_min) & (dfp["año"] <= cfg.año_max)].reset_index(
        drop=True
    )
    fuente = dfp["fuente"].values
    año = dfp["año"].values.astype(int)
    base = ~dfp["es_breve"].values & ~dfp["es_carta"].values
    L_forma = dfp["tokens_forma"].map(len).values.astype(np.int64)
    Xf, vf = cr.matriz_conteos(dfp["tokens_forma"], min_df=cfg.min_df_formas)
    SERIES = {
        "eltiempo_base": (fuente == ag.ET) & base & (año >= cfg.et_desde),
        "semana_base": (fuente == ag.SEMANA) & base,
    }
    NG = pd.DataFrame(
        {
            "fuente": [ag.CORTO[f] for f in fuente],
            "año": año,
            "tokens": L_forma,
            "base": base,
        }
    )
    NG = (
        NG[NG.base]
        .groupby(["fuente", "año"])
        .agg(docs=("tokens", "size"), tokens=("tokens", "sum"))
        .reset_index()
    )

    def demo_ic(serie, y, nombre):
        idx = np.where(SERIES[serie] & (año == y))[0]
        if len(idx) == 0:
            return None
        Xg, Lg = Xf[idx], L_forma[idx]
        n_t = np.asarray(Xg.sum(0)).ravel()
        pres = np.where(n_t > 0)[0]
        bins = pd.cut(
            n_t[pres],
            [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 10**9],
            right=False,
            labels=False,
        )
        rg = np.random.default_rng(cfg.seed)
        por_bin = max(1, cfg.n_terminos_demo // 10)
        sel = np.concatenate(
            [
                rg.choice(
                    pres[bins == b], min(por_bin, (bins == b).sum()), replace=False
                )
                for b in np.unique(bins[~np.isnan(bins)])
            ]
        )
        lo, hi, _ = cr.bootstrap_tasa(Xg[:, sel], Lg, cfg.b_boot, cfg.seed)
        ar = cr.ancho_relativo(n_t[sel], Lg.sum(), lo, hi)
        ar["termino"] = vf[sel]
        ar["dp"] = cr.dispersion_dp(Xg[:, sel], Lg)["dp"].values
        ar["grupo"] = nombre
        ar["fuente"] = serie.split("_")[0]
        return ar

    et = NG[NG.fuente == "eltiempo"].sort_values("tokens")
    y_et = int(et.iloc[len(et) // 2]["año"])
    sem = NG[(NG.fuente == "semana") & (NG.año < 1990)].sort_values("tokens")
    y_se = (
        int(sem.iloc[0]["año"])
        if len(sem)
        else int(NG[NG.fuente == "semana"].sort_values("tokens").iloc[0]["año"])
    )
    AR = pd.concat(
        [
            x
            for x in [
                demo_ic("eltiempo_base", y_et, f"El Tiempo {y_et}"),
                demo_ic("semana_base", y_se, f"Semana {y_se}"),
            ]
            if x is not None
        ]
    )
    cmin = cr.conteo_minimo(AR[AR.fuente == "eltiempo"], umbral=cfg.hw_rel_max)[0]
    obs = {
        "conteo_minimo_eltiempo": cmin,
        "cociente_ic_boot_poisson_mediana": float(AR["cociente"].median()),
        "cociente_ic_boot_poisson_mediana_n10": float(
            AR[AR["n"] >= 10]["cociente"].median()
        ),
        "spearman_cociente_dp": float(
            stats.spearmanr(AR["cociente"], AR["dp"], nan_policy="omit").statistic
        ),
    }
    ref = {k: d.r9["rareza"][k] for k in obs}
    ok = obs["conteo_minimo_eltiempo"] == ref["conteo_minimo_eltiempo"] and all(
        abs(obs[k] - ref[k]) <= 1e-6 * max(1, abs(ref[k]))
        for k in obs
        if k != "conteo_minimo_eltiempo"
    )
    AR.attrs.update({"y_et": y_et, "y_se": y_se, "hw_rel_max": cfg.hw_rel_max})
    return AR, ok, {"obtenido": obs, "resumen_09": ref}


# ---------------------------------------------------------------------------
# Figuras del 09
# ---------------------------------------------------------------------------


def piso_zipf_grupos(d) -> tuple[pd.DataFrame, dict]:
    """Piso Zipf = log10(1e6 / N) + 3 por fuente×año, con N de pmw_zipf (base sin breves)."""
    pz = pd.read_parquet(
        d.R9 / "pmw_zipf.parquet", columns=["fuente", "año", "N_grupo"]
    ).drop_duplicates()
    pz["piso_zipf"] = np.log10(1e6 / pz["N_grupo"]) + 3
    pz["rara_medible"] = pz["piso_zipf"] < 3.0
    obs = sorted(
        f"{f} {a}" for f, a in pz.loc[~pz["rara_medible"], ["fuente", "año"]].values
    )
    ref = sorted(d.r9["rareza"]["grupos_sin_banda_rara_medible"])
    return pz.sort_values(["fuente", "año"]), {
        "coincide_con_resumen_09": obs == ref,
        "n_no_medibles": len(obs),
    }


def f11_piso_zipf(d, pz: pd.DataFrame, den: pd.DataFrame | None):
    n = 2 if den is not None else 1
    fig, ax = plt.subplots(1, n, figsize=tamaño(6.8), squeeze=False)
    ax = ax.ravel()
    a = ax[0]
    marcar_artefactos(a, rotular=True)
    a.axhspan(0, 3, color=GRIS_FONDO, lw=0, zorder=0)
    a.axhline(3, color=GRIS, ls="--", lw=0.7)
    a.axhline(4, color=GRIS, ls=":", lw=0.7)
    a.text(1980.2, 2.9, "banda rara (< 3)", va="top", fontsize=6.3, color="#555555")
    a.text(1980.2, 3.95, "media 3–4", va="top", fontsize=6.3, color="#555555")
    for f in ["eltiempo", "semana", "dinero"]:
        g = pz[pz["fuente"] == f]
        a.plot(
            g["año"], g["piso_zipf"], "-", color=FUENTE[f], lw=1, label=NOMBRE_FUENTE[f]
        )
        nm = g[~g["rara_medible"]]
        a.plot(
            nm["año"],
            nm["piso_zipf"],
            "o",
            color=FUENTE[f],
            ms=2.6,
            mfc="white",
            mew=0.8,
        )
        a.plot(
            g.loc[g["rara_medible"], "año"],
            g.loc[g["rara_medible"], "piso_zipf"],
            "o",
            color=FUENTE[f],
            ms=2.6,
        )
    a.set_ylim(2.4, max(5.3, pz["piso_zipf"].max() + 0.1))
    a.set_ylabel("Zipf de una sola aparición")
    a.set_title("(a) Piso medible del Zipf por fuente y año")
    a.legend(loc="upper right", fontsize=6)
    ejes_años(a, 1980, 2011)
    if den is not None:
        a = ax[1]
        marcar_artefactos(a)
        for f in ["eltiempo", "semana", "dinero"]:
            g = den[den["fuente"] == f].sort_values("año")
            a.plot(
                g["año"],
                g["desplazamiento_zipf_base"],
                "-",
                color=FUENTE[f],
                lw=1,
                label=f"{NOMBRE_FUENTE[f]} base",
            )
            a.plot(
                g["año"], g["desplazamiento_zipf_todos"], ":", color=FUENTE[f], lw=0.9
            )
        a.set_ylabel("Δ Zipf (contenido − todos)")
        a.set_title("(b) Efecto del denominador")
        a.legend(
            handles=[
                *a.get_legend_handles_labels()[0],
                Line2D([], [], color=GRIS, ls=":", label="con breves"),
            ],
            loc="lower left",
            fontsize=6,
        )
        ejes_años(a, 1980, 2011)
    leyenda_marcas(fig, cambios=False)
    nota(
        fig,
        "Formas de tokens_forma, base sin breves ni cartas. Círculo hueco: año en que la banda rara (Zipf < 3) no se "
        "puede medir porque una sola aparición ya vale ≥ 3. (b) Δ Zipf = log10(N_tokens / N_contenido): al cambiar "
        "el denominador a lemas de contenido el Zipf sube ≈ 0,4 y varía por año y fuente.",
    )
    return fig


def f12_ic_poisson_bootstrap(d, AR: pd.DataFrame):
    hw = AR.attrs.get("hw_rel_max", 0.5)
    cmin = d.r9["rareza"]["conteo_minimo_eltiempo"]
    fig, ax = plt.subplots(1, 2, figsize=tamaño(6.2))
    a = ax[0]
    for f, g in AR.groupby("fuente"):
        a.scatter(
            g["n"],
            g["hw_rel_boot"],
            s=2,
            alpha=0.35,
            color=FUENTE[f],
            lw=0,
            label=f"{g['grupo'].iloc[0]} (bootstrap)",
        )
    ss = AR.sort_values("n")
    a.plot(ss["n"], ss["hw_rel_poisson"], color=TINTA, lw=1, label="Poisson (Garwood)")
    a.axhline(hw, ls="--", color=GRIS, lw=0.8)
    a.axvline(cmin, ls=":", color=FUENTE["eltiempo"], lw=0.9)
    a.text(
        cmin * 1.08,
        hw * 1.15,
        f"conteo mínimo ET ≈ {cmin}",
        fontsize=6.3,
        color=FUENTE["eltiempo"],
    )
    a.set(
        xscale="log",
        yscale="log",
        xlabel="apariciones en el año",
        ylabel="semiancho relativo del IC 95 %",
    )
    a.set_title("(a) Incertidumbre vs. conteo")
    a.legend(loc="lower left", fontsize=6, markerscale=3)
    a = ax[1]
    A10 = AR[AR["n"] >= 10]
    bins = [10, 20, 50, 100, 200, 500, 10**9]
    for f, g in A10.groupby("fuente"):
        med = g.groupby(pd.cut(g["n"], bins, right=False), observed=True)[
            "cociente"
        ].median()
        x = [b.left for b in med.index]
        a.plot(x, med.values, "o-", color=FUENTE[f], ms=3, label=g["grupo"].iloc[0])
    a.axhline(1, color=GRIS, lw=0.7)
    a.set(
        xscale="log",
        xlabel="apariciones en el año (inicio del bin)",
        ylabel="ancho bootstrap / ancho Poisson",
    )
    a.set_title("(b) Cuánto subestima Poisson (mediana, n ≥ 10)")
    a.legend(loc="upper left", fontsize=6)
    nota(
        fig,
        f"Bootstrap de Poisson por documento (B = 1000). Regla declarada en el 09: conteo mínimo = menor bin con "
        f"mediana del semiancho relativo bootstrap ≤ {coma(hw, 1)}. Re-derivado con el código y las semillas del 09 y "
        f"verificado contra resumen_09.json.",
    )
    return fig


def f13_zipf_frecuencia_documental(d, n_pares: int = 3):
    """Regla de anotación (la misma del 09, declarada allí): Zipf redondeado a 0,1 entre 3 y 4,5;
    en cada valor, la más repartida y la más concentrada; pares con razón de df ≥ 5×.
    Se anotan los `n_pares` de mayor razón."""
    D = d.disp[a_bool(d.disp["contenido"]) & (d.disp["n"] >= 50)].copy()
    D["zr"] = D["zipf"].round(1)
    pares = []
    for z, g in D[D["zipf"].between(3, 4.5)].groupby("zr"):
        if len(g) >= 2:
            a_, b_ = g.loc[g["df_por_mil"].idxmax()], g.loc[g["df_por_mil"].idxmin()]
            if a_["df_por_mil"] >= 5 * b_["df_por_mil"]:
                pares.append((a_["df_por_mil"] / b_["df_por_mil"], a_, b_))
    pares = sorted(pares, key=lambda x: -x[0])[:n_pares]
    fig, ax = plt.subplots(
        1, 2, figsize=tamaño(6.8), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    a = ax[0]
    a.hexbin(
        D["zipf"],
        np.log10(D["df_por_mil"]),
        gridsize=45,
        cmap="Greys",
        mincnt=1,
        bins="log",
        lw=0,
    )
    for k, (r, rep, con) in enumerate(pares):
        c = PALETA_VARIANTES[1:][k % 3]
        for w, dy in [(rep, 8), (con, -10)]:
            a.plot(w["zipf"], np.log10(w["df_por_mil"]), "o", color=c, ms=3.5)
            a.annotate(
                w["termino"],
                (w["zipf"], np.log10(w["df_por_mil"])),
                xytext=(4, dy),
                textcoords="offset points",
                fontsize=6.3,
                color=c,
            )
        a.plot(
            [rep["zipf"], con["zipf"]],
            np.log10([rep["df_por_mil"], con["df_por_mil"]]),
            color=c,
            lw=0.7,
        )
        a.text(
            rep["zipf"] + 0.03,
            np.log10(np.sqrt(rep["df_por_mil"] * con["df_por_mil"])),
            f"×{r:.0f}",
            fontsize=6,
            color=c,
        )
    a.set_xlabel("Zipf (El Tiempo base)")
    a.set_ylabel("log10 docs por mil")
    a.set_title("(a) Mismo Zipf, difusión muy distinta")
    a = ax[1]
    a.scatter(D["zipf"], D["dp"], s=2, alpha=0.25, color="0.35", lw=0, rasterized=True)
    med = D.groupby(D["zipf"].round(1))["dp"].median()
    a.plot(med.index, med.values, color="black", lw=1, label="mediana por Zipf")
    a.set_ylim(0, 1.02)
    a.legend(loc="lower left", fontsize=6)
    a.set_xlabel("Zipf (El Tiempo base)")
    a.set_ylabel("DP de Gries (documentos como partes)")
    a.set_title("(b) La DP se satura")
    nota(
        fig,
        f"Formas de contenido con ≥ 50 apariciones en El Tiempo base. Pares anotados: regla del 09 (mismo Zipf "
        f"a 0,1 entre 3 y 4,5; la más repartida y la más concentrada; razón de df ≥ 5×), los {len(pares)} de mayor razón.",
    )
    return fig


F14_CONJUNTOS = [
    ("santafe", "Santafé de Bogotá → Bogotá"),
    ("reinsertados", "reinsertado → desmovilizado"),
    ("paramilitares", "paramilitares / autodefensas / bacrim"),
    ("insurgencia", "insurgencia / subversión"),
    ("tercera_edad", "tercera edad → adulto mayor"),
    ("periodo", "período → periodo"),
]


def _variantes_a_dibujar(d, conj, serie) -> list[str]:
    """Regla declarada: si hay hipótesis de cruce, la forma nueva; si son 2 variantes sin
    hipótesis, la minoritaria (menor cuota media: es la que muestra el reemplazo, p. ej.
    «santafé de bogotá»); si son más, todas (máx. 4 por n total)."""
    q = d.cuotas[(d.cuotas["conjunto"] == conj) & (d.cuotas["serie"] == serie)]
    r = d.cruces[(d.cruces["conjunto"] == conj) & (d.cruces["serie"] == serie)]
    if len(r) and pd.notna(r["nueva"].iloc[0]):
        return [r["nueva"].iloc[0]]
    vs = q.groupby("variante")["n"].sum().sort_values(ascending=False).index.tolist()
    if len(vs) == 2:
        m = q.dropna(subset=["cuota"]).groupby("variante")["cuota"].mean()
        return [m.idxmin()] if len(m) else vs[:1]
    return vs[:4]


def _marcar_cruce(a, d, conj, serie, color, y_txt=0.97):
    r = d.cruces[(d.cruces["conjunto"] == conj) & (d.cruces["serie"] == serie)]
    if not len(r):
        return
    r = r.iloc[0]
    if r["estado"] == "cruce" and pd.notna(r["cruce"]):
        a.axvline(r["cruce"], color=color, ls=":", lw=0.9)
        if pd.notna(r["lo"]) and pd.notna(r["hi"]):
            a.plot(
                [r["lo"], r["hi"]],
                [y_txt, y_txt],
                color=color,
                lw=2,
                transform=a.get_xaxis_transform(),
                solid_capstyle="butt",
                clip_on=False,
            )
    for c in años_de(r["cambios"]):
        a.plot(
            c - 0.5,
            1.0,
            "v",
            color=color,
            ms=3.5,
            transform=a.get_xaxis_transform(),
            clip_on=False,
        )
    if a_bool(pd.Series([r["cerca_formato"], r["cambio_cerca_formato"]])).any():
        a.text(
            0.98,
            0.04,
            "a ±1 de 2006",
            transform=a.transAxes,
            ha="right",
            fontsize=6,
            color=color,
            bbox={"fc": "white", "ec": color, "lw": 0.5, "pad": 1.5},
        )


def f14_cuotas_variantes(d):
    fig, axs = plt.subplots(4, 2, figsize=tamaño(19.5))
    axs = axs.ravel()
    for a, (conj, titulo) in zip(axs, F14_CONJUNTOS):
        s = "eltiempo_base"
        q = d.cuotas[(d.cuotas["conjunto"] == conj) & (d.cuotas["serie"] == s)].dropna(
            subset=["cuota"]
        )
        vs = _variantes_a_dibujar(d, conj, s)
        marcar_artefactos(a)
        for k, v in enumerate(vs):
            g = completar_años(q[q["variante"] == v], 1990, 2011)
            c = PALETA_VARIANTES[k] if len(vs) > 1 else FUENTE["eltiempo"]
            a.fill_between(g["año"], g["lo"], g["hi"], color=c, alpha=0.2, lw=0)
            a.plot(g["año"], g["cuota"], "-", color=c, lw=1.1, label=v)
        _marcar_cruce(a, d, conj, s, TINTA)
        a.set_title(titulo, fontsize=7)
        a.legend(
            loc="best",
            fontsize=5.8,
            title="cuota de",
            title_fontsize=5.8,
            frameon=True,
            framealpha=0.85,
            facecolor="white",
            edgecolor="none",
        )
        if len(vs) == 1 and len(q) and q.loc[q["variante"] == vs[0], "hi"].max() < 0.3:
            a.set_ylim(0, None)  # forma minoritaria: escala propia
            a._f14_libre = True
    # sólo/solo: El Tiempo vs Semana
    a = axs[len(F14_CONJUNTOS)]
    marcar_artefactos(a)
    for s, f in [("eltiempo_base", "eltiempo"), ("semana_base", "semana")]:
        g = d.cuotas[
            (d.cuotas["conjunto"] == "solo_tilde")
            & (d.cuotas["serie"] == s)
            & (d.cuotas["variante"] == "solo")
        ]
        g = completar_años(g.dropna(subset=["cuota"]), 1980, 2011)
        a.fill_between(g["año"], g["lo"], g["hi"], color=FUENTE[f], alpha=0.2, lw=0)
        a.plot(g["año"], g["cuota"], color=FUENTE[f], lw=1.1, label=NOMBRE_FUENTE[f])
        _marcar_cruce(a, d, "solo_tilde", s, FUENTE[f])
    a.set_title("sólo → solo: cuota de «solo»", fontsize=7)
    a.legend(loc="center left", fontsize=6)
    # éste / sólo con tilde (pmw)
    a = axs[len(F14_CONJUNTOS) + 1]
    marcar_artefactos(a)
    for s, f in [("eltiempo_base", "eltiempo"), ("semana_base", "semana")]:
        for t, ls in [("éste/ésta (tilde)", "-"), ("sólo (tilde)", ":")]:
            g = completar_años(
                d.tildes[(d.tildes["serie"] == s) & (d.tildes["termino"] == t)].dropna(
                    subset=["pmw"]
                ),
                1980,
                2011,
            )
            if ls == "-":
                a.fill_between(
                    g["año"], g["lo"], g["hi"], color=FUENTE[f], alpha=0.15, lw=0
                )
            a.plot(g["año"], g["pmw"], ls, color=FUENTE[f], lw=1)
    a.set_title("formas tildadas (pmw): éste/ésta (—) y sólo (···)", fontsize=7)
    a.legend(
        handles=[
            Line2D([], [], color=FUENTE[f], label=NOMBRE_FUENTE[f])
            for f in ("eltiempo", "semana")
        ],
        loc="upper right",
        fontsize=6,
    )
    for k, a in enumerate(axs):
        ejes_años(
            a,
            1990 if k < len(F14_CONJUNTOS) else 1980,
            2011,
            5 if k < len(F14_CONJUNTOS) else 10,
        )
    leyenda_marcas(fig, cambios=False)
    for a in axs[:-1]:
        if not getattr(a, "_f14_libre", False):
            a.set_ylim(0, 1)
    marcar_artefactos(axs[0], rotular=True)
    fig.supylabel("cuota dentro del conjunto (pmw en el último panel)", fontsize=7)
    nota(
        fig,
        "Base sin breves ni cartas; IC 95 % por bootstrap de documentos. Línea punteada vertical: año de cruce (la "
        "forma nueva supera a la vieja 3 años seguidos tras haber estado debajo), con su IC en la barra superior. "
        "▼ cambios de la cuota (binseg sobre √cuota). Recuadro: cruce o cambio a ±1 año de 2006. Las formas "
        "minoritarias (p. ej. «santafé de bogotá») van con escala propia. Los términos del conflicto son marcos, "
        "no sinónimos.",
    )
    return fig


PAZ_ORDEN = [
    "paz",
    "proceso de paz",
    "diálogo",
    "negociación",
    "zona de distensión",
    "acuerdo humanitario",
    "desmovilización",
    "reconciliación",
    "acuerdo de paz",
    "posconflicto",
]


def f15_paz_lexica(d):
    terms = [t for t in PAZ_ORDEN if t in set(d.paz["termino"])]
    terms += sorted(set(d.paz["termino"]) - set(terms))
    nrow = int(np.ceil(len(terms) / 2))
    fig, axs = plt.subplots(nrow, 2, figsize=tamaño(3.4 * nrow + 0.8), sharex=True)
    axs = axs.ravel()
    for a, t in zip(axs, terms):
        marcar_artefactos(a)
        for s, f in [("eltiempo_base", "eltiempo"), ("semana_base", "semana")]:
            g = completar_años(
                d.paz[(d.paz["serie"] == s) & (d.paz["termino"] == t)].dropna(
                    subset=["pmw"]
                ),
                1980,
                2011,
            )
            a.fill_between(g["año"], g["lo"], g["hi"], color=FUENTE[f], alpha=0.2, lw=0)
            a.plot(g["año"], g["pmw"], color=FUENTE[f], lw=1, label=NOMBRE_FUENTE[f])
        a.set_title(f"«{t}»", fontsize=7)
        a.tick_params(labelbottom=True)
        a.set_ylim(bottom=0)
        ejes_años(a, 1980, 2011, 10)
    for a in axs[len(terms) :]:
        a.axis("off")
    marcar_artefactos(axs[0], rotular=True)
    axs[0].legend(loc="upper left", fontsize=6)
    leyenda_marcas(fig, cambios=False)
    fig.supylabel("apariciones por millón de tokens (IC 95 %)", fontsize=7)
    nota(
        fig,
        "Base sin breves ni cartas. Conteo de la forma (o expresión) en tokens_forma; IC por bootstrap de documentos. "
        "Cerca de la mitad de «paz» cae en el tópico residual T10, por eso la agenda de paz se lee aquí y no en el 08.",
    )
    return fig


F16_PALABRAS = [
    "teléfono",
    "celular",
    "banda",
    "víctima",
    "desplazado",
    "negociación",
    "paz",
]


def f16_vecinos_estables(d, k: int = 5):
    v = d.vecinos.dropna(subset=["rango"]).copy()
    v = v[v["palabra"].isin(F16_PALABRAS)].sort_values("rango")
    v = v.drop_duplicates(["palabra", "periodo", "vecino"])
    periodos = [p for p in ["P1", "P2", "P3", "P4"] if p in set(v["periodo"])]
    rot_p = {
        "P1": "P1 1990–94",
        "P2": "P2 1995–99",
        "P3": "P3 2000–05",
        "P4": "P4 2006–11",
    }
    palabras = [p for p in F16_PALABRAS if p in set(v["palabra"])]
    fig, ax = _figura_tabla(tamaño(1.55 * len(palabras) + 1.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(len(palabras) + 0.2, -0.6)
    x0, w = 0.14, (1 - 0.14) / max(len(periodos), 1)
    fs = mpl.rcParams["font.size"] - 0.5
    for j, p in enumerate(periodos):
        ax.text(
            x0 + j * w, -0.25, rot_p.get(p, p), fontsize=fs, weight="bold", va="center"
        )
    ax.plot([0, 1], [0.0, 0.0], color=TINTA, lw=0.6)
    for i, pal in enumerate(palabras):
        y = i + 0.5
        ax.text(0, y, pal, fontsize=fs, weight="bold", va="center")
        for j, p in enumerate(periodos):
            g = v[(v["palabra"] == pal) & (v["periodo"] == p)].head(k)
            txt = "\n".join(x.replace("_", " ") for x in g["vecino"]) if len(g) else "—"
            ax.text(
                x0 + j * w,
                y,
                txt,
                fontsize=fs - 0.8,
                va="center",
                linespacing=1.05,
                color=TINTA if len(g) else GRIS,
            )
        ax.plot([0, 1], [i + 1, i + 1], color=GRIS_CLARO, lw=0.4)
    nota(
        fig,
        f"Top-{k} vecinos que aparecen en el top-25 de ≥ 3 réplicas bootstrap de documentos (word2vec SGNS, mismo "
        "número de tokens por período, sección 4b del 09). El estado de cambio por palabra es frágil; la evidencia "
        "está en estos vecinos.",
    )
    return fig


def f17_neologismos(d):
    import cronolecto as cr

    per = cr.ConfigCrono().periodos
    c = d.cat[d.cat["categoria"].astype(str).str.strip() != ""].copy()
    n_art = int((c["categoria"] == "artefacto").sum())
    c = c[c["categoria"] != "artefacto"]
    c["periodo"] = cr.periodo_de(c["año"].astype(int).values, per)
    cats = c["categoria"].value_counts().index.tolist()
    nomp = [p for p, _, _ in per]
    tab = c.groupby(["categoria", "tipo", "periodo"]).size()
    ncol = 3
    nrow = int(np.ceil(len(cats) / ncol))
    fig, axs = plt.subplots(
        nrow, ncol, figsize=tamaño(3.5 * nrow + 1.2), sharex=True, sharey=True
    )
    axs = axs.ravel()
    x = np.arange(len(nomp))
    for a, cat in zip(axs, cats):
        neo = [tab.get((cat, "neologismo", p), 0) for p in nomp]
        obs = [tab.get((cat, "obsolescencia", p), 0) for p in nomp]
        a.bar(x, neo, color=NEO, width=0.7)
        a.bar(x, [-o for o in obs], color=OBS, width=0.7)
        a.axhline(0, color=TINTA, lw=0.5)
        a.set_title(cat, fontsize=7)
        a.set_xticks(x, nomp)
        a.grid(axis="x", visible=False)
    for a in axs[len(cats) :]:
        a.axis("off")
    lim = max(1, int(tab.max()))
    axs[0].set_ylim(-lim - 0.5, lim + 0.5)
    axs[0].yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f"{abs(int(v))}")
    )
    axs[0].legend(
        handles=[
            Patch(color=NEO, label="neologismos (↑)"),
            Patch(color=OBS, label="obsolescencias (↓)"),
        ],
        loc="upper left",
        fontsize=6,
    )
    rp = ", ".join(f"{p} {a}–{b}" for p, a, b in per)
    nota(
        fig,
        f"Regla de uso sostenido del 09 (umbral 5 pmw, ventana de 5 años), El Tiempo base; categorías a mano en "
        f"categorias_lexico.csv. Se excluyen {n_art} artefactos de archivo o procesamiento (variantes de ü, tildes, "
        f"alias/ext). Períodos: {rp}; el año es el primer (o último) año de uso sostenido.",
    )
    return fig
