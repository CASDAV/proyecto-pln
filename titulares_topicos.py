"""Titulares representativos por tópico (insumo para el informe E3).
Uso, desde la raíz del repo:  uv run python titulares_topicos.py > titulares.txt
Solo lee archivos; no escribe nada.

Criterio (declarado): misma población del ajuste del 08 (no breves, no cartas);
top-5 por θ_k; empates de θ se rompen por longitud del cuerpo (más larga primero).
"""

import pandas as pd

th = pd.read_parquet("results/08/theta_documentos.parquet")
cp = pd.read_parquet(
    "data/corpus_procesado.parquet",
    columns=["id", "fecha", "titulo", "n_palabras_cuerpo"],
)
df = th.merge(cp, on="id", how="left")
df = df[~df["es_breve"] & ~df["es_carta"] & df["ok"]]
print(f"población: {len(df):,} docs (no breves, no cartas)")

corto = {
    "www.eltiempo.com": "ET",
    "www.semana.com": "Semana",
    "www.dinero.com": "Dinero",
}
tcols = [c for c in th.columns if c.startswith("T") and c[1:3].isdigit()]
for t in tcols:
    top = df.sort_values([t, "n_palabras_cuerpo"], ascending=False).head(5)
    print(f"\n== {t}")
    for _, r in top.iterrows():
        print(
            f"{r[t]:.2f} | {corto.get(r['fuente'], r['fuente'])} {pd.Timestamp(r['fecha']):%Y-%m} "
            f"| {r['n_palabras_cuerpo']:>5} pal. | id {r['id']} | {r['titulo']}"
        )
