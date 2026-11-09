"""Qué tópicos cambian en 1995 y 2006 (insumo para el informe E3, sección 5.1).
Uso, desde la raíz del repo:  uv run python cambios_por_topico.py > cambios_topicos.txt
Solo lee results/08/prevalencias_ic.csv; no escribe nada.

Criterio: serie principal de la F05 (El Tiempo, sin breves, igual peso por mes).
Para cada cambio se compara la media de los 2 años previos con la de los 2 años
siguientes (min_seg = 2, como el detector). Se reporta Δ en puntos porcentuales y
si los IC 95 % de los dos años contiguos (año−1 y año) no se solapan.
"""
import pandas as pd

P = pd.read_csv("results/08/prevalencias_ic.csv")
print("series disponibles:", sorted(P["serie"].unique()))
serie = next(s for s in P["serie"].unique()
             if "eltiempo" in s and "sin_breves" in s and s.endswith("mes"))
print("serie usada:", serie)
S = P[P["serie"] == serie]
W = S.pivot(index="año", columns="topico", values="prev")
LO = S.pivot(index="año", columns="topico", values="lo")
HI = S.pivot(index="año", columns="topico", values="hi")

for c in (1995, 2006):
    antes, despues = [c - 2, c - 1], [c, c + 1]
    d = (W.loc[despues].mean() - W.loc[antes].mean()) * 100
    sep = (LO.loc[c] > HI.loc[c - 1]) | (HI.loc[c] < LO.loc[c - 1])
    t = pd.DataFrame({
        "prev_antes_%": W.loc[antes].mean() * 100,
        "prev_despues_%": W.loc[despues].mean() * 100,
        "delta_pp": d,
        f"IC_{c-1}_vs_{c}_separados": sep,
    }).sort_values("delta_pp", key=abs, ascending=False)
    print(f"\n== Cambio {c}: media {antes} frente a {despues}")
    print(t.round(2).to_string())
