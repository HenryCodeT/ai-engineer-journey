"""Genera datasets sintéticos y reproducibles para practicar (no requieren internet).

    python data/generar_datos.py

- viviendas_lima.csv  → EDA, faltantes, outliers, encoding, selección de variables (cursos 01-02)
- clientes_banco.csv  → clasificación desbalanceada (módulo 4)
- canasta_compras.csv → market basket analysis (clase 25)

Los problemas de calidad están puestos a propósito: nulos con distintos mecanismos
(MCAR / MAR / MNAR), outliers, una columna casi constante y variables correlacionadas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 42
SALIDA = Path(__file__).resolve().parent / "raw"

DISTRITOS = {  # distrito: (precio base USD/m2, distancia al centro km)
    "Miraflores": (2300, 9), "San Isidro": (2500, 8), "Barranco": (2100, 12),
    "Surco": (1800, 13), "La Molina": (1700, 16), "San Borja": (1900, 10),
    "Jesús María": (1600, 4), "Lince": (1500, 5), "Pueblo Libre": (1450, 6),
    "San Miguel": (1400, 9), "Magdalena": (1550, 7), "Los Olivos": (950, 14),
    "SJL": (800, 13), "Ate": (850, 15), "Chorrillos": (1200, 15),
}


def generar_viviendas(n: int = 2000, rng: np.random.Generator | None = None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(SEMILLA)
    nombres = list(DISTRITOS)
    pesos = np.array([8, 6, 5, 10, 6, 6, 7, 6, 5, 7, 5, 9, 8, 6, 6], dtype=float)
    distrito = rng.choice(nombres, size=n, p=pesos / pesos.sum())
    tipo = rng.choice(["Departamento", "Casa", "Oficina"], size=n, p=[0.68, 0.24, 0.08])

    area = np.where(tipo == "Casa", rng.normal(180, 50, n),
                    np.where(tipo == "Oficina", rng.normal(90, 30, n), rng.normal(85, 25, n)))
    area = np.clip(area, 30, None).round(1)
    habitaciones = np.clip(np.round(area / 35 + rng.normal(0, 0.7, n)), 1, 7).astype(int)
    habitaciones = np.where(tipo == "Oficina", 0, habitaciones)
    banos = np.clip(np.round(habitaciones * 0.7 + rng.normal(0.3, 0.5, n)), 1, 5).astype(int)
    antiguedad = np.clip(rng.gamma(2.0, 6.0, n), 0, 60).round().astype(int)
    piso = np.where(tipo == "Casa", np.nan, np.clip(rng.geometric(0.18, n), 1, 25)).astype(float)
    cochera = np.where(rng.random(n) < 0.35 + 0.002 * area, "Si", "No")
    estado = np.where(antiguedad < 2, "Nuevo",
                      rng.choice(["Bueno", "Regular", "Remodelar"], size=n, p=[0.55, 0.32, 0.13]))
    estado = np.where((antiguedad < 2), "Nuevo", estado)
    base = np.array([DISTRITOS[d][0] for d in distrito])
    distancia = np.array([DISTRITOS[d][1] for d in distrito]) + rng.normal(0, 1.5, n)
    distancia = np.clip(distancia, 0.5, None).round(2)

    factor_estado = pd.Series(estado).map({"Nuevo": 1.15, "Bueno": 1.0, "Regular": 0.88, "Remodelar": 0.72}).to_numpy()
    precio = (area * base * factor_estado * (1 - 0.004 * antiguedad)
              * np.where(cochera == "Si", 1.06, 1.0) * rng.lognormal(0, 0.12, n))
    precio = precio.round(-2)

    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "distrito": distrito,
        "tipo_inmueble": tipo,
        "area_m2": area,
        "area_construida_m2": (area * rng.uniform(0.92, 0.99, n)).round(1),  # muy correlacionada con area_m2
        "habitaciones": habitaciones,
        "banos": banos,
        "antiguedad_anios": antiguedad.astype(float),
        "piso": piso,
        "cochera": cochera,
        "estado": estado,
        "distancia_centro_km": distancia,
        "pais": np.where(rng.random(n) < 0.997, "Peru", "PE"),  # casi constante (baja varianza)
        "precio_usd": precio,
    })

    # --- Problemas de calidad a propósito -------------------------------------------------
    # MCAR: antigüedad faltante al azar (8%)
    df.loc[rng.random(n) < 0.08, "antiguedad_anios"] = np.nan
    # MAR: distancia falta más en distritos periféricos (depende de otra variable observada)
    periferia = df["distrito"].isin(["SJL", "Ate", "Los Olivos", "Chorrillos"])
    df.loc[periferia & (rng.random(n) < 0.25), "distancia_centro_km"] = np.nan
    # MNAR: los dueños de inmuebles caros ocultan el número de baños
    caro = df["precio_usd"] > df["precio_usd"].quantile(0.85)
    df.loc[caro & (rng.random(n) < 0.30), "banos"] = np.nan
    # Categórica con nulos y texto sucio
    df.loc[rng.random(n) < 0.04, "estado"] = np.nan
    sucios = rng.random(n) < 0.03
    df.loc[sucios, "cochera"] = rng.choice(["si", "SI", "no ", "N/A"], size=sucios.sum())
    # Outliers: errores de digitación (área ×10) y una mansión real
    idx_err = rng.choice(n, size=12, replace=False)
    df.loc[idx_err, "area_m2"] = (df.loc[idx_err, "area_m2"] * 10).round(1)
    idx_lujo = rng.choice(n, size=5, replace=False)
    df.loc[idx_lujo, "precio_usd"] = df.loc[idx_lujo, "precio_usd"] * rng.uniform(5, 9, 5)
    df.loc[rng.choice(n, size=3, replace=False), "antiguedad_anios"] = 150  # imposible
    df["banos"] = df["banos"].astype("Int64")
    # Filas duplicadas
    df = pd.concat([df, df.sample(15, random_state=SEMILLA)], ignore_index=True)
    return df


def generar_clientes_banco(n: int = 5000, rng: np.random.Generator | None = None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(SEMILLA + 1)
    edad = np.clip(rng.normal(41, 12, n), 18, 85).round().astype(int)
    ingreso = np.round(rng.lognormal(8.0, 0.5, n), -1)
    antig = np.clip(rng.exponential(4, n), 0, 30).round(1)
    productos = rng.choice([1, 2, 3, 4], size=n, p=[0.45, 0.4, 0.12, 0.03])
    reclamos = rng.poisson(0.4, n)
    activo = rng.random(n) < 0.55
    segmento = rng.choice(["Masivo", "Preferente", "Premium"], size=n, p=[0.7, 0.23, 0.07])
    logit = (-2.9 + 0.9 * reclamos - 0.35 * antig / 4 - 0.8 * activo
             + 0.6 * (productos >= 3) + 0.02 * (edad - 41))
    abandono = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)  # ≈ 5–8 % positivos
    return pd.DataFrame({
        "cliente_id": np.arange(1, n + 1), "edad": edad, "ingreso_mensual": ingreso,
        "antiguedad_anios": antig, "num_productos": productos, "reclamos_ultimo_anio": reclamos,
        "es_activo": activo.astype(int), "segmento": segmento, "abandono": abandono,
    })


def generar_canasta(n_tickets: int = 1500, rng: np.random.Generator | None = None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(SEMILLA + 2)
    productos = ["pan", "leche", "huevos", "mantequilla", "cafe", "azucar", "arroz", "aceite",
                 "pollo", "papa", "cebolla", "gaseosa", "galletas", "queso", "yogurt"]
    reglas = [({"pan"}, "mantequilla", 0.55), ({"cafe"}, "azucar", 0.6),
              ({"pollo"}, "papa", 0.5), ({"arroz"}, "aceite", 0.45), ({"leche"}, "galletas", 0.3)]
    filas = []
    for t in range(1, n_tickets + 1):
        cesta = set(rng.choice(productos, size=rng.integers(1, 5), replace=False))
        for antecedente, consecuente, p in reglas:
            if antecedente <= cesta and rng.random() < p:
                cesta.add(consecuente)
        filas += [{"ticket_id": t, "producto": prod} for prod in sorted(cesta)]
    return pd.DataFrame(filas)


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    generar_viviendas().to_csv(SALIDA / "viviendas_lima.csv", index=False)
    generar_clientes_banco().to_csv(SALIDA / "clientes_banco.csv", index=False)
    generar_canasta().to_csv(SALIDA / "canasta_compras.csv", index=False)
    print(f"✅ Datasets generados en {SALIDA}")


if __name__ == "__main__":
    main()
