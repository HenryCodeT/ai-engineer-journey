"""Utilidades para los tests de cada clase."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data" / "raw"


def cargar_ejercicio(archivo_test: str, nombre: str = "ejercicio"):
    """Importa el ejercicio.py de la misma carpeta que el test (sin choques de nombres entre clases)."""
    ruta = Path(archivo_test).resolve().parent / f"{nombre}.py"
    spec = importlib.util.spec_from_file_location(f"{nombre}_{abs(hash(ruta))}", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def ruta_dataset(nombre: str) -> Path:
    ruta = DATOS / nombre
    if not ruta.exists():
        from data.generar_datos import main as generar

        generar()
    return ruta


def leer_dataset(nombre: str) -> pd.DataFrame:
    return pd.read_csv(ruta_dataset(nombre))


def conexion_sqlite(**tablas: str):
    """Crea una base SQLite en memoria con los CSV indicados.

    Ejemplo: con = conexion_sqlite(viviendas="viviendas_lima.csv")
    """
    import sqlite3

    con = sqlite3.connect(":memory:")
    for tabla, archivo in tablas.items():
        leer_dataset(archivo).to_sql(tabla, con, index=False)
    return con
