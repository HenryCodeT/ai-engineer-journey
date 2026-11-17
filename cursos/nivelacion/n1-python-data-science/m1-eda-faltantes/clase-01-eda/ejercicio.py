"""
Clase 01 · Análisis Exploratorio de Datos (EDA)
================================================

Implementa cada función reemplazando `raise NotImplementedError`.
Evalúa con:

    make evaluar CLASE=cursos/nivelacion/n1-python-data-science/m1-eda-faltantes/clase-01-eda

Reglas:
- No modifiques el DataFrame de entrada (trabaja sobre copias o con operaciones que devuelven uno nuevo).
- Evita bucles `for` sobre filas: usa operaciones de pandas.
- Dataset de práctica: data/raw/viviendas_lima.csv  (objetivo: `precio_usd`)
"""

from __future__ import annotations

import pandas as pd
from pandas._libs.hashtable import value_count
from pandas.core import series


def resumen_general(df: pd.DataFrame) -> dict:
    """Primera foto del dataset: ¿qué tan grande es y qué tan "sucio" está?

    Devuelve un diccionario con:
        "filas"            -> int, número de filas
        "columnas"         -> int, número de columnas
        "duplicados"       -> int, filas que son copia exacta de una fila anterior
        "celdas_faltantes" -> int, total de celdas vacías (NaN) en todo el DataFrame

    Ejemplo:
        >>> df = pd.DataFrame({"a": [1, 1, None], "b": ["x", "x", "y"]})
        >>> resumen_general(df)
        {'filas': 3, 'columnas': 2, 'duplicados': 1, 'celdas_faltantes': 1}

    💡 Pistas: df.shape, df.duplicated(), df.isna()
    """
    return {
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "duplicados": df.duplicated().sum(),
        "celdas_faltantes": df.isna().sum().sum(),
    }


def clasificar_variables(df: pd.DataFrame, excluir: list[str] | None = None) -> dict:
    """Separa las columnas en numéricas y categóricas (paso 2 del EDA).

    Args:
        df: DataFrame a analizar.
        excluir: columnas que NO deben aparecer en ninguna lista
                 (típicamente el id y la variable objetivo). Por defecto ninguna.

    Devuelve:
        {"numericas": [...], "categoricas": [...]}
        Cada lista conserva el orden en que las columnas aparecen en `df`.
        Numérica = cualquier dtype numérico (int, float, Int64...). Todo lo demás es categórica.

    Ejemplo:
        >>> df = pd.DataFrame({"id": [1], "zona": ["A"], "precio": [10.5]})
        >>> clasificar_variables(df, excluir=["id"])
        {'numericas': ['precio'], 'categoricas': ['zona']}

    💡 Pistas: df.select_dtypes(include="number")
    """
    if excluir is None:
        df_new = df
    else:
        df_new = df.drop(columns=excluir)
    return {
        "numericas": list(df_new.select_dtypes(include="number").columns),
        "categoricas": list(df_new.select_dtypes(exclude="number").columns),
    }


def resumen_numericas(df: pd.DataFrame) -> pd.DataFrame:
    """Estadística descriptiva de las columnas numéricas (ignorando NaN).

    Devuelve un DataFrame donde:
        - el índice son los nombres de las columnas numéricas (en su orden original)
        - las columnas son exactamente: ["media", "mediana", "std", "min", "max", "asimetria"]

    La asimetría (skewness) indica hacia dónde se "estira" la distribución:
        > 0  cola a la derecha (p. ej. precios: muchos baratos, pocos carísimos)
        ≈ 0  simétrica
        < 0  cola a la izquierda

    💡 Pistas: df.agg([...]) acepta "mean", "median", "std", "min", "max", "skew"; luego .T
    """
    num = df.select_dtypes(include="number")
    table = num.agg(["mean", "median", "std", "min", "max", "skew"]).T
    table.columns = ["media", "mediana", "std", "min", "max", "asimetria"]
    return table


def conteo_categorias(
    df: pd.DataFrame, columna: str, normalizar: bool = False
) -> pd.Series:
    """Cuántas veces aparece cada categoría de `columna`, de mayor a menor.

    - Los NaN deben contarse también (como su propia categoría).
    - Si normalizar=True, devuelve proporciones (suman 1.0) en lugar de conteos.

    Ejemplo:
        >>> df = pd.DataFrame({"c": ["a", "b", "a", None]})
        >>> conteo_categorias(df, "c").to_dict()
        {'a': 2, 'b': 1, nan: 1}

    💡 Pistas: Series.value_counts tiene parámetros dropna y normalize
    """

    count = df[columna].value_counts(dropna=False, normalize=normalizar)

    return count


def tabla_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de datos faltantes por columna (adelanto del paso 4, clase 02).

    Devuelve un DataFrame con:
        - índice: SOLO las columnas que tienen al menos un NaN
        - columnas: ["faltantes", "porcentaje"]
            faltantes  -> int, cantidad de NaN
            porcentaje -> float, % sobre el total de filas, redondeado a 2 decimales (0–100)
        - ordenado de mayor a menor por "faltantes"

    Si no hay faltantes, devuelve un DataFrame vacío con esas dos columnas.

    Ejemplo:
        >>> df = pd.DataFrame({"a": [1, None, None, 4], "b": [1, 2, 3, None], "c": [1, 2, 3, 4]})
        >>> tabla_faltantes(df)
           faltantes  porcentaje
        a          2        50.0
        b          1        25.0
    """
    conteo = df.isna().sum()
    porcentaje = (conteo / len(df) * 100).round(2)
    tabla = pd.DataFrame({"faltantes": conteo, "porcentaje": porcentaje})
    tabla = tabla[tabla["faltantes"] > 0]
    tabla = tabla.sort_values("faltantes", ascending=False)
    return tabla


def bivariante_categorica(
    df: pd.DataFrame, columna: str, objetivo: str
) -> pd.DataFrame:
    """Cruce de una variable categórica con la variable objetivo numérica (análisis bivariante).

    Sirve para responder: "¿esta variable ayuda a diferenciar el objetivo?"
    (si las medias por categoría son muy distintas, probablemente sí aporta al modelo).

    Devuelve un DataFrame con:
        - índice: cada categoría de `columna` (sin incluir NaN)
        - columnas: ["cantidad", "media_objetivo", "mediana_objetivo"]
        - ordenado de mayor a menor por "media_objetivo"

    Ejemplo:
        >>> df = pd.DataFrame({"zona": ["A", "A", "B"], "precio": [10, 20, 100]})
        >>> bivariante_categorica(df, "zona", "precio")
              cantidad  media_objetivo  mediana_objetivo
        zona
        B            1           100.0             100.0
        A            2            15.0              15.0

    💡 Pistas: df.groupby(columna)[objetivo].agg(...)
    """
    table = df.groupby(columna)[objetivo].agg(["count", "mean", "median"])  
    table.columns = ["cantidad", "media_objetivo", "mediana_objetivo"]           # paso 2: ¿qué 3 nombres, y en qué orden?
    table = table.sort_values("media_objetivo", ascending=False)     # paso 3: ¿por qué columna, y en qué dirección?
    return table
