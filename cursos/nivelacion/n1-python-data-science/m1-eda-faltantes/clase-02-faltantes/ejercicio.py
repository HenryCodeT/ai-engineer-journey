"""
Clase 02 · Estrategias para datos faltantes (MCAR, MAR, MNAR)
==============================================================

Implementa cada función reemplazando `raise NotImplementedError`.
Evalúa con:

    python -m framework evaluar cursos/nivelacion/n1-python-data-science/m1-eda-faltantes/clase-02-faltantes

(o solo la nota: python -m pytest cursos/nivelacion/n1-python-data-science/m1-eda-faltantes/clase-02-faltantes)

Reglas:
- No modifiques el DataFrame de entrada (trabaja sobre una copia: df.copy()).
- Evita bucles `for` sobre filas: usa operaciones de pandas.
- Dataset de práctica: data/raw/viviendas_lima.csv

Orden sugerido (de fácil a difícil): 1 → 2 → 3 → 4 → 5.
La función 5 reutiliza la 3: si la 3 está bien, la 5 es corta.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------- 1
def agregar_indicador_faltante(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Devuelve una COPIA de `df` con una columna nueva `<columna>_falta`.

    La columna nueva vale True donde `columna` es NaN y False donde tiene dato.
    Es la herramienta básica para estudiar faltantes: convierte "hay un hueco"
    en una variable que se puede contar, agrupar y comparar.

    Args:
        df: DataFrame original (NO se modifica).
        columna: nombre de la columna a revisar.

    Devuelve:
        Un DataFrame nuevo con todas las columnas de `df` + `<columna>_falta` (dtype bool) al final.

    Ejemplo:
        >>> df = pd.DataFrame({"edad": [20, None, 40]})
        >>> agregar_indicador_faltante(df, "edad")
           edad  edad_falta
        0  20.0       False
        1   NaN        True
        2  40.0       False

    💡 Pistas:
        - df.copy() crea la copia.
        - df[columna].isna() devuelve una Serie de True/False.
        - El nombre nuevo se arma con un f-string: f"{columna}_falta"

    Esqueleto (una variable por paso):
        resultado = ...                       # paso 1: copia de df
        nombre_nuevo = ...                    # paso 2: "<columna>_falta"
        resultado[nombre_nuevo] = ...         # paso 3: ¿qué método marca los NaN?
        return resultado
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 2
def imputar_mediana(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Devuelve una COPIA de `df` donde los NaN de `columna` se reemplazan por su mediana.

    El profesor: en datos MCAR (faltan completamente al azar) y si son POCOS,
    imputar con la mediana "no les va a afectar mucho".
    Solo se toca `columna`; las demás quedan igual (aunque tengan NaN).

    Args:
        df: DataFrame original (NO se modifica).
        columna: columna numérica a imputar.

    Devuelve:
        DataFrame nuevo, igual a `df` salvo que `columna` ya no tiene NaN.

    Ejemplo:
        >>> df = pd.DataFrame({"edad": [20.0, None, 40.0, 100.0], "zona": ["A", None, "B", "A"]})
        >>> imputar_mediana(df, "edad")
            edad  zona
        0   20.0     A
        1   40.0  None      <- 40 es la mediana de [20, 40, 100]; zona sigue con su NaN
        2   40.0     B
        3  100.0     A

    💡 Pistas:
        - df[columna].median() ignora los NaN automáticamente.
        - Series.fillna(valor) devuelve una Serie NUEVA con los huecos rellenados.
        - Ojo: fillna no cambia nada si no guardas el resultado (hay que asignarlo).

    Esqueleto:
        resultado = ...                       # paso 1: copia
        mediana = ...                         # paso 2: mediana de la columna
        resultado[columna] = ...              # paso 3: la columna con los NaN rellenados
        return resultado
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 3
def tasa_faltantes_por_grupo(df: pd.DataFrame, columna: str, grupo: str) -> pd.Series:
    """Porcentaje de NaN de `columna` dentro de cada categoría de `grupo`.

    Es LA herramienta para sospechar MAR: si en un grupo faltan muchos datos y
    en otro casi ninguno, la ausencia depende de otra variable (no es al azar).
    Es lo que hizo el profesor con el IQ: "los datos faltan cuando el IQ es bajo".

    Args:
        df: DataFrame (NO se modifica).
        columna: columna cuyos faltantes queremos estudiar.
        grupo: columna categórica por la que agrupamos.

    Devuelve:
        Serie con:
          - índice: cada categoría de `grupo` (las filas con `grupo` NaN se ignoran)
          - valores: % de NaN de `columna` en esa categoría (0–100), redondeado a 2 decimales
          - ordenada de mayor a menor

    Ejemplo:
        >>> df = pd.DataFrame({"zona": ["A", "A", "B", "B"], "ingreso": [10, 20, None, 40]})
        >>> tasa_faltantes_por_grupo(df, "ingreso", "zona")
        zona
        B    50.0
        A     0.0

    💡 Pistas:
        - Truco clave: la media de una columna de True/False es la PROPORCIÓN de True
          (True cuenta como 1, False como 0). Ej: [True, False, False, False] -> media 0.25
        - Puedes reutilizar tu función 1 para tener la columna `<columna>_falta`
          y luego hacer groupby sobre ella (como en bivariante_categorica de la clase 01).
        - Proporción -> porcentaje: multiplica por 100.

    Esqueleto:
        con_indicador = ...                   # paso 1: usa agregar_indicador_faltante
        proporcion = ...                      # paso 2: groupby(grupo)[...].mean()
        porcentaje = ...                      # paso 3: x 100 y redondear a 2
        porcentaje = ...                      # paso 4: ordenar de mayor a menor
        return porcentaje
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 4
def comparar_media_por_faltante(df: pd.DataFrame, columna: str, variable: str) -> dict:
    """Compara la media de `variable` entre filas CON y SIN faltante en `columna`.

    Otra forma de buscar relación: si las filas donde falta `columna` tienen una
    `variable` muy distinta (p. ej. un precio mucho más alto), los faltantes NO son al azar.

    Args:
        df: DataFrame (NO se modifica).
        columna: columna cuyos faltantes definen los dos grupos.
        variable: columna numérica cuya media comparamos.

    Devuelve:
        dict con tres floats redondeados a 2 decimales:
          "media_con_faltante" -> media de `variable` en filas donde `columna` es NaN
          "media_sin_faltante" -> media de `variable` en filas donde `columna` tiene dato
          "diferencia"         -> media_con_faltante - media_sin_faltante

    Ejemplo:
        >>> df = pd.DataFrame({"banos": [1, None, 2, None], "precio": [100, 500, 200, 700]})
        >>> comparar_media_por_faltante(df, "banos", "precio")
        {'media_con_faltante': 600.0, 'media_sin_faltante': 150.0, 'diferencia': 450.0}

    💡 Pistas:
        - mascara = df[columna].isna()        -> True en las filas con hueco
        - df.loc[mascara, variable]           -> la variable SOLO en esas filas
        - ~mascara                            -> invierte la máscara (filas SIN hueco)
        - float(x) convierte un número de numpy a float "normal" de Python.

    Esqueleto:
        mascara = ...                         # paso 1
        media_con = ...                       # paso 2: filtro con la máscara + .mean()
        media_sin = ...                       # paso 3: filtro con ~mascara + .mean()
        return {...}                          # paso 4: redondea con round(valor, 2)
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 5
def clasificar_mecanismo(
    df: pd.DataFrame, columna: str, grupo: str, umbral: float = 15.0
) -> str:
    """Clasificación heurística del tipo de faltante de `columna` mirando `grupo`.

    Regla (simplificada, sirve para el reporte del EDA, no es una prueba estadística):
      1. Si `columna` no tiene ningún NaN               -> "sin faltantes"
      2. Calcula la tasa de faltantes por grupo (función 3) y su RANGO = máximo - mínimo
         (en puntos porcentuales).
         - Si rango > umbral -> "posible MAR"  (la ausencia cambia mucho según el grupo)
         - Si no             -> "MCAR"         (parecido en todos los grupos: compatible con azar)

    ⚠️ MNAR no se puede detectar con esta regla: por definición no se ve relación
    con las otras variables. Por eso el profesor dice que requiere un estudio aparte.

    Args:
        df: DataFrame (NO se modifica).
        columna: columna con faltantes a estudiar.
        grupo: columna categórica con la que buscamos relación.
        umbral: diferencia máxima tolerada entre grupos, en puntos porcentuales (0–100).

    Devuelve:
        Uno de estos tres strings exactos: "sin faltantes", "MCAR", "posible MAR".

    Ejemplo:
        >>> df = pd.DataFrame({"zona": ["A", "A", "B", "B"], "ingreso": [10, 20, None, None]})
        >>> clasificar_mecanismo(df, "ingreso", "zona")      # A: 0 %, B: 100 % -> rango 100
        'posible MAR'
        >>> clasificar_mecanismo(df, "zona", "ingreso")      # zona no tiene NaN
        'sin faltantes'

    💡 Pistas:
        - df[columna].isna().sum() == 0  -> no hay faltantes
        - Reutiliza tasa_faltantes_por_grupo(df, columna, grupo)
        - Una Serie tiene .max() y .min()

    Esqueleto:
        if ...:                               # paso 1: ¿hay faltantes?
            return "sin faltantes"
        tasas = ...                           # paso 2: función 3
        rango = ...                           # paso 3: máximo - mínimo
        if ...:                               # paso 4: compara con umbral
            return "posible MAR"
        return "MCAR"
    """
    raise NotImplementedError
