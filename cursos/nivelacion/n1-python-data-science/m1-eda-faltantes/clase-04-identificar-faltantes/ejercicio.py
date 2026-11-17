"""
Clase 04 · Cómo identifico datos faltantes (ejercicio)
=======================================================

Implementa cada función reemplazando `raise NotImplementedError`.
Evalúa con:

    python -m framework evaluar cursos/nivelacion/n1-python-data-science/m1-eda-faltantes/clase-04-identificar-faltantes

(o solo la nota: python -m pytest cursos/nivelacion/n1-python-data-science/m1-eda-faltantes/clase-04-identificar-faltantes)

Reglas:
- No modifiques el DataFrame de entrada (trabaja sobre una copia: df.copy()).
- Evita bucles `for` sobre filas: usa operaciones de pandas (un `for` sobre COLUMNAS sí está bien).
- Dataset de práctica: data/raw/clientes_banco.csv (no trae nulos: los simulamos, como el profesor).

Es el mismo flujo de la clase:
    1. reporte_nulos          -> el "informe" de nulos que armó el profesor
    2. columnas_descartables  -> el debate: con 60 % de nulos, la variable sale del análisis
    3. simular_faltantes      -> cómo el profesor metió nulos en las MISMAS filas
    4. correlacion_nulidad    -> lo que dibuja el heatmap de missingno
    5. diagnostico_nulidad    -> leer ese heatmap: correlación alta = posible MAR
    6. imputar_knn            -> la imputación "avanzada" con KNNImputer

Orden sugerido: 1 → 2 → 3 → 4 → 5 → 6. La 2 reutiliza la 1 y la 5 reutiliza la 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 1
def reporte_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Reporte de nulos por columna, como el que armó el profesor en la clase.

    Es más ordenado que `df.info()`: una fila por variable con cuántos nulos tiene,
    cuántos datos sí tiene y qué porcentaje del total falta.

    Args:
        df: DataFrame a analizar (NO se modifica).

    Devuelve:
        Un DataFrame NUEVO con exactamente estas 4 columnas, en este orden:
          "variable"   -> nombre de la columna de `df` (str)
          "nulos"      -> cantidad de NaN de esa columna (int)
          "no_nulos"   -> cantidad de valores presentes = len(df) - nulos (int)
          "porcentaje" -> nulos / len(df) * 100, redondeado a 3 decimales (float)
        - Una fila por CADA columna de `df` (también las que tienen 0 nulos).
        - Ordenado de MÁS nulos a MENOS nulos. Si hay empate, se respeta el orden
          original de las columnas (usa sort_values(..., kind="stable")).
        - Índice limpio 0, 1, 2, ... (reset_index(drop=True)).

    Ejemplo:
        >>> df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 6], "c": [7, 8, 9]})
        >>> reporte_nulos(df)
          variable  nulos  no_nulos  porcentaje
        0        b      2         1      66.667
        1        a      1         2      33.333
        2        c      0         3       0.000

    💡 Pistas:
        - df.isna().sum() -> Serie con los nulos por columna (el índice son los nombres).
        - Puedes armar el resultado con pd.DataFrame({"variable": ..., "nulos": ..., ...}).
        - Serie.index te da los nombres; .values o .tolist() te dan los números.
        - round(x, 3) o Serie.round(3) para los 3 decimales.

    Esqueleto (un paso por línea):
        nulos = ...                           # paso 1: nulos por columna
        reporte = pd.DataFrame({...})         # paso 2: columnas variable / nulos
        reporte["no_nulos"] = ...             # paso 3: len(df) - nulos
        reporte["porcentaje"] = ...           # paso 4: % con 3 decimales
        reporte = ...                         # paso 5: ordenar (stable) y reset_index(drop=True)
        return reporte
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 2
def columnas_descartables(df: pd.DataFrame, umbral: float = 50.0) -> list[str]:
    """Columnas con DEMASIADOS nulos para imputarlas: su porcentaje de nulos es MAYOR que `umbral`.

    El profesor: con un 60 % de faltantes "no puedes reconstruir la mitad de un mapa
    con la otra mitad". Primero se revisa el script SQL / se avisa al ingeniero de datos;
    si el problema sigue, se REPORTA y la variable sale del análisis.

    Args:
        df: DataFrame a analizar (NO se modifica).
        umbral: porcentaje (0–100). Se descarta la columna si su % de nulos es > umbral
                (estrictamente mayor: exactamente 50 % con umbral=50 NO se descarta).

    Devuelve:
        Lista de nombres de columnas, en el MISMO orden en que aparecen en `df`.
        Lista vacía si ninguna supera el umbral.

    Ejemplo:
        >>> df = pd.DataFrame({"a": [None, None, None, 1], "b": [None, None, 1, 1], "c": [1, 2, 3, 4]})
        >>> columnas_descartables(df)              # a: 75 %, b: 50 %, c: 0 %
        ['a']
        >>> columnas_descartables(df, umbral=40)
        ['a', 'b']

    💡 Pistas:
        - El porcentaje de nulos por columna es df.isna().mean() * 100
          (la media de True/False es la proporción de True).
        - Filtra la Serie con una condición: serie[serie > umbral]
        - .index.tolist() convierte el índice en una lista de Python.
        - Ojo con el orden: tu reporte_nulos ordena por nulos; aquí piden el orden original.

    Esqueleto:
        porcentaje = ...                      # paso 1: % de nulos por columna (orden original)
        altas = ...                           # paso 2: solo las que superan el umbral
        return ...                            # paso 3: lista de nombres
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 3
def simular_faltantes(
    df: pd.DataFrame, columnas: list[str], fraccion: float, semilla: int = 123
) -> pd.DataFrame:
    """Devuelve una COPIA de `df` con NaN en las MISMAS filas para todas las `columnas`.

    Es lo que hizo el profesor: el dataset de churn no tenía nulos, así que "con fines
    didácticos" borró el 60 % de Dependent_count y Marital_Status. Como borró las mismas
    filas en ambas, después el heatmap mostró correlación de nulidad = 1 (patrón MAR).

    Para que el resultado sea REPRODUCIBLE (y el test pueda revisarlo), las filas se
    eligen OBLIGATORIAMENTE así:
        filas = df.sample(frac=fraccion, random_state=semilla).index

    Args:
        df: DataFrame original (NO se modifica).
        columnas: lista de columnas donde se pondrán los NaN.
        fraccion: proporción de filas a vaciar (0.6 = 60 %).
        semilla: random_state para que siempre salgan las mismas filas.

    Devuelve:
        DataFrame nuevo, igual a `df` salvo que en `filas` × `columnas` hay NaN.
        Las demás columnas no cambian. (Las columnas enteras pasarán a float: es normal,
        NaN es un float.)

    Ejemplo:
        >>> df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "c": [9, 9, 9, 9]})
        >>> simular_faltantes(df, ["a", "b"], fraccion=0.5, semilla=0)
             a    b  c
        0  1.0  5.0  9
        1  2.0  6.0  9
        2  NaN  NaN  9      <- "a" y "b" vacías en las MISMAS filas
        3  NaN  NaN  9      <- "c" no se toca

    💡 Pistas:
        - df.loc[filas, columnas] = np.nan  asigna NaN a la vez en varias filas Y columnas.
        - Haz la copia ANTES de asignar, si no modificarías el df del usuario.

    Esqueleto:
        resultado = ...                       # paso 1: copia
        filas = ...                           # paso 2: la línea obligatoria de arriba
        resultado.loc[..., ...] = ...         # paso 3: NaN en esas filas y columnas
        return resultado
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 4
def correlacion_nulidad(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de correlación de NULIDAD: lo que dibuja `missingno.heatmap(df)`.

    Idea: convierte cada columna en "¿falta? True/False" y calcula la correlación entre
    esas columnas. Un 1 entre A y B significa: "cada vez que falta A, también falta B".
    Solo tiene sentido para columnas que tienen ALGUNOS nulos (ni 0 %, ni 100 %): si una
    columna nunca falta (o siempre falta), su indicador es constante y la correlación no
    se puede calcular.

    Args:
        df: DataFrame a analizar (NO se modifica).

    Devuelve:
        DataFrame cuadrado (columnas × columnas) con la correlación de Pearson entre los
        indicadores de nulidad, redondeada a 2 decimales.
        - Solo incluye las columnas con al menos 1 NaN y al menos 1 dato (orden original).
        - La diagonal vale 1.0.
        - Si no queda ninguna columna, devuelve un DataFrame vacío (pd.DataFrame()).

    Ejemplo:
        >>> df = pd.DataFrame({
        ...     "a": [None, None, 1, 1],
        ...     "b": [None, None, 2, 2],     # falta justo donde falta "a"
        ...     "c": [None, 3, None, 3],     # sin relación con "a"
        ...     "d": [1, 2, 3, 4],           # nunca falta -> fuera de la matriz
        ... })
        >>> correlacion_nulidad(df)
             a    b    c
        a  1.0  1.0  0.0
        b  1.0  1.0  0.0
        c  0.0  0.0  1.0

    💡 Pistas:
        - df.isna() -> DataFrame de True/False del mismo tamaño que df.
        - Con .any() y .all() sobre df.isna() sabes qué columnas tienen algún nulo / son todo nulos.
        - Un DataFrame de True/False también tiene .corr() (True=1, False=0).

    Esqueleto:
        indicadores = ...                     # paso 1: df.isna()
        con_nulos = ...                       # paso 2: máscara de columnas con algún nulo y no todo nulo
        indicadores = ...                     # paso 3: quedarte solo con esas columnas
        if ...:                               # paso 4: ¿no quedó ninguna?
            return pd.DataFrame()
        return ...                            # paso 5: .corr() y redondear a 2
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 5
def diagnostico_nulidad(df: pd.DataFrame, umbral: float = 0.5) -> dict:
    """Lee la matriz de correlación de nulidad como lo hizo el profesor.

    - Si la nulidad de una columna está correlacionada (en valor absoluto) >= `umbral`
      con la de ALGUNA OTRA columna -> "posible MAR" (su ausencia va de la mano con otra).
    - Si no está correlacionada con nadie -> "potencial MCAR" (el profesor dice "potencial":
      hay que confirmarlo, por ejemplo mezclando las filas y volviendo a mirar).
    MNAR no se detecta así: por definición no muestra correlación con lo que vemos.

    Args:
        df: DataFrame a analizar (NO se modifica).
        umbral: correlación absoluta mínima (0–1) para decir "posible MAR".

    Devuelve:
        dict {columna: "posible MAR" | "potencial MCAR"} SOLO con las columnas que aparecen
        en correlacion_nulidad(df) (las que tienen algunos nulos), en ese mismo orden.
        dict vacío si no hay columnas con nulos.

    Ejemplo (mismo df que en correlacion_nulidad):
        >>> diagnostico_nulidad(df)
        {'a': 'posible MAR', 'b': 'posible MAR', 'c': 'potencial MCAR'}

    ⚠️ La diagonal (cada columna consigo misma) siempre vale 1: ¡no la cuentes!

    💡 Pistas:
        - Reutiliza correlacion_nulidad(df).
        - .abs() para el valor absoluto.
        - Para ignorar la diagonal: matriz.mask(np.eye(len(matriz), dtype=bool)) la vuelve NaN.
          (o, columna por columna: matriz[col].drop(col))
        - .max() ignora los NaN. Un `for` sobre las columnas está bien aquí.

    Esqueleto:
        matriz = ...                          # paso 1: función 4 en valor absoluto
        sin_diagonal = ...                    # paso 2: quitar la diagonal
        resultado = {}
        for col in ...:                       # paso 3: recorrer las columnas de la matriz
            maximo = ...                      #   mayor correlación con OTRA columna
            resultado[col] = ...              #   "posible MAR" o "potencial MCAR"
        return resultado

        (Cuidado: si solo hay UNA columna con nulos, su máximo "con otra" será NaN
         -> eso cuenta como "potencial MCAR".)
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 6
def imputar_knn(df: pd.DataFrame, columnas: list[str], k: int = 10) -> pd.DataFrame:
    """Imputa los NaN de `columnas` con KNNImputer (k vecinos, pesos uniformes).

    El profesor: es uno de los métodos avanzados más usados. Para cada fila con hueco
    busca las `k` filas más parecidas (mirando las demás `columnas`) y rellena con el
    PROMEDIO de lo que esas vecinas tienen en la columna vacía. Resultado: el histograma
    queda más suave, sin el "pico" que deja la mediana.

    Args:
        df: DataFrame original (NO se modifica).
        columnas: columnas NUMÉRICAS que se usan para medir distancias y que se imputan.
                  Las demás columnas de `df` no se usan ni se tocan.
        k: número de vecinos (n_neighbors).

    Devuelve:
        DataFrame nuevo con las mismas columnas, el mismo orden y el MISMO índice que `df`;
        `columnas` ya sin NaN, el resto igual que antes.

    Ejemplo:
        >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0], "y": [10.0, 20.0, None, 100.0],
        ...                    "zona": ["A", "A", "B", "B"]})
        >>> imputar_knn(df, ["x", "y"], k=2)
              x      y zona
        0   1.0   10.0    A
        1   2.0   20.0    A
        2   3.0   15.0    B    <- las 2 vecinas más cercanas en x son x=2 y x=1 -> (20 + 10) / 2
        3  10.0  100.0    B

    💡 Pistas:
        - from sklearn.impute import KNNImputer
        - imputer = KNNImputer(n_neighbors=k, weights="uniform")
        - imputer.fit_transform(df[columnas]) devuelve un ARRAY de numpy (no un DataFrame):
          puedes asignarlo directamente a resultado[columnas] = ...
        - Si conviertes el array a DataFrame, pásale index=df.index, ¡o perderás el índice!

    Esqueleto:
        resultado = ...                       # paso 1: copia
        imputer = ...                         # paso 2: KNNImputer con k vecinos, pesos uniformes
        valores = ...                         # paso 3: fit_transform sobre las columnas
        resultado[columnas] = ...             # paso 4: devolver los valores a su sitio
        return resultado
    """
    raise NotImplementedError
