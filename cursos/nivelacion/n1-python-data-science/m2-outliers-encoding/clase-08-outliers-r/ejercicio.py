"""
Clase 08 · Outliers multivariados con algoritmos de ML (PyOD -> equivalentes en scikit-learn)
===========================================================================================

(El video se titula "Outliers en R", pero todo es Python: es la continuación de la clase 07.)

Implementa cada función reemplazando `raise NotImplementedError`.
Evalúa con:

    python -m framework evaluar cursos/nivelacion/n1-python-data-science/m2-outliers-encoding/clase-08-outliers-r

(o solo la nota: python -m pytest cursos/nivelacion/n1-python-data-science/m2-outliers-encoding/clase-08-outliers-r)

Reglas:
- No modifiques el DataFrame de entrada (trabaja sobre una copia cuando haga falta).
- Evita bucles `for` sobre filas: usa numpy / pandas / scikit-learn.
- Convención de etiquetas de ESTE ejercicio = la de PyOD:  0 = inlier,  1 = outlier.
  (scikit-learn usa otra: 1 = inlier, -1 = outlier. La función 3 traduce de una a otra.)
- Solo puedes usar pandas, numpy y scikit-learn (PyOD no está instalado).
- Dataset de práctica: data/raw/viviendas_lima.csv

Hilo del ejercicio (sigue la clase):
  1. Puntaje kNN: qué tan lejos está cada punto de sus vecinos (lo que hace KNN de PyOD),
     con las tres formas que mencionó el profesor: "largest", "mean", "median".
  2. `contamination`: convertir puntajes en etiquetas 0/1 sacrificando como máximo
     un porcentaje del dataset (el 0.2 del profesor).
  3. Traducir etiquetas de scikit-learn (-1/1) a la convención de PyOD (1/0).
  4. Otro algoritmo distinto: IsolationForest, con la misma `contamination`.
  5. Comparar dos detectores: "cada algoritmo marca puntos distintos".
  6. Quedarse solo con los inliers y medir cuánto dataset conservaste
     (el profesor: 417 de 506 -> ~80 %).

Orden sugerido: 1 -> 2 -> 3 -> 4 -> 5 -> 6. La 4 usa la 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------- 1
def puntaje_knn(X, k: int = 5, metodo: str = "largest") -> np.ndarray:
    """Devuelve el puntaje de "outlier" kNN de cada fila: grande = punto aislado.

    El profesor: "los vecinos que están más próximos a cierto punto son considerados
    inliers, y los puntitos más alejados, muy aislados, muy lejanos, son outliers".
    Cada fila es un PUNTO en R^n (n = número de columnas). Para cada punto se miran
    sus k vecinos más cercanos (sin contarse a sí mismo) y se resume la distancia:

        metodo="largest" -> distancia al k-ésimo vecino (el más lejano de los k)
        metodo="mean"    -> promedio de las k distancias
        metodo="median"  -> mediana de las k distancias

    (Es lo mismo que `pyod.models.knn.KNN(n_neighbors=k, method=metodo)` calcula
    en su atributo `decision_scores_`.)

    Args:
        X: DataFrame o array 2D solo con columnas numéricas y SIN NaN.
        k: número de vecinos (sin contar al propio punto).
        metodo: "largest", "mean" o "median". Cualquier otro -> ValueError.

    Devuelve:
        np.ndarray 1D de floats, un puntaje por fila, en el mismo orden que X.
        Distancia euclidiana (la que usa NearestNeighbors por defecto).

    Ejemplo (1 columna, k=3):
        >>> X = pd.DataFrame({"x": [0, 1, 2, 3, 10]})
        >>> puntaje_knn(X, k=3, metodo="largest")
        array([3., 2., 2., 3., 9.])
        # El 10: sus 3 vecinos son 3, 2, 1 -> distancias 7, 8, 9 -> la mayor es 9.
        >>> puntaje_knn(X, k=3, metodo="mean")
        array([2.  , 1.33, 1.33, 2.  , 8.  ])   (redondeado)
        >>> puntaje_knn(X, k=3, metodo="median")
        array([2., 1., 1., 2., 8.])

    💡 Pistas:
        - ¡Trampa! Si le preguntas a NearestNeighbors por los vecinos de los MISMOS
          puntos con los que entrenó, el vecino número 0 de cada punto es él mismo
          (distancia 0). Pide k + 1 vecinos y descarta la primera columna.
        - NearestNeighbors(n_neighbors=...).fit(datos) y luego
          .kneighbors(datos) devuelve (distancias, indices); cada fila de
          `distancias` viene ordenada de menor a mayor.
        - Por filas: .mean(axis=1), np.median(..., axis=1); la última columna es
          la k-ésima distancia.

    Esqueleto (una variable por paso):
        datos = ...                     # paso 1: np.asarray(X, dtype=float)
        nn = ...                        # paso 2: NearestNeighbors con k + 1 vecinos, .fit(datos)
        distancias, _ = ...             # paso 3: nn.kneighbors(datos)
        distancias = ...                # paso 4: quitar la columna 0 (el propio punto)
        if metodo == "largest": ...     # paso 5: última columna
        if metodo == "mean": ...        # paso 6: promedio por fila
        if metodo == "median": ...      # paso 7: mediana por fila
        raise ValueError(...)           # paso 8: método desconocido
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 2
def marcar_por_contaminacion(puntajes, contamination: float = 0.1) -> np.ndarray:
    """Convierte puntajes en etiquetas 0/1 usando `contamination` (como PyOD).

    El profesor: con `contamination=0.2` le dices al algoritmo "compromete a lo
    mucho el 20 % de la base de datos, no elimines más que eso, y elimina los
    outliers más perjudiciales, los que están más alejados".

    Regla (la misma que usa PyOD por dentro):
        umbral = percentil (1 - contamination) * 100 de los puntajes
        outlier (1) si puntaje > umbral   (ESTRICTAMENTE mayor)
        inlier  (0) en otro caso

    Args:
        puntajes: lista / array / Serie de números (más alto = más raro).
        contamination: proporción entre 0 y 0.5 (0.2 = "hasta el 20 %").

    Devuelve:
        np.ndarray 1D de enteros (0 o 1), mismo orden y largo que `puntajes`.

    Ejemplo:
        >>> marcar_por_contaminacion([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.2)
        array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        # percentil 80 de 1..10 = 8.2 -> solo 9 y 10 superan el umbral.

    💡 Pistas:
        - np.percentile(valores, q) con q entre 0 y 100.
        - Una comparación da booleanos; .astype(int) los vuelve 0/1.

    Esqueleto:
        valores = ...        # paso 1: np.asarray(puntajes, dtype=float)
        umbral = ...         # paso 2: percentil 100 * (1 - contamination)
        return ...           # paso 3: (valores > umbral) como enteros
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 3
def convertir_etiquetas_sklearn(pred) -> np.ndarray:
    """Traduce etiquetas de scikit-learn (1 = inlier, -1 = outlier) a PyOD (0 / 1).

    En la clase, `clf.predict(X)` de PyOD devolvía 0 = inlier y 1 = outlier
    (417 ceros y 89 unos en Boston). Los detectores de scikit-learn
    (IsolationForest, LocalOutlierFactor...) devuelven 1 = inlier y -1 = outlier.
    Para comparar ambos mundos hay que hablar el mismo idioma.

        -1  ->  1   (outlier)
         1  ->  0   (inlier)

    Args:
        pred: lista / array / Serie con valores 1 y -1.

    Devuelve:
        np.ndarray 1D de enteros 0/1, mismo orden.

    Ejemplo:
        >>> convertir_etiquetas_sklearn([1, -1, 1, 1, -1])
        array([0, 1, 0, 0, 1])

    💡 Pista: np.where(condición, valor_si_true, valor_si_false).

    Esqueleto:
        arr = ...            # paso 1: np.asarray(pred)
        return ...           # paso 2: np.where(arr == -1, ..., ...)
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 4
def detectar_isolation_forest(
    df: pd.DataFrame, columnas: list, contamination: float = 0.1, random_state: int = 42
) -> pd.Series:
    """Detecta outliers multivariados con IsolationForest y devuelve etiquetas 0/1.

    El profesor insistió: "cada algoritmo tiene un comportamiento diferente".
    IsolationForest es otro algoritmo (también existe en PyOD como IForest): en
    vez de medir distancias, intenta "aislar" cada punto con cortes al azar; los
    puntos raros se aíslan con pocos cortes.

    Pasos:
        1. Toma solo `columnas` y rellena los NaN de cada columna con SU mediana
           (técnica de la clase 03; IsolationForest no acepta NaN).
        2. Crea IsolationForest(contamination=contamination, random_state=random_state).
        3. fit_predict sobre esos datos -> etiquetas 1 / -1.
        4. Tradúcelas a 0 / 1 con `convertir_etiquetas_sklearn`.

    Args:
        df: DataFrame (NO se modifica).
        columnas: lista de columnas numéricas a usar (sin la variable objetivo).
        contamination: proporción máxima de outliers.
        random_state: semilla (hace el resultado reproducible).

    Devuelve:
        pd.Series de enteros 0/1 con el MISMO índice que `df` y name="outlier".

    Ejemplo:
        >>> df = pd.DataFrame({"a": [50, 51, 49, 50, 100], "b": [50, 49, 51, 50, 100]})
        >>> detectar_isolation_forest(df, ["a", "b"], contamination=0.2)
        0    0
        1    0
        2    0
        3    0
        4    1        <- (100, 100) está lejísimos del grupo
        Name: outlier, dtype: int64

    💡 Pistas:
        - df[columnas].median() da la mediana de cada columna; .fillna(eso) rellena.
        - pd.Series(valores, index=df.index, name="outlier").

    Esqueleto:
        X = ...              # paso 1: columnas con NaN rellenados por la mediana
        modelo = ...         # paso 2: IsolationForest(...)
        pred = ...           # paso 3: modelo.fit_predict(X)
        etiquetas = ...      # paso 4: convertir_etiquetas_sklearn(pred)
        return ...           # paso 5: Serie con el índice de df y name="outlier"
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 5
def comparar_detectores(etiquetas_a, etiquetas_b) -> dict:
    """Compara qué filas marca como outlier cada detector (etiquetas 0/1).

    En la clase, ABOD marcó 19 puntos y kNN 11 con la misma contaminación, y
    CBLOF marcó como outliers una región que los demás veían como inliers. Esta
    función lo mide con números:

        outliers_a = cuántos 1 tiene A
        outliers_b = cuántos 1 tiene B
        coinciden  = cuántas filas son 1 en A Y en B a la vez (intersección)
        jaccard    = coinciden / (filas que son 1 en A O en B)   (unión)
                     redondeado a 4 decimales. Si la unión es 0 (ninguno marca
                     nada) -> 1.0 (coinciden totalmente).

    Jaccard = 1 -> marcan exactamente las mismas filas; 0 -> ninguna en común.

    Args:
        etiquetas_a, etiquetas_b: listas / arrays / Series de 0/1 del mismo largo
            (se comparan por POSICIÓN).

    Devuelve:
        dict {"outliers_a": int, "outliers_b": int, "coinciden": int, "jaccard": float}
        (int y float de Python, no de numpy).

    Ejemplo:
        >>> comparar_detectores([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
        {'outliers_a': 3, 'outliers_b': 3, 'coinciden': 2, 'jaccard': 0.5}
        # unión = filas 0, 1, 3, 4 = 4 -> 2 / 4 = 0.5

    💡 Pistas:
        - a = np.asarray(etiquetas_a) == 1 da booleanos.
        - (a & b).sum() es la intersección, (a | b).sum() la unión.

    Esqueleto:
        a = ...              # paso 1: booleanos de A
        b = ...              # paso 2: booleanos de B
        interseccion = ...   # paso 3
        union = ...          # paso 4
        jaccard = ...        # paso 5: cuidado con union == 0
        return {...}         # paso 6
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 6
def filtrar_inliers(df: pd.DataFrame, etiquetas) -> tuple:
    """Se queda solo con las filas inlier (etiqueta 0) y dice qué % del dataset conservaste.

    Es el último paso de la clase: `df_tmp[pred == 0]` -> 417 filas de 506 (~80 %),
    "eliminé los puntos que estaban mucho más alejados del comportamiento normal".

    Args:
        df: DataFrame original (NO se modifica).
        etiquetas: lista / array / Serie de 0/1, mismo largo que df, alineada por
            POSICIÓN con sus filas.

    Devuelve:
        Tupla (df_limpio, porcentaje_conservado):
          - df_limpio: COPIA de df con solo las filas de etiqueta 0, conservando
            el índice original (no lo reinicies).
          - porcentaje_conservado: 100 * filas_limpio / filas_original, redondeado
            a 2 decimales (float).

    Ejemplo:
        >>> df = pd.DataFrame({"area_m2": [60, 75, 2746, 90]}, index=[10, 11, 12, 13])
        >>> limpio, pct = filtrar_inliers(df, [0, 0, 1, 0])
        >>> limpio.index.tolist(), pct
        ([10, 11, 13], 75.0)

    💡 Pistas:
        - Una máscara booleana de numpy sirve en df.loc[mascara] (o df[mascara]).
        - .copy() para no devolver una "vista" del original.

    Esqueleto:
        mascara = ...        # paso 1: np.asarray(etiquetas) == 0
        limpio = ...         # paso 2: filas de la máscara, .copy()
        porcentaje = ...     # paso 3: round(100 * len(limpio) / len(df), 2)
        return limpio, porcentaje
    """
    raise NotImplementedError
