"""
Clase 15 · Eliminar características altamente correlacionadas
=============================================================

Implementa cada función reemplazando `raise NotImplementedError`.
Evalúa con:

    python -m framework evaluar cursos/nivelacion/n1-python-data-science/m3-seleccion-variables/clase-15-correlacion

(o solo la nota: python -m pytest cursos/nivelacion/n1-python-data-science/m3-seleccion-variables/clase-15-correlacion)

Reglas:
- No modifiques las entradas (DataFrames, Series, listas): trabaja sobre copias si hace falta.
- Solo necesitas pandas, numpy y scikit-learn.
- Datasets: los tests usan una variante SINTÉTICA del dataset de "móviles" de la clase 13
  (mismas 20 columnas) en la que, como en el Kaggle real, hay 4 PARES de variables
  correlacionadas: fc~pc, px_height~px_width, sc_h~sc_w y four_g~three_g.
  También usan data/raw/viviendas_lima.csv (area_m2 ~ area_construida_m2, etc.).

Hilo del ejercicio (sigue la clase):
  1. Correlación de Pearson "a mano" con numpy (qué número mira la matriz de correlación).
  2. Encontrar los PARES de columnas muy correlacionadas (solo el triángulo superior:
     "la parte de arriba es igual a la parte de abajo").
  3. Decidir qué columnas eliminar con un umbral (el profesor usó 0.4), usando SOLO X_train.
  4. Ver por qué el profesor "no lo recomienda mucho": Pearson solo mide relación LINEAL
     (comparación con Spearman).
  5. Entrenar el mismo Random Forest de la clase 13 con y sin las columnas eliminadas.

Orden sugerido: 1 -> 5. La 3 puede usar la idea de la 2; la 5 usa la 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 1
def correlacion_pearson_manual(x, y) -> float:
    """Calcula el coeficiente de correlación de Pearson entre x e y SIN usar .corr() ni np.corrcoef.

    Fórmula (con mx = media de x, my = media de y):

                    suma( (x - mx) * (y - my) )
        r = -----------------------------------------------
             raiz( suma((x - mx)**2) * suma((y - my)**2) )

    - r = +1: relación lineal perfecta creciente (y = 2x + 1).
    - r = -1: relación lineal perfecta decreciente (y = -3x).
    - r ≈  0: no hay relación LINEAL (¡puede haber otra relación!).

    Se compara POSICIÓN por posición, aunque x e y sean Series con índices distintos.

    Args:
        x: valores numéricos (lista, array de numpy o Series de pandas).
        y: valores numéricos de la misma longitud.

    Devuelve:
        float entre -1 y 1 (debe coincidir con np.corrcoef(x, y)[0, 1]).

    Errores:
        ValueError si x e y no tienen la misma longitud.

    Ejemplo:
        >>> correlacion_pearson_manual([1, 2, 3, 4], [3, 5, 7, 9])    # y = 2x + 1
        1.0
        >>> round(correlacion_pearson_manual([1, 2, 3, 4], [1, 3, 2, 4]), 2)
        0.8

    💡 Pistas:
        - np.asarray(x, dtype=float) convierte listas y Series en arrays y "olvida" el índice.
        - .mean(), np.sum(...) y np.sqrt(...) son todo lo que necesitas.
    """
    # 1. convierte x e y a arrays de float
    # 2. si las longitudes son distintas -> raise ValueError("...")
    # 3. centra cada uno restando su media (dx, dy)
    # 4. numerador = suma de dx * dy
    # 5. denominador = raíz de (suma de dx**2) * (suma de dy**2)
    # 6. devuelve float(numerador / denominador)
    raise NotImplementedError


# ---------------------------------------------------------------- 2
def pares_correlacionados(df: pd.DataFrame, umbral: float = 0.4) -> pd.DataFrame:
    """Lista los PARES de columnas cuya correlación de Pearson, en VALOR ABSOLUTO, supera el umbral.

    La matriz de correlación es simétrica: corr(a, b) = corr(b, a) y la diagonal es 1
    (cada columna consigo misma). Por eso solo se mira el TRIÁNGULO SUPERIOR, sin la
    diagonal: cada par aparece UNA sola vez.

    Reglas:
        - Todas las columnas de df son numéricas.
        - Se usa la correlación de Pearson (df.corr()).
        - Se cumple la condición si |corr| > umbral (estrictamente mayor).
          Una correlación de -0.9 también cuenta: es una relación lineal fuerte (decreciente).
        - col_a es la columna que aparece ANTES en df; col_b la que aparece después.
        - La columna "corr" guarda el valor CON signo.
        - Orden: de mayor a menor |corr|. Índice 0, 1, 2, ...
        - Si no hay pares, devuelve un DataFrame vacío con las mismas 3 columnas.

    Args:
        df: DataFrame con columnas numéricas (por ejemplo X_train).
        umbral: valor entre 0 y 1 (en la clase, 0.4).

    Devuelve:
        pd.DataFrame con columnas ["col_a", "col_b", "corr"].

    Ejemplo:
        >>> df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 9], "c": [3, 1, 4, 2]})
        >>> pares_correlacionados(df, umbral=0.4)
          col_a col_b      corr
        0     a     b  0.994...
        (a~c = 0.0 y b~c = -0.04 no superan 0.4, así que no aparecen)

    💡 Pistas:
        - corr = df.corr()
        - np.triu(np.ones(corr.shape, dtype=bool), k=1) es una máscara True SOLO arriba
          de la diagonal (k=1 deja fuera la diagonal).
        - Opción A: recorre i < j con dos for y guarda (columna_i, columna_j, corr.iloc[i, j]).
        - Opción B: corr.where(mascara).stack() da una Series con índice (col_a, col_b).
        - .abs() para el valor absoluto; sort_values(..., ascending=False);
          reset_index(drop=True).
    """
    # 1. calcula la matriz de correlación
    # 2. recorre SOLO el triángulo superior (i < j)
    # 3. guarda los pares con |corr| > umbral
    # 4. arma el DataFrame col_a / col_b / corr
    # 5. ordénalo por |corr| de mayor a menor y reinicia el índice
    raise NotImplementedError


# ---------------------------------------------------------------- 3
def columnas_a_eliminar(X_train: pd.DataFrame, umbral: float = 0.4) -> list:
    """Decide qué columnas eliminar por estar muy correlacionadas, usando SOLO X_train.

    Estrategia (la receta más usada, con el triángulo superior):
        Una columna se elimina si tiene |corr| > umbral con ALGUNA columna que aparece
        ANTES que ella en X_train.
    Es decir: de cada par (col_a, col_b) se elimina siempre col_b, la que va después.
    Así nunca se eliminan las dos columnas de un mismo par.

    Ojo con las "cadenas": si a~b y b~c superan el umbral, se eliminan b y c
    (c tiene una columna anterior, b, con la que supera el umbral), aunque b ya esté
    en la lista. Cada columna aparece una sola vez en el resultado.

    El profesor: "toda la experimentación... tiene que hacerse con este dato [X_train] y
    luego eso tiene que generalizarse aquí en el dato de prueba". Por eso esta función
    solo recibe X_train.

    Args:
        X_train: predictores de entrenamiento (todas numéricas).
        umbral: valor entre 0 y 1 (en la clase, 0.4).

    Devuelve:
        list de nombres de columnas a eliminar, en el ORDEN en que aparecen en X_train.
        Lista vacía si no hay ninguna.

    Ejemplo:
        >>> X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 9], "c": [3, 1, 4, 2]})
        >>> columnas_a_eliminar(X, umbral=0.4)
        ['b']        # a~b supera 0.4 -> se elimina la que va después (b); a se queda

    💡 Pistas:
        - corr = X_train.corr().abs()
        - superior = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        - En `superior`, la COLUMNA c contiene la correlación de c con las columnas
          anteriores. Elimina c si (superior[c] > umbral).any().
    """
    # 1. matriz de correlación en valor absoluto (solo con X_train)
    # 2. quédate con el triángulo superior (k=1)
    # 3. recorre las columnas en orden: si alguna correlación con una anterior > umbral -> a la lista
    # 4. devuelve la lista
    raise NotImplementedError


# ---------------------------------------------------------------- 4
def comparar_pearson_spearman(x, y) -> dict:
    """Calcula la correlación de Pearson y la de Spearman entre x e y.

    - Pearson mide relación LINEAL (lo que usa la matriz de correlación de la clase).
    - Spearman = Pearson aplicado a los RANKINGS (posiciones) de x y de y. Mide si la
      relación es MONÓTONA: "cuando x sube, y siempre sube" (o siempre baja), aunque
      no sea en línea recta.

    Casos que verás en los tests:
        - y = e^x (curva que siempre sube): Spearman = 1, Pearson < 1.
        - y = x**2 con x simétrico (-3..3): forma de U, relación PERFECTA, pero
          Pearson ≈ 0 y también Spearman ≈ 0 (no es monótona). Por eso "a veces falla".

    Rankings: el más pequeño recibe 1, el siguiente 2, ... Si hay EMPATES, cada uno recibe
    el PROMEDIO de las posiciones (pd.Series(...).rank() hace eso por defecto).

    Args:
        x, y: valores numéricos de la misma longitud (lista, array o Series).

    Devuelve:
        dict {"pearson": float, "spearman": float}.

    Errores:
        ValueError si x e y no tienen la misma longitud.

    Ejemplo:
        >>> comparar_pearson_spearman([1, 2, 3, 4, 5], [1, 4, 9, 16, 100])
        {'pearson': 0.795..., 'spearman': 1.0}

    💡 Pistas:
        - Puedes reutilizar tu correlacion_pearson_manual.
        - pd.Series(np.asarray(x)).rank() devuelve los rankings (empates -> promedio).
        - Spearman = Pearson(ranking de x, ranking de y).
    """
    # 1. convierte x e y a arrays y valida longitudes
    # 2. pearson = correlación de Pearson de x e y
    # 3. rx, ry = rankings de x y de y
    # 4. spearman = correlación de Pearson de rx y ry
    # 5. devuelve el diccionario (floats)
    raise NotImplementedError


# ---------------------------------------------------------------- 5
def evaluar_eliminacion(X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series,
                        umbral: float = 0.4) -> dict:
    """Compara el baseline de la clase 13 con TODAS las columnas vs. SIN las correlacionadas.

    Pasos:
        1. Decide las columnas a eliminar con columnas_a_eliminar(X_train, umbral)
           (¡solo con X_train!).
        2. Quita ESAS MISMAS columnas de X_train y de X_test.
        3. Entrena el modelo baseline dos veces (todas / reducido) y mide accuracy en test:
               RandomForestClassifier(max_depth=5, max_leaf_nodes=20, random_state=1)

    Args:
        X_train, X_test: predictores de train y test.
        y_train, y_test: objetivo de train y test.
        umbral: umbral de |corr| (en la clase, 0.4).

    Devuelve:
        dict con 4 claves:
            {"todas": accuracy (float) con todas las columnas,
             "reducido": accuracy (float) sin las columnas eliminadas,
             "n_columnas": int, número de columnas que QUEDAN,
             "eliminadas": list, las columnas eliminadas (orden de X_train)}

    Ejemplo:
        >>> evaluar_eliminacion(X_tr, X_te, y_tr, y_te, umbral=0.4)
        {'todas': 0.74, 'reducido': 0.7375, 'n_columnas': 16,
         'eliminadas': ['pc', 'px_width', 'sc_w', 'three_g']}
        (valores del dataset de móviles con pares correlacionados de los tests)

    💡 Pistas:
        - from sklearn.ensemble import RandomForestClassifier
        - from sklearn.metrics import accuracy_score
        - X_train.drop(columns=eliminadas) no modifica X_train (devuelve uno nuevo).
        - Si te ayuda, escribe una función auxiliar _accuracy_rf(Xtr, ytr, Xte, yte).
    """
    # 1. eliminadas = columnas_a_eliminar(X_train, umbral)
    # 2. X_train_red, X_test_red = quitar esas columnas de ambos
    # 3. acc_todas = RF baseline con todas las columnas
    # 4. acc_reducido = RF baseline con las columnas que quedan
    # 5. devuelve el diccionario con las 4 claves
    raise NotImplementedError
