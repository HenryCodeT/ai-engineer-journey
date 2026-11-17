"""
Clase 11 · Caso Panorama de la vivienda, parte 2: escenario avanzado
=====================================================================

Implementa cada función reemplazando `raise NotImplementedError`.
Evalúa con:

    python -m framework evaluar cursos/nivelacion/n1-python-data-science/m2-outliers-encoding/clase-11-vivienda-2

(o solo la nota: python -m pytest cursos/nivelacion/n1-python-data-science/m2-outliers-encoding/clase-11-vivienda-2)

Reglas:
- No modifiques los DataFrames de entrada (trabaja sobre copias).
- Evita bucles `for` sobre filas: usa pandas / scikit-learn.
- Dataset de práctica: data/raw/viviendas_lima.csv  (objetivo: "precio_usd"; excluye "id").

Hilo del ejercicio (sigue la clase, que usaba Melbourne con objetivo "Price"):
  1. Separar X (predictores numéricos) e y (objetivo), comprobando que y no tenga nulos.
  2. Imputar X con KNNImputer SIN perder filas, columnas ni índice.
  3. Detectar outliers POR FILA y devolver una máscara de inliers (True = me quedo con la fila).
  4. Filtrar X e y con la MISMA máscara (que no se desalineen).
  5. Encadenar 1 → 4: el "escenario avanzado" del profesor (imputar → quitar outliers).
  6. (Reto) Medir con números el debate "¿imputar antes o después de dividir train/test?".

La clase 10 ya tiene `dividir`, `score_dataset`, `imputar_train_test`… Aquí NO se repiten:
esta clase aporta el escenario avanzado y la comparación antes/después.

Orden sugerido: 1 → 2 → 3 → 4 → 5 → 6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 1
def separar_x_y(df: pd.DataFrame, objetivo: str, excluir: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Separa los predictores numéricos (X) de la variable objetivo (y).

    El profesor: *"primero quiero mantener separado mi X de mi Y"*. Se queda solo con las
    columnas numéricas (las categóricas "las vemos después") y quita el objetivo de X.
    Y advierte: *"la variable objetivo no debería tener nulos; si hay nulos tenemos un
    problema, porque tendríamos que imputar la variable objetivo, que ya es otro dolor de cabeza"*.
    Por eso, si `y` tiene algún NaN, esta función NO sigue: lanza un error.

    Args:
        df: DataFrame original (NO se modifica).
        objetivo: nombre de la columna a predecir (p. ej. "precio_usd").
        excluir: columnas numéricas que tampoco deben ir en X (p. ej. ["id"]).
                 Si es None, no se excluye nada más. Si alguna no existe, se ignora.

    Devuelve:
        (X, y):
          X -> DataFrame con las columnas NUMÉRICAS de df, sin `objetivo` ni `excluir`,
               en el mismo orden en que aparecen en df y con el mismo índice.
          y -> Serie df[objetivo] (con el mismo índice).

    Errores:
        ValueError si `objetivo` no es una columna de df.
        ValueError si df[objetivo] tiene al menos un NaN.

    Ejemplo:
        >>> df = pd.DataFrame({"id": [1, 2], "distrito": ["A", "B"],
        ...                    "area": [50.0, None], "precio": [100.0, 200.0]})
        >>> X, y = separar_x_y(df, "precio", excluir=["id"])
        >>> list(X.columns), list(y)
        (['area'], [100.0, 200.0])
        # "distrito" sale por no ser numérica; "id" por estar en excluir; los NaN de X se quedan

    💡 Pistas:
        - df.select_dtypes(include="number") deja solo las numéricas.
        - .drop(columns=[...], errors="ignore") no falla si una columna no existe.
        - df[objetivo].isna().any() es True si hay algún nulo.

    Esqueleto (un paso por línea):
        # paso 1: si objetivo no está en df.columns -> raise ValueError("...")
        # paso 2: si y tiene nulos -> raise ValueError("...")
        numericas = ...        # paso 3: solo columnas numéricas
        X = ...                # paso 4: quitar objetivo + excluir (y .copy())
        y = ...                # paso 5: la columna objetivo (.copy())
        return X, y
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 2
def imputar_knn_df(X: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Rellena los NaN de X con KNNImputer y devuelve un DataFrame del MISMO tamaño.

    El profesor: *"la imputación no elimina nada de registros ni tampoco elimina columnas,
    simplemente rellena valores nulos; por lo tanto los tamaños coinciden"* y *"por favor no
    se olviden de que estos tamaños coincidan sí o sí"*.

    Ojo: `KNNImputer.fit_transform` devuelve un array de numpy (sin nombres de columnas ni
    índice). Tienes que reconstruir el DataFrame con las columnas Y el índice de X; si
    pierdes el índice, después no podrás alinear X con y.

    Args:
        X: DataFrame numérico con NaN (NO se modifica).
        k: número de vecinos (n_neighbors) del KNNImputer. Por defecto 5.

    Devuelve:
        DataFrame sin NaN, con las mismas columnas (mismo orden) y el mismo índice que X.
        Si la forma (shape) del resultado no coincide con la de X, lanza ValueError.

    Ejemplo:
        >>> X = pd.DataFrame({"a": [1.0, 2.0, None], "b": [10.0, 20.0, 30.0]}, index=[7, 8, 9])
        >>> imputar_knn_df(X, k=2)
             a     b
        7  1.0  10.0
        8  2.0  20.0
        9  1.5  30.0
        # la fila 9 toma el promedio de "a" de sus 2 vecinos más cercanos (filas 7 y 8)

    💡 Pistas:
        - from sklearn.impute import KNNImputer
        - pd.DataFrame(matriz, columns=X.columns, index=X.index)

    Esqueleto:
        imputador = ...        # paso 1: KNNImputer(n_neighbors=k)
        matriz = ...           # paso 2: fit_transform(X)
        X_imp = ...            # paso 3: volver a DataFrame (columnas e índice de X)
        # paso 4: si X_imp.shape != X.shape -> raise ValueError("...")
        return X_imp
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 3
def mascara_inliers(X: pd.DataFrame, metodo: str = "isolation", contamination: float = 0.1,
                    random_state: int = 0) -> pd.Series:
    """Detecta outliers FILA POR FILA y devuelve True para las filas que se QUEDAN (inliers).

    En clase se usó PyOD: `KNN(contamination=0.1).fit(X_imp)` y luego `labels_`, donde
    0 = inlier y 1 = outlier. PyOD no está instalado aquí, así que usamos los equivalentes
    de scikit-learn, que usan OTRA convención: `fit_predict` devuelve 1 = inlier y
    -1 = outlier. Para no confundirnos, devolvemos directamente True/False:
    True = inlier (me quedo con la fila), False = outlier (la quito).

    Recuerda lo que respondió el profesor a Samuel: *"no lo hace por variable, lo hace por
    registro, porque una fila es un vector de características en R^n"*: cada fila es UN punto.

    Args:
        X: DataFrame numérico SIN NaN (primero se imputa; NO se modifica).
        metodo: "isolation" -> IsolationForest(contamination=..., random_state=...)
                "lof"       -> LocalOutlierFactor(n_neighbors=20, contamination=...)
                               (LOF, como el KNN de PyOD, se basa en distancias a vecinos)
        contamination: fracción esperada de outliers (el "grado" 0.1 de la clase = 10 %).
        random_state: semilla (solo la usa IsolationForest).

    Devuelve:
        Serie de bool, con el MISMO índice que X y nombre "inlier".

    Errores:
        ValueError si X tiene algún NaN (los detectores no aceptan nulos: imputa antes).
        ValueError si `metodo` no es "isolation" ni "lof".

    Ejemplo:
        >>> X = pd.DataFrame({"a": [1.0, 1.2, 0.9, 1.1, 0.8, 1.0, 50.0]})
        >>> mascara_inliers(X, metodo="isolation", contamination=0.1).tolist()
        [True, True, True, True, True, True, False]
        # (LOF necesita más filas que vecinos: con 7 filas no es fiable; pruébalo con el dataset)

    💡 Pistas:
        - from sklearn.ensemble import IsolationForest
        - from sklearn.neighbors import LocalOutlierFactor
        - pred = detector.fit_predict(X)  ->  array de 1 y -1
        - pd.Series(pred == 1, index=X.index, name="inlier")

    Esqueleto:
        # paso 1: si X tiene nulos -> raise ValueError
        # paso 2: según metodo, crear el detector (o raise ValueError si no existe)
        pred = ...             # paso 3: fit_predict
        return ...             # paso 4: Serie booleana con el índice de X
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 4
def filtrar_xy(X: pd.DataFrame, y: pd.Series, mascara: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Aplica la MISMA máscara a X y a y, para que sigan alineados fila a fila.

    El profesor: *"en ningún momento he tocado a y [...] y aquí simplemente va a ser un
    filtrado, porque los índices coinciden perfectamente"*. Melbourne pasó de 18 396 a
    16 842 filas, y X e y tenían que quedar con 16 842 las dos.

    Antes de filtrar, comprueba que X, y y la máscara tengan EXACTAMENTE el mismo índice
    (mismos valores y mismo orden). Si no, lanza ValueError: filtrar con índices distintos
    mezclaría el precio de una casa con las características de otra.

    Args:
        X: predictores (NO se modifica).
        y: objetivo (NO se modifica).
        mascara: Serie booleana (True = me quedo con la fila).

    Devuelve:
        (X_filtrado, y_filtrado): solo las filas con True, conservando sus índices
        originales (NO reinicies el índice) y las columnas de X.

    Errores:
        ValueError si los índices de X, y y mascara no son iguales.

    Ejemplo:
        >>> X = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 11, 12])
        >>> y = pd.Series([100, 200, 300], index=[10, 11, 12])
        >>> m = pd.Series([True, False, True], index=[10, 11, 12])
        >>> Xf, yf = filtrar_xy(X, y, m)
        >>> list(Xf.index), list(yf)
        ([10, 12], [100, 300])

    💡 Pistas:
        - X.index.equals(y.index) compara valores Y orden.
        - X.loc[mascara] se queda con las filas True.

    Esqueleto:
        # paso 1: comprobar índices (X vs y, X vs mascara) -> raise ValueError
        X_f = ...              # paso 2: filtrar X (.copy())
        y_f = ...              # paso 3: filtrar y con la misma máscara (.copy())
        return X_f, y_f
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 5
def limpiar_escenario_avanzado(df: pd.DataFrame, objetivo: str, excluir: list[str] | None = None,
                               k: int = 5, metodo: str = "isolation", contamination: float = 0.1,
                               random_state: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """El "escenario avanzado" del profesor, de principio a fin (antes de dividir train/test).

    Pasos (los mismos que en clase):
      1) separar X e y                       -> separar_x_y
      2) imputar X con KNN                   -> imputar_knn_df
      3) detectar outliers sobre X imputado  -> mascara_inliers
      4) filtrar X imputado e y              -> filtrar_xy

    El resultado es lo que el profesor llamó `X_r` e `y_r` (r de "removed outliers"), listo
    para pasar al `train_test_split` + `score_dataset` de la clase 10.

    Args:
        df, objetivo, excluir: como en separar_x_y.
        k: vecinos del KNNImputer.
        metodo, contamination, random_state: como en mascara_inliers.

    Devuelve:
        (X_r, y_r): X imputado y sin outliers (sin NaN) e y filtrado, con índices alineados.

    Ejemplo (con viviendas_lima.csv, excluir=["id"], contamination=0.1):
        X_r tiene ~90 % de las filas, 7 columnas y ningún NaN; y_r tiene las mismas filas.

    💡 Pista: aquí no hay nada nuevo; solo llama a tus 4 funciones en orden.

    Esqueleto:
        X, y = ...             # paso 1
        X_imp = ...            # paso 2
        mascara = ...          # paso 3
        return ...             # paso 4
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 6 (reto)
def comparar_antes_vs_despues_split(df: pd.DataFrame, objetivo: str, excluir: list[str] | None = None,
                                    test_size: float = 0.2, random_state: int = 0) -> dict:
    """Mide el debate de la clase: ¿ajustar la imputación ANTES o DESPUÉS de dividir?

    Usamos imputación por MEDIANA (SimpleImputer) porque así se puede ver exactamente qué
    "aprendió" el imputador: una mediana por columna.

    Pasos:
      1) X, y = separar_x_y(df, objetivo, excluir)
      2) X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                             random_state=random_state)
      3) ANTES (postura del profesor): SimpleImputer(strategy="median").fit(X)   <- toda la data
         y transformar X_train y X_test con ese imputador.
      4) DESPUÉS (práctica estándar de scikit-learn): SimpleImputer(strategy="median").fit(X_train)
         <- solo train, y transformar X_train y X_test con ese imputador.
      5) En cada caso: entrenar RandomForestRegressor(n_estimators=50, max_depth=6,
         random_state=0) con (X_train imputado, y_train), predecir X_test imputado y
         calcular el MAE contra y_test (el mismo modelo que `score_dataset` de la clase 10).

    El split es EL MISMO en los dos casos: lo único que cambia es con qué datos se ajustó
    el imputador. Así la diferencia se debe solo a eso.

    Args:
        df, objetivo, excluir: como en separar_x_y.
        test_size: fracción para test (0.2 = 80/20, como en clase).
        random_state: semilla del train_test_split.

    Devuelve:
        dict con:
          "mae_antes"        -> float, MAE con el imputador ajustado en TODO X, redondeado a 2
          "mae_despues"      -> float, MAE con el imputador ajustado SOLO en X_train, redondeado a 2
          "diferencia"       -> float, mae_despues - mae_antes (con los valores SIN redondear),
                                redondeado a 2
          "medianas_antes"   -> pd.Series: mediana aprendida por columna (índice = columnas de X)
          "medianas_despues" -> pd.Series: ídem, pero aprendida solo de X_train

    Ejemplo (con viviendas_lima.csv, excluir=["id"]):
        r = comparar_antes_vs_despues_split(df, "precio_usd", excluir=["id"])
        r["medianas_despues"] es igual a X_train.median()  (¡no mira el test!)
        r["medianas_antes"]   es igual a X.median()        (incluye filas de test)

    💡 Pistas:
        - from sklearn.impute import SimpleImputer ; imputador.statistics_ son las medianas.
        - pd.Series(imputador.statistics_, index=X.columns)
        - from sklearn.metrics import mean_absolute_error
        - float(...) para que el MAE sea un float de Python.

    Esqueleto:
        X, y = ...                                   # paso 1
        X_train, X_test, y_train, y_test = ...       # paso 2
        imp_antes = ...                              # paso 3: fit con X
        mae_antes = ...                              #         entrenar + MAE
        imp_despues = ...                            # paso 4: fit con X_train
        mae_despues = ...                            #         entrenar + MAE
        return {...}                                 # paso 5
    """
    raise NotImplementedError
