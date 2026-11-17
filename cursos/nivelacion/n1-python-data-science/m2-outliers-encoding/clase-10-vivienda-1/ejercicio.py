"""
Clase 10 · Caso Panorama de la vivienda (1): comparar escenarios de preprocesamiento
===================================================================================

Implementa cada función reemplazando `raise NotImplementedError`.
Evalúa con:

    python -m framework evaluar cursos/nivelacion/n1-python-data-science/m2-outliers-encoding/clase-10-vivienda-1

(o solo la nota: python -m pytest cursos/nivelacion/n1-python-data-science/m2-outliers-encoding/clase-10-vivienda-1)

Reglas:
- No modifiques los DataFrames de entrada (trabaja sobre copias).
- Solo pandas, numpy y scikit-learn (PyOD no está instalado: para outliers usamos
  IsolationForest de scikit-learn, que también es uno de los algoritmos de PyOD).
- Dataset de práctica: data/raw/viviendas_lima.csv (objetivo: `precio_usd`).
  Es el análogo del dataset de Melbourne que usó el profesor (objetivo: `Price`).

Hilo del ejercicio (la metodología de la clase):
  1. Separar y (lo que quiero predecir) de X (los predictores), solo numéricas.
  2. Dividir UNA sola vez en train/test 80/20 con random_state=0 (igual para todos).
  3. `score_dataset`: entrena un RandomForest fijo y devuelve el MAE. Tampoco cambia.
  4. Imputar: fit_transform SOLO en train, transform en test (la regla clave).
  5. Quitar outliers SOLO del train (X_train e y_train a la vez), máximo 10 %.
  6. Juntar todo: correr varios escenarios y ordenarlos por MAE.

Orden sugerido: 1 -> 2 -> 3 -> 4 -> 5 -> 6. La 6 usa todas las anteriores.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------- 1
def preparar_xy(
    df: pd.DataFrame,
    objetivo: str,
    solo_numericas: bool = True,
    descartar: list | None = None,
) -> tuple:
    """Separa el DataFrame en X (predictores) e y (variable objetivo).

    El profesor: "y es igual a df['Price'] ... y X es el dataframe quitándole la
    variable objetivo y quitándole todos los tipos de datos que son object, para
    quedarme únicamente con datos numéricos por el momento".

    Args:
        df: DataFrame completo (NO se modifica).
        objetivo: nombre de la columna a predecir (p. ej. "precio_usd").
        solo_numericas: si es True, X se queda sin las columnas de tipo object
            (usa select_dtypes(exclude="object"), como en la clase).
            Si es False, X conserva todas las columnas (menos objetivo y descartar).
        descartar: lista opcional de columnas que también salen de X
            (p. ej. ["id"], que es un número pero no dice nada del precio,
            o las columnas con demasiados faltantes). None = no descartar nada.

    Devuelve:
        Tupla (X, y):
          - X: DataFrame nuevo sin `objetivo` ni las columnas de `descartar`,
            con las columnas en el MISMO orden que en df.
          - y: Serie con la columna objetivo (su .name es `objetivo`).
        Ambos conservan el índice original de df.

    Ejemplo:
        >>> df = pd.DataFrame({"id": [1, 2], "distrito": ["Surco", "Lince"],
        ...                    "area_m2": [80.0, 60.0], "precio_usd": [150000.0, 90000.0]})
        >>> X, y = preparar_xy(df, "precio_usd", descartar=["id"])
        >>> list(X.columns), y.tolist()
        (['area_m2'], [150000.0, 90000.0])

    💡 Pistas:
        - df.drop(columns=[...]) devuelve un DataFrame NUEVO (no toca df).
        - `descartar or []` convierte None en lista vacía.
        - df.select_dtypes(exclude="object") deja solo lo que no es texto.

    Esqueleto (un paso por línea):
        y = ...          # paso 1: la columna objetivo
        quitar = ...     # paso 2: [objetivo] + lista de descartar
        X = ...          # paso 3: df sin esas columnas
        if solo_numericas:
            X = ...      # paso 4: quitar columnas object
        return X, y
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 2
def dividir(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 0) -> tuple:
    """Divide X e y en entrenamiento y prueba (80/20, random_state=0 por defecto).

    El profesor: "80 20 y un random state de 0, por favor mantengan esta
    configuración para que todos tengamos lo mismo". Y además: "esta división no
    tiene nada que ver con ningún escenario": se hace UNA vez, arriba, y todos los
    escenarios usan los mismos train y test. Así la comparación es justa.

    Args:
        X: predictores.
        y: objetivo (mismo largo que X).
        test_size: fracción para prueba (0.2 = 20 %).
        random_state: semilla del barajado.

    Devuelve:
        Tupla (X_train, X_test, y_train, y_test), en ESE orden (el mismo que
        devuelve train_test_split), conservando los índices originales.

    Ejemplo:
        >>> X = pd.DataFrame({"a": range(10)}); y = pd.Series(range(10))
        >>> X_train, X_test, y_train, y_test = dividir(X, y)
        >>> len(X_train), len(X_test)
        (8, 2)

    💡 Pistas:
        - sklearn.model_selection.train_test_split ya hace todo; pásale
          test_size y random_state.

    Esqueleto:
        return ...       # una sola línea con train_test_split
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 3
def score_dataset(X_train, X_valid, y_train, y_valid) -> float:
    """Entrena un RandomForestRegressor y devuelve el MAE sobre los datos de validación.

    Es la función "base" de la clase, la que no cambia entre escenarios:
    "va a entrenar con X_train, luego hacemos una predicción y vamos a retornar
    el mean absolute error".

    Modelo (hiperparámetros FIJOS, los de la clase):
        RandomForestRegressor(n_estimators=50, max_depth=6, random_state=0)
          - n_estimators=50: 50 árboles; cada uno predice y se promedian.
          - max_depth=6: árboles poco profundos para no sobreajustar (overfitting).
          - random_state=0: mismo resultado cada vez que lo corres.

    Args:
        X_train, y_train: datos para ENTRENAR (sin NaN).
        X_valid, y_valid: datos para MEDIR (sin NaN). El modelo nunca los ve al entrenar.

    Devuelve:
        float: mean_absolute_error(y_valid, predicciones). En las mismas unidades
        que y (aquí, dólares): "en promedio me equivoco por tantos USD".

    Ejemplo (si y es constante, el bosque aprende esa constante):
        >>> X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}); y = pd.Series([5.0] * 4)
        >>> score_dataset(X, X, y, y)
        0.0

    💡 Pistas:
        - modelo.fit(X_train, y_train) y luego modelo.predict(X_valid).
        - sklearn.metrics.mean_absolute_error(reales, predichos).
        - Devuelve float(...) para no entregar un tipo numpy.

    Esqueleto:
        modelo = ...       # paso 1: RandomForestRegressor con los 3 hiperparámetros
        ...                # paso 2: entrenar con train
        pred = ...         # paso 3: predecir sobre valid
        return ...         # paso 4: float del MAE
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 4
def imputar_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame, imputador) -> tuple:
    """Imputa train y test SIN filtrar información del test (la regla clave de la clase).

    El profesor: "fit_transform es para que aprendan los parámetros de los datos de
    entrenamiento, y luego los aplico a los datos de test: ahí ya NO puedo utilizar
    fit_transform, voy a utilizar solamente transform".

    Args:
        X_train: predictores de entrenamiento (numéricos, pueden tener NaN).
        X_test: predictores de prueba (mismas columnas, pueden tener NaN).
        imputador: un imputador de scikit-learn SIN entrenar, por ejemplo
            SimpleImputer(strategy="median"), KNNImputer(n_neighbors=5) o
            IterativeImputer(random_state=0). Se entrena DENTRO de esta función.

    Devuelve:
        Tupla (X_train_imp, X_test_imp) de DataFrames sin NaN, con las MISMAS
        columnas y el MISMO índice que X_train y X_test respectivamente.
        (scikit-learn devuelve arrays de numpy: hay que volver a envolverlos.)

    Ejemplo (la mediana sale SOLO del train):
        >>> tr = pd.DataFrame({"banos": [1.0, 2.0, 3.0, None]}, index=[0, 1, 2, 3])
        >>> te = pd.DataFrame({"banos": [None, 9.0]}, index=[7, 8])
        >>> a, b = imputar_train_test(tr, te, SimpleImputer(strategy="median"))
        >>> a["banos"].tolist(), b["banos"].tolist()
        ([1.0, 2.0, 3.0, 2.0], [2.0, 9.0])
        # El NaN del test se rellena con 2.0 (mediana del train), no con 9.0.

    💡 Pistas:
        - imputador.fit_transform(X_train)  -> aprende y transforma el train.
        - imputador.transform(X_test)       -> SOLO transforma el test.
        - pd.DataFrame(array, columns=X_train.columns, index=X_train.index).

    Esqueleto:
        arr_train = ...    # paso 1: fit_transform sobre train
        arr_test = ...     # paso 2: transform sobre test
        X_train_imp = ...  # paso 3: DataFrame con columnas e índice de train
        X_test_imp = ...   # paso 4: DataFrame con columnas e índice de test
        return X_train_imp, X_test_imp
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 5
def quitar_outliers_train(X_train: pd.DataFrame, y_train: pd.Series, contamination: float = 0.1) -> tuple:
    """Elimina outliers SOLO del entrenamiento con IsolationForest, filtrando X e y a la vez.

    La consigna de la clase: eliminar outliers "con un máximo de 10 % de
    contaminación". Usamos IsolationForest (está en PyOD y en scikit-learn).
    El test NUNCA se toca: en la vida real no puedes "borrar" a un cliente que te
    pide una predicción solo porque es raro.

    Args:
        X_train: predictores de entrenamiento YA imputados (sin NaN).
        y_train: objetivo de entrenamiento, alineado con X_train.
        contamination: fracción máxima de filas a marcar como outlier.
            Debe cumplir 0 < contamination <= 0.1; si no -> ValueError.

    Devuelve:
        Tupla (X_filtrado, y_filtrado): copias con solo las filas inlier,
        conservando el índice original (no lo reinicies). X e y deben quedar
        con exactamente los mismos índices.

    Detector (FIJO):
        IsolationForest(contamination=contamination, random_state=0)
        .fit_predict(X_train) devuelve 1 = inlier, -1 = outlier
        (ojo: es la convención de scikit-learn, distinta de PyOD).

    Ejemplo:
        20 viviendas normales alrededor de (50, 50) y una en (100, 100):
        con contamination=0.05 se elimina solo la de (100, 100), de X y de y.

    💡 Pistas:
        - Valida contamination ANTES de entrenar nada.
        - mascara = predicciones == 1   (array booleano).
        - X_train[mascara] y y_train[mascara] usan la MISMA máscara -> quedan alineados.

    Esqueleto:
        if ...: raise ValueError(...)   # paso 1: validar contamination
        detector = ...                  # paso 2: IsolationForest(...)
        pred = ...                      # paso 3: detector.fit_predict(X_train)
        mascara = ...                   # paso 4: pred == 1
        return ..., ...                 # paso 5: X e y filtrados, .copy()
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 6
def evaluar_escenarios(
    df: pd.DataFrame,
    escenarios: dict,
    objetivo: str = "precio_usd",
    descartar: list | None = None,
) -> pd.DataFrame:
    """Corre varios escenarios de preprocesamiento y los ordena por MAE (menor = mejor).

    Es la "automatización" que anunció el profesor: lo que cambia entre escenarios
    es el imputador y si se quitan outliers; la división y el modelo son fijos.

    Args:
        df: dataset completo (NO se modifica).
        escenarios: dict  nombre -> (imputador, quitar_outliers)
            - imputador: imputador de scikit-learn sin entrenar.
            - quitar_outliers: bool; True = aplicar quitar_outliers_train con
              contamination=0.1 DESPUÉS de imputar.
        objetivo: columna a predecir.
        descartar: columnas extra a sacar de X (se pasa a preparar_xy).

    Pasos (los de la clase):
        0. Una sola vez: preparar_xy (solo numéricas) y dividir (80/20, rs=0).
        Para cada escenario:
        1. Definir el imputador -> usa clone(imputador) para estrenar uno nuevo
           (así el objeto que te pasaron no queda entrenado y el resultado no
           depende del orden de los escenarios).
        2. imputar_train_test.
        3. Si quitar_outliers: quitar_outliers_train (solo train).
        4. score_dataset -> MAE.

    Devuelve:
        DataFrame con columnas ["escenario", "mae", "filas_train"]:
          - escenario: el nombre (clave del dict).
          - mae: float devuelto por score_dataset.
          - filas_train: filas con las que se entrenó (menos si se quitaron outliers).
        Ordenado por "mae" de menor a mayor, con índice 0, 1, 2, ...

    Ejemplo:
        >>> esc = {"basico_mediana": (SimpleImputer(strategy="median"), False),
        ...        "mediana_sin_outliers": (SimpleImputer(strategy="median"), True)}
        >>> evaluar_escenarios(df, esc, descartar=["id"])
                     escenario       mae  filas_train
        0  ...                  ...        ...
        1  ...                  ...        ...

    💡 Pistas:
        - Acumula un dict por escenario en una lista y al final pd.DataFrame(lista).
        - .sort_values("mae").reset_index(drop=True).

    Esqueleto:
        X, y = ...                                  # paso 0a
        X_train, X_test, y_train, y_test = ...      # paso 0b
        filas = []
        for nombre, (imputador, quitar) in escenarios.items():
            imp = ...                               # paso 1: clone
            Xtr, Xte = ...                          # paso 2
            ytr = y_train
            if quitar:
                Xtr, ytr = ...                      # paso 3
            mae = ...                               # paso 4
            filas.append({...})
        return ...                                  # ordenar y reiniciar índice
    """
    raise NotImplementedError
