"""
Clase 14 · Eliminación de variables con baja varianza
=====================================================

Implementa cada función reemplazando `raise NotImplementedError`.
Evalúa con:

    python -m framework evaluar cursos/nivelacion/n1-python-data-science/m3-seleccion-variables/clase-14-baja-varianza

(o solo la nota: python -m pytest cursos/nivelacion/n1-python-data-science/m3-seleccion-variables/clase-14-baja-varianza)

Reglas:
- No modifiques las entradas (X_train, X_test, ...): trabaja sobre copias si hace falta.
- Solo necesitas pandas, numpy y scikit-learn.
- Dataset: los tests usan el MISMO dataset sintético de "móviles" de la clase 13
  (20 variables, 4 rangos de precio, semilla 13) y la misma división
  train_test_split(test_size=0.2, stratify=y, random_state=1). Así comparas con tu baseline.

Hilo del ejercicio (sigue la clase):
  1. La varianza a mano: suma de diferencias al cuadrado / n  (el "-5 y 5" de la pizarra).
  2. Ranking de varianza de cada columna (de mayor a menor).
  3. VarianceThreshold: fit SOLO en train, transform en train y test.
  4. Curva "número de variables vs umbral" (el "método del codo" del profesor).
  5. 📌 Extra: la varianza depende de la ESCALA (original vs MinMax vs estandarizada).
  6. Comparar el RF baseline con todas las variables vs las seleccionadas por varianza.

Orden sugerido: 1 -> 6. La 2 puede usar la 1; la 6 debe usar la 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 1
def varianza_manual(valores, ddof: int = 0) -> float:
    """Calcula la varianza SIN usar .var(), np.var(), .std() ni np.std().

    Fórmula de la clase: "se sacan las diferencias [respecto a la media], se elevan al
    cuadrado, se suman y se dividen sobre la cantidad de puntos".

        varianza = suma( (x - media)^2 ) / (n - ddof)

    - ddof=0 -> divide entre n     (varianza "poblacional"; la que usa VarianceThreshold).
    - ddof=1 -> divide entre n - 1 (varianza "muestral"; la que usa pandas .var() por defecto).

    Args:
        valores: lista, array de numpy o Series de pandas con números.
        ddof: 0 o 1 (ver arriba).

    Devuelve:
        float.

    Errores:
        ValueError si `valores` está vacío.

    Ejemplo:
        >>> varianza_manual([-5, 5])      # media 0; ((-5-0)^2 + (5-0)^2) / 2
        25.0
        >>> varianza_manual([-10, 10])    # misma media 0, pero más dispersos
        100.0

    💡 Pistas:
        - np.asarray(valores, dtype=float) convierte todo en un array de floats.
        - La media sí puedes calcularla con .mean() o con suma / n.
        - (arr - media) ** 2 eleva al cuadrado cada diferencia; .sum() las suma.
    """
    # 1. convierte `valores` en un array de floats
    # 2. si está vacío -> raise ValueError("...")
    # 3. calcula la media
    # 4. calcula las diferencias al cuadrado y súmalas
    # 5. divide entre (n - ddof) y devuelve un float
    raise NotImplementedError


# ---------------------------------------------------------------- 2
def ranking_varianza(df: pd.DataFrame, ddof: int = 0) -> pd.Series:
    """Devuelve la varianza de CADA columna, ordenada de MAYOR a MENOR.

    Es lo que mira VarianceThreshold antes de decidir: cada columna por separado,
    sin mirar la variable objetivo.

    Args:
        df: DataFrame con columnas numéricas (por ejemplo X_train).
        ddof: igual que en varianza_manual (0 por defecto, como VarianceThreshold).

    Devuelve:
        pd.Series con índice = nombre de la columna y valores = varianza (float),
        ordenada de forma descendente.

    Ejemplo (la pizarra del profesor):
        >>> pizarra = pd.DataFrame({"x1": [2, 10, 3, 15], "x2": [1, 2, 1, 1],
        ...                         "x3": [100, 50, 130, 45]})
        >>> ranking_varianza(pizarra)
        x3    1254.6875
        x1      28.2500
        x2       0.1875
        dtype: float64

    💡 Pistas:
        - Puedes recorrer las columnas y usar tu varianza_manual, o usar df.var(ddof=ddof)
          (aquí sí se permite: ¡ya sabes lo que hace por dentro!).
        - .sort_values(ascending=False) ordena de mayor a menor.
    """
    # 1. calcula la varianza de cada columna (con el ddof que te pasan)
    # 2. ponla en una Series con el nombre de la columna como índice
    # 3. ordénala de mayor a menor
    raise NotImplementedError


# ---------------------------------------------------------------- 3
def seleccionar_por_varianza(X_train: pd.DataFrame, X_test: pd.DataFrame,
                             umbral: float = 0.5) -> tuple:
    """Elimina las columnas con baja varianza usando VarianceThreshold (como en la clase).

    Paso a paso:
      - El selector se AJUSTA (fit) SOLO con X_train: ahí calcula la varianza de cada
        columna y decide cuáles conserva (las de varianza ESTRICTAMENTE mayor que `umbral`).
      - Luego TRANSFORMA X_train y X_test con esa misma decisión, para que ambos tengan
        exactamente las mismas columnas.
      - VarianceThreshold devuelve arrays de numpy; tú debes devolver DataFrames con los
        nombres de columna y el índice originales.

    Args:
        X_train: predictores de entrenamiento.
        X_test: predictores de prueba (mismas columnas que X_train).
        umbral: varianza mínima (se conserva si varianza > umbral).

    Devuelve:
        tupla (X_train_sel, X_test_sel, columnas_conservadas):
            X_train_sel, X_test_sel: DataFrames solo con las columnas conservadas,
                                     con el mismo índice que X_train / X_test.
            columnas_conservadas: list de str, en el orden original de X_train.

    Ejemplo:
        >>> X_tr = pd.DataFrame({"a": [1, 2, 1, 1], "b": [0, 10, 0, 10], "c": [5, 5, 5, 5]})
        >>> X_tr_sel, X_te_sel, cols = seleccionar_por_varianza(X_tr, X_tr, umbral=0.5)
        >>> cols                      # var(a)=0.1875, var(b)=25, var(c)=0
        ['b']

    💡 Pistas:
        - from sklearn.feature_selection import VarianceThreshold
        - selector = VarianceThreshold(threshold=umbral); selector.fit(X_train)
        - selector.get_support() da un array de True/False, uno por columna.
        - X_train.columns[mascara] da los nombres; X_train.loc[:, cols] o X_train[cols]
          selecciona columnas (equivale a transform, pero conserva nombres e índice).
    """
    # 1. crea el VarianceThreshold con el umbral
    # 2. ajústalo SOLO con X_train
    # 3. obtén la máscara de columnas conservadas y sus nombres (lista)
    # 4. arma X_train_sel y X_test_sel con esas columnas (DataFrames)
    # 5. devuelve la tupla (X_train_sel, X_test_sel, columnas)
    raise NotImplementedError


# ---------------------------------------------------------------- 4
def curva_umbral(X: pd.DataFrame, umbrales) -> pd.DataFrame:
    """Cuenta cuántas variables sobreviven para cada umbral (la gráfica de la clase).

    El profesor probó umbrales "desde cero hasta dos, contando de 0.1 en 0.1" y graficó
    el número de variables seleccionadas contra el umbral: "se parece bastante al
    método del codo".

    Regla: una columna sobrevive si su varianza (ddof=0) es ESTRICTAMENTE mayor que el
    umbral (igual que VarianceThreshold). Si ninguna sobrevive, el conteo es 0
    (ojo: VarianceThreshold lanza ValueError en ese caso; tu función NO debe fallar).

    Args:
        X: DataFrame numérico (normalmente X_train).
        umbrales: lista o array de umbrales, en el orden que quieras.

    Devuelve:
        pd.DataFrame con una fila por umbral (en el mismo orden recibido) y 2 columnas:
            "umbral"      (float)
            "n_variables" (int)
        y un índice simple 0, 1, 2, ...

    Ejemplo:
        >>> pizarra = pd.DataFrame({"x1": [2, 10, 3, 15], "x2": [1, 2, 1, 1],
        ...                         "x3": [100, 50, 130, 45]})
        >>> curva_umbral(pizarra, [0, 0.5, 100, 5000])
           umbral  n_variables
        0     0.0            3
        1     0.5            2
        2   100.0            1
        3  5000.0            0

    💡 Pistas:
        - Calcula UNA vez las varianzas de todas las columnas (X.var(ddof=0) o tu ranking).
        - Para cada umbral: (varianzas > umbral).sum() cuenta cuántas pasan.
        - Arma una lista de diccionarios {"umbral": ..., "n_variables": ...} y pásala a pd.DataFrame.
    """
    # 1. calcula la varianza (ddof=0) de cada columna
    # 2. recorre los umbrales
    # 3. para cada uno, cuenta cuántas varianzas son > umbral
    # 4. devuelve el DataFrame con columnas "umbral" y "n_variables"
    raise NotImplementedError


# ---------------------------------------------------------------- 5
def varianza_por_escala(X_train: pd.DataFrame) -> pd.DataFrame:
    """📌 Extra: muestra que la varianza depende de la ESCALA de cada columna.

    Calcula la varianza (ddof=0) de cada columna en tres versiones de X_train:
      - "original":   los datos tal cual (ram en MB, blue en 0/1, ...).
      - "minmax":     tras MinMaxScaler (cada columna llevada al rango [0, 1]).
      - "estandar":   tras StandardScaler (cada columna con media 0 y desviación 1).
    Cada escalador se ajusta con X_train (fit_transform).

    Lo que verás: en "original" la ram "gana" por estar en miles; en "minmax" las
    binarias pasan arriba (0.25); en "estandar" TODAS valen 1 y VarianceThreshold
    ya no puede distinguir nada.

    Args:
        X_train: DataFrame numérico.

    Devuelve:
        pd.DataFrame con índice = columnas de X_train (en su orden original) y
        3 columnas: "original", "minmax", "estandar" (floats).

    Ejemplo:
        >>> df = pd.DataFrame({"binaria": [0, 1, 0, 1], "miles": [1000, 3000, 2000, 4000]})
        >>> varianza_por_escala(df).round(4)
                   original  minmax  estandar
        binaria        0.25  0.2500       1.0
        miles    1250000.00  0.1389       1.0

    💡 Pistas:
        - from sklearn.preprocessing import MinMaxScaler, StandardScaler
        - escalador.fit_transform(X_train) devuelve un array: np.var(arr, axis=0) da la
          varianza por columna (con ddof=0), o conviértelo a DataFrame y usa .var(ddof=0).
        - Arma el resultado con pd.DataFrame({...}, index=X_train.columns).
    """
    # 1. varianza de cada columna de los datos originales
    # 2. aplica MinMaxScaler y calcula la varianza de cada columna
    # 3. aplica StandardScaler y calcula la varianza de cada columna
    # 4. junta todo en un DataFrame con columnas "original", "minmax", "estandar"
    raise NotImplementedError


# ---------------------------------------------------------------- 6
def comparar_accuracy_umbral(X_train: pd.DataFrame, X_test: pd.DataFrame,
                             y_train: pd.Series, y_test: pd.Series,
                             umbral: float = 0.5) -> dict:
    """Compara el baseline de la clase 13 con todas las variables vs las elegidas por varianza.

    El profesor: "esta configuración la vamos a mantener para ser equitativos" y pasó de
    R^20 (accuracy 0.83) a R^13 (accuracy 0.8475): "el modelo es menos complejo pero
    clasifica mejor".

    Ambos modelos son el MISMO Random Forest de la clase 13:
        RandomForestClassifier(max_depth=5, max_leaf_nodes=20, random_state=1)
    Las columnas se eligen con tu seleccionar_por_varianza(X_train, X_test, umbral).

    Args:
        X_train, X_test: predictores (con todas las columnas).
        y_train, y_test: objetivo.
        umbral: umbral de varianza.

    Devuelve:
        dict con exactamente 3 claves:
            {"todas": accuracy en test con todas las columnas (float),
             "seleccion": accuracy en test solo con las columnas conservadas (float),
             "n_columnas": cuántas columnas se conservaron (int)}

    Ejemplo:
        >>> comparar_accuracy_umbral(X_tr, X_te, y_tr, y_te, umbral=0.5)
        {'todas': 0.7525, 'seleccion': ..., 'n_columnas': 13}

    💡 Pistas:
        - from sklearn.ensemble import RandomForestClassifier
        - from sklearn.metrics import accuracy_score
        - Entrena dos modelos (o escribe una función auxiliar que entrene y evalúe).
    """
    # 1. accuracy del RF con todas las columnas
    # 2. selecciona columnas con seleccionar_por_varianza
    # 3. accuracy del RF solo con las columnas conservadas
    # 4. devuelve el diccionario con las 3 claves
    raise NotImplementedError
