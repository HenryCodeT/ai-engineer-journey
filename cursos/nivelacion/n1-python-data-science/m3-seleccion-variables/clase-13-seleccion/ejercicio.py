"""
Clase 13 · Método de selección de variables (motivación y BASELINE)
===================================================================

Implementa cada función reemplazando `raise NotImplementedError`.
Evalúa con:

    python -m framework evaluar cursos/nivelacion/n1-python-data-science/m3-seleccion-variables/clase-13-seleccion

(o solo la nota: python -m pytest cursos/nivelacion/n1-python-data-science/m3-seleccion-variables/clase-13-seleccion)

Reglas:
- No modifiques las entradas (X, y): trabaja sobre copias si hace falta.
- Solo necesitas pandas, numpy y scikit-learn.
- Datasets: los tests usan un dataset SINTÉTICO de "móviles" (20 variables, 4 rangos de
  precio, como el caso de Kaggle de la clase) y data/raw/clientes_banco.csv (target
  `abandono`, desbalanceado: ~5 % de positivos).

Hilo del ejercicio (sigue la clase):
  1. accuracy a mano: correctos / total (el "10 de 12" de la pizarra).
  2. Matriz de confusión a mano: la diagonal son los aciertos.
  3. Proporción de cada clase en y (para saber si está balanceado).
  4. División train/test ESTRATIFICADA (stratify=y, random_state=1).
  5. Entrenar el BASELINE: Random Forest "podado" (max_depth=5, max_leaf_nodes=20).
  6. Comparar el baseline (todas las variables) contra un subconjunto de variables:
     la pregunta con la que termina la clase, "¿lo puedo mejorar?".

Orden sugerido: 1 -> 6. La 5 puede usar la 1; la 6 puede usar la 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 1
def accuracy_manual(y_true, y_pred) -> float:
    """Calcula el accuracy SIN scikit-learn: muestras bien clasificadas / total de muestras.

    El profesor: "de 12 puntos he clasificado correctamente 10, porque hay dos errores...
    entonces 10 de 12". Accuracy = 10 / 12 ≈ 0.833.

    Se compara POSICIÓN por posición (el 1.º real con el 1.º predicho, etc.), aunque
    y_true sea una Series con un índice desordenado (como el y_test que devuelve
    train_test_split) y y_pred sea un array de numpy.

    Args:
        y_true: etiquetas reales (lista, array de numpy o Series de pandas).
        y_pred: etiquetas predichas (lista, array de numpy o Series de pandas).

    Devuelve:
        float entre 0 y 1.

    Errores:
        ValueError si y_true e y_pred no tienen la misma longitud.

    Ejemplo:
        >>> accuracy_manual([0, 1, 1, 0], [0, 1, 0, 0])
        0.75

    💡 Pistas:
        - np.asarray(...) convierte listas y Series en arrays y "olvida" el índice.
        - (a == b) da un array de True/False; True cuenta como 1 al sumar o promediar.
    """
    # 1. convierte y_true y y_pred a arrays de numpy
    # 2. si las longitudes son distintas -> raise ValueError("...")
    # 3. compara elemento a elemento (==)
    # 4. devuelve la proporción de aciertos como float
    raise NotImplementedError


# ---------------------------------------------------------------- 2
def matriz_confusion_manual(y_true, y_pred, clases: list) -> pd.DataFrame:
    """Construye la matriz de confusión SIN scikit-learn.

    - FILAS = clase REAL, COLUMNAS = clase PREDICHA, en el orden de `clases`.
    - La celda (fila i, columna j) cuenta cuántas muestras eran de la clase i y el
      modelo dijo que eran de la clase j.
    - La DIAGONAL son los aciertos; fuera de la diagonal, los errores.
    - Si una clase nunca aparece (ni real ni predicha), su fila/columna va con ceros.
    - Valores enteros (int).

    Debe coincidir con sklearn.metrics.confusion_matrix(y_true, y_pred, labels=clases).

    Args:
        y_true: etiquetas reales (lista, array o Series).
        y_pred: etiquetas predichas (lista, array o Series).
        clases: lista con TODAS las clases, en el orden deseado.

    Devuelve:
        pd.DataFrame de tamaño len(clases) x len(clases), con index=clases y columns=clases.

    Ejemplo:
        >>> matriz_confusion_manual([0, 0, 1, 1, 1], [0, 1, 1, 1, 0], clases=[0, 1])
           0  1
        0  1  1
        1  1  2

        (Fila 0: de los 2 que eran clase 0, 1 acertado y 1 confundido con la clase 1.)

    💡 Pistas:
        - Opción A: pd.crosstab(serie_real, serie_predicha) cuenta las combinaciones;
          luego .reindex(index=clases, columns=clases, fill_value=0) completa lo que falte.
          Ojo: crea las Series con np.asarray para que no choquen los índices.
        - Opción B: crea una matriz de ceros con np.zeros((k, k), dtype=int) y recorre
          los pares (real, predicho) sumando 1 en la celda correcta.
        - Al final, quita los nombres de index/columns si crosstab los pone
          (.rename_axis(index=None, columns=None)).
    """
    # 1. convierte y_true / y_pred a arrays
    # 2. cuenta cada combinación (real, predicho)
    # 3. asegúrate de que estén TODAS las clases, en el orden de `clases`, con 0 si falta
    # 4. devuelve un DataFrame de enteros con index=clases y columns=clases
    raise NotImplementedError


# ---------------------------------------------------------------- 3
def proporciones_por_clase(y) -> pd.Series:
    """Devuelve la proporción (0 a 1) de cada clase en y, ordenada por la etiqueta de la clase.

    Sirve para dos cosas de la clase:
      - saber si los datos están balanceados (el accuracy engaña si no lo están);
      - comprobar que stratify mantiene las proporciones en train y test.

    Args:
        y: etiquetas (lista, array o Series).

    Devuelve:
        pd.Series con índice = clases (orden ascendente) y valores = proporción.
        Los valores suman 1.

    Ejemplo:
        >>> proporciones_por_clase([1, 0, 0, 0])
        0    0.75
        1    0.25
        dtype: float64

    💡 Pistas:
        - pd.Series(y).value_counts(normalize=True) da proporciones en vez de conteos.
        - .sort_index() las ordena por la etiqueta.
    """
    # 1. convierte y a Series
    # 2. cuenta cada clase como proporción
    # 3. ordena por la etiqueta de la clase
    raise NotImplementedError


# ---------------------------------------------------------------- 4
def dividir_estratificado(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                          random_state: int = 1) -> tuple:
    """Divide en train y test de forma ESTRATIFICADA, como en la clase.

    El profesor: con stratify=y "trata de forma proporcionada de extraer de cada categoría
    cierta cantidad de muestras... de esa forma garantizo que siempre hay muestras de todas
    las clases". Y random_state=1 "para que ustedes puedan replicar los resultados".

    Args:
        X: predictores (DataFrame).
        y: variable objetivo (Series).
        test_size: fracción para test (0.2 = 20 %).
        random_state: semilla.

    Devuelve:
        tupla (X_train, X_test, y_train, y_test), EN ESE ORDEN (el mismo de sklearn).
        Debe ser idéntica a lo que da train_test_split con stratify=y y esa semilla.

    Ejemplo:
        >>> X_tr, X_te, y_tr, y_te = dividir_estratificado(X, y)
        >>> len(X_te) / len(X)
        0.2

    💡 Pistas:
        - from sklearn.model_selection import train_test_split
        - Pásale test_size, random_state y stratify=y.
    """
    # 1. importa train_test_split
    # 2. llama con X, y, test_size, random_state y stratify
    # 3. devuelve la tupla tal cual
    raise NotImplementedError


# ---------------------------------------------------------------- 5
def entrenar_baseline(X_train: pd.DataFrame, y_train: pd.Series,
                      X_test: pd.DataFrame, y_test: pd.Series,
                      max_depth: int = 5, max_leaf_nodes: int = 20,
                      random_state: int = 1) -> float:
    """Entrena el modelo BASELINE de la clase y devuelve su accuracy en TEST.

    Modelo: RandomForestClassifier "podado" para que los árboles no crezcan sin control:
        RandomForestClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes,
                               random_state=random_state)
    (el resto de hiperparámetros, por defecto). Se entrena con X_train / y_train, se
    predice X_test y se mide el accuracy contra y_test.

    Args:
        X_train, y_train: datos de entrenamiento.
        X_test, y_test: datos de prueba.
        max_depth: profundidad máxima de cada árbol (número de niveles).
        max_leaf_nodes: máximo de nodos hoja (nodos finales) por árbol.
        random_state: semilla del bosque (para que el resultado sea reproducible).

    Devuelve:
        float: accuracy en test (entre 0 y 1).

    Ejemplo:
        >>> acc = entrenar_baseline(X_train, y_train, X_test, y_test)
        >>> round(acc, 2)      # en el dataset sintético de móviles, con las 20 variables
        0.75

    💡 Pistas:
        - from sklearn.ensemble import RandomForestClassifier
        - modelo.fit(X_train, y_train) y luego modelo.predict(X_test)
        - Puedes usar tu accuracy_manual (o sklearn.metrics.accuracy_score).
    """
    # 1. crea el RandomForestClassifier con los 3 hiperparámetros
    # 2. entrena con X_train, y_train
    # 3. predice sobre X_test
    # 4. devuelve el accuracy (float)
    raise NotImplementedError


# ---------------------------------------------------------------- 6
def comparar_con_subconjunto(X_train: pd.DataFrame, X_test: pd.DataFrame,
                             y_train: pd.Series, y_test: pd.Series,
                             columnas: list) -> dict:
    """Compara el baseline con TODAS las variables contra un SUBCONJUNTO de variables.

    Es la pregunta con la que cierra la clase: "si este fue mi primer resultado, ¿lo puedo
    mejorar o no?" usando menos dimensiones (pasar de R^20 a R^k). En las próximas
    clases aprenderás CÓMO elegir esas columnas; aquí solo mides el efecto.

    Ambos modelos usan entrenar_baseline con sus valores por defecto
    (max_depth=5, max_leaf_nodes=20, random_state=1).

    Args:
        X_train, X_test: predictores de train y test (con todas las columnas).
        y_train, y_test: objetivo de train y test.
        columnas: lista de nombres de columnas del subconjunto.

    Devuelve:
        dict con 3 claves (floats):
            {"todas": accuracy con todas las columnas,
             "subconjunto": accuracy solo con `columnas`,
             "diferencia": subconjunto - todas}   (positiva = el subconjunto mejora)

    Errores:
        KeyError si alguna de `columnas` no existe en X_train.

    Ejemplo:
        >>> comparar_con_subconjunto(X_tr, X_te, y_tr, y_te, ["ram", "battery_power"])
        {'todas': 0.75, 'subconjunto': ..., 'diferencia': ...}

    💡 Pistas:
        - X_train[columnas] selecciona solo esas columnas (y lanza KeyError si alguna no existe).
        - Llama dos veces a entrenar_baseline.
    """
    # 1. accuracy con todas las columnas
    # 2. accuracy solo con X_train[columnas] / X_test[columnas]
    # 3. devuelve el diccionario con las 3 claves
    raise NotImplementedError
