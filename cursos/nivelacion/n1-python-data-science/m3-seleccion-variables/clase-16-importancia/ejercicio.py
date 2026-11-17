"""
Clase 16 · Métodos basados en la importancia (SelectFromModel)
==============================================================

Implementa cada función reemplazando `raise NotImplementedError`.
Evalúa con:

    python -m framework evaluar cursos/nivelacion/n1-python-data-science/m3-seleccion-variables/clase-16-importancia

(o solo la nota: python -m pytest cursos/nivelacion/n1-python-data-science/m3-seleccion-variables/clase-16-importancia)

Reglas:
- No modifiques las entradas (X, y, el modelo): trabaja sobre copias si hace falta.
- Solo necesitas pandas, numpy y scikit-learn.
- Datasets: los tests usan el MISMO dataset sintético de "móviles" de la clase 13
  (20 variables, 4 rangos de precio; solo ram, battery_power, px_height y px_width
  influyen en el precio) y data/raw/clientes_banco.csv.

Dos modelos aparecen en este ejercicio (los dos de la clase):

  MODELO DE IMPORTANCIA (el que va DENTRO de SelectFromModel):
      RandomForestClassifier(n_estimators=100, max_depth=15, max_leaf_nodes=20,
                             random_state=random_state)

  MODELO BASELINE (el de la clase 13, para medir el accuracy y comparar):
      RandomForestClassifier(max_depth=5, max_leaf_nodes=20, random_state=1)

Hilo del ejercicio (sigue la clase):
  1. Gini a mano: cuánto "desorden" hay en un nodo del árbol.
  2. Ranking de importancia de las variables según el Random Forest entrenado.
  3. SelectFromModel con un umbral (0.01 en la clase) -> columnas que se quedan.
  4. 📌 Extra: importancia por permutación (otra forma de medir la importancia).
  5. Tabla resumen del módulo: qué método dio mejor accuracy con menos columnas.
  6. El flujo completo de la clase: seleccionar, transformar, reentrenar y comparar.

Orden sugerido: 1 -> 6. La 3 puede apoyarse en la idea de la 2; la 6 usa la 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 1
def gini_manual(conteos) -> float:
    """Calcula la impureza de Gini de UN nodo a partir de cuántas muestras hay de cada clase.

    El profesor: el árbol trabaja con "la entropía o Gini", que miden "el nivel de pureza
    dentro de cada partición". Un nodo PURO (todas las muestras de la misma clase) tiene
    Gini 0. Mientras más mezcladas estén las clases, mayor es el Gini.

        Gini = 1 - suma(p_k ** 2)      donde p_k = conteo_k / total

    Args:
        conteos: cuántas muestras hay de cada clase en el nodo (lista o array de enteros
                 >= 0). Ejemplo: [5, 5] = 5 equis y 5 triángulos.

    Devuelve:
        float entre 0 y 1.

    Errores:
        ValueError si algún conteo es negativo o si el total es 0 (nodo vacío).

    Ejemplo:
        >>> gini_manual([5, 5])     # mitad y mitad: lo más mezclado con 2 clases
        0.5
        >>> gini_manual([10, 0])    # nodo puro
        0.0
        >>> gini_manual([3, 1])     # 1 - (0.75**2 + 0.25**2)
        0.375

    💡 Pistas:
        - np.asarray(conteos, dtype=float) convierte la lista en array.
        - p = conteos / conteos.sum() da las proporciones.
        - (p ** 2).sum() suma los cuadrados.
    """
    # 1. convierte conteos a array de floats
    # 2. si hay algún negativo -> raise ValueError("...")
    # 3. si el total es 0 -> raise ValueError("...")
    # 4. calcula las proporciones p
    # 5. devuelve float(1 - suma de p al cuadrado)
    raise NotImplementedError


# ---------------------------------------------------------------- 2
def ranking_importancia(X_train: pd.DataFrame, y_train: pd.Series,
                        random_state: int = 1) -> pd.Series:
    """Entrena el MODELO DE IMPORTANCIA y devuelve la importancia de cada variable, ordenada.

    El MODELO DE IMPORTANCIA es el de la clase:
        RandomForestClassifier(n_estimators=100, max_depth=15, max_leaf_nodes=20,
                               random_state=random_state)
    Después de entrenarlo con X_train / y_train, el atributo `feature_importances_`
    dice cuánto aportó cada variable a separar las clases (cuánto bajó el Gini en los
    cortes que usan esa variable). Esas importancias SUMAN 1.

    Args:
        X_train: predictores de entrenamiento (DataFrame).
        y_train: objetivo de entrenamiento (Series).
        random_state: semilla del bosque.

    Devuelve:
        pd.Series con índice = nombres de las columnas de X_train y valores = importancia,
        ordenada de MAYOR a MENOR. Tiene una entrada por columna y suma 1.
        (El nombre de la Series no importa.)

    Ejemplo:
        >>> ranking_importancia(X_train, y_train).head(3)
        ram              0.6...
        battery_power    0.1...
        px_height        0.0...

    💡 Pistas:
        - from sklearn.ensemble import RandomForestClassifier
        - modelo.fit(X_train, y_train) y luego modelo.feature_importances_ (un array).
        - pd.Series(array, index=X_train.columns) le pone nombre a cada valor.
        - .sort_values(ascending=False) ordena de mayor a menor.
    """
    # 1. crea el RandomForestClassifier con n_estimators=100, max_depth=15,
    #    max_leaf_nodes=20 y random_state
    # 2. entrénalo con X_train, y_train
    # 3. arma una Series con feature_importances_ y los nombres de las columnas
    # 4. ordénala de mayor a menor y devuélvela
    raise NotImplementedError


# ---------------------------------------------------------------- 3
def seleccionar_desde_modelo(X_train: pd.DataFrame, X_test: pd.DataFrame,
                             y_train: pd.Series, threshold=0.01,
                             random_state: int = 1) -> tuple:
    """Selecciona variables con SelectFromModel, como en la clase.

    El profesor: "este modelito lo coloco dentro de un SelectFromModel... vamos a definir
    un umbral, aunque ustedes también pueden quitar el umbral y que trabaje con su umbral
    por defecto". Luego "llamo el support" (máscara de True/False) y filtra las columnas.

    Pasos:
      - Crea el MODELO DE IMPORTANCIA:
            RandomForestClassifier(n_estimators=100, max_depth=15, max_leaf_nodes=20,
                                   random_state=random_state)
      - Mételo en SelectFromModel(modelo, threshold=threshold) y haz .fit SOLO con
        X_train, y_train (el test no se usa para decidir nada).
      - get_support() -> máscara booleana: True = la variable se queda
        (su importancia es >= threshold).
      - Aplica la máscara a X_train y a X_test.

    Args:
        X_train, X_test: predictores de train y test (DataFrames con las mismas columnas).
        y_train: objetivo de train.
        threshold: umbral de importancia. 0.01 como en la clase. Con None, SelectFromModel
                   usa su umbral por defecto (la MEDIA de las importancias = 1 / n_columnas).
        random_state: semilla del bosque.

    Devuelve:
        tupla (X_train_sel, X_test_sel, columnas):
            - X_train_sel, X_test_sel: DataFrames solo con las columnas elegidas, con los
              mismos nombres de columna e índice que las entradas, en el ORDEN original.
            - columnas: lista de nombres de las columnas elegidas (mismo orden).

    Ejemplo:
        >>> X_tr_sel, X_te_sel, cols = seleccionar_desde_modelo(X_tr, X_te, y_tr, threshold=None)
        >>> cols
        ['battery_power', 'px_height', 'px_width', 'ram']

    💡 Pistas:
        - from sklearn.feature_selection import SelectFromModel
        - selector = SelectFromModel(modelo, threshold=threshold).fit(X_train, y_train)
        - mascara = selector.get_support()
        - X_train.columns[mascara] da los nombres; X_train.loc[:, mascara] filtra y
          conserva el DataFrame (selector.transform devuelve un array de numpy SIN nombres).
    """
    # 1. crea el modelo de importancia
    # 2. crea el SelectFromModel con el threshold y entrénalo con X_train, y_train
    # 3. obtén la máscara con get_support()
    # 4. lista de columnas elegidas
    # 5. filtra X_train y X_test con esas columnas
    # 6. devuelve (X_train_sel, X_test_sel, columnas)
    raise NotImplementedError


# ---------------------------------------------------------------- 4
def importancia_permutacion(modelo, X_test: pd.DataFrame, y_test: pd.Series,
                            random_state: int = 1, n_repeats: int = 5) -> pd.Series:
    """📌 Extra · Importancia por PERMUTACIÓN de un modelo YA entrenado, medida en TEST.

    Idea: si "barajo" (desordeno) una columna y el accuracy cae mucho, esa columna era
    importante. Si no cambia nada, el modelo no la usaba. Se mide en test, con datos que
    el modelo no vio, y no tiene el sesgo de feature_importances_ hacia variables con
    muchos valores distintos.

    Usa sklearn.inspection.permutation_importance(modelo, X_test, y_test,
    n_repeats=n_repeats, random_state=random_state) y quédate con `importances_mean`
    (la caída media del accuracy al barajar cada columna).

    Args:
        modelo: un clasificador de scikit-learn YA entrenado (no lo vuelvas a entrenar).
        X_test, y_test: datos de prueba.
        random_state: semilla de las permutaciones.
        n_repeats: cuántas veces se baraja cada columna.

    Devuelve:
        pd.Series con índice = columnas de X_test y valores = importances_mean,
        ordenada de MAYOR a MENOR. Puede tener valores 0 o negativos
        (columnas que no aportan).

    Ejemplo:
        >>> importancia_permutacion(modelo_baseline, X_test, y_test).head(2)
        ram              0.47...
        battery_power    0.11...

    💡 Pistas:
        - from sklearn.inspection import permutation_importance
        - r = permutation_importance(...); r.importances_mean es un array.
        - Igual que en la 2: Series con index=X_test.columns y .sort_values(ascending=False).
    """
    # 1. llama a permutation_importance con el modelo, X_test, y_test,
    #    n_repeats y random_state
    # 2. arma una Series con importances_mean y los nombres de las columnas
    # 3. ordénala de mayor a menor y devuélvela
    raise NotImplementedError


# ---------------------------------------------------------------- 5
def comparar_metodos(resultados: dict) -> pd.DataFrame:
    """Arma la tabla resumen del módulo: qué método de selección funcionó mejor.

    El profesor repasa sus resultados: "R^20 → 0.83; varianza R^13 → 0.8475;
    correlación R^13 → 0.8475; importancia R^10 → 0.8775... la dimensión ha bajado
    y la performance ha subido".

    Args:
        resultados: dict  nombre_del_método -> (n_columnas, accuracy)
            Ejemplo: {"todas": (20, 0.83), "importancia": (10, 0.8775)}

    Devuelve:
        pd.DataFrame con EXACTAMENTE las columnas ["metodo", "n_columnas", "accuracy"]
        (n_columnas entero, accuracy float), ordenado así:
            1. accuracy de MAYOR a MENOR;
            2. si empatan en accuracy, primero el de MENOS columnas (modelo más simple);
            3. si empatan en las dos, por "metodo" en orden alfabético.
        Con índice 0, 1, 2, ... (reinicia el índice).

    Errores:
        ValueError si `resultados` está vacío.

    Ejemplo:
        >>> comparar_metodos({"todas": (20, 0.83), "varianza": (13, 0.8475),
        ...                   "importancia": (10, 0.8775)})
                metodo  n_columnas  accuracy
        0  importancia          10    0.8775
        1     varianza          13    0.8475
        2        todas          20    0.8300

    💡 Pistas:
        - Recorre resultados.items() y arma una lista de dicts (una fila por método).
        - df.sort_values([...], ascending=[False, True, True])
        - .reset_index(drop=True)
    """
    # 1. si resultados está vacío -> raise ValueError("...")
    # 2. arma una fila por método: {"metodo": ..., "n_columnas": ..., "accuracy": ...}
    # 3. crea el DataFrame
    # 4. ordena por accuracy (desc), n_columnas (asc) y metodo (asc)
    # 5. reinicia el índice y devuelve
    raise NotImplementedError


# ---------------------------------------------------------------- 6
def evaluar_seleccion(X_train: pd.DataFrame, X_test: pd.DataFrame,
                      y_train: pd.Series, y_test: pd.Series,
                      threshold=0.01) -> dict:
    """El flujo completo de la clase: seleccionar con importancia y reentrenar el BASELINE.

    1. Selecciona columnas con seleccionar_desde_modelo(X_train, X_test, y_train,
       threshold=threshold) (random_state=1).
    2. Entrena el MODELO BASELINE de la clase 13
           RandomForestClassifier(max_depth=5, max_leaf_nodes=20, random_state=1)
       DOS veces: con todas las columnas y solo con las seleccionadas.
       ("Bajo la misma configuración he llegado a 0.8775": misma config, menos columnas).
    3. Mide el accuracy de cada uno en TEST.

    Args:
        X_train, X_test, y_train, y_test: la división de siempre.
        threshold: umbral para SelectFromModel (0.01, None = media, etc.).

    Devuelve:
        dict con 4 claves:
            {"todas": float (accuracy con todas las columnas),
             "seleccion": float (accuracy solo con las columnas elegidas),
             "n_columnas": int (cuántas columnas se eligieron),
             "columnas": list (sus nombres, en el orden original)}

    Ejemplo:
        >>> evaluar_seleccion(X_tr, X_te, y_tr, y_te, threshold=None)
        {'todas': 0.7525, 'seleccion': 0.815, 'n_columnas': 4,
         'columnas': ['battery_power', 'px_height', 'px_width', 'ram']}

    💡 Pistas:
        - Reutiliza tu seleccionar_desde_modelo.
        - from sklearn.metrics import accuracy_score
        - Crea un modelo NUEVO para cada entrenamiento (no reutilices el mismo objeto).
    """
    # 1. llama a seleccionar_desde_modelo -> X_train_sel, X_test_sel, columnas
    # 2. entrena el baseline con TODAS las columnas y mide el accuracy en test
    # 3. entrena otro baseline solo con las seleccionadas y mide el accuracy en test
    # 4. devuelve el dict con las 4 claves
    raise NotImplementedError
