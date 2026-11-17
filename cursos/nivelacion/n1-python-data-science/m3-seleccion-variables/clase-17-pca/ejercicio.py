"""
Clase 17 · Reducción de la dimensionalidad (PCA)
================================================

Implementa cada función reemplazando `raise NotImplementedError`.
Evalúa con:

    python -m framework evaluar cursos/nivelacion/n1-python-data-science/m3-seleccion-variables/clase-17-pca

(o solo la nota: python -m pytest cursos/nivelacion/n1-python-data-science/m3-seleccion-variables/clase-17-pca)

Reglas:
- No modifiques las entradas (X_train, X_test, y_train, y_test): trabaja sobre copias si hace falta.
- Solo necesitas pandas, numpy y scikit-learn.
- Datasets: los tests usan el dataset SINTÉTICO de "móviles" de la clase 13 (20 variables,
  4 rangos de precio, split estratificado test_size=0.2, random_state=1) y las columnas
  numéricas de data/raw/clientes_banco.csv.

Hilo del ejercicio (sigue la clase):
  1. Proyectar con PCA: fit SOLO en train, transform en train y en test (la pregunta de Samuel).
  2. Ver cuánta información conserva cada componente (explained_variance_ratio_).
  3. La "segunda opción" del profe: pedir un % de información (n_components=0.90).
  4. 📌 Extra: PCA es MUY sensible a la escala (con y sin StandardScaler).
  5. Medir el accuracy del baseline de la clase 13 sobre las componentes.
  6. 📌 Extra: ¿qué variables "forman" cada componente? (cargas o loadings).

Orden sugerido: 1 -> 6. La 3 y la 4 pueden reutilizar la idea de la 1; la 5 usa la 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 1
def aplicar_pca(X_train: pd.DataFrame, X_test: pd.DataFrame, n_components,
                escalar: bool = True) -> tuple:
    """Proyecta X_train y X_test en un espacio nuevo de menor dimensión con PCA.

    Es el código de la clase, con un paso previo opcional de escalado (📌 Extra):

        1. (si escalar=True) StandardScaler: fit con X_train, transform a X_train y a X_test.
        2. PCA(n_components=n_components): fit con X_train (ya escalado si corresponde),
           transform a X_train y a X_test.

    ⚠️ Lo que preguntó Samuel: a X_test se le aplica SOLO transform, nunca fit ni
    fit_transform. Si hicieras fit con X_test estarías "aprendiendo" los autovectores
    (o la media y la desviación del scaler) con datos que el modelo no debe ver.

    Args:
        X_train: predictores de entrenamiento (DataFrame numérico).
        X_test: predictores de test (mismas columnas).
        n_components: int (cuántas componentes quieres, p. ej. 3) o float entre 0 y 1
            (el % mínimo de información a conservar, p. ej. 0.90). Se pasa tal cual a PCA.
        escalar: si True, estandariza antes de PCA (StandardScaler ajustado solo con train).

    Devuelve:
        tupla (Z_train, Z_test, pca):
            - Z_train, Z_test: DataFrames con columnas "PC1", "PC2", …, "PCk"
              y el MISMO índice que X_train / X_test.
            - pca: el objeto PCA ya ajustado (sirve para mirar explained_variance_ratio_).

    Ejemplo:
        >>> X_tr = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 9]})
        >>> X_te = pd.DataFrame({"a": [5], "b": [10]}, index=[99])
        >>> Z_tr, Z_te, pca = aplicar_pca(X_tr, X_te, n_components=1, escalar=False)
        >>> list(Z_tr.columns), list(Z_te.index)
        (['PC1'], [99])

    💡 Pistas:
        - from sklearn.preprocessing import StandardScaler
        - from sklearn.decomposition import PCA
        - Después del fit, pca.n_components_ te dice cuántas componentes quedaron
          (útil cuando n_components es 0.90).
        - pd.DataFrame(matriz, columns=[...], index=X_train.index)
    """
    # 1. si escalar: crea un StandardScaler, fit SOLO con X_train, transforma train y test
    #    (si no: usa X_train y X_test tal cual)
    # 2. crea PCA(n_components=n_components) y haz fit SOLO con el train
    # 3. transforma train y test con el pca ajustado
    # 4. arma los nombres "PC1".."PCk" con pca.n_components_
    # 5. envuelve las dos matrices en DataFrames con esos nombres y el índice original
    # 6. devuelve (Z_train, Z_test, pca)
    raise NotImplementedError


# ---------------------------------------------------------------- 2
def varianza_explicada(pca) -> pd.DataFrame:
    """Resume cuánta información (varianza) conserva cada componente de un PCA ya ajustado.

    En la clase: "la primera componente explica el 67 % de la información, luego el 16 y
    luego el 10 … sumo todo y tengo alrededor de 93 %".

    Args:
        pca: objeto PCA ya ajustado (cualquier objeto con el atributo
            explained_variance_ratio_ sirve).

    Devuelve:
        DataFrame con una fila por componente (índice 0, 1, 2, …) y 3 columnas, en este orden:
            - "componente": "PC1", "PC2", …
            - "varianza":   proporción de varianza explicada por esa componente (0 a 1)
            - "acumulada":  suma acumulada de "varianza" hasta esa componente

    Ejemplo:
        >>> varianza_explicada(pca_con_ratios_05_03_02)
          componente  varianza  acumulada
        0        PC1       0.5        0.5
        1        PC2       0.3        0.8
        2        PC3       0.2        1.0

    💡 Pistas:
        - pca.explained_variance_ratio_ es un array de numpy.
        - np.cumsum(...) o .cumsum() de pandas da la suma acumulada.
    """
    # 1. toma pca.explained_variance_ratio_
    # 2. arma los nombres "PC1".."PCk"
    # 3. calcula la suma acumulada
    # 4. devuelve el DataFrame con las columnas componente / varianza / acumulada
    raise NotImplementedError


# ---------------------------------------------------------------- 3
def componentes_para(X_train: pd.DataFrame, umbral: float = 0.9,
                     escalar: bool = True) -> int:
    """¿Cuántas componentes necesito para conservar al menos `umbral` de la información?

    Es la "segunda opción" de la clase: PCA(n_components=0.90) "va a buscar todas las
    componentes que en conjunto llegan a 90 o más de conservación de información".

    Solo mira X_train (no necesita test ni y). Si escalar=True, estandariza antes
    con StandardScaler (ajustado con X_train).

    Args:
        X_train: predictores de entrenamiento.
        umbral: fracción de información a conservar (entre 0 y 1, sin incluir 1).
        escalar: si True, estandariza antes de PCA.

    Devuelve:
        int: número de componentes que elige PCA.

    Ejemplo:
        >>> componentes_para(X_train_moviles, umbral=0.9, escalar=False)
        3

    💡 Pistas:
        - PCA(n_components=umbral) con un float entre 0 y 1 hace justo esto.
        - Tras el fit, la respuesta está en pca.n_components_ (conviértela a int).
    """
    # 1. si escalar: estandariza X_train (fit_transform aquí está bien: SOLO es train)
    # 2. crea PCA(n_components=umbral) y haz fit
    # 3. devuelve int(pca.n_components_)
    raise NotImplementedError


# ---------------------------------------------------------------- 4
def comparar_con_y_sin_escala(X_train: pd.DataFrame) -> dict:
    """📌 Extra: muestra con números que PCA es MUY sensible a la escala.

    PCA busca las direcciones de MAYOR varianza. Si una columna está en miles (ram,
    ingreso_mensual…) y las otras en unidades, esa columna tiene una varianza enorme y
    se "come" la primera componente aunque no sea la más útil.

    Ajusta dos PCA sobre X_train (con todas las componentes, PCA() sin argumentos):
        - sin escalar: directamente sobre X_train
        - con escala: sobre StandardScaler().fit_transform(X_train)
    y devuelve la proporción de varianza explicada por la PRIMERA componente (PC1) en cada caso.

    Args:
        X_train: predictores de entrenamiento (numéricos).

    Devuelve:
        dict con 2 claves (floats entre 0 y 1):
            {"sin_escala": varianza de PC1 sin escalar,
             "con_escala": varianza de PC1 con StandardScaler}

    Ejemplo:
        >>> X = pd.DataFrame({"sueldo": [1000, 3000, 2000, 5000], "hijos": [1, 0, 2, 1]})
        >>> comparar_con_y_sin_escala(X)
        {'sin_escala': 0.99999…, 'con_escala': 0.5…}

    💡 Pistas:
        - explained_variance_ratio_[0] es la de PC1.
        - Convierte a float de Python con float(...).
    """
    # 1. PCA() sobre X_train tal cual -> guarda la varianza de PC1
    # 2. estandariza X_train con StandardScaler
    # 3. PCA() sobre los datos estandarizados -> guarda la varianza de PC1
    # 4. devuelve el diccionario
    raise NotImplementedError


# ---------------------------------------------------------------- 5
def evaluar_pca(X_train: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.Series, y_test: pd.Series,
                n_components, escalar: bool = True) -> float:
    """Entrena el baseline de la clase 13 sobre las componentes de PCA y devuelve el accuracy en test.

    Es el experimento de la clase: "tengo un R^20 … y ahora he bajado a un R'^3".

    Pasos:
        1. Z_train, Z_test = proyección con PCA (fit solo en train; usa tu aplicar_pca
           con los mismos n_components y escalar).
        2. RandomForestClassifier(max_depth=5, max_leaf_nodes=20, random_state=1)
           entrenado con Z_train / y_train.
        3. accuracy de las predicciones sobre Z_test contra y_test.

    Args:
        X_train, X_test: predictores originales.
        y_train, y_test: objetivo.
        n_components: int o float (igual que en aplicar_pca).
        escalar: si True, estandariza antes de PCA.

    Devuelve:
        float: accuracy en test (0 a 1).

    Ejemplo:
        >>> round(evaluar_pca(X_tr, X_te, y_tr, y_te, n_components=3, escalar=False), 3)
        0.705     # en el dataset sintético de móviles (el profe obtuvo ≈ 0.91 con el de Kaggle)

    💡 Pistas:
        - Reutiliza aplicar_pca (devuelve 3 cosas; aquí solo necesitas las 2 primeras).
        - from sklearn.ensemble import RandomForestClassifier
        - from sklearn.metrics import accuracy_score
    """
    # 1. proyecta train y test con aplicar_pca
    # 2. crea el RandomForest del baseline (max_depth=5, max_leaf_nodes=20, random_state=1)
    # 3. entrena con Z_train, y_train
    # 4. predice Z_test
    # 5. devuelve el accuracy como float
    raise NotImplementedError


# ---------------------------------------------------------------- 6
def cargas_principales(pca, columnas, componente: int = 0, top: int = 3) -> pd.Series:
    """📌 Extra: ¿qué variables originales pesan más en una componente?

    Cada componente es una MEZCLA (combinación lineal) de las columnas originales:
        PC1 = w1·x1 + w2·x2 + … + w20·x20
    Esos pesos w son las "cargas" (loadings) y están en pca.components_: una FILA por
    componente y una COLUMNA por variable original. Mirarlas es la forma de recuperar
    un poco de interpretabilidad (la gran desventaja de PCA).

    Args:
        pca: objeto PCA ajustado (sirve cualquier objeto con el atributo components_).
        columnas: nombres de las variables originales, en el orden en que entraron al PCA.
        componente: qué componente mirar (0 = PC1, 1 = PC2, …).
        top: cuántas variables devolver.

    Devuelve:
        pd.Series con las `top` variables de MAYOR |carga| (valor absoluto), ordenadas de
        mayor a menor |carga|. Índice = nombre de la variable; valor = carga CON SU SIGNO.

    Ejemplo:
        >>> # components_ = [[0.1, -0.9, 0.4]] con columnas a, b, c
        >>> cargas_principales(pca, ["a", "b", "c"], componente=0, top=2)
        b   -0.9
        c    0.4
        dtype: float64

    💡 Pistas:
        - pd.Series(pca.components_[componente], index=columnas)
        - Ordena por valor absoluto: s.abs().sort_values(ascending=False).index
          (o s.sort_values(key=abs, ascending=False)).
        - .head(top) o .iloc[:top].
    """
    # 1. toma la fila `componente` de pca.components_
    # 2. conviértela en Series con índice = columnas
    # 3. ordénala por valor absoluto, de mayor a menor (conservando el signo)
    # 4. quédate con las `top` primeras
    raise NotImplementedError
