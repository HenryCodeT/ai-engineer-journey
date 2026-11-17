"""
Clase 09 · Codificación de variables (encoding) y escalamiento
==============================================================

Implementa cada función reemplazando `raise NotImplementedError`.
Evalúa con:

    python -m framework evaluar cursos/nivelacion/n1-python-data-science/m2-outliers-encoding/clase-09-encoding

(o solo la nota: python -m pytest cursos/nivelacion/n1-python-data-science/m2-outliers-encoding/clase-09-encoding)

Reglas:
- No modifiques el DataFrame de entrada (trabaja sobre una copia: df.copy()).
- Evita bucles `for` sobre filas: usa operaciones de pandas / numpy.
- Solo necesitas pandas, numpy y scikit-learn (NO uses category_encoders).
- Dataset de práctica: data/raw/viviendas_lima.csv

Hilo del ejercicio (sigue la clase):
  1. Separar numéricas y categóricas (select_dtypes).
  2. Decidir el encoder según el número de categorías (2-3 -> OneHot; muchas -> binario).
  3. Calcular cuántas columnas genera cada encoder (la "maldición de la dimensionalidad").
  4. Hacer a mano la codificación BINARIA (log2 de n columnas).
  5. ESCALAR por columna (MinMax / Standard): cambia el rango, NO la forma.
  6. NORMALIZAR por fila (norma L2): cada punto queda con longitud 1.
  7. Juntarlo todo en un ColumnTransformer (como el pipeline de la clase).

Orden sugerido: 1 -> 7. Son independientes entre sí (la 4 puede usar la 3).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 1
def separar_variables(df: pd.DataFrame, objetivo: str | None = None) -> dict:
    """Separa los nombres de columnas en numéricas y categóricas, sin la variable objetivo.

    El profesor: "una forma de seleccionar de forma automática mis variables numéricas
    es simplemente df.select_dtypes(include=...) .columns ... y para categóricas EXCLUYO
    todo lo que es numérico y me quedo con el complemento".
    Además: la variable objetivo (y) NO se transforma, así que no debe aparecer en
    ninguna de las dos listas.

    Args:
        df: DataFrame original (NO se modifica).
        objetivo: nombre de la columna objetivo, o None si no hay.

    Devuelve:
        dict con dos claves, cada una con una LISTA de nombres en el MISMO orden que en df:
            {"numericas": [...], "categoricas": [...]}

    Ejemplo:
        >>> df = pd.DataFrame({"edad": [30, 40], "trabajo": ["admin", "obrero"], "y": ["no", "yes"]})
        >>> separar_variables(df, objetivo="y")
        {'numericas': ['edad'], 'categoricas': ['trabajo']}

    💡 Pistas:
        - df.select_dtypes(include="number").columns  -> numéricas
        - df.select_dtypes(exclude="number").columns  -> el complemento
        - Convierte a list y quita `objetivo` si está.
    """
    # 1. nombres de columnas numéricas (como lista)
    # 2. nombres de columnas NO numéricas (como lista)
    # 3. quita `objetivo` de ambas listas (si no es None)
    # 4. devuelve el diccionario
    raise NotImplementedError


# ---------------------------------------------------------------- 2
def elegir_encoder(df: pd.DataFrame, columna: str, max_onehot: int = 3) -> str:
    """Decide qué encoder usar para una columna categórica según su número de categorías.

    Regla de la clase: con pocas categorías (default, housing, loan: 2; marital: 3)
    "puedo aplicar un OneHotEncoder con mucha tranquilidad". Con muchas (job: 12)
    "no me conviene, me va a generar muchas dimensiones" -> mejor la codificación binaria.

    Se cuentan las categorías DISTINTAS sin contar los NaN.

    Args:
        df: DataFrame (NO se modifica).
        columna: columna categórica a analizar.
        max_onehot: máximo de categorías para usar OneHot (inclusive).

    Devuelve:
        "onehot" si nº de categorías <= max_onehot, si no "binary".

    Ejemplo:
        >>> df = pd.DataFrame({"loan": ["yes", "no", "no", "yes"], "job": ["admin", "obrero", "tecnico", "gerente"]})
        >>> elegir_encoder(df, "loan")      # 2 categorías
        'onehot'
        >>> elegir_encoder(df, "job")       # 4 categorías > 3
        'binary'

    💡 Pista: Series.nunique() ya ignora los NaN por defecto.
    """
    # 1. cuenta las categorías distintas de la columna
    # 2. compara con max_onehot y devuelve el texto
    raise NotImplementedError


# ---------------------------------------------------------------- 3
def columnas_resultantes(n_categorias: int, metodo: str) -> int:
    """Cuántas columnas nuevas genera codificar UNA variable con `n_categorias`.

    Lo que explicó el profesor (y la pregunta de Carlos):
        - "onehot": una columna por categoría            -> n
        - "dummy":  una categoría queda como 0 0 ... 0   -> n - 1
        - "binary": se cuenta en binario los códigos 0, 1, ..., n-1
                    -> los bits necesarios = ceil(log2(n))
                    ("si tienes 8 categorías, log2(8) = 3 columnas"; con 20 -> 5)
        - "ordinal": todo cabe en UNA columna (0, 1, 2, ...)  -> 1

    Casos borde (definidos así en este curso):
        - binary con n = 1 o n = 2 -> 1 columna (con un solo bit ya distingues 2 categorías;
          y nunca se generan 0 columnas).
        - dummy con n = 1 -> 0 columnas (la única categoría es la de referencia).

    Args:
        n_categorias: número de categorías distintas (entero >= 1).
        metodo: "onehot", "dummy", "binary" u "ordinal".

    Devuelve:
        int con el número de columnas.

    Lanza:
        ValueError si `metodo` no es uno de los cuatro.

    Ejemplo:
        >>> columnas_resultantes(8, "onehot"), columnas_resultantes(8, "dummy"), columnas_resultantes(8, "binary")
        (8, 7, 3)
        >>> columnas_resultantes(20, "binary")
        5

    💡 Pistas: math.ceil(math.log2(n)); max(1, ...) para el caso borde.
       ¡Ojo! Devuelve un int de Python, no un float.
    """
    # 1. si metodo == "onehot" -> n
    # 2. si metodo == "dummy" -> n - 1
    # 3. si metodo == "binary" -> max(1, ceil(log2(n)))
    # 4. si metodo == "ordinal" -> 1
    # 5. en otro caso -> raise ValueError("...")
    raise NotImplementedError


# ---------------------------------------------------------------- 4
def codificar_binario(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Codificación BINARIA hecha a mano (lo que hace BinaryEncoder de category_encoders).

    Pasos:
        1. Ordena las categorías distintas ALFABÉTICAMENTE (sorted) y dales un código
           0, 1, 2, ..., n-1  (la primera alfabética = 0).
        2. Escribe cada código en binario con k = max(1, ceil(log2(n))) bits.
        3. Cada bit es una columna nueva: f"{columna}_0" es el bit MÁS significativo
           (el de la izquierda), ..., f"{columna}_{k-1}" el menos significativo.
        4. La columna original se ELIMINA y las nuevas se agregan AL FINAL, en orden _0, _1, ...
           Las nuevas columnas son enteros (0 o 1). El resto del DataFrame no cambia.

    Supón que la columna no tiene NaN.

    Args:
        df: DataFrame original (NO se modifica).
        columna: columna categórica a codificar.

    Devuelve:
        DataFrame nuevo, con el mismo índice que df.

    Ejemplo (3 categorías -> 2 bits; A=0=00, B=1=01, C=2=10):
        >>> df = pd.DataFrame({"precio": [1, 2, 3, 4], "team": ["B", "A", "C", "B"]})
        >>> codificar_binario(df, "team")
           precio  team_0  team_1
        0       1       0       1
        1       2       0       0
        2       3       1       0
        3       4       0       1

    💡 Pistas:
        - codigos = df[columna].map({cat: i for i, cat in enumerate(sorted(...))})
        - El bit j (contando desde la izquierda) de un número c es: (c >> (k - 1 - j)) & 1
          En numpy funciona sobre arrays enteros completos, sin bucle sobre filas
          (un bucle sobre los k bits sí está bien: son pocas columnas).
    """
    # 1. copia el DataFrame
    # 2. categorías ordenadas y diccionario categoría -> código
    # 3. códigos enteros con .map(...)
    # 4. k = número de bits
    # 5. para cada j en range(k): nueva columna f"{columna}_{j}" con el bit j
    # 6. elimina la columna original y devuelve
    raise NotImplementedError


# ---------------------------------------------------------------- 5
def escalar(df: pd.DataFrame, columnas: list, metodo: str = "minmax") -> pd.DataFrame:
    """ESCALA columnas numéricas (columna por columna). No cambia la forma de la distribución.

    El profesor: "un escalador no afecta la distribución de la variable... la variable
    simplemente se va a un rango menor de datos, pero nada más".
        - "minmax":   x' = (x - min) / (max - min)       -> queda en [0, 1]  (MinMaxScaler)
        - "standard": x' = (x - media) / desviación      -> media 0, varianza 1 (StandardScaler)
                      Usa la desviación POBLACIONAL (ddof=0), igual que scikit-learn.

    Solo se transforman las `columnas` indicadas; las demás quedan igual.
    Los NaN se mantienen como NaN (min, max, media y std se calculan ignorándolos).

    Args:
        df: DataFrame original (NO se modifica).
        columnas: lista de columnas numéricas a escalar.
        metodo: "minmax" o "standard".

    Devuelve:
        DataFrame nuevo, con las mismas columnas y el mismo orden.

    Lanza:
        ValueError si `metodo` no es "minmax" ni "standard".

    Ejemplo:
        >>> df = pd.DataFrame({"area": [50.0, 100.0, 150.0], "zona": ["a", "b", "c"]})
        >>> escalar(df, ["area"], "minmax")["area"].tolist()
        [0.0, 0.5, 1.0]
        >>> escalar(df, ["area"], "standard")["area"].round(4).tolist()
        [-1.2247, 0.0, 1.2247]

    💡 Pistas: df[columnas].min(), .max(), .mean(), .std(ddof=0) devuelven una Series por
       columna, y pandas alinea por nombre al restar/dividir: (X - X.min()) / (X.max() - X.min())
    """
    # 1. copia el DataFrame
    # 2. X = df[columnas]
    # 3. según el método, calcula la versión escalada de X
    # 4. asigna de vuelta en la copia y devuelve
    raise NotImplementedError


# ---------------------------------------------------------------- 6
def normalizar_filas_l2(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """NORMALIZA por FILA (cada muestra = un punto) dividiendo por su norma L2.

    El profesor: "normalizar es que yo no trabajo con las variables, trabajo con los
    puntos... B sobre módulo de B". Para la fila (x1, ..., xn):
        norma = sqrt(x1² + ... + xn²)      y      fila' = fila / norma
    Así cada fila queda con norma (longitud) 1. Esto SÍ cambia la distribución de cada
    variable, porque cada valor depende de los demás valores de su fila.

    Si una fila es todo ceros (norma 0), se deja en ceros (no dividas entre 0).
    Supón que no hay NaN en `columnas`.

    Args:
        df: DataFrame original (NO se modifica).
        columnas: columnas numéricas que forman el "punto".

    Devuelve:
        DataFrame nuevo con TODAS las columnas de df; solo cambian `columnas`.

    Ejemplo:
        >>> df = pd.DataFrame({"x": [3.0, 0.0], "y": [4.0, 0.0], "id": [1, 2]})
        >>> normalizar_filas_l2(df, ["x", "y"])
             x    y  id
        0  0.6  0.8   1
        1  0.0  0.0   2

    💡 Pistas: np.sqrt((X ** 2).sum(axis=1)) da la norma de cada fila;
       X.div(norma, axis=0) divide cada fila por su número. Reemplaza norma 0 por 1.
    """
    # 1. copia el DataFrame y toma X = df[columnas]
    # 2. norma de cada fila (axis=1)
    # 3. donde la norma sea 0, usa 1
    # 4. divide fila por fila y asigna de vuelta
    raise NotImplementedError


# ---------------------------------------------------------------- 7
def construir_preprocesador(
    cols_onehot: list, cols_ordinal: list, cols_num: list, escalador: str = "power"
):
    """Construye (SIN entrenar) un ColumnTransformer como el pipeline de la clase.

    Tres transformadores, cada uno con su NOMBRE exacto:
        ("onehot",  OneHotEncoder(handle_unknown="ignore"), cols_onehot)
        ("ordinal", OrdinalEncoder(),                       cols_ordinal)
        ("num",     <numérico>,                             cols_num)
    donde <numérico> es:
        - PowerTransformer()  (Yeo-Johnson por defecto)   si escalador == "power"
        - StandardScaler()                                 si escalador == "standard"
        - MinMaxScaler()                                   si escalador == "minmax"
    Cualquier otra columna del DataFrame se descarta (remainder="drop", el valor por defecto).
    Para que la salida sea una matriz normal (no "sparse"), crea el ColumnTransformer
    con sparse_threshold=0.

    El profesor: "lo bueno de este pipeline es que le cambié el encoder por aquí nada más
    y no pasó nada: las numéricas siguen con su propia transformación".

    Args:
        cols_onehot: columnas categóricas con pocas categorías.
        cols_ordinal: columnas categóricas a codificar como 0, 1, 2...
        cols_num: columnas numéricas.
        escalador: "power", "standard" o "minmax".

    Devuelve:
        sklearn.compose.ColumnTransformer sin entrenar (sin llamar a fit).

    Lanza:
        ValueError si `escalador` no es una de las tres opciones.

    Ejemplo:
        >>> pre = construir_preprocesador(["tipo_inmueble"], ["estado"], ["area_m2"])
        >>> X = pre.fit_transform(viviendas.dropna(subset=["estado"]))
        >>> X.shape[1]          # 3 tipos (onehot) + 1 (ordinal) + 1 (num)
        5
        >>> pre.get_feature_names_out()[0]
        'onehot__tipo_inmueble_Casa'

    💡 Pistas: from sklearn.compose import ColumnTransformer
              from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, ...
    """
    # 1. elige el transformador numérico según `escalador` (o ValueError)
    # 2. arma la lista de tuplas (nombre, transformador, columnas)
    # 3. devuelve ColumnTransformer(lista, sparse_threshold=0)
    raise NotImplementedError
