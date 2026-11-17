"""
Clase 06 · Qué son los outliers (y cómo afectan a un modelo)
=============================================================

Implementa cada función reemplazando `raise NotImplementedError`.
Evalúa con:

    python -m framework evaluar cursos/nivelacion/n1-python-data-science/m2-outliers-encoding/clase-06-outliers

(o solo la nota: python -m pytest cursos/nivelacion/n1-python-data-science/m2-outliers-encoding/clase-06-outliers)

Reglas:
- No modifiques el DataFrame de entrada (trabaja sobre una copia: df.copy()).
- Evita bucles `for` sobre filas: usa operaciones de pandas / numpy.
- Dataset de práctica: data/raw/viviendas_lima.csv

Hilo del ejercicio (sigue la clase):
  1. Contar cuántos valores escapan de unos límites CONOCIDOS (la "edad de 200 años").
  2. Llevarlos a sus límites (capping), como los sensores de voltaje 0-100.
  3. Ver por qué un outlier "jala" la media pero no la mediana.
  4. Ver cómo los outliers "jalan" la recta de una regresión lineal.
  5. Acercar una variable a la normal con PowerTransformer (Box-Cox / Yeo-Johnson).
  6. (Reto) Medir la asociación entre dos categóricas con la V de Cramér.

Orden sugerido: 1 → 2 → 3 → 4 → 5 → 6. Son independientes entre sí.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 1
def contar_fuera_de_rango(df: pd.DataFrame, columna: str, minimo: float, maximo: float) -> dict:
    """Cuenta cuántos valores de `columna` están por DEBAJO de `minimo` y por ENCIMA de `maximo`.

    El profesor: lo primero es "identificar las columnas que sabemos que tienen límites
    que no pueden superar". Una edad de -10 o de 200 años es imposible: es un ERROR.
    Antes de corregir, hay que saber CUÁNTOS hay (si son 3 no es lo mismo que si son 300).

    Los límites son "válidos": un valor igual a `minimo` o a `maximo` NO está fuera.
    Los NaN no se cuentan (no son outliers, son faltantes: eso fue el módulo 1).

    Args:
        df: DataFrame (NO se modifica).
        columna: columna numérica a revisar.
        minimo: límite inferior conocido (incluido en el rango válido).
        maximo: límite superior conocido (incluido en el rango válido).

    Devuelve:
        dict con tres enteros (int de Python):
          "por_debajo" -> cuántos valores son < minimo
          "por_encima" -> cuántos valores son > maximo
          "total"      -> por_debajo + por_encima

    Ejemplo:
        >>> df = pd.DataFrame({"edad": [-10, 0, 35, 125, 200, None]})
        >>> contar_fuera_de_rango(df, "edad", 0, 125)
        {'por_debajo': 1, 'por_encima': 1, 'total': 2}
        # -10 está debajo; 200 está encima; 0 y 125 son los límites (válidos); None no cuenta

    💡 Pistas:
        - df[columna] < minimo devuelve una Serie de True/False (NaN < 0 da False).
        - .sum() sobre True/False cuenta los True.
        - int(x) convierte el número de numpy a int "normal".

    Esqueleto (una variable por paso):
        serie = ...                          # paso 1: la columna
        abajo = ...                          # paso 2: cuántos < minimo (con int)
        arriba = ...                         # paso 3: cuántos > maximo (con int)
        return {...}                         # paso 4: arma el dict
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 2
def recortar_a_limites(df: pd.DataFrame, columna: str, minimo: float, maximo: float) -> pd.DataFrame:
    """Devuelve una COPIA de `df` con `columna` llevada a sus límites (capping).

    La primera forma de tratar outliers según el profesor: "si tienes una variable
    conocida, donde sabes cuáles son sus límites superior e inferior, llévala a sus
    límites": toda edad menor que 0 pasa a 0 y toda edad mayor que 125 pasa a 125.
    Así lo hicieron con los sensores de voltaje (rango 0-100 que llegaban a 150 por
    las tormentas): "para no dejar un hueco" en vez de eliminar la fila.

    Args:
        df: DataFrame original (NO se modifica).
        columna: columna numérica a recortar.
        minimo: todo valor < minimo pasa a valer minimo.
        maximo: todo valor > maximo pasa a valer maximo.

    Devuelve:
        DataFrame nuevo con las MISMAS filas (no elimina ninguna) y las mismas columnas.
        En `columna`, los valores dentro del rango y los NaN quedan igual.

    Ejemplo:
        >>> df = pd.DataFrame({"voltaje": [50.0, 150.0, -5.0, 100.0, None]})
        >>> recortar_a_limites(df, "voltaje", 0, 100)["voltaje"].tolist()
        [50.0, 100.0, 0.0, 100.0, nan]

    💡 Pistas:
        - Serie.clip(lower=..., upper=...) hace exactamente esto (y respeta los NaN).
        - Recuerda asignar el resultado a la columna de la COPIA.

    Esqueleto:
        resultado = ...                      # paso 1: copia de df
        resultado[columna] = ...             # paso 2: la columna recortada con clip
        return resultado
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 3
def comparar_media_mediana(serie: pd.Series) -> dict:
    """Calcula la media y la mediana de una Serie y cuánto se separan.

    Por qué importa: el profesor dice que los outliers "jalan" al modelo. La MEDIA es
    el ejemplo más simple de ese jalón: un solo valor gigante la mueve mucho. La
    MEDIANA (el valor del medio) casi no se entera. Si media y mediana están muy
    lejos, sospecha de outliers (o de una variable muy asimétrica).

    Args:
        serie: Serie numérica (puede tener NaN: pandas los ignora).

    Devuelve:
        dict con tres floats de Python redondeados a 2 decimales:
          "media"      -> serie.mean()
          "mediana"    -> serie.median()
          "diferencia" -> media - mediana  (calcúlala SIN redondear y redondea al final)

    Ejemplo:
        >>> comparar_media_mediana(pd.Series([10, 11, 12, 13, 1000]))
        {'media': 209.2, 'mediana': 12.0, 'diferencia': 197.2}
        # un solo 1000 subió la media de ~11.5 a 209.2; la mediana sigue en 12

    💡 Pistas:
        - Serie.mean() y Serie.median() ya ignoran los NaN.
        - round(float(x), 2) deja un float de Python con 2 decimales.

    Esqueleto:
        media = ...                          # paso 1
        mediana = ...                        # paso 2
        diferencia = ...                     # paso 3: con los valores sin redondear
        return {...}                         # paso 4: todo redondeado a 2
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 4
def pendiente_con_y_sin_outliers(x, y, es_outlier) -> dict:
    """Ajusta una recta y = a·x + b dos veces (con y sin outliers) y devuelve las pendientes `a`.

    Es el gráfico de la clase: "esta recta es el modelo de regresión [...] los outliers
    jalan demasiado el modelo". Con todos los puntos la recta se tuerce; quitando los
    outliers vuelve a la tendencia real.

    La pendiente `a` dice cuánto sube y cuando x sube 1. En viviendas: cuántos USD
    vale cada m² extra. Si los outliers la cambian mucho, el modelo aprende un precio
    por m² equivocado.

    Args:
        x: valores de la variable explicativa (lista, array o Serie), sin NaN.
        y: valores de la variable objetivo, mismo largo que x, sin NaN.
        es_outlier: lista/array/Serie de True/False del mismo largo: True = es outlier.

    Devuelve:
        dict con dos floats de Python redondeados a 2 decimales:
          "pendiente_con" -> pendiente usando TODOS los puntos
          "pendiente_sin" -> pendiente usando solo los puntos con es_outlier == False

    Ejemplo:
        >>> x = [1, 2, 3, 4, 5, 6]
        >>> y = [2, 4, 6, 8, 10, 40]          # el último punto es un outlier
        >>> pendiente_con_y_sin_outliers(x, y, [False, False, False, False, False, True])
        {'pendiente_con': 6.0, 'pendiente_sin': 2.0}
        # sin el outlier, y = 2·x exacto; con él, la recta se empina hasta 6

    💡 Pistas:
        - np.polyfit(x, y, 1) ajusta una recta y devuelve [pendiente, intercepto].
        - Convierte todo a arrays para filtrar sin líos de índices:
              x = np.asarray(x, dtype=float)
              mascara = np.asarray(es_outlier, dtype=bool)
          y luego x[~mascara] se queda con los que NO son outliers (~ = "no").

    Esqueleto:
        x = ...                              # paso 1: array de floats
        y = ...                              # paso 2: array de floats
        mascara = ...                        # paso 3: array de bool
        pendiente_con = ...                  # paso 4: polyfit con todo, toma [0]
        pendiente_sin = ...                  # paso 5: polyfit con x[~mascara], y[~mascara]
        return {...}                         # paso 6: redondeado a 2
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 5
def aplicar_power_transformer(df: pd.DataFrame, columnas: list, metodo: str = "yeo-johnson") -> pd.DataFrame:
    """Devuelve una COPIA de `df` con `columnas` transformadas por un PowerTransformer de sklearn.

    El profesor se lo recomendó a Arturo: PowerTransformer "trata de llevar la variable
    a una normal" con dos transformaciones:
      - "box-cox":     SOLO acepta valores estrictamente positivos (> 0).
      - "yeo-johnson": acepta positivos, cero y negativos.
    Sirve, por ejemplo, porque PCA necesita variables cercanas a la normal.
    Una variable con una cola larga (como precio_usd) queda mucho más simétrica y
    sus valores extremos dejan de estar tan lejos del resto.

    Usa los valores por defecto de PowerTransformer (standardize=True: el resultado
    queda con media ~0 y desviación ~1). Los NaN se mantienen como NaN (sklearn los
    ignora al ajustar y los deja en su sitio al transformar).

    Args:
        df: DataFrame original (NO se modifica).
        columnas: lista de columnas numéricas a transformar.
        metodo: "box-cox" o "yeo-johnson".

    Devuelve:
        DataFrame nuevo, igual a `df` salvo que las `columnas` están transformadas.

    Errores (lanza ValueError con un mensaje en español):
        - si `metodo` no es "box-cox" ni "yeo-johnson".
        - si metodo == "box-cox" y alguna de las columnas tiene algún valor <= 0.

    Ejemplo:
        >>> df = pd.DataFrame({"precio": [100.0, 200.0, 400.0, 10000.0], "zona": list("ABCD")})
        >>> r = aplicar_power_transformer(df, ["precio"], "box-cox")
        >>> round(r["precio"].mean(), 6), r["zona"].tolist()
        (0.0, ['A', 'B', 'C', 'D'])
        >>> aplicar_power_transformer(pd.DataFrame({"x": [0.0, 1.0]}), ["x"], "box-cox")
        ValueError: box-cox solo acepta valores positivos ...

    💡 Pistas:
        - from sklearn.preprocessing import PowerTransformer
        - PowerTransformer(method=metodo).fit_transform(df[columnas]) devuelve un array
          con la misma forma que df[columnas]; asígnalo a resultado[columnas].
        - (df[columnas] <= 0).any().any() es True si hay algún valor no positivo
          (el primer .any() revisa cada columna, el segundo junta las columnas).
        - raise ValueError("...") para los casos inválidos. Valida ANTES de transformar.

    Esqueleto:
        if metodo not in (...):              # paso 1: método válido
            raise ...
        if metodo == "box-cox" and ...:      # paso 2: box-cox con valores <= 0
            raise ...
        resultado = ...                      # paso 3: copia
        transformador = ...                  # paso 4: PowerTransformer(method=metodo)
        resultado[columnas] = ...            # paso 5: fit_transform de df[columnas]
        return resultado
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 6 (reto)
def cramers_v(df: pd.DataFrame, col_a: str, col_b: str) -> float:
    """(Reto) Calcula la V de Cramér entre dos columnas CATEGÓRICAS.

    El profesor, sobre correlacionar variables binarias (como "hubo campaña: 1/0"):
    una binaria es una categórica de dos categorías, y entre categóricas la asociación
    se mide con la V de Cramér, no con la correlación de Pearson. Y avisó: "no lo
    mezclen": categóricas con categóricas, numéricas con numéricas.

    V va de 0 (no hay ninguna relación) a 1 (una determina por completo a la otra).

    Receta (📌 Extra, la fórmula no se vio en clase):
        1. tabla = tabla de contingencia (cuántas filas hay de cada combinación a, b).
        2. chi2  = estadístico chi-cuadrado de esa tabla, SIN corrección de Yates.
        3. n     = total de observaciones de la tabla.
        4. k     = min(número de filas de la tabla, número de columnas) - 1.
        5. V     = raíz cuadrada de chi2 / (n · k).
       Si k == 0 (una de las columnas tiene una sola categoría), devuelve 0.0.

    Args:
        df: DataFrame (NO se modifica).
        col_a, col_b: nombres de dos columnas categóricas (texto, bool o 0/1).

    Devuelve:
        float de Python redondeado a 4 decimales, entre 0 y 1.

    Ejemplo:
        >>> df = pd.DataFrame({"campania": [1, 1, 0, 0], "venta_alta": ["Si", "Si", "No", "No"]})
        >>> cramers_v(df, "campania", "venta_alta")
        1.0       # cada vez que hay campaña, la venta es alta: asociación perfecta
        >>> df = pd.DataFrame({"a": ["x", "x", "y", "y"], "b": ["p", "q", "p", "q"]})
        >>> cramers_v(df, "a", "b")
        0.0       # saber a no dice nada sobre b

    💡 Pistas:
        - pd.crosstab(df[col_a], df[col_b]) arma la tabla de contingencia.
        - from scipy.stats import chi2_contingency
          chi2_contingency(tabla, correction=False) devuelve varias cosas; la PRIMERA es chi2.
        - tabla.to_numpy().sum() es n; tabla.shape es (filas, columnas).
        - np.sqrt(...) para la raíz.

    Esqueleto:
        tabla = ...                          # paso 1
        chi2 = ...                           # paso 2: [0] del resultado de chi2_contingency
        n = ...                              # paso 3
        k = ...                              # paso 4
        if k == 0:
            return 0.0
        v = ...                              # paso 5
        return ...                           # paso 6: float redondeado a 4
    """
    raise NotImplementedError
