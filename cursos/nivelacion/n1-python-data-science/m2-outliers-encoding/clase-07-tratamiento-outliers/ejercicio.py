"""
Clase 07 · Tratamiento de outliers (regla IQR, z-score y outliers multivariados)
=================================================================================

Implementa cada función reemplazando `raise NotImplementedError`.
Evalúa con:

    python -m framework evaluar cursos/nivelacion/n1-python-data-science/m2-outliers-encoding/clase-07-tratamiento-outliers

(o solo la nota: python -m pytest cursos/nivelacion/n1-python-data-science/m2-outliers-encoding/clase-07-tratamiento-outliers)

Reglas:
- No modifiques el DataFrame de entrada (trabaja sobre una copia: df.copy()).
- Evita bucles `for` sobre filas: usa operaciones de pandas / numpy.
  (Un `for` sobre COLUMNAS sí está permitido.)
- Los NaN NO son outliers: nunca se marcan como atípicos ni provocan que se borre una fila.
- Dataset de práctica: data/raw/viviendas_lima.csv

Hilo del ejercicio (sigue la clase):
  1. Calcular los límites de la regla IQR (Q1 - 1.5·IQR, Q3 + 1.5·IQR).
  2. Marcar qué filas son outliers en UNA columna (como el ejemplo de CRIM).
  3. Eliminar outliers en VARIAS columnas a la vez (la "automatización" del profesor).
  4. Medir cuántas filas se perdieron (Boston: de 506 a 268, casi la mitad).
  5. Otra regla univariada: ±3 desviaciones estándar (z-score).
  6. (Reto) Detectar un outlier MULTIVARIADO con la distancia de Mahalanobis
     (el ejemplo de altura y peso: normal en cada eje, raro en conjunto).

Orden sugerido: 1 → 2 → 3 → 4 → 5 → 6. La 2 y la 3 se apoyan en la 1
(puedes llamar a `limites_iqr` dentro de ellas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 1
def limites_iqr(serie: pd.Series, k: float = 1.5) -> tuple:
    """Devuelve los límites (inferior, superior) de la regla del rango intercuartílico.

    El profesor: "calculo el rango intercuartílico, que es el cuartil 3 menos el
    cuartil 1, y simplemente el límite superior es el cuartil 3 + 1.5 veces el rango
    intercuartílico y el límite inferior es cuartil 1 - 1.5 veces el rango".
    Es el método "más clásico" porque no hay que buscar ningún hiperparámetro.

        Q1  = percentil 25        Q3 = percentil 75
        IQR = Q3 - Q1
        inferior = Q1 - k * IQR
        superior = Q3 + k * IQR

    Args:
        serie: valores numéricos (puede tener NaN; se ignoran).
        k: multiplicador del IQR. El clásico es 1.5 (3.0 = solo outliers "extremos").

    Devuelve:
        Tupla (inferior, superior) de floats. Usa los cuantiles de pandas tal cual
        (`Serie.quantile`, interpolación lineal por defecto). No redondees.

    Ejemplo:
        >>> limites_iqr(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 100]))
        (-3.0, 13.0)
        # Q1 = 3, Q3 = 7, IQR = 4 -> 3 - 6 = -3  y  7 + 6 = 13. El 100 queda fuera.

    💡 Pistas:
        - serie.quantile(0.25) y serie.quantile(0.75) dan Q1 y Q3 (ignoran NaN solos).
        - Devuelve floats normales: float(valor).

    Esqueleto (una variable por paso):
        q1 = ...            # paso 1: percentil 25
        q3 = ...            # paso 2: percentil 75
        iqr = ...           # paso 3: Q3 - Q1
        inferior = ...      # paso 4
        superior = ...      # paso 5
        return (float(inferior), float(superior))
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 2
def marcar_outliers_iqr(df: pd.DataFrame, columna: str, k: float = 1.5) -> pd.Series:
    """Devuelve una Serie booleana: True en las filas donde `columna` es outlier según IQR.

    Es lo que hizo el profesor con CRIM (índice de criminalidad): calcula los
    límites y filtra "los puntos que están sobre el cuartil 3 + 1.5 el IQR". Por
    debajo no había nada; por arriba, varios.

    Un valor es outlier si es ESTRICTAMENTE menor que el límite inferior o
    ESTRICTAMENTE mayor que el límite superior. Un valor exactamente igual al
    límite NO es outlier. Los NaN dan False.

    Args:
        df: DataFrame (NO se modifica).
        columna: nombre de la columna numérica a revisar.
        k: multiplicador del IQR (por defecto 1.5).

    Devuelve:
        pd.Series de dtype bool, con el MISMO índice que `df`.

    Ejemplo:
        >>> df = pd.DataFrame({"area_m2": [60, 70, 80, 90, 100, 110, 120, 1200]},
        ...                   index=[10, 11, 12, 13, 14, 15, 16, 17])
        >>> marcar_outliers_iqr(df, "area_m2")
        10    False
        11    False
        ...
        17     True      <- 1200 m²: el típico error de digitación (120 × 10)
        Name: area_m2, dtype: bool

    💡 Pistas:
        - Reutiliza limites_iqr(df[columna], k).
        - (s < inferior) | (s > superior) ya devuelve False para los NaN.

    Esqueleto:
        s = ...                          # paso 1: la columna
        inferior, superior = ...         # paso 2: límites
        return ...                       # paso 3: condición "por debajo O por encima"
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 3
def eliminar_outliers_iqr(df: pd.DataFrame, columnas: list, k: float = 1.5) -> pd.DataFrame:
    """Devuelve una COPIA de `df` sin las filas que son outlier en ALGUNA de `columnas`.

    Es la "automatización" del profesor: en vez de tratar una variable, "de todas
    las columnas extraigo su cuartil 1 y su cuartil 3, calculo el IQR de todo y
    filtro directamente de todos". En Boston eso dejó 268 de 506 filas.

    Reglas importantes:
        - Los límites de CADA columna se calculan UNA sola vez, sobre el `df`
          ORIGINAL (no se recalculan después de ir borrando).
        - Una fila se elimina si es outlier en AL MENOS UNA de las columnas.
        - Un NaN no es outlier: una fila con NaN se conserva (si no es outlier
          en otra columna).
        - Se conserva el índice original (NO hagas reset_index) y el orden.
        - Se conservan TODAS las columnas de `df`, no solo las de `columnas`.

    Args:
        df: DataFrame original (NO se modifica).
        columnas: lista de columnas numéricas a limpiar.
        k: multiplicador del IQR.

    Devuelve:
        DataFrame nuevo con las filas que sobrevivieron.

    Ejemplo:
        >>> df = pd.DataFrame({"area": [50, 60, 70, 80, 90, 900],
        ...                    "precio": [100, 9000, 120, 130, 140, 150]})
        >>> eliminar_outliers_iqr(df, ["area", "precio"])
           area  precio
        0    50     100
        2    70     120
        3    80     130
        4    90     140
        # Fila 5 cae por "area" (900) y fila 1 por "precio" (9000).
        # Perdimos 2 de 6 filas aunque cada columna tenía UN solo outlier.

    💡 Pistas:
        - Empieza con una máscara "todas se quedan":
          conservar = pd.Series(True, index=df.index)
        - Recorre las columnas y combina con &:  conservar = conservar & ~es_outlier
        - Puedes reutilizar marcar_outliers_iqr(df, col, k).
        - df.loc[conservar].copy() devuelve la copia filtrada.

    Esqueleto:
        conservar = ...                      # paso 1: máscara inicial (todo True)
        for col in columnas:                 # paso 2: por cada columna...
            es_outlier = ...                 #   ...¿qué filas son outlier en col?
            conservar = ...                  #   ...quítalas de la máscara
        return ...                           # paso 3: copia filtrada
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 4
def porcentaje_filas_perdidas(original: pd.DataFrame, filtrado: pd.DataFrame) -> float:
    """Devuelve qué porcentaje (0-100) de filas se perdió al pasar de `original` a `filtrado`.

    El profesor se alarma: "mi dataset se ha reducido a 268 registros,
    originalmente eran 506 ... prácticamente el 50%" y eso "me puede comprometer
    las otras variables". Antes de borrar outliers, MIDE cuánto pierdes.

        porcentaje = (filas_original - filas_filtrado) / filas_original * 100

    Args:
        original: DataFrame antes de limpiar.
        filtrado: DataFrame después de limpiar.

    Devuelve:
        float redondeado a 2 decimales. Si `original` no tiene filas, devuelve 0.0.

    Ejemplo:
        >>> porcentaje_filas_perdidas(pd.DataFrame({"a": range(506)}),
        ...                           pd.DataFrame({"a": range(268)}))
        47.04

    💡 Pistas:
        - len(df) da el número de filas.
        - round(valor, 2).

    Esqueleto:
        n_original = ...
        n_filtrado = ...
        if n_original == 0: return 0.0
        return round(..., 2)
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 5
def outliers_zscore(serie: pd.Series, umbral: float = 3.0) -> pd.Series:
    """Devuelve una Serie booleana: True donde el valor está a MÁS de `umbral` desviaciones
    estándar de la media.

    El profesor: "los puntos de datos a tres o más desviaciones estándar de la media
    se consideran también atípicos".

        z = (x - media) / desviacion_estandar
        outlier  <=>  |z| > umbral

    Usa la media y la desviación estándar de pandas (`serie.mean()`, `serie.std()`;
    `std` usa ddof=1 por defecto, déjalo así). Los NaN dan False.
    Si la desviación estándar es 0 (todos iguales), ningún valor es outlier.

    Args:
        serie: valores numéricos.
        umbral: cuántas desviaciones estándar se toleran (por defecto 3).

    Devuelve:
        pd.Series de dtype bool con el MISMO índice que `serie`.

    Ejemplo:
        >>> s = pd.Series([10] * 20 + [100])
        >>> outliers_zscore(s).tolist()[-1]
        True       # el 100 está a ~4.4 desviaciones estándar de la media
        >>> outliers_zscore(s).sum()
        1

    ⚠️ Ojo (lo que dijo el profesor sobre la media): el outlier también "infla" la
    media y la desviación estándar con las que se calcula z. Por eso, con muy pocos
    datos, un outlier puede "esconderse" y no superar el umbral.

    💡 Pistas:
        - (serie - serie.mean()) / serie.std() calcula z para todos a la vez.
        - .abs() y luego > umbral.
        - Para el caso std == 0: devuelve pd.Series(False, index=serie.index).

    Esqueleto:
        media = ...
        desv = ...
        if desv == 0: return ...
        z = ...
        return ...
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 6 (reto)
def distancia_mahalanobis(df: pd.DataFrame, columnas: list) -> pd.Series:
    """Devuelve, para cada fila, su distancia de Mahalanobis al "centro" de los datos.

    El profesor: los outliers "no solamente son univariados, también pueden ser
    multivariados" (altura y peso: ninguna sale del boxplot, pero al cruzarlas
    aparecen puntos raros). "Los valores atípicos bivariados y multivariados
    generalmente se miden utilizando un índice de influencia o la distancia".

    📌 Extra: la distancia de Mahalanobis es esa "distancia". Mide qué tan lejos está
    un punto del centro TENIENDO EN CUENTA cómo se relacionan las variables
    (su covarianza). Una persona de 182 cm y 52 kg no es rara por su altura ni por
    su peso, pero sí por la COMBINACIÓN, y Mahalanobis lo detecta.

        X      = matriz (n filas × p columnas) con las filas SIN NaN en `columnas`
        mu     = media de cada columna                       (vector de p)
        S      = matriz de covarianza de X (ddof=1)          (p × p)
        S_inv  = inversa de S
        d_i    = x_i - mu                                    (para cada fila i)
        dist_i = raíz( d_i · S_inv · d_iᵀ )

    Reglas:
        - mu y S se calculan solo con las filas que tienen TODAS las `columnas` sin NaN.
        - Las filas con algún NaN en `columnas` reciben NaN como distancia.
        - El resultado tiene el MISMO índice que `df` (incluidas las filas con NaN).

    Args:
        df: DataFrame (NO se modifica).
        columnas: lista de 2 o más columnas numéricas.

    Devuelve:
        pd.Series de floats (distancias >= 0), índice igual al de `df`.

    Ejemplo (altura/peso, la última persona es alta pero muy liviana):
        >>> df = pd.DataFrame({
        ...     "altura_cm": [150, 160, 170, 180, 190, 185],
        ...     "peso_kg":   [ 50,  60,  70,  80,  90,  55]})
        >>> distancia_mahalanobis(df, ["altura_cm", "peso_kg"]).round(2)
        La fila 5 (185 cm, 55 kg) tiene la distancia MÁS GRANDE de todas,
        aunque 185 y 55 están dentro del rango de cada columna.

    💡 Pistas:
        - limpio = df[columnas].dropna()  ->  X = limpio.to_numpy(dtype=float)
        - mu = X.mean(axis=0)
        - S = np.cov(X, rowvar=False)   (rowvar=False: las columnas son variables)
        - S_inv = np.linalg.inv(S)
        - d = X - mu   (numpy resta mu a cada fila automáticamente)
        - Para todas las filas a la vez: np.sum((d @ S_inv) * d, axis=1) da d_i·S_inv·d_iᵀ.
          Luego np.sqrt(...).
        - pd.Series(valores, index=limpio.index).reindex(df.index) pone NaN donde faltaba.

    Esqueleto:
        limpio = ...          # paso 1: filas sin NaN en columnas
        X = ...               # paso 2: a numpy
        mu = ...              # paso 3: medias
        S_inv = ...           # paso 4: inversa de la covarianza
        d = ...               # paso 5: desvíos respecto a la media
        dist = ...            # paso 6: raíz de la forma cuadrática, fila por fila
        return ...            # paso 7: Serie con el índice de df (reindex)
    """
    raise NotImplementedError
