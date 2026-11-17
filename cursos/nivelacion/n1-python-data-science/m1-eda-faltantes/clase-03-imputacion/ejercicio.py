"""
Clase 03 · Datos faltantes y métodos de imputación
===================================================

Implementa cada función reemplazando `raise NotImplementedError`.
Evalúa con:

    python -m framework evaluar cursos/nivelacion/n1-python-data-science/m1-eda-faltantes/clase-03-imputacion

(o solo la nota: python -m pytest cursos/nivelacion/n1-python-data-science/m1-eda-faltantes/clase-03-imputacion)

Reglas:
- No modifiques el DataFrame de entrada (trabaja sobre una copia: df.copy()).
- Evita bucles `for` sobre filas: usa operaciones de pandas.
- Dataset de práctica: data/raw/viviendas_lima.csv

Hilo del ejercicio (sigue la clase):
  1. Encontrar los vacíos "disfrazados" (la 6.ª categoría "blanco / no reportó").
  2. Imputar una categórica (moda o "Desconocido").
  3. Imputar una numérica por grupo (mediana por distrito).
  4. Ver cómo un LEFT JOIN "fabrica" nulos (la anécdota del script SQL).
  5. Medir cuánto cambia la distribución al imputar (sesgo).

Orden sugerido: 1 → 2 → 3 → 4 → 5. Son independientes entre sí.
"""

from __future__ import annotations

import pandas as pd

# Textos que en la práctica significan "no hay dato".
# Se comparan DESPUÉS de quitar espacios y pasar a minúsculas.
MARCADORES_VACIO = ("", "n/a", "na", "-", "null", "none", "sin dato", "no reporto")


# ---------------------------------------------------------------- 1
def normalizar_vacios(
    df: pd.DataFrame, columna: str, marcadores: tuple = MARCADORES_VACIO
) -> pd.DataFrame:
    """Devuelve una COPIA de `df` donde los vacíos "disfrazados" de `columna` pasan a NaN.

    El profesor: al revisar nulos aparece "una sexta categoría que dice blanco, vacío,
    no reportó". Para pandas eso es un texto normal, NO un NaN: `isna()` no lo cuenta.
    Primero hay que convertirlo en NaN de verdad; recién ahí se puede contar e imputar.

    Un valor se considera vacío si, tras quitarle espacios (strip) y pasarlo a
    minúsculas (lower), está en `marcadores`. Ej: "  N/A " -> "n/a" -> vacío.
    Los valores que no son texto (números, NaN) no se tocan.

    Args:
        df: DataFrame original (NO se modifica).
        columna: columna de texto a limpiar.
        marcadores: textos (en minúsculas, sin espacios) que significan "no hay dato".

    Devuelve:
        DataFrame nuevo, igual a `df` salvo que en `columna` los vacíos disfrazados son NaN.
        Los demás valores quedan EXACTAMENTE como estaban (no los pases a minúsculas).

    Ejemplo:
        >>> df = pd.DataFrame({"email": ["a@x.com", " ", "N/A", "No reporto", None]})
        >>> normalizar_vacios(df, "email")
             email
        0  a@x.com
        1      NaN     <- " " -> "" -> vacío
        2      NaN     <- "N/A" -> "n/a" -> vacío
        3      NaN     <- "No reporto" -> "no reporto" -> vacío
        4      NaN     <- ya era NaN

    💡 Pistas:
        - df[columna].str.strip().str.lower() normaliza el texto (los NaN siguen NaN).
        - Serie.isin(marcadores) devuelve True/False: True donde el texto es un marcador.
        - resultado.loc[mascara, columna] = np.nan  (o pd.NA / None) pone el hueco.
          Usa la máscara para ELEGIR filas, pero escribe sobre la columna original.

    Esqueleto (una variable por paso):
        resultado = ...                       # paso 1: copia de df
        texto = ...                           # paso 2: columna con strip + lower
        mascara = ...                         # paso 3: ¿qué filas son marcadores?
        resultado.loc[mascara, columna] = ... # paso 4: ponles NaN
        return resultado
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 2
def imputar_categorica(
    df: pd.DataFrame, columna: str, estrategia: str = "moda", valor: str = "Desconocido"
) -> pd.DataFrame:
    """Devuelve una COPIA de `df` con los NaN de una columna categórica rellenados.

    Dos estrategias clásicas (📌 Extra, el profesor las verá en clases siguientes):
      - "moda":      rellena con la categoría más frecuente. Rápido, pero "infla" esa
                     categoría y borra la señal de que el dato faltaba.
      - "constante": rellena con `valor` (por defecto "Desconocido"). Convierte el
                     hueco en una categoría explícita: el modelo PUEDE aprender si
                     "no reportó" significa algo.

    Args:
        df: DataFrame original (NO se modifica).
        columna: columna categórica a imputar.
        estrategia: "moda" o "constante". Cualquier otra cosa -> ValueError.
        valor: texto a usar cuando estrategia == "constante".

    Devuelve:
        DataFrame nuevo, igual a `df` salvo que `columna` ya no tiene NaN.

    Ejemplo:
        >>> df = pd.DataFrame({"estado": ["Bueno", None, "Bueno", "Regular"]})
        >>> imputar_categorica(df, "estado")["estado"].tolist()
        ['Bueno', 'Bueno', 'Bueno', 'Regular']
        >>> imputar_categorica(df, "estado", estrategia="constante")["estado"].tolist()
        ['Bueno', 'Desconocido', 'Bueno', 'Regular']

    💡 Pistas:
        - df[columna].mode() devuelve una SERIE (puede haber empates, vienen ordenados);
          toma el primero con .iloc[0].
        - Serie.fillna(relleno) rellena los huecos (recuerda asignar el resultado).
        - raise ValueError(f"Estrategia no válida: {estrategia}") para el caso inválido.

    Esqueleto:
        if estrategia == "moda":
            relleno = ...                     # paso 1a: la moda
        elif estrategia == "constante":
            relleno = ...                     # paso 1b: el parámetro valor
        else:
            raise ...                         # paso 1c: estrategia inválida
        resultado = ...                       # paso 2: copia
        resultado[columna] = ...              # paso 3: fillna con relleno
        return resultado
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 3
def imputar_mediana_por_grupo(df: pd.DataFrame, columna: str, grupo: str) -> pd.DataFrame:
    """Rellena los NaN de `columna` con la mediana DE SU GRUPO (no la global).

    En la clase 02 vimos que `distancia_centro_km` falta más en distritos periféricos
    (MAR). Si rellenas con la mediana global (~10 km), a una casa de Ate (~15 km) le
    pones la distancia de Miraflores: eso es SESGO. Con la mediana de su distrito,
    el valor imputado es mucho más creíble.

    Regla extra: si un grupo tiene TODOS sus valores en NaN (no hay mediana de grupo),
    esos huecos se rellenan con la mediana GLOBAL de la columna.

    Args:
        df: DataFrame original (NO se modifica).
        columna: columna numérica a imputar.
        grupo: columna categórica que define los grupos.

    Devuelve:
        DataFrame nuevo, igual a `df` salvo que `columna` ya no tiene NaN.
        Los valores que ya existían no cambian.

    Ejemplo:
        >>> df = pd.DataFrame({"zona": ["A", "A", "A", "B", "B", "C"],
        ...                    "km":   [1.0, 3.0, None, 10.0, None, None]})
        >>> imputar_mediana_por_grupo(df, "km", "zona")["km"].tolist()
        [1.0, 3.0, 2.0, 10.0, 10.0, 3.0]
        # A -> mediana 2.0 ; B -> 10.0 ; C no tiene datos -> mediana global de [1, 3, 10] = 3.0

    💡 Pistas:
        - df.groupby(grupo)[columna].transform("median") devuelve una Serie del MISMO
          largo que df: en cada fila, la mediana de su grupo. ¡Es la clave!
        - Serie.fillna(otra_serie) rellena cada hueco con el valor de la misma fila.
        - Puedes encadenar dos fillna: primero por grupo, luego con la mediana global.

    Esqueleto:
        resultado = ...                       # paso 1: copia
        mediana_grupo = ...                   # paso 2: groupby + transform("median")
        mediana_global = ...                  # paso 3: mediana de toda la columna
        resultado[columna] = ...              # paso 4: fillna(grupo) y luego fillna(global)
        return resultado
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 4
def nulos_por_join(izquierda: pd.DataFrame, derecha: pd.DataFrame, clave: str) -> dict:
    """Compara un INNER JOIN con un LEFT JOIN y cuenta cuántas filas quedaron SIN pareja.

    La anécdota del profesor: "en vez de un INNER JOIN hizo un LEFT JOIN [...] me sacaste
    nulos". En un LEFT JOIN, las filas de la izquierda que no encuentran pareja en la
    derecha se quedan igual, pero con NaN en TODAS las columnas que venían de la derecha.
    Ese NaN no es un dato "no reportado": lo fabricó la consulta.

    En pandas: pd.merge(izquierda, derecha, on=clave, how="inner" | "left").

    Args:
        izquierda: tabla principal (p. ej. viviendas).
        derecha: tabla que se le pega (p. ej. datos de cada distrito). Su `clave` no se repite.
        clave: columna común por la que se une.

    Devuelve:
        dict con tres enteros (int de Python):
          "filas_inner"      -> número de filas del INNER JOIN
          "filas_left"       -> número de filas del LEFT JOIN
          "filas_sin_pareja" -> filas del LEFT JOIN cuya clave NO está en `derecha`
                                (las que quedaron con NaN fabricados)

    Ejemplo:
        >>> casas = pd.DataFrame({"distrito": ["Lince", "Ate", "Lince", "SJL"]})
        >>> info = pd.DataFrame({"distrito": ["Lince", "Ate"], "zona": ["Centro", "Este"]})
        >>> nulos_por_join(casas, info, "distrito")
        {'filas_inner': 3, 'filas_left': 4, 'filas_sin_pareja': 1}     # SJL no tiene pareja

    💡 Pistas:
        - len(df) da el número de filas.
        - pd.merge(..., how="left", indicator=True) agrega la columna "_merge" que vale
          "left_only" en las filas sin pareja. (Otra forma: ~izquierda[clave].isin(derecha[clave])).
        - int(x) convierte un número de numpy a int "normal".

    Esqueleto:
        inner = ...                           # paso 1: merge how="inner"
        left = ...                            # paso 2: merge how="left", indicator=True
        sin_pareja = ...                      # paso 3: cuenta las "left_only"
        return {...}                          # paso 4: arma el dict con int(...)
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 5
def impacto_imputacion(original: pd.Series, imputada: pd.Series) -> dict:
    """Mide cuánto cambió la distribución de una variable al imputarla.

    Imputar NO es gratis: si rellenas muchos huecos con el MISMO número, la variable se
    "aplana" (baja la desviación estándar) y la media puede moverse. Eso es el SESGO del
    que habla el profesor: el modelo aprende de datos que ya no se parecen a la realidad.

    Args:
        original: la Serie ANTES de imputar (con NaN).
        imputada: la misma Serie DESPUÉS de imputar (sin NaN).

    Devuelve:
        dict con cinco floats redondeados a 2 decimales:
          "media_antes"       -> media de `original` (pandas ignora los NaN)
          "media_despues"     -> media de `imputada`
          "std_antes"         -> desviación estándar de `original` (la de pandas, .std())
          "std_despues"       -> desviación estándar de `imputada`
          "cambio_std_pct"    -> (std_despues - std_antes) / std_antes * 100
                                 Negativo = la variable se aplanó; positivo = se dispersó.
        Calcula cambio_std_pct con los valores SIN redondear y redondea solo al final.

    Ejemplo:
        >>> antes = pd.Series([2.0, 4.0, None, 6.0])
        >>> despues = pd.Series([2.0, 4.0, 4.0, 6.0])        # imputada con la media (4.0)
        >>> impacto_imputacion(antes, despues)
        {'media_antes': 4.0, 'media_despues': 4.0, 'std_antes': 2.0, 'std_despues': 1.63, 'cambio_std_pct': -18.35}
        # la media no se movió, pero la dispersión bajó un 18 %

    💡 Pistas:
        - Serie.mean() y Serie.std() ya ignoran los NaN.
        - round(float(x), 2) para dejar un float de Python con 2 decimales.

    Esqueleto:
        media_antes = ...                     # paso 1
        media_despues = ...                   # paso 2
        std_antes = ...                       # paso 3
        std_despues = ...                     # paso 4
        cambio = ...                          # paso 5: fórmula del cambio porcentual
        return {...}                          # paso 6: todo redondeado a 2
    """
    raise NotImplementedError
