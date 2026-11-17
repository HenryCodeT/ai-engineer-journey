"""Evaluador automático — Clase 06 · Qué son los outliers.  NO MODIFICAR.

Cada test tiene un peso (@pytest.mark.puntos). La suma es 100.
"""
import math

import numpy as np
import pandas as pd
import pytest

from framework.testing import cargar_ejercicio, leer_dataset

ej = cargar_ejercicio(__file__)


@pytest.fixture
def edades():
    # Edades con errores de carga: un negativo (bebé mal convertido) y una edad imposible.
    return pd.DataFrame({
        "persona": ["Ana", "Bebé", "Luis", "Rosa", "Juan", "Eva", "Lia"],
        "edad": [34.0, -1.0, 0.0, 125.0, 200.0, np.nan, 61.0],
        "ciudad": ["Lima", "Lima", "Cusco", "Lima", "Piura", "Cusco", "Lima"],
    })


@pytest.fixture
def sensores():
    # Sensores de voltaje (rango válido 0-100) que las tormentas sacaban de rango.
    return pd.DataFrame({
        "sensor": ["S1", "S1", "S2", "S2", "S3", "S3"],
        "voltaje": [55.0, 150.0, 98.5, -3.0, 100.0, 120.0],
    })


@pytest.fixture(scope="module")
def viviendas():
    return leer_dataset("viviendas_lima.csv")


# ---------------------------------------------------------------- contar_fuera_de_rango (15)
@pytest.mark.puntos(8)
def test_contar_fuera_de_rango_mini(edades, sensores):
    copia = edades.copy()
    r = ej.contar_fuera_de_rango(edades, "edad", 0, 125)
    pd.testing.assert_frame_equal(edades, copia, obj="No modifiques el DataFrame de entrada")
    assert r == {"por_debajo": 1, "por_encima": 1, "total": 2}, (
        "Edades válidas entre 0 y 125: -1 está por debajo, 200 por encima. 0 y 125 son los propios "
        f"límites (válidos) y el NaN no cuenta. Esperaba {{'por_debajo': 1, 'por_encima': 1, 'total': 2}}, obtuve {r}"
    )
    assert all(type(v) is int for v in r.values()), "Los tres valores deben ser int de Python (usa int(...))"
    r2 = ej.contar_fuera_de_rango(sensores, "voltaje", 0, 100)
    assert r2 == {"por_debajo": 1, "por_encima": 2, "total": 3}, (
        f"Voltaje 0-100: -3 por debajo; 150 y 120 por encima (100 es el límite, es válido). Obtuve {r2}"
    )


@pytest.mark.puntos(7)
def test_contar_fuera_de_rango_viviendas(viviendas):
    r = ej.contar_fuera_de_rango(viviendas, "antiguedad_anios", 0, 100)
    assert r == {"por_debajo": 0, "por_encima": 3, "total": 3}, (
        "En viviendas_lima hay 3 inmuebles con antigüedad 150 años (imposible en este dataset) y ninguno "
        f"negativo. Obtuve {r}"
    )
    r2 = ej.contar_fuera_de_rango(viviendas, "area_m2", 30, 1000)
    assert r2["total"] == 6 and r2["por_encima"] == 6, (
        f"area_m2 tiene 6 valores mayores a 1000 m² (errores de digitación ×10). Obtuve {r2}"
    )


# ---------------------------------------------------------------- recortar_a_limites (20)
@pytest.mark.puntos(12)
def test_recortar_a_limites_mini(edades, sensores):
    copia = edades.copy()
    r = ej.recortar_a_limites(edades, "edad", 0, 125)
    pd.testing.assert_frame_equal(edades, copia, obj="No modifiques el DataFrame de entrada (usa df.copy())")
    assert len(r) == len(edades), "Capping NO elimina filas: deben quedar las 7 personas"
    assert list(r.columns) == list(edades.columns), "No agregues ni quites columnas"
    esperado = [34.0, 0.0, 0.0, 125.0, 125.0, 61.0]
    obtenido = r["edad"].drop(index=5).tolist()
    assert obtenido == esperado, (
        "La edad -1 debe pasar a 0 y la de 200 a 125; las demás quedan igual. "
        f"Esperaba (sin la fila 5) {esperado}, obtuve {obtenido}"
    )
    assert pd.isna(r.loc[5, "edad"]), "El NaN de la fila 5 debe seguir siendo NaN (capping no imputa)"
    assert r["ciudad"].tolist() == edades["ciudad"].tolist(), "Solo se recorta la columna pedida"
    v = ej.recortar_a_limites(sensores, "voltaje", 0, 100)["voltaje"].tolist()
    assert v == [55.0, 100.0, 98.5, 0.0, 100.0, 100.0], (
        f"Sensores 0-100: 150 y 120 -> 100, -3 -> 0. Esperaba [55.0, 100.0, 98.5, 0.0, 100.0, 100.0], obtuve {v}"
    )


@pytest.mark.puntos(8)
def test_recortar_a_limites_viviendas(viviendas):
    nulos_antes = viviendas["antiguedad_anios"].isna().sum()
    r = ej.recortar_a_limites(viviendas, "antiguedad_anios", 0, 100)
    assert len(r) == len(viviendas) == 2015, "No se elimina ninguna vivienda: deben seguir siendo 2015 filas"
    assert r["antiguedad_anios"].max() == 100, "Después del capping ninguna antigüedad puede superar 100"
    assert (r["antiguedad_anios"] == 100).sum() == 3, "Las 3 viviendas de 150 años deben quedar en 100"
    assert r["antiguedad_anios"].isna().sum() == nulos_antes == 170, "Los 170 NaN de antigüedad deben seguir ahí"
    assert viviendas["antiguedad_anios"].max() == 150, "El DataFrame original debe conservar sus 150 años"
    normales = viviendas["antiguedad_anios"] <= 100
    assert r.loc[normales, "antiguedad_anios"].equals(viviendas.loc[normales, "antiguedad_anios"]), (
        "Los valores que ya estaban dentro del rango no deben cambiar"
    )


# ---------------------------------------------------------------- comparar_media_mediana (15)
@pytest.mark.puntos(8)
def test_comparar_media_mediana_mini():
    r = ej.comparar_media_mediana(pd.Series([10, 11, 12, 13, 1000]))
    assert r == {"media": 209.2, "mediana": 12.0, "diferencia": 197.2}, (
        "Con [10, 11, 12, 13, 1000]: media 209.2 (la jala el 1000), mediana 12.0, diferencia 197.2. "
        f"Obtuve {r}"
    )
    assert all(type(v) is float for v in r.values()), "Los valores deben ser float de Python (usa float(...))"
    r2 = ej.comparar_media_mediana(pd.Series([10, 11, 12, 13, 14, np.nan]))
    assert r2 == {"media": 12.0, "mediana": 12.0, "diferencia": 0.0}, (
        f"Sin outliers (y con un NaN que se ignora) media = mediana = 12.0. Obtuve {r2}"
    )
    r3 = ej.comparar_media_mediana(pd.Series([0.990, 0.996, 1.026]))
    assert r3["diferencia"] == pytest.approx(0.01), (
        "Media 1.004 y mediana 0.996: la diferencia real es 0.008 -> 0.01. Si redondeas antes "
        f"(1.0 - 1.0) sale 0.0. Calcula SIN redondear y redondea al final. Obtuve {r3['diferencia']}"
    )


@pytest.mark.puntos(7)
def test_comparar_media_mediana_viviendas(viviendas):
    r = ej.comparar_media_mediana(viviendas["precio_usd"])
    assert r["media"] == pytest.approx(157610.75, abs=0.01), f"Media de precio_usd: 157610.75. Obtuve {r['media']}"
    assert r["mediana"] == pytest.approx(130500.0), f"Mediana de precio_usd: 130500.0. Obtuve {r['mediana']}"
    assert r["diferencia"] == pytest.approx(27110.75, abs=0.01), (
        "La media de precio_usd supera a la mediana en 27110.75 USD: la jalan las mansiones del dataset"
    )
    r2 = ej.comparar_media_mediana(viviendas["antiguedad_anios"])
    assert r2["mediana"] == pytest.approx(10.0) and r2["media"] == pytest.approx(12.26, abs=0.01), (
        f"antiguedad_anios (con NaN): media 12.26, mediana 10.0. Obtuve {r2}"
    )


# ---------------------------------------------------------------- pendiente_con_y_sin_outliers (20)
@pytest.mark.puntos(10)
def test_pendiente_mini():
    x = [1, 2, 3, 4, 5, 6]
    y = [2, 4, 6, 8, 10, 40]
    r = ej.pendiente_con_y_sin_outliers(x, y, [False, False, False, False, False, True])
    assert set(r) == {"pendiente_con", "pendiente_sin"}, f"Claves esperadas: pendiente_con y pendiente_sin. Obtuve {set(r)}"
    assert r["pendiente_sin"] == pytest.approx(2.0), (
        f"Sin el outlier los puntos cumplen y = 2·x: la pendiente debe ser 2.0. Obtuve {r['pendiente_sin']}"
    )
    assert r["pendiente_con"] == pytest.approx(6.0), (
        f"Con el punto (6, 40) la recta se empina: la pendiente debe ser 6.0. Obtuve {r['pendiente_con']}"
    )
    assert all(type(v) is float for v in r.values()), "Los valores deben ser float de Python"
    # Series con índice desordenado: la máscara se aplica por POSICIÓN.
    xs = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=[7, 3, 9, 1, 5])
    ys = pd.Series([5.0, 10.0, 15.0, 0.0, 25.0], index=[7, 3, 9, 1, 5])
    r2 = ej.pendiente_con_y_sin_outliers(xs, ys, pd.Series([False, False, False, True, False]))
    assert r2["pendiente_sin"] == pytest.approx(0.5), (
        "Sin el 4.º punto (40, 0) los demás cumplen y = 0.5·x. ¿Filtraste por posición (np.asarray)?"
    )
    assert r2["pendiente_con"] == pytest.approx(0.3), f"Con todos los puntos la pendiente es 0.3. Obtuve {r2['pendiente_con']}"


@pytest.mark.puntos(10)
def test_pendiente_viviendas(viviendas):
    errores = viviendas["area_m2"] > 400  # los 12 errores de digitación (área ×10)
    assert errores.sum() == 12, "(control) deberían ser 12 viviendas con área > 400 m²"
    r = ej.pendiente_con_y_sin_outliers(viviendas["area_m2"], viviendas["precio_usd"], errores)
    assert r["pendiente_con"] == pytest.approx(381.51, abs=0.05), (
        f"Con los 12 errores, cada m² 'vale' solo ~381.51 USD. Obtuve {r['pendiente_con']}"
    )
    assert r["pendiente_sin"] == pytest.approx(1527.33, abs=0.05), (
        f"Sin los 12 errores, cada m² vale ~1527.33 USD (¡4 veces más!). Obtuve {r['pendiente_sin']}"
    )


# ---------------------------------------------------------------- aplicar_power_transformer (20)
@pytest.mark.puntos(12)
def test_power_transformer_mini():
    df = pd.DataFrame({
        "precio": [100.0, 200.0, 400.0, 800.0, 10000.0],
        "temp": [-5.0, 0.0, 3.0, 4.0, 30.0],
        "zona": list("ABCDE"),
    })
    copia = df.copy()
    r = ej.aplicar_power_transformer(df, ["precio"], "box-cox")
    pd.testing.assert_frame_equal(df, copia, obj="No modifiques el DataFrame de entrada (usa df.copy())")
    assert list(r.columns) == list(df.columns) and len(r) == 5, "Mismas filas y columnas que la entrada"
    assert r["precio"].mean() == pytest.approx(0.0, abs=1e-6), "Con standardize=True la media queda en ~0"
    assert r["precio"].std(ddof=0) == pytest.approx(1.0, abs=1e-6), "Con standardize=True la desviación queda en ~1"
    assert r["precio"].is_monotonic_increasing, "La transformación conserva el orden de los valores"
    assert r["temp"].tolist() == df["temp"].tolist() and r["zona"].tolist() == list("ABCDE"), (
        "Solo se transforman las columnas pedidas"
    )
    assert abs(r["precio"].skew()) < abs(df["precio"].skew()), "Box-Cox debe reducir la asimetría de 'precio'"
    yj = ej.aplicar_power_transformer(df, ["precio", "temp"])  # yeo-johnson por defecto: acepta negativos
    assert yj["temp"].mean() == pytest.approx(0.0, abs=1e-6), "Yeo-Johnson (por defecto) debe aceptar negativos y estandarizar"
    with pytest.raises(ValueError):
        ej.aplicar_power_transformer(df, ["temp"], "box-cox")  # tiene -5 y 0
    with pytest.raises(ValueError):
        ej.aplicar_power_transformer(df, ["precio"], "logaritmo")
    con_nan = pd.DataFrame({"x": [1.0, 2.0, np.nan, 8.0, 50.0]})
    rn = ej.aplicar_power_transformer(con_nan, ["x"])
    assert rn["x"].isna().tolist() == [False, False, True, False, False], "El NaN debe seguir en su sitio"


@pytest.mark.puntos(8)
def test_power_transformer_viviendas(viviendas):
    cols = ["precio_usd", "area_m2"]
    r = ej.aplicar_power_transformer(viviendas, cols, "box-cox")
    asim_antes = viviendas[cols].skew()
    asim_despues = r[cols].skew()
    assert asim_antes["precio_usd"] > 4 and abs(asim_despues["precio_usd"]) < 0.1, (
        f"precio_usd pasa de asimetría ~4.12 a ~0 con Box-Cox. Obtuve {asim_despues['precio_usd']:.2f}"
    )
    assert abs(asim_despues["area_m2"]) < 0.1, (
        f"area_m2 pasa de asimetría ~12.7 a ~0 con Box-Cox. Obtuve {asim_despues['area_m2']:.2f}"
    )
    assert r["precio_usd"].max() < 6, (
        "La mansión de 2.1 millones queda a menos de 6 desviaciones de la media (antes estaba a ~17)"
    )
    rn = ej.aplicar_power_transformer(viviendas, ["antiguedad_anios"])
    assert rn["antiguedad_anios"].isna().sum() == 170, "Los 170 NaN de antigüedad deben mantenerse"
    with pytest.raises(ValueError):
        ej.aplicar_power_transformer(viviendas, ["habitaciones"], "box-cox")  # las oficinas tienen 0


# ---------------------------------------------------------------- cramers_v (10, reto)
@pytest.mark.puntos(6)
def test_cramers_v_mini():
    perfecto = pd.DataFrame({"campania": [1, 1, 0, 0, 1, 0], "venta_alta": ["Si", "Si", "No", "No", "Si", "No"]})
    v = ej.cramers_v(perfecto, "campania", "venta_alta")
    assert type(v) is float, "Debe devolver un float de Python"
    assert v == pytest.approx(1.0), f"Campaña y venta alta van siempre juntas: V debe ser 1.0. Obtuve {v}"
    indep = pd.DataFrame({"a": ["x", "x", "y", "y"] * 3, "b": ["p", "q", "p", "q"] * 3})
    v0 = ej.cramers_v(indep, "a", "b")
    assert v0 == pytest.approx(0.0, abs=1e-9), f"Saber 'a' no dice nada de 'b': V debe ser 0.0. Obtuve {v0}"
    medio = pd.DataFrame({"a": ["x"] * 6 + ["y"] * 4, "b": ["p"] * 5 + ["q"] + ["p"] + ["q"] * 3})
    vm = ej.cramers_v(medio, "a", "b")
    assert vm == pytest.approx(0.5833, abs=1e-4), (
        "Tabla x:(p=5, q=1), y:(p=1, q=3): chi2 = 3.4028 sin corrección de Yates, n = 10, k = 1 -> "
        f"V = sqrt(3.4028/10) = 0.5833. Obtuve {vm} (¿usaste correction=False?)"
    )
    assert ej.cramers_v(medio, "b", "a") == vm, "La V de Cramér es simétrica: cramers_v(a, b) == cramers_v(b, a)"
    una_cat = pd.DataFrame({"a": ["x"] * 4, "b": ["p", "q", "p", "q"]})
    assert ej.cramers_v(una_cat, "a", "b") == 0.0, "Si una columna tiene una sola categoría (k = 0) devuelve 0.0"


@pytest.mark.puntos(4)
def test_cramers_v_viviendas(viviendas):
    df = viviendas.assign(sin_habitaciones=viviendas["habitaciones"] == 0)
    v1 = ej.cramers_v(df, "tipo_inmueble", "sin_habitaciones")
    assert v1 == pytest.approx(1.0), (
        f"Solo las oficinas tienen 0 habitaciones: tipo_inmueble determina 'sin_habitaciones' -> V = 1.0. Obtuve {v1}"
    )
    v2 = ej.cramers_v(df, "tipo_inmueble", "estado")
    assert v2 == pytest.approx(0.0376, abs=1e-4) and not math.isnan(v2), (
        f"tipo_inmueble y estado casi no se relacionan: V ≈ 0.0376. Obtuve {v2}"
    )
