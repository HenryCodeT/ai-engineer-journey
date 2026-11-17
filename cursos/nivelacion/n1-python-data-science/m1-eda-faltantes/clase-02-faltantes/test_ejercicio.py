"""Evaluador automático — Clase 02 · Estrategias para datos faltantes.  NO MODIFICAR.

Cada test tiene un peso (@pytest.mark.puntos). La suma es 100.
"""
import numpy as np
import pandas as pd
import pytest

from framework.testing import cargar_ejercicio, leer_dataset

ej = cargar_ejercicio(__file__)


@pytest.fixture
def mini():
    # edad    -> falta 1 de 4 en A y 1 de 4 en B (25 % y 25 %): parecido en ambos grupos ≈ MCAR
    # ingreso -> falta 0 de 4 en A y 3 de 4 en B (0 % y 75 %): depende de la zona ≈ MAR
    return pd.DataFrame({
        "zona": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "edad": [20.0, 30.0, np.nan, 50.0, 25.0, 35.0, 45.0, np.nan],
        "ingreso": [100.0, 200.0, 300.0, 400.0, np.nan, np.nan, np.nan, 800.0],
        "precio": [10, 20, 30, 40, 50, 60, 70, 80],
    })


@pytest.fixture(scope="module")
def viviendas():
    return leer_dataset("viviendas_lima.csv")


# ---------------------------------------------------------------- agregar_indicador_faltante (15)
@pytest.mark.puntos(8)
def test_indicador_mini(mini):
    copia = mini.copy()
    r = ej.agregar_indicador_faltante(mini, "edad")
    pd.testing.assert_frame_equal(mini, copia, obj="No modifiques el DataFrame de entrada (usa df.copy())")
    assert list(r.columns) == ["zona", "edad", "ingreso", "precio", "edad_falta"], (
        f"Esperaba las columnas originales + 'edad_falta' al final. Obtuve {list(r.columns)}"
    )
    assert r["edad_falta"].dtype == bool, f"'edad_falta' debe ser de tipo bool, es {r['edad_falta'].dtype}"
    assert r["edad_falta"].tolist() == [False, False, True, False, False, False, False, True], (
        "'edad_falta' debe ser True exactamente en las filas 2 y 7 (donde edad es NaN)"
    )


@pytest.mark.puntos(7)
def test_indicador_viviendas(viviendas):
    r = ej.agregar_indicador_faltante(viviendas, "piso")
    assert "piso_falta" in r.columns, "Falta la columna 'piso_falta'"
    assert len(r) == len(viviendas), "No debes agregar ni quitar filas"
    assert int(r["piso_falta"].sum()) == int(viviendas["piso"].isna().sum()) == 454, (
        f"En viviendas faltan 454 valores de 'piso'; tu indicador marca {int(r['piso_falta'].sum())}"
    )
    assert "piso_falta" not in viviendas.columns, "No modifiques el DataFrame original"


# ---------------------------------------------------------------- imputar_mediana (15)
@pytest.mark.puntos(8)
def test_imputar_mediana_mini(mini):
    copia = mini.copy()
    r = ej.imputar_mediana(mini, "edad")
    pd.testing.assert_frame_equal(mini, copia, obj="No modifiques el DataFrame de entrada (usa df.copy())")
    assert r["edad"].isna().sum() == 0, "Después de imputar, 'edad' no debe tener NaN"
    assert r.loc[2, "edad"] == pytest.approx(32.5) and r.loc[7, "edad"] == pytest.approx(32.5), (
        "La mediana de edad (sin NaN) es 32.5: los huecos de las filas 2 y 7 deben valer 32.5"
    )
    assert r.loc[0, "edad"] == 20.0, "Los valores que ya existían no deben cambiar"
    assert r["ingreso"].isna().sum() == 3, "Solo se imputa la columna pedida: 'ingreso' debe seguir con sus 3 NaN"


@pytest.mark.puntos(7)
def test_imputar_mediana_viviendas(viviendas):
    faltaban = viviendas["antiguedad_anios"].isna()
    r = ej.imputar_mediana(viviendas, "antiguedad_anios")
    assert r["antiguedad_anios"].isna().sum() == 0, "antiguedad_anios no debe quedar con NaN"
    assert (r.loc[faltaban, "antiguedad_anios"] == 10.0).all(), (
        "La mediana de antiguedad_anios en viviendas es 10.0: todos los huecos deben valer 10.0"
    )
    assert viviendas["antiguedad_anios"].isna().sum() == 170, "El DataFrame original debe conservar sus 170 NaN"
    assert r["distancia_centro_km"].isna().sum() == viviendas["distancia_centro_km"].isna().sum(), (
        "No toques otras columnas: distancia_centro_km debe seguir igual"
    )


# ---------------------------------------------------------------- tasa_faltantes_por_grupo (25)
@pytest.mark.puntos(13)
def test_tasa_por_grupo_mini(mini):
    copia = mini.copy()
    r = ej.tasa_faltantes_por_grupo(mini, "ingreso", "zona")
    pd.testing.assert_frame_equal(mini, copia, obj="No modifiques el DataFrame de entrada")
    assert isinstance(r, pd.Series), f"Debe devolver una Serie de pandas, devolvió {type(r).__name__}"
    assert list(r.index) == ["B", "A"], f"Orden esperado de mayor a menor: ['B', 'A']. Obtuve {list(r.index)}"
    assert r["B"] == pytest.approx(75.0), f"En B faltan 3 de 4 ingresos -> 75.0 %. Obtuve {r['B']}"
    assert r["A"] == pytest.approx(0.0), f"En A no falta ningún ingreso -> 0.0 %. Obtuve {r['A']}"
    r2 = ej.tasa_faltantes_por_grupo(mini, "edad", "zona")
    assert r2.to_dict() == {"A": pytest.approx(25.0), "B": pytest.approx(25.0)}, (
        f"Para edad, A y B tienen 25.0 % cada uno (en porcentaje 0-100, no proporción). Obtuve {r2.to_dict()}"
    )


@pytest.mark.puntos(12)
def test_tasa_por_grupo_viviendas(viviendas):
    piso = ej.tasa_faltantes_por_grupo(viviendas, "piso", "tipo_inmueble")
    assert piso.index[0] == "Casa" and piso["Casa"] == pytest.approx(100.0), (
        "A todas las casas les falta 'piso' (100 %): ¡una casa no está en un piso!"
    )
    assert piso["Departamento"] == 0 and piso["Oficina"] == 0, "A departamentos y oficinas no les falta 'piso'"
    dist = ej.tasa_faltantes_por_grupo(viviendas, "distancia_centro_km", "distrito")
    assert len(dist) == viviendas["distrito"].nunique(), "Debe haber una fila por distrito"
    assert dist.is_monotonic_decreasing, "La Serie debe estar ordenada de mayor a menor"
    assert set(dist.index[:4]) == {"SJL", "Ate", "Los Olivos", "Chorrillos"}, (
        f"Los 4 distritos con más faltantes de distancia deberían ser los periféricos. Obtuve {list(dist.index[:4])}"
    )
    assert (dist.round(2) == dist).all(), "Redondea los porcentajes a 2 decimales"


# ---------------------------------------------------------------- comparar_media_por_faltante (20)
@pytest.mark.puntos(10)
def test_comparar_media_mini(mini):
    copia = mini.copy()
    r = ej.comparar_media_por_faltante(mini, "edad", "precio")
    pd.testing.assert_frame_equal(mini, copia, obj="No modifiques el DataFrame de entrada")
    assert set(r) == {"media_con_faltante", "media_sin_faltante", "diferencia"}, (
        f"Claves esperadas: media_con_faltante, media_sin_faltante, diferencia. Obtuve {sorted(r)}"
    )
    assert r["media_con_faltante"] == pytest.approx(55.0), (
        f"Donde falta edad (filas 2 y 7) el precio es 30 y 80 -> media 55.0. Obtuve {r['media_con_faltante']}"
    )
    assert r["media_sin_faltante"] == pytest.approx(41.67), (
        f"Donde hay edad el precio medio es 41.67 (redondeado a 2). Obtuve {r['media_sin_faltante']}"
    )
    assert r["diferencia"] == pytest.approx(13.33, abs=0.011), (
        f"diferencia = con - sin = 55.0 - 41.67 ≈ 13.33. Obtuve {r['diferencia']}"
    )
    assert all(isinstance(v, float) for v in r.values()), "Los tres valores deben ser float de Python (usa float(...))"


@pytest.mark.puntos(10)
def test_comparar_media_viviendas(viviendas):
    r = ej.comparar_media_por_faltante(viviendas, "banos", "precio_usd")
    assert r["media_con_faltante"] > 2 * r["media_sin_faltante"], (
        "En viviendas, las filas sin dato de 'banos' son mucho más caras (más del doble): "
        f"obtuve con={r['media_con_faltante']} y sin={r['media_sin_faltante']}"
    )
    assert r["diferencia"] == pytest.approx(r["media_con_faltante"] - r["media_sin_faltante"], abs=0.011), (
        "diferencia debe ser media_con_faltante - media_sin_faltante (en ese orden)"
    )
    assert r["media_con_faltante"] == round(r["media_con_faltante"], 2), "Redondea a 2 decimales"
    mini_r = ej.comparar_media_por_faltante(
        pd.DataFrame({"a": [1.0, None, 3.0], "b": [10, 40, 20]}), "a", "b"
    )
    assert mini_r == {"media_con_faltante": 40.0, "media_sin_faltante": 15.0, "diferencia": 25.0}, (
        f"Con a=[1, NaN, 3] y b=[10, 40, 20] esperaba con=40, sin=15, diferencia=25. Obtuve {mini_r}"
    )


# ---------------------------------------------------------------- clasificar_mecanismo (25)
@pytest.mark.puntos(10)
def test_clasificar_mini(mini):
    assert ej.clasificar_mecanismo(mini, "ingreso", "zona") == "posible MAR", (
        "ingreso: A 0 % vs B 75 % -> rango 75 > 15 -> 'posible MAR'"
    )
    assert ej.clasificar_mecanismo(mini, "edad", "zona") == "MCAR", (
        "edad: A 25 % vs B 25 % -> rango 0 -> 'MCAR'"
    )
    assert ej.clasificar_mecanismo(mini, "precio", "zona") == "sin faltantes", (
        "precio no tiene NaN -> 'sin faltantes'"
    )
    assert ej.clasificar_mecanismo(mini, "ingreso", "zona", umbral=80) == "MCAR", (
        "Con umbral=80, un rango de 75 no lo supera -> 'MCAR' (¿usaste el parámetro umbral?)"
    )


@pytest.mark.puntos(15)
def test_clasificar_viviendas(viviendas):
    copia = viviendas.copy()
    casos = {
        ("piso", "tipo_inmueble"): "posible MAR",
        ("distancia_centro_km", "distrito"): "posible MAR",
        ("antiguedad_anios", "tipo_inmueble"): "MCAR",
        ("precio_usd", "distrito"): "sin faltantes",
    }
    for (col, grupo), esperado in casos.items():
        obtenido = ej.clasificar_mecanismo(viviendas, col, grupo)
        assert obtenido == esperado, f"{col} agrupado por {grupo}: esperaba '{esperado}' y obtuve '{obtenido}'"
    pd.testing.assert_frame_equal(viviendas, copia, obj="No modifiques el DataFrame de entrada")
