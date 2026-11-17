"""Evaluador automático — Clase 01 · EDA.  NO MODIFICAR.

Cada test tiene un peso (@pytest.mark.puntos). La suma es 100.
"""
import numpy as np
import pandas as pd
import pytest

from framework.testing import cargar_ejercicio, leer_dataset

ej = cargar_ejercicio(__file__)


@pytest.fixture
def mini():
    return pd.DataFrame({
        "id": [1, 2, 3, 3, 5],
        "zona": ["A", "B", "A", "A", None],
        "area": [50.0, np.nan, 80.0, 80.0, 120.0],
        "precio": [100, 200, 160, 160, 400],
    })


@pytest.fixture(scope="module")
def viviendas():
    return leer_dataset("viviendas_lima.csv")


# ---------------------------------------------------------------- resumen_general (15)
@pytest.mark.puntos(8)
def test_resumen_general_mini(mini):
    r = ej.resumen_general(mini)
    assert r == {"filas": 5, "columnas": 4, "duplicados": 1, "celdas_faltantes": 2}, (
        f"Esperaba filas=5, columnas=4, duplicados=1, celdas_faltantes=2 y obtuve {r}"
    )


@pytest.mark.puntos(7)
def test_resumen_general_viviendas(viviendas):
    r = ej.resumen_general(viviendas)
    assert r["filas"] == len(viviendas)
    assert r["duplicados"] == 15, "El dataset de viviendas tiene 15 filas duplicadas escondidas"
    assert all(isinstance(v, (int, np.integer)) for v in r.values()), "Todos los valores deben ser enteros"


# ---------------------------------------------------------------- clasificar_variables (20)
@pytest.mark.puntos(10)
def test_clasificar_variables_basico(mini):
    copia = mini.copy()
    r = ej.clasificar_variables(mini)
    pd.testing.assert_frame_equal(mini, copia, obj="No modifiques el DataFrame de entrada")
    assert r == {"numericas": ["id", "area", "precio"], "categoricas": ["zona"]}, f"Obtuve {r}"


@pytest.mark.puntos(10)
def test_clasificar_variables_excluir(viviendas):
    r = ej.clasificar_variables(viviendas, excluir=["id", "precio_usd"])
    assert "id" not in r["numericas"] and "precio_usd" not in r["numericas"], "`excluir` no se respetó"
    assert r["categoricas"] == ["distrito", "tipo_inmueble", "cochera", "estado", "pais"], (
        f"Categóricas incorrectas o en otro orden: {r['categoricas']}"
    )
    assert r["numericas"][:3] == ["area_m2", "area_construida_m2", "habitaciones"]
    assert len(r["numericas"]) == 7


# ---------------------------------------------------------------- resumen_numericas (20)
@pytest.mark.puntos(10)
def test_resumen_numericas_forma(mini):
    r = ej.resumen_numericas(mini)
    assert list(r.columns) == ["media", "mediana", "std", "min", "max", "asimetria"], (
        f"Columnas esperadas: media, mediana, std, min, max, asimetria. Obtuve {list(r.columns)}"
    )
    assert list(r.index) == ["id", "area", "precio"], "El índice deben ser las columnas numéricas en orden"


@pytest.mark.puntos(10)
def test_resumen_numericas_valores(mini):
    r = ej.resumen_numericas(mini)
    assert r.loc["area", "media"] == pytest.approx(82.5), "La media debe ignorar los NaN"
    assert r.loc["area", "mediana"] == pytest.approx(80.0)
    assert r.loc["precio", "max"] == 400
    assert r.loc["precio", "asimetria"] > 0, "precio tiene cola a la derecha → asimetría positiva"


# ---------------------------------------------------------------- conteo_categorias (15)
@pytest.mark.puntos(8)
def test_conteo_categorias_con_nan(mini):
    r = ej.conteo_categorias(mini, "zona")
    assert r.iloc[0] == 3 and r.index[0] == "A", "La categoría más frecuente debe ir primero"
    assert r.isna().sum() == 0 and r.index.isna().sum() == 1, "Los NaN deben contarse como una categoría"
    assert r.sum() == len(mini)


@pytest.mark.puntos(7)
def test_conteo_categorias_normalizado(viviendas):
    r = ej.conteo_categorias(viviendas, "tipo_inmueble", normalizar=True)
    assert r.sum() == pytest.approx(1.0), "Con normalizar=True las proporciones deben sumar 1"
    assert r.index[0] == "Departamento"
    assert r.is_monotonic_decreasing, "Debe estar ordenado de mayor a menor"


# ---------------------------------------------------------------- tabla_faltantes (15)
@pytest.mark.puntos(9)
def test_tabla_faltantes_mini():
    df = pd.DataFrame({"a": [1, None, None, 4], "b": [1, 2, 3, None], "c": [1, 2, 3, 4]})
    r = ej.tabla_faltantes(df)
    assert list(r.columns) == ["faltantes", "porcentaje"]
    assert list(r.index) == ["a", "b"], "Solo columnas con faltantes, ordenadas de mayor a menor"
    assert r.loc["a", "faltantes"] == 2 and r.loc["b", "porcentaje"] == pytest.approx(25.0)


@pytest.mark.puntos(6)
def test_tabla_faltantes_viviendas_y_vacia(viviendas):
    r = ej.tabla_faltantes(viviendas)
    assert r.index[0] == "piso", "En viviendas, `piso` es la columna con más faltantes (¿por qué será?)"
    assert r["faltantes"].is_monotonic_decreasing
    assert (r["porcentaje"] == r["porcentaje"].round(2)).all(), "Redondea el porcentaje a 2 decimales"
    vacia = ej.tabla_faltantes(pd.DataFrame({"x": [1, 2]}))
    assert vacia.empty and list(vacia.columns) == ["faltantes", "porcentaje"]


# ---------------------------------------------------------------- bivariante_categorica (15)
@pytest.mark.puntos(8)
def test_bivariante_mini(mini):
    r = ej.bivariante_categorica(mini, "zona", "precio")
    assert list(r.columns) == ["cantidad", "media_objetivo", "mediana_objetivo"]
    assert list(r.index) == ["B", "A"], "Ordena por media_objetivo descendente (B=200 > A=140) y excluye NaN"
    assert r.loc["A", "cantidad"] == 3
    assert r.loc["A", "media_objetivo"] == pytest.approx(140.0)
    assert r.loc["A", "mediana_objetivo"] == pytest.approx(160.0)


@pytest.mark.puntos(7)
def test_bivariante_viviendas(viviendas):
    r = ej.bivariante_categorica(viviendas, "distrito", "precio_usd")
    assert len(r) == viviendas["distrito"].nunique()
    assert r["media_objetivo"].is_monotonic_decreasing
    assert r["cantidad"].sum() == len(viviendas)
    assert r.index[-1] in {"SJL", "Ate", "Los Olivos"}, "Los distritos más baratos deberían quedar al final"
