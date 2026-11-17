"""Evaluador automático — Track SQL · Clase 01. NO MODIFICAR."""
import pandas as pd
import pytest

from framework.testing import cargar_ejercicio, conexion_sqlite, leer_dataset

ej = cargar_ejercicio(__file__)


@pytest.fixture(scope="module")
def con():
    return conexion_sqlite(viviendas="viviendas_lima.csv")


@pytest.fixture(scope="module")
def df():
    return leer_dataset("viviendas_lima.csv")


def correr(con, sql):
    assert isinstance(sql, str) and sql.strip(), "La función debe devolver un string con la consulta SQL"
    try:
        return pd.read_sql(sql, con)
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"La consulta no se pudo ejecutar: {e}") from e


@pytest.mark.puntos(20)
def test_conteo_por_distrito(con, df):
    r = correr(con, ej.consulta_conteo_por_distrito())
    assert list(r.columns) == ["distrito", "cantidad"], f"Columnas: {list(r.columns)}"
    esperado = (df.groupby("distrito").size().reset_index(name="cantidad")
                .sort_values(["cantidad", "distrito"], ascending=[False, True]).reset_index(drop=True))
    pd.testing.assert_frame_equal(r, esperado, check_dtype=False, obj="Resultado")


@pytest.mark.puntos(20)
def test_precio_promedio_por_tipo(con, df):
    r = correr(con, ej.consulta_precio_promedio_por_tipo())
    assert list(r.columns) == ["tipo_inmueble", "precio_promedio"], f"Columnas: {list(r.columns)}"
    esperado = df.groupby("tipo_inmueble")["precio_usd"].mean().round(0).sort_values(ascending=False)
    assert list(r["tipo_inmueble"]) == list(esperado.index), "Orden incorrecto"
    assert r["precio_promedio"].tolist() == pytest.approx(esperado.tolist(), abs=1)


@pytest.mark.puntos(20)
def test_faltantes(con, df):
    r = correr(con, ej.consulta_faltantes())
    assert list(r.columns) == ["faltantes_antiguedad", "faltantes_piso"], f"Columnas: {list(r.columns)}"
    assert len(r) == 1, "Debe devolver una sola fila"
    assert r.iloc[0, 0] == df["antiguedad_anios"].isna().sum()
    assert r.iloc[0, 1] == df["piso"].isna().sum()


@pytest.mark.puntos(20)
def test_top_departamentos(con, df):
    for n in (3, 7):
        r = correr(con, ej.consulta_top_departamentos_caros(n))
        assert list(r.columns) == ["id", "distrito", "area_m2", "precio_usd"], f"Columnas: {list(r.columns)}"
        esperado = (df[df["tipo_inmueble"] == "Departamento"]
                    .sort_values(["precio_usd", "id"], ascending=[False, True]).head(n))
        assert len(r) == n, f"Con n={n} esperaba {n} filas y obtuve {len(r)}: ¿usas el parámetro n?"
        assert r["id"].tolist() == esperado["id"].tolist()


@pytest.mark.puntos(20)
def test_distritos_caros(con, df):
    for umbral in (200000, 250000):
        r = correr(con, ej.consulta_distritos_caros(umbral))
        assert list(r.columns) == ["distrito", "precio_promedio", "cantidad"], f"Columnas: {list(r.columns)}"
        g = df[df["cochera"] == "Si"].groupby("distrito")["precio_usd"].agg(["mean", "size"])
        g = g[g["mean"] > umbral].sort_values("mean", ascending=False)
        assert r["distrito"].tolist() == g.index.tolist(), f"Con umbral={umbral} los distritos no coinciden"
        assert r["cantidad"].tolist() == g["size"].tolist()
