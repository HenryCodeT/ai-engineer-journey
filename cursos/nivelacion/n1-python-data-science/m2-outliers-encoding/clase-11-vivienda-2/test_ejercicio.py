"""Evaluador automático — Clase 11 · Caso Panorama de la vivienda, parte 2.  NO MODIFICAR.

Cada test tiene un peso (@pytest.mark.puntos). La suma es 100.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from framework.testing import cargar_ejercicio, leer_dataset

ej = cargar_ejercicio(__file__)

OBJETIVO = "precio_usd"
EXCLUIR = ["id"]
COLUMNAS_X = ["area_m2", "area_construida_m2", "habitaciones", "banos",
              "antiguedad_anios", "piso", "distancia_centro_km"]


@pytest.fixture
def mini():
    # 6 casas: una categórica, un id, NaN en los predictores y un objetivo completo.
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "distrito": ["A", "B", "A", "C", "B", "A"],
        "area": [50.0, np.nan, 70.0, 80.0, 60.0, 55.0],
        "banos": [1.0, 2.0, np.nan, 2.0, 1.0, 1.0],
        "precio": [100.0, 150.0, 170.0, 200.0, 130.0, 110.0],
    }, index=[10, 11, 12, 13, 14, 15])


@pytest.fixture
def nube():
    # 40 puntos agrupados alrededor de (0, 0) y 2 puntos muy lejanos (filas 100 y 101).
    rng = np.random.default_rng(0)
    datos = rng.normal(0, 1, size=(40, 2))
    datos = np.vstack([datos, [[25.0, 25.0], [-30.0, 20.0]]])
    return pd.DataFrame(datos, columns=["x1", "x2"], index=list(range(60, 100)) + [100, 101])


@pytest.fixture(scope="module")
def viviendas():
    return leer_dataset("viviendas_lima.csv")


# ---------------------------------------------------------------- separar_x_y (15)
@pytest.mark.puntos(9)
def test_separar_x_y_columnas_e_indice(mini):
    original = mini.copy()
    X, y = ej.separar_x_y(mini, "precio", excluir=["id"])
    assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series), "Debe devolver (DataFrame, Serie)"
    assert list(X.columns) == ["area", "banos"], (
        f"X debe tener solo las numéricas sin 'precio' ni 'id' (en orden): se esperaba ['area', 'banos'], "
        f"llegó {list(X.columns)}")
    assert y.tolist() == [100.0, 150.0, 170.0, 200.0, 130.0, 110.0], "y debe ser la columna 'precio'"
    assert list(X.index) == list(mini.index) and list(y.index) == list(mini.index), "X e y deben conservar el índice de df"
    assert X.isna().sum().sum() == 2, "separar_x_y NO imputa: los 2 NaN de X deben seguir ahí"
    pd.testing.assert_frame_equal(mini, original, obj="No debes modificar el df de entrada")
    X2, _ = ej.separar_x_y(mini, "precio")
    assert list(X2.columns) == ["id", "area", "banos"], "Con excluir=None solo se quita el objetivo (y las no numéricas)"


@pytest.mark.puntos(6)
def test_separar_x_y_valida_objetivo(mini):
    con_nulo = mini.copy()
    con_nulo.loc[12, "precio"] = np.nan
    with pytest.raises(ValueError):
        ej.separar_x_y(con_nulo, "precio")
    with pytest.raises(ValueError):
        ej.separar_x_y(mini, "no_existe")


# ---------------------------------------------------------------- imputar_knn_df (15)
@pytest.mark.puntos(9)
def test_imputar_knn_df_conserva_forma_e_indice():
    X = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [10.0, 20.0, 30.0]}, index=[7, 8, 9])
    original = X.copy()
    res = ej.imputar_knn_df(X, k=2)
    assert isinstance(res, pd.DataFrame), "Debe devolver un DataFrame (no un array de numpy)"
    assert res.shape == X.shape, f"La imputación no quita filas ni columnas: se esperaba {X.shape}, llegó {res.shape}"
    assert list(res.index) == [7, 8, 9], f"Debe conservar el índice de X [7, 8, 9], llegó {list(res.index)}"
    assert list(res.columns) == ["a", "b"], "Debe conservar las columnas de X"
    assert res.loc[9, "a"] == pytest.approx(1.5), (
        f"Con k=2, la fila 9 debe tomar el promedio de sus 2 vecinos (1.0 y 2.0) = 1.5, llegó {res.loc[9, 'a']}")
    assert res.loc[7, "a"] == 1.0 and res.loc[8, "b"] == 20.0, "Los valores que no eran NaN no deben cambiar"
    pd.testing.assert_frame_equal(X, original, obj="No debes modificar X")


@pytest.mark.puntos(6)
def test_imputar_knn_df_dataset_real(viviendas):
    X = viviendas[COLUMNAS_X]
    res = ej.imputar_knn_df(X)
    assert res.shape == X.shape, f"Se esperaba {X.shape}, llegó {res.shape}"
    assert res.isna().sum().sum() == 0, "No debe quedar ningún NaN tras imputar"
    assert res.index.equals(X.index), "Debe conservar el índice"
    assert X.isna().sum().sum() > 0, "No debes modificar X (sus NaN originales deben seguir)"


# ---------------------------------------------------------------- mascara_inliers (20)
@pytest.mark.puntos(4)  # x2 métodos = 8
@pytest.mark.parametrize("metodo", ["isolation", "lof"])
def test_mascara_inliers_marca_los_lejanos(nube, metodo):
    m = ej.mascara_inliers(nube, metodo=metodo, contamination=0.05, random_state=0)
    assert isinstance(m, pd.Series), "Debe devolver una Serie"
    assert m.dtype == bool, f"La Serie debe ser booleana (True = inlier), llegó dtype {m.dtype}"
    assert m.index.equals(nube.index), "La máscara debe tener el mismo índice que X"
    assert not m.loc[100] and not m.loc[101], (
        f"[{metodo}] Las filas 100 y 101 están lejísimos de la nube: deben ser False (outliers)")
    assert m.sum() >= 38, f"[{metodo}] Con contamination=0.05 casi todas deben ser inliers; llegaron {int(m.sum())}"


@pytest.mark.puntos(6)
def test_mascara_inliers_dataset_real_y_determinismo(viviendas):
    X = viviendas[COLUMNAS_X].fillna(viviendas[COLUMNAS_X].median())
    m1 = ej.mascara_inliers(X, metodo="isolation", contamination=0.1, random_state=0)
    m2 = ej.mascara_inliers(X, metodo="isolation", contamination=0.1, random_state=0)
    assert m1.equals(m2), "Con la misma semilla el resultado debe ser idéntico"
    proporcion = 1 - m1.mean()
    assert 0.08 <= proporcion <= 0.12, f"Con contamination=0.1 se espera ~10 % de outliers, llegó {proporcion:.1%}"
    assert m1.name == "inlier", f"La Serie debe llamarse 'inlier', se llama {m1.name!r}"


@pytest.mark.puntos(6)
def test_mascara_inliers_errores(nube):
    con_nulo = nube.copy()
    con_nulo.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        ej.mascara_inliers(con_nulo)
    with pytest.raises(ValueError):
        ej.mascara_inliers(nube, metodo="abod")


# ---------------------------------------------------------------- filtrar_xy (10)
@pytest.mark.puntos(6)
def test_filtrar_xy_misma_mascara():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}, index=[10, 11, 12, 13])
    y = pd.Series([100, 200, 300, 400], index=[10, 11, 12, 13], name="precio")
    m = pd.Series([True, False, True, False], index=[10, 11, 12, 13])
    Xf, yf = ej.filtrar_xy(X, y, m)
    assert list(Xf.index) == [10, 12] and list(yf.index) == [10, 12], (
        "Debe quedarse con las filas True y CONSERVAR sus índices originales (sin reset_index)")
    assert yf.tolist() == [100, 300], "y debe filtrarse con la MISMA máscara que X"
    assert list(Xf.columns) == ["a", "b"], "Filtrar quita filas, no columnas"
    assert len(X) == 4 and len(y) == 4, "No debes modificar X ni y"


@pytest.mark.puntos(4)
def test_filtrar_xy_indices_desalineados():
    X = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
    y = pd.Series([10, 20, 30], index=[0, 1, 5])
    m = pd.Series([True, True, False], index=[0, 1, 2])
    with pytest.raises(ValueError):
        ej.filtrar_xy(X, y, m)
    y_ok = pd.Series([10, 20, 30], index=[0, 1, 2])
    m_mal = pd.Series([True, True, False], index=[2, 1, 0])
    with pytest.raises(ValueError):
        ej.filtrar_xy(X, y_ok, m_mal)


# ---------------------------------------------------------------- limpiar_escenario_avanzado (15)
@pytest.mark.puntos(15)
def test_limpiar_escenario_avanzado_real(viviendas):
    original = viviendas.copy()
    for metodo in ("isolation", "lof"):
        _revisar_escenario(viviendas, metodo)
    pd.testing.assert_frame_equal(viviendas, original, obj="No debes modificar el df de entrada")


def _revisar_escenario(viviendas, metodo):
    X_r, y_r = ej.limpiar_escenario_avanzado(viviendas, OBJETIVO, excluir=EXCLUIR, metodo=metodo,
                                             contamination=0.1, random_state=0)
    assert list(X_r.columns) == COLUMNAS_X, f"[{metodo}] Columnas esperadas {COLUMNAS_X}, llegaron {list(X_r.columns)}"
    assert X_r.isna().sum().sum() == 0, "X_r debe estar imputado (sin NaN)"
    assert X_r.index.equals(y_r.index), "X_r e y_r deben tener exactamente el mismo índice"
    frac = len(X_r) / len(viviendas)
    assert 0.88 <= frac <= 0.92, f"[{metodo}] Con contamination=0.1 debe quedar ~90 % de filas, quedó {frac:.1%}"
    assert (y_r == viviendas.loc[y_r.index, OBJETIVO]).all(), f"[{metodo}] Cada precio debe seguir con SU vivienda"


# ---------------------------------------------------------------- comparar_antes_vs_despues_split (25)
def _split(df):
    X = df[COLUMNAS_X]
    y = df[OBJETIVO]
    return train_test_split(X, y, test_size=0.2, random_state=0)


@pytest.mark.puntos(8)
def test_comparar_estructura(viviendas):
    r = ej.comparar_antes_vs_despues_split(viviendas, OBJETIVO, excluir=EXCLUIR)
    claves = {"mae_antes", "mae_despues", "diferencia", "medianas_antes", "medianas_despues"}
    assert isinstance(r, dict) and set(r) == claves, f"Claves esperadas {sorted(claves)}, llegaron {sorted(r)}"
    for c in ("mae_antes", "mae_despues", "diferencia"):
        assert isinstance(r[c], float), f"'{c}' debe ser float de Python, llegó {type(r[c]).__name__}"
    assert 10_000 < r["mae_antes"] < 200_000 and 10_000 < r["mae_despues"] < 200_000, (
        f"MAE fuera de rango razonable para precios de viviendas: {r['mae_antes']}, {r['mae_despues']}")
    assert r["diferencia"] == pytest.approx(r["mae_despues"] - r["mae_antes"], abs=0.02), (
        "diferencia debe ser mae_despues - mae_antes")


@pytest.mark.puntos(10)
def test_comparar_despues_no_mira_el_test(viviendas):
    r = ej.comparar_antes_vs_despues_split(viviendas, OBJETIVO, excluir=EXCLUIR)
    X_train, _, _, _ = _split(viviendas)
    esperado_despues = X_train.median()
    esperado_antes = viviendas[COLUMNAS_X].median()
    md, ma = r["medianas_despues"], r["medianas_antes"]
    assert isinstance(md, pd.Series) and list(md.index) == COLUMNAS_X, "medianas_despues: Serie con índice = columnas de X"
    assert np.allclose(md.values, esperado_despues.values), (
        "La versión 'después' debe aprender las medianas SOLO de X_train (mismo split 80/20, random_state=0). "
        f"Esperado {esperado_despues.round(2).to_dict()}, llegó {md.round(2).to_dict()}")
    assert np.allclose(ma.values, esperado_antes.values), (
        "La versión 'antes' debe aprender las medianas de TODO X (incluye el test). "
        f"Esperado {esperado_antes.round(2).to_dict()}, llegó {ma.round(2).to_dict()}")


@pytest.mark.puntos(7)
def test_comparar_determinista_y_mismo_modelo(viviendas):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error

    r1 = ej.comparar_antes_vs_despues_split(viviendas, OBJETIVO, excluir=EXCLUIR)
    r2 = ej.comparar_antes_vs_despues_split(viviendas, OBJETIVO, excluir=EXCLUIR)
    assert r1["mae_antes"] == r2["mae_antes"] and r1["mae_despues"] == r2["mae_despues"], (
        "Con las mismas semillas el resultado debe ser idéntico")

    X_train, X_test, y_train, y_test = _split(viviendas)
    med = X_train.median()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=0)
        modelo.fit(X_train.fillna(med).values, y_train)
        esperado = mean_absolute_error(y_test, modelo.predict(X_test.fillna(med).values))
    assert r1["mae_despues"] == pytest.approx(esperado, abs=1.0), (
        "mae_despues debe salir de RandomForestRegressor(n_estimators=50, max_depth=6, random_state=0) "
        f"con el imputador ajustado en train: se esperaba ~{esperado:.2f}, llegó {r1['mae_despues']}")
