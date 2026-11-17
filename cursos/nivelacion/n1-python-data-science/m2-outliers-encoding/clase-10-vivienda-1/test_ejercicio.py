"""Evaluador automático — Clase 10 · Caso Panorama de la vivienda (1).  NO MODIFICAR.

Cada test tiene un peso (@pytest.mark.puntos). La suma es 100.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split

from framework.testing import cargar_ejercicio, leer_dataset

ej = cargar_ejercicio(__file__)

NUMERICAS_VIVIENDAS = [
    "area_m2", "area_construida_m2", "habitaciones", "banos",
    "antiguedad_anios", "piso", "distancia_centro_km",
]


@pytest.fixture
def mini():
    # 6 viviendas con texto, un id y el objetivo. Índice NO estándar.
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "distrito": ["Surco", "Lince", "Surco", "Miraflores", "Lince", "Surco"],
            "area_m2": [80.0, 60.0, np.nan, 120.0, 55.0, 90.0],
            "banos": [2.0, 1.0, 2.0, np.nan, 1.0, 2.0],
            "precio_usd": [150000.0, 90000.0, 160000.0, 300000.0, 85000.0, 170000.0],
        },
        index=[10, 11, 12, 13, 14, 15],
    )


@pytest.fixture
def grupo_y_lejano():
    # 20 viviendas "normales" alrededor de (50, 50) y una en (100, 100). Índice NO estándar.
    pts = [[50.3, 49.7], [51.3, 50.2], [48.9, 50.7], [52.6, 51.9], [48.6, 47.5], [48.8, 50.1],
           [45.3, 49.6], [47.5, 48.5], [48.9, 49.4], [50.8, 52.1], [49.7, 52.7], [48.7, 50.7],
           [51.8, 50.2], [48.5, 48.2], [49.1, 50.4], [48.0, 49.6], [49.7, 51.1], [50.4, 50.7],
           [48.7, 49.7], [51.6, 53.0], [100.0, 100.0]]
    X = pd.DataFrame(pts, columns=["a", "b"], index=range(100, 121))
    y = pd.Series(np.arange(21, dtype=float) * 1000, index=X.index, name="precio_usd")
    return X, y


@pytest.fixture(scope="module")
def viviendas():
    return leer_dataset("viviendas_lima.csv")


# ---------------------------------------------------------------- 1 · preparar_xy (15)
@pytest.mark.puntos(8)
def test_preparar_xy_mini(mini):
    copia = mini.copy()
    X, y = ej.preparar_xy(mini, "precio_usd", descartar=["id"])
    assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series), "Debe devolver (DataFrame, Serie)"
    assert list(X.columns) == ["area_m2", "banos"], (
        f"X debe quedarse solo con las numéricas sin objetivo ni 'id' (en el orden original); obtuve {list(X.columns)}"
    )
    assert y.name == "precio_usd" and y.tolist() == mini["precio_usd"].tolist(), "y debe ser la columna objetivo"
    assert list(X.index) == list(mini.index) and list(y.index) == list(mini.index), "Debe conservar el índice original"
    pd.testing.assert_frame_equal(mini, copia, obj="El DataFrame de entrada no debe modificarse")


@pytest.mark.puntos(4)
def test_preparar_xy_opciones(mini):
    X, _ = ej.preparar_xy(mini, "precio_usd")
    assert list(X.columns) == ["id", "area_m2", "banos"], (
        f"Sin `descartar`, 'id' (numérica) se queda en X; obtuve {list(X.columns)}"
    )
    X2, _ = ej.preparar_xy(mini, "precio_usd", solo_numericas=False, descartar=["id"])
    assert list(X2.columns) == ["distrito", "area_m2", "banos"], (
        f"Con solo_numericas=False se conservan las columnas de texto; obtuve {list(X2.columns)}"
    )


@pytest.mark.puntos(3)
def test_preparar_xy_viviendas(viviendas):
    X, y = ej.preparar_xy(viviendas, "precio_usd", descartar=["id"])
    assert list(X.columns) == NUMERICAS_VIVIENDAS, (
        f"En viviendas_lima.csv X debe tener las 7 numéricas {NUMERICAS_VIVIENDAS}; obtuve {list(X.columns)}"
    )
    assert len(y) == len(viviendas), "y debe tener una fila por vivienda"


# ---------------------------------------------------------------- 2 · dividir (10)
@pytest.mark.puntos(10)
def test_dividir():
    X = pd.DataFrame({"a": np.arange(20.0)}, index=range(50, 70))
    y = pd.Series(np.arange(20.0) * 10, index=X.index)
    X_train, X_test, y_train, y_test = ej.dividir(X, y)
    assert (len(X_train), len(X_test)) == (16, 4), (
        f"Con 20 filas y 80/20 se esperaban 16 de train y 4 de test; obtuve {len(X_train)} y {len(X_test)}"
    )
    assert list(X_train.index) == list(y_train.index), "X_train e y_train deben tener los mismos índices"
    assert list(X_test.index) == list(y_test.index), "X_test e y_test deben tener los mismos índices"
    ref = train_test_split(X, y, test_size=0.2, random_state=0)
    assert list(X_test.index) == list(ref[1].index), (
        "La división debe ser la de train_test_split con test_size=0.2 y random_state=0 (la de la clase)"
    )
    otra = ej.dividir(X, y)
    assert list(otra[1].index) == list(X_test.index), "Dos llamadas iguales deben dar la misma división"


# ---------------------------------------------------------------- 3 · score_dataset (15)
@pytest.mark.puntos(5)
def test_score_dataset_constante():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    y = pd.Series([7.0] * 6)
    mae = ej.score_dataset(X.iloc[:4], X.iloc[4:], y.iloc[:4], y.iloc[4:])
    assert isinstance(mae, float), f"Debe devolver un float de Python; obtuve {type(mae)}"
    assert mae == pytest.approx(0.0), f"Si y es constante el modelo acierta siempre: MAE esperado 0; obtuve {mae}"


@pytest.mark.puntos(10)
def test_score_dataset_viviendas(viviendas):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error

    num = viviendas.drop(columns=["id", "precio_usd"]).select_dtypes(exclude="object").fillna(0)
    y = viviendas["precio_usd"]
    Xtr, Xte, ytr, yte = train_test_split(num, y, test_size=0.2, random_state=0)
    mae = ej.score_dataset(Xtr, Xte, ytr, yte)
    modelo = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=0).fit(Xtr, ytr)
    esperado = mean_absolute_error(yte, modelo.predict(Xte))
    assert mae == pytest.approx(esperado, rel=1e-9), (
        f"El MAE no coincide con RandomForestRegressor(n_estimators=50, max_depth=6, random_state=0) "
        f"entrenado en train y medido en valid: esperado {esperado:.1f}, obtuve {mae}"
    )
    assert ej.score_dataset(Xtr, Xte, ytr, yte) == mae, "Debe ser determinista (random_state=0)"


# ---------------------------------------------------------------- 4 · imputar_train_test (25)
@pytest.mark.puntos(12)
def test_imputar_mediana_solo_del_train():
    tr = pd.DataFrame({"banos": [1.0, 2.0, 3.0, np.nan], "piso": [1.0, np.nan, 5.0, 3.0]}, index=[0, 1, 2, 3])
    te = pd.DataFrame({"banos": [np.nan, 9.0, 9.0], "piso": [100.0, np.nan, 200.0]}, index=[7, 8, 9])
    tr0, te0 = tr.copy(), te.copy()
    a, b = ej.imputar_train_test(tr, te, SimpleImputer(strategy="median"))
    assert isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame), "Debe devolver dos DataFrames"
    assert a["banos"].tolist() == [1.0, 2.0, 3.0, 2.0], "El NaN de train se rellena con la mediana del train (2.0)"
    assert b["banos"].tolist() == [2.0, 9.0, 9.0], (
        "El NaN de test debe rellenarse con la mediana del TRAIN (2.0), no con la del test (9.0): "
        "en test solo se usa transform"
    )
    assert b["piso"].tolist() == [100.0, 3.0, 200.0], (
        "piso: el NaN de test debe ser 3.0 (mediana de train 1, 5, 3), no 150 (mediana del test)"
    )
    pd.testing.assert_frame_equal(tr, tr0, obj="X_train de entrada no debe modificarse")
    pd.testing.assert_frame_equal(te, te0, obj="X_test de entrada no debe modificarse")


@pytest.mark.puntos(5)
def test_imputar_conserva_columnas_e_indices():
    tr = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0], "z": [10.0, 20.0, np.nan, 40.0]}, index=[40, 41, 42, 43])
    te = pd.DataFrame({"x": [np.nan, 2.0], "z": [30.0, np.nan]}, index=[90, 91])
    a, b = ej.imputar_train_test(tr, te, KNNImputer(n_neighbors=2))
    assert list(a.columns) == ["x", "z"] and list(b.columns) == ["x", "z"], "Deben conservarse los nombres de columnas"
    assert list(a.index) == [40, 41, 42, 43] and list(b.index) == [90, 91], (
        "Deben conservarse los índices originales de train y de test"
    )
    assert not a.isna().any().any() and not b.isna().any().any(), "No deben quedar NaN"


@pytest.mark.puntos(8)
def test_imputar_viviendas_imputador_aprende_del_train(viviendas):
    X = viviendas[NUMERICAS_VIVIENDAS]
    y = viviendas["precio_usd"]
    Xtr, Xte, _, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    imp = SimpleImputer(strategy="median")
    a, b = ej.imputar_train_test(Xtr, Xte, imp)
    assert hasattr(imp, "statistics_"), "El imputador que recibes debe quedar entrenado (fit dentro de la función)"
    np.testing.assert_allclose(
        imp.statistics_, Xtr.median().to_numpy(),
        err_msg="Las medianas aprendidas deben salir SOLO de X_train (fit_transform en train)",
    )
    assert a.shape == Xtr.shape and b.shape == Xte.shape, "Las formas de train y test no deben cambiar"
    assert not a.isna().any().any() and not b.isna().any().any(), "No deben quedar NaN tras imputar"


# ---------------------------------------------------------------- 5 · quitar_outliers_train (15)
@pytest.mark.puntos(8)
def test_quitar_outliers_mini(grupo_y_lejano):
    X, y = grupo_y_lejano
    X0, y0 = X.copy(), y.copy()
    Xf, yf = ej.quitar_outliers_train(X, y, contamination=0.05)
    assert 120 not in Xf.index, "La vivienda en (100, 100) (índice 120) debe eliminarse"
    assert len(Xf) == 20, f"Con contamination=0.05 sobre 21 filas solo se elimina 1; quedaron {len(Xf)}"
    assert list(Xf.index) == list(yf.index), "X e y filtrados deben tener exactamente los mismos índices"
    assert yf.tolist() == y.drop(120).tolist(), "y debe filtrarse con la MISMA máscara que X"
    pd.testing.assert_frame_equal(X, X0, obj="X_train de entrada no debe modificarse")
    pd.testing.assert_series_equal(y, y0, obj="y_train de entrada no debe modificarse")


@pytest.mark.puntos(3)
def test_quitar_outliers_maximo_10(grupo_y_lejano):
    X, y = grupo_y_lejano
    with pytest.raises(ValueError):
        ej.quitar_outliers_train(X, y, contamination=0.2)
    with pytest.raises(ValueError):
        ej.quitar_outliers_train(X, y, contamination=0)


@pytest.mark.puntos(4)
def test_quitar_outliers_viviendas(viviendas):
    X = viviendas[NUMERICAS_VIVIENDAS].fillna(viviendas[NUMERICAS_VIVIENDAS].median())
    y = viviendas["precio_usd"]
    Xf, yf = ej.quitar_outliers_train(X, y)
    frac = len(Xf) / len(X)
    assert 0.88 <= frac <= 0.92, f"Con contamination=0.1 debe conservarse ~90 % de las filas; conservé {frac:.1%}"
    assert list(Xf.index) == list(yf.index), "X e y filtrados deben quedar alineados"
    Xf2, _ = ej.quitar_outliers_train(X, y)
    assert list(Xf2.index) == list(Xf.index), "Debe ser determinista (random_state=0)"


# ---------------------------------------------------------------- 6 · evaluar_escenarios (20)
@pytest.mark.puntos(20)
def test_evaluar_escenarios(viviendas):
    copia = viviendas.copy()
    imp_basico = SimpleImputer(strategy="median")
    escenarios = {
        "basico_mediana": (imp_basico, False),
        "mediana_sin_outliers": (SimpleImputer(strategy="median"), True),
        "knn": (KNNImputer(n_neighbors=5), False),
    }
    res = ej.evaluar_escenarios(viviendas, escenarios, descartar=["id"])
    assert isinstance(res, pd.DataFrame), "Debe devolver un DataFrame"
    assert list(res.columns) == ["escenario", "mae", "filas_train"], (
        f"Columnas esperadas ['escenario', 'mae', 'filas_train']; obtuve {list(res.columns)}"
    )
    assert sorted(res["escenario"]) == sorted(escenarios), "Debe haber una fila por escenario"
    assert res["mae"].is_monotonic_increasing, "Debe estar ordenado por MAE de menor a mayor"
    assert list(res.index) == list(range(len(res))), "El índice debe ser 0, 1, 2, ..."

    n_train = int(round(len(viviendas) * 0.8))
    filas = dict(zip(res["escenario"], res["filas_train"]))
    assert filas["basico_mediana"] == n_train and filas["knn"] == n_train, (
        f"Sin quitar outliers se entrena con todo el train ({n_train} filas)"
    )
    assert 0.88 * n_train <= filas["mediana_sin_outliers"] < n_train, (
        "Al quitar outliers (10 %) el train debe perder ~10 % de filas (y el test NO se toca)"
    )

    # El escenario básico debe coincidir con hacer los pasos a mano.
    X, y = ej.preparar_xy(viviendas, "precio_usd", descartar=["id"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    med = Xtr.median()
    esperado = ej.score_dataset(Xtr.fillna(med), Xte.fillna(med), ytr, yte)
    maes = dict(zip(res["escenario"], res["mae"]))
    assert maes["basico_mediana"] == pytest.approx(esperado, rel=1e-9), (
        "El MAE del escenario básico no coincide con: preparar_xy -> dividir(80/20, rs=0) -> "
        "imputar con la mediana del train -> score_dataset"
    )
    assert 1_000 < maes["basico_mediana"] < 200_000, "El MAE en dólares debe estar en un rango razonable"

    # Determinismo y el imputador recibido no debe quedar "gastado".
    res2 = ej.evaluar_escenarios(viviendas, escenarios, descartar=["id"])
    pd.testing.assert_frame_equal(res, res2, obj="Dos llamadas iguales deben dar el mismo resultado")
    assert not hasattr(imp_basico, "statistics_"), (
        "Usa clone(imputador): el imputador que te pasan no debe quedar entrenado"
    )
    pd.testing.assert_frame_equal(viviendas, copia, obj="El DataFrame de entrada no debe modificarse")
