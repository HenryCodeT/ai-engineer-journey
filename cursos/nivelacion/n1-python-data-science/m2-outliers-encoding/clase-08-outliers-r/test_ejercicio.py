"""Evaluador automático — Clase 08 · Outliers multivariados (PyOD -> scikit-learn).  NO MODIFICAR.

Cada test tiene un peso (@pytest.mark.puntos). La suma es 100.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from framework.testing import cargar_ejercicio, leer_dataset

ej = cargar_ejercicio(__file__)

COLS = ["area_m2", "area_construida_m2", "habitaciones", "banos", "antiguedad_anios", "distancia_centro_km"]


@pytest.fixture
def recta():
    # 5 puntos en una sola dimensión: 0, 1, 2, 3 juntos y el 10 aislado.
    return pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 10.0]})


@pytest.fixture
def cuadrado():
    # Las 4 esquinas de un cuadrado de lado 1 y un punto lejano en (5, 5).
    return pd.DataFrame({"a": [0.0, 0.0, 1.0, 1.0, 5.0], "b": [0.0, 1.0, 0.0, 1.0, 5.0]})


@pytest.fixture
def grupo_y_lejano():
    # 20 viviendas "normales" alrededor de (50, 50) y una en (100, 100). Índice NO estándar.
    pts = [[50.3, 49.7], [51.3, 50.2], [48.9, 50.7], [52.6, 51.9], [48.6, 47.5], [48.8, 50.1],
           [45.3, 49.6], [47.5, 48.5], [48.9, 49.4], [50.8, 52.1], [49.7, 52.7], [48.7, 50.7],
           [51.8, 50.2], [48.5, 48.2], [49.1, 50.4], [48.0, 49.6], [49.7, 51.1], [50.4, 50.7],
           [48.7, 49.7], [51.6, 53.0], [100.0, 100.0]]
    return pd.DataFrame(pts, columns=["a", "b"], index=range(100, 121))


@pytest.fixture(scope="module")
def viviendas():
    return leer_dataset("viviendas_lima.csv")


@pytest.fixture(scope="module")
def viviendas_completas(viviendas):
    # Solo filas sin NaN en las 6 columnas numéricas (precio_usd es el objetivo: se excluye).
    return viviendas.dropna(subset=COLS)


# ---------------------------------------------------------------- puntaje_knn (25)
@pytest.mark.puntos(12)
def test_puntaje_knn_tres_metodos(recta):
    copia = recta.copy()
    r = ej.puntaje_knn(recta, k=3, metodo="largest")
    pd.testing.assert_frame_equal(recta, copia, obj="No modifiques el DataFrame de entrada")
    assert isinstance(r, np.ndarray), f"Debe devolver un np.ndarray, devolvió {type(r).__name__}"
    assert r.shape == (5,), f"Debe haber un puntaje por fila (5), obtuve forma {r.shape}"
    assert np.allclose(r, [3, 2, 2, 3, 9]), (
        "metodo='largest' con k=3 sobre [0, 1, 2, 3, 10]: distancia al 3er vecino SIN contarse a sí mismo. "
        f"Esperaba [3, 2, 2, 3, 9], obtuve {np.round(r, 3).tolist()}. "
        "Si te sale [2, 1, 1, 2, 8], estás contando al propio punto como vecino 0."
    )
    m = ej.puntaje_knn(recta, k=3, metodo="mean")
    assert np.allclose(m, [2, 4 / 3, 4 / 3, 2, 8]), (
        f"metodo='mean': promedio de las 3 distancias. Esperaba [2, 1.333, 1.333, 2, 8], obtuve {np.round(m, 3).tolist()}"
    )
    md = ej.puntaje_knn(recta, k=3, metodo="median")
    assert np.allclose(md, [2, 1, 1, 2, 8]), (
        f"metodo='median': mediana de las 3 distancias. Esperaba [2, 1, 1, 2, 8], obtuve {np.round(md, 3).tolist()}"
    )
    with pytest.raises(ValueError):
        ej.puntaje_knn(recta, k=3, metodo="maximo")


@pytest.mark.puntos(5)
def test_puntaje_knn_dos_dimensiones(cuadrado):
    r = ej.puntaje_knn(cuadrado, k=1)
    esperado = [1, 1, 1, 1, np.sqrt(32)]
    assert np.allclose(r, esperado), (
        "Con k=1 (metodo por defecto 'largest'), cada esquina del cuadrado tiene su vecino a distancia 1 y "
        f"(5, 5) está a √32 ≈ 5.657 de (1, 1) (distancia euclidiana en 2D). Esperaba {np.round(esperado, 3).tolist()}, "
        f"obtuve {np.round(r, 3).tolist()}"
    )
    r_arr = ej.puntaje_knn(cuadrado.to_numpy(), k=1)
    assert np.allclose(r_arr, esperado), "También debe funcionar si X es un array de numpy (usa np.asarray)"


@pytest.mark.puntos(8)
def test_puntaje_knn_viviendas(viviendas_completas):
    r = ej.puntaje_knn(viviendas_completas[COLS], k=5)
    assert len(r) == len(viviendas_completas) == 1622, f"Esperaba 1622 puntajes, obtuve {len(r)}"
    pos_max = int(np.argmax(r))
    idx_max = viviendas_completas.index[pos_max]
    assert idx_max == 1902, (
        f"La vivienda más aislada debería ser la de índice 1902 (area_m2 = 2746: error ×10), obtuve índice {idx_max}"
    )
    assert r[pos_max] == pytest.approx(1804.3086, abs=1e-3), (
        f"Su puntaje (distancia al 5º vecino) debería ser ≈ 1804.309, obtuve {r[pos_max]:.3f}"
    )


# ---------------------------------------------------------------- marcar_por_contaminacion (15)
@pytest.mark.puntos(10)
def test_marcar_por_contaminacion_mini():
    r = ej.marcar_por_contaminacion([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.2)
    assert isinstance(r, np.ndarray), f"Debe devolver un np.ndarray, devolvió {type(r).__name__}"
    assert r.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1], (
        "contamination=0.2 sobre 1..10: el percentil 80 es 8.2, solo 9 y 10 lo superan. "
        f"Esperaba [0,0,0,0,0,0,0,0,1,1], obtuve {r.tolist()}"
    )
    assert np.issubdtype(r.dtype, np.integer), f"Las etiquetas deben ser enteros 0/1, no {r.dtype}"
    desordenado = ej.marcar_por_contaminacion(pd.Series([5.0, 1.0, 9.0, 3.0, 7.0]), 0.1)
    assert desordenado.tolist() == [0, 0, 1, 0, 0], (
        "Con [5, 1, 9, 3, 7] y contamination=0.1 el percentil 90 es 8.2: solo el 9 (posición 2) es outlier. "
        f"Respeta el orden original. Obtuve {desordenado.tolist()}"
    )


@pytest.mark.puntos(5)
def test_marcar_por_contaminacion_viviendas(viviendas_completas):
    puntajes = ej.puntaje_knn(viviendas_completas[COLS], k=5)
    etiquetas = ej.marcar_por_contaminacion(puntajes, 0.2)
    assert int(etiquetas.sum()) == 325, (
        f"kNN (k=5) + contamination=0.2 sobre 1622 viviendas debería marcar 325 outliers (~20 %), obtuve {int(etiquetas.sum())}"
    )
    assert etiquetas[int(np.argmax(puntajes))] == 1, "La vivienda con mayor puntaje debe quedar marcada como outlier (1)"


# ---------------------------------------------------------------- convertir_etiquetas_sklearn (10)
@pytest.mark.puntos(10)
def test_convertir_etiquetas_sklearn():
    r = ej.convertir_etiquetas_sklearn([1, -1, 1, 1, -1])
    assert isinstance(r, np.ndarray), f"Debe devolver un np.ndarray, devolvió {type(r).__name__}"
    assert r.tolist() == [0, 1, 0, 0, 1], (
        f"scikit-learn: 1 = inlier, -1 = outlier. PyOD: 0 = inlier, 1 = outlier. Esperaba [0, 1, 0, 0, 1], obtuve {r.tolist()}"
    )
    todos_in = ej.convertir_etiquetas_sklearn(np.array([1, 1, 1]))
    assert todos_in.tolist() == [0, 0, 0], f"Si todos son inliers (1) debe devolver ceros, obtuve {todos_in.tolist()}"


# ---------------------------------------------------------------- detectar_isolation_forest (20)
@pytest.mark.puntos(12)
def test_detectar_isolation_forest_mini(grupo_y_lejano):
    copia = grupo_y_lejano.copy()
    r = ej.detectar_isolation_forest(grupo_y_lejano, ["a", "b"], contamination=0.05, random_state=42)
    pd.testing.assert_frame_equal(grupo_y_lejano, copia, obj="No modifiques el DataFrame de entrada")
    assert isinstance(r, pd.Series), f"Debe devolver una pd.Series, devolvió {type(r).__name__}"
    assert r.name == "outlier", f"La Serie debe llamarse 'outlier', se llama {r.name!r}"
    assert r.index.equals(grupo_y_lejano.index), "La Serie debe tener el MISMO índice que df (100..120)"
    assert set(r.unique()) <= {0, 1}, f"Las etiquetas deben ser 0/1 (convención PyOD), obtuve {sorted(r.unique())}"
    assert r.loc[120] == 1 and r.sum() == 1, (
        "Con contamination=0.05 solo el punto (100, 100) (índice 120) debería ser outlier. "
        f"Outliers obtenidos en índices: {r[r == 1].index.tolist()}"
    )
    con_nan = grupo_y_lejano.copy()
    con_nan.loc[105, "a"] = np.nan
    r2 = ej.detectar_isolation_forest(con_nan, ["a", "b"], contamination=0.05, random_state=42)
    assert r2.loc[120] == 1, "Con un NaN en el df debe seguir funcionando (rellena con la mediana de la columna)"


@pytest.mark.puntos(8)
def test_detectar_isolation_forest_viviendas(viviendas):
    r = ej.detectar_isolation_forest(viviendas, COLS, contamination=0.1, random_state=42)
    X = viviendas[COLS].fillna(viviendas[COLS].median())
    ref = IsolationForest(contamination=0.1, random_state=42).fit_predict(X)
    esperado = np.where(ref == -1, 1, 0)
    assert len(r) == len(viviendas), f"Debe haber una etiqueta por vivienda ({len(viviendas)}), obtuve {len(r)}"
    assert np.array_equal(np.asarray(r), esperado), (
        "Las etiquetas no coinciden con IsolationForest(contamination=0.1, random_state=42) sobre las columnas "
        f"con NaN rellenados por la mediana. Esperaba {int(esperado.sum())} outliers, obtuve {int(np.asarray(r).sum())}"
    )


# ---------------------------------------------------------------- comparar_detectores (15)
@pytest.mark.puntos(10)
def test_comparar_detectores_mini():
    r = ej.comparar_detectores([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    assert r == {"outliers_a": 3, "outliers_b": 3, "coinciden": 2, "jaccard": 0.5}, (
        "A marca filas 0,1,4; B marca 0,3,4 -> coinciden 2 (filas 0 y 4), unión 4 -> Jaccard 0.5. "
        f"Obtuve {r}"
    )
    assert all(type(r[c]) is int for c in ["outliers_a", "outliers_b", "coinciden"]), "Los conteos deben ser int de Python"
    assert type(r["jaccard"]) is float, "jaccard debe ser float de Python"
    r2 = ej.comparar_detectores(np.array([1, 0, 0]), np.array([0, 1, 1]))
    assert r2 == {"outliers_a": 1, "outliers_b": 2, "coinciden": 0, "jaccard": 0.0}, (
        f"Sin filas en común el Jaccard es 0.0. Obtuve {r2}"
    )
    r3 = ej.comparar_detectores([0, 0, 0], [0, 0, 0])
    assert r3["jaccard"] == 1.0, f"Si ninguno marca nada (unión = 0) el Jaccard debe ser 1.0, obtuve {r3['jaccard']}"
    r4 = ej.comparar_detectores([1, 1, 0], [1, 0, 0])
    assert r4["jaccard"] == 0.5, f"A = {{0, 1}}, B = {{0}} -> 1/2 = 0.5, obtuve {r4['jaccard']}"
    r5 = ej.comparar_detectores([1, 1, 1, 0], [1, 0, 0, 0])
    assert r5["jaccard"] == 0.3333, f"1/3 redondeado a 4 decimales es 0.3333, obtuve {r5['jaccard']}"


@pytest.mark.puntos(5)
def test_comparar_detectores_viviendas(viviendas_completas):
    knn = ej.marcar_por_contaminacion(ej.puntaje_knn(viviendas_completas[COLS], k=5), 0.2)
    iso = ej.detectar_isolation_forest(viviendas_completas, COLS, contamination=0.2, random_state=42)
    r = ej.comparar_detectores(knn, iso)
    assert r == {"outliers_a": 325, "outliers_b": 325, "coinciden": 244, "jaccard": 0.601}, (
        "kNN e IsolationForest con contamination=0.2 marcan 325 viviendas cada uno, pero solo coinciden en 244 "
        f"(Jaccard 0.601): 'cada algoritmo tiene un comportamiento diferente'. Obtuve {r}"
    )


# ---------------------------------------------------------------- filtrar_inliers (15)
@pytest.mark.puntos(10)
def test_filtrar_inliers_mini():
    df = pd.DataFrame({"area_m2": [60, 75, 2746, 90], "distrito": ["Surco", "Lince", "Surco", "Breña"]},
                      index=[10, 11, 12, 13])
    copia = df.copy()
    limpio, pct = ej.filtrar_inliers(df, [0, 0, 1, 0])
    pd.testing.assert_frame_equal(df, copia, obj="No modifiques el DataFrame de entrada")
    assert isinstance(limpio, pd.DataFrame), "El primer elemento debe ser un DataFrame"
    assert limpio.index.tolist() == [10, 11, 13], (
        f"Deben quedar las filas de etiqueta 0 con su índice ORIGINAL [10, 11, 13], obtuve {limpio.index.tolist()}"
    )
    assert list(limpio.columns) == ["area_m2", "distrito"], "No elimines columnas"
    assert pct == 75.0 and isinstance(pct, float), f"3 de 4 filas conservadas -> 75.0 (float), obtuve {pct!r}"
    limpio.loc[10, "area_m2"] = -1
    assert df.loc[10, "area_m2"] == 60, "df_limpio debe ser una COPIA: cambiarlo no debe tocar el original"
    _, pct2 = ej.filtrar_inliers(df.iloc[:3], pd.Series([0, 1, 1], index=["x", "y", "z"]))
    assert pct2 == 33.33, f"1 de 3 -> 33.33 (redondeado a 2 decimales; etiquetas por POSICIÓN), obtuve {pct2!r}"


@pytest.mark.puntos(5)
def test_filtrar_inliers_viviendas(viviendas_completas):
    etiquetas = ej.marcar_por_contaminacion(ej.puntaje_knn(viviendas_completas[COLS], k=5), 0.2)
    limpio, pct = ej.filtrar_inliers(viviendas_completas, etiquetas)
    assert limpio.shape == (1297, 14), f"De 1622 viviendas deben quedar 1297 (14 columnas), obtuve {limpio.shape}"
    assert pct == 79.96, f"1297 / 1622 -> 79.96 % conservado (el profe: ~80 %), obtuve {pct}"
    assert 1902 not in limpio.index, "La vivienda de 2746 m² (índice 1902) debería haber sido eliminada"
