"""Evaluador automático — Clase 07 · Tratamiento de outliers.  NO MODIFICAR.

Cada test tiene un peso (@pytest.mark.puntos). La suma es 100.
"""
import numpy as np
import pandas as pd
import pytest

from framework.testing import cargar_ejercicio, leer_dataset

ej = cargar_ejercicio(__file__)


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def mini_area():
    # Índice "raro" a propósito: el resultado debe conservarlo.
    return pd.DataFrame(
        {"area_m2": [60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 1200.0, np.nan]},
        index=[10, 11, 12, 13, 14, 15, 16, 17, 18],
    )


@pytest.fixture
def altura_peso():
    # 14 personas donde altura y peso suben juntos + 1 persona alta pero muy liviana.
    # 182 cm y 52 kg están DENTRO del rango de cada columna, pero la combinación es rara.
    return pd.DataFrame(
        {
            "altura_cm": [150, 155, 158, 160, 163, 165, 168, 170, 173, 175, 178, 180, 183, 185, 182],
            "peso_kg": [50, 53, 55, 57, 60, 62, 64, 66, 69, 71, 74, 76, 79, 82, 52],
        },
        index=[f"p{i:02d}" for i in range(1, 16)],
    )


@pytest.fixture(scope="module")
def viviendas():
    return leer_dataset("viviendas_lima.csv")


COLS_REALES = ["area_m2", "precio_usd", "antiguedad_anios", "distancia_centro_km"]


# ------------------------------------------------------------------ limites_iqr (15)
@pytest.mark.puntos(6)
def test_limites_iqr_mini():
    r = ej.limites_iqr(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 100]))
    assert isinstance(r, tuple) and len(r) == 2, "Debe devolver una tupla (inferior, superior)"
    assert r == pytest.approx((-3.0, 13.0)), (
        f"Con Q1=3, Q3=7, IQR=4 se esperaba (3 - 1.5*4, 7 + 1.5*4) = (-3.0, 13.0); obtuviste {r}"
    )


@pytest.mark.puntos(4)
def test_limites_iqr_k_y_nan():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 100, np.nan, np.nan])
    r = ej.limites_iqr(s, k=3)
    assert r == pytest.approx((-9.0, 19.0)), (
        f"Con k=3 se esperaba (3 - 3*4, 7 + 3*4) = (-9.0, 19.0), ignorando los NaN; obtuviste {r}"
    )


@pytest.mark.puntos(5)
def test_limites_iqr_dataset_real(viviendas):
    r = ej.limites_iqr(viviendas["area_m2"])
    assert r == pytest.approx((-6.2, 205.4), abs=1e-6), (
        f"En viviendas_lima.csv, area_m2 tiene Q1=73.15 y Q3=126.05 -> límites (-6.2, 205.4); obtuviste {r}"
    )


# ------------------------------------------------------------------ marcar_outliers_iqr (15)
@pytest.mark.puntos(9)
def test_marcar_outliers_iqr_mini(mini_area):
    copia = mini_area.copy()
    r = ej.marcar_outliers_iqr(mini_area, "area_m2")
    pd.testing.assert_frame_equal(mini_area, copia, obj="No modifiques el DataFrame de entrada")
    assert isinstance(r, pd.Series), "Debe devolver una pd.Series"
    assert r.dtype == bool, f"La Serie debe ser booleana (dtype bool); es {r.dtype}"
    assert list(r.index) == list(mini_area.index), "La Serie debe tener el MISMO índice que el DataFrame"
    esperado = [False] * 7 + [True, False]
    assert r.tolist() == esperado, (
        "Solo 1200 (índice 17) es outlier; el NaN (índice 18) debe ser False. "
        f"Se esperaba {esperado}, obtuviste {r.tolist()}"
    )


@pytest.mark.puntos(3)
def test_marcar_outliers_iqr_limite_exacto():
    # Q1=3, Q3=7 -> límites (-3, 13). El 13 está JUSTO en el límite: no es outlier.
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 13]})
    r = ej.marcar_outliers_iqr(df, "x")
    assert r.sum() == 0, (
        "Un valor IGUAL al límite (13) no es outlier: usa < y >, no <= ni >="
    )


@pytest.mark.puntos(3)
def test_marcar_outliers_iqr_dataset_real(viviendas):
    r = ej.marcar_outliers_iqr(viviendas, "area_m2")
    assert int(r.sum()) == 147, (
        f"En area_m2 la regla IQR marca 147 filas (muchas son casas grandes reales); obtuviste {int(r.sum())}"
    )
    errores = viviendas["area_m2"] > 1000
    assert r[errores].all(), "Todas las áreas mayores a 1000 m² (errores ×10) deben quedar marcadas"


# ------------------------------------------------------------------ eliminar_outliers_iqr (20)
@pytest.mark.puntos(8)
def test_eliminar_outliers_iqr_mini():
    df = pd.DataFrame({
        "area": [50, 60, 70, 80, 90, 900],
        "precio": [100, 9000, 120, 130, 140, 150],
        "distrito": ["A", "B", "C", "D", "E", "F"],
    })
    copia = df.copy()
    r = ej.eliminar_outliers_iqr(df, ["area", "precio"])
    pd.testing.assert_frame_equal(df, copia, obj="No modifiques el DataFrame de entrada (usa df.copy())")
    assert isinstance(r, pd.DataFrame), "Debe devolver un DataFrame"
    assert list(r.columns) == ["area", "precio", "distrito"], (
        "Debe conservar TODAS las columnas (también las que no se limpian, como 'distrito')"
    )
    assert r.index.tolist() == [0, 2, 3, 4], (
        "Se deben eliminar la fila 1 (precio 9000) y la 5 (area 900), conservando el índice original "
        f"(sin reset_index). Se esperaba [0, 2, 3, 4], obtuviste {r.index.tolist()}"
    )


@pytest.mark.puntos(4)
def test_eliminar_outliers_iqr_limites_sobre_original():
    # Si recalculas los límites de 'b' DESPUÉS de borrar la fila de 'a' = 90,
    # la fila 1 (b = 1) pasaría a ser outlier. La regla de la clase calcula
    # todos los límites UNA vez sobre el DataFrame original.
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 90],
                       "b": [17, 1, 28, 21, 34, 18, 14, 2]})
    r = ej.eliminar_outliers_iqr(df, ["a", "b"])
    assert r.index.tolist() == [0, 1, 2, 3, 4, 5, 6], (
        "Solo debe caer la fila 7 (a = 90). Calcula los límites de cada columna sobre el df ORIGINAL, "
        f"no sobre el ya filtrado. Obtuviste el índice {r.index.tolist()}"
    )


@pytest.mark.puntos(3)
def test_eliminar_outliers_iqr_conserva_nan():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 100, np.nan],
                       "y": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]})
    r = ej.eliminar_outliers_iqr(df, ["x", "y"])
    assert 9 in r.index, "Una fila con NaN NO es outlier: debe conservarse"
    assert 8 not in r.index, "La fila 8 (x = 100) es outlier y debe eliminarse"
    assert len(r) == 9, f"Deben quedar 9 filas; quedaron {len(r)}"


@pytest.mark.puntos(5)
def test_eliminar_outliers_iqr_dataset_real(viviendas):
    r = ej.eliminar_outliers_iqr(viviendas, COLS_REALES)
    assert len(r) == 1763, (
        f"Aplicando IQR a {COLS_REALES} a la vez deben quedar 1763 de 2015 filas; quedaron {len(r)}"
    )
    # Lo que observó el profesor con CRIM: tras borrar, los cuartiles cambian y aparecen
    # "nuevos" outliers si vuelves a aplicar la regla.
    segunda = ej.eliminar_outliers_iqr(r, ["area_m2"])
    assert len(segunda) < len(r), (
        "Al aplicar la regla otra vez sobre el resultado deberían caer más filas "
        "(los cuartiles se recalculan sobre los datos nuevos)"
    )


# ------------------------------------------------------------------ porcentaje_filas_perdidas (10)
@pytest.mark.puntos(4)
def test_porcentaje_boston():
    r = ej.porcentaje_filas_perdidas(pd.DataFrame({"a": range(506)}), pd.DataFrame({"a": range(268)}))
    assert r == pytest.approx(47.04), (
        f"Boston: de 506 a 268 filas se pierde (506-268)/506*100 = 47.04 %; obtuviste {r}"
    )


@pytest.mark.puntos(3)
def test_porcentaje_casos_borde():
    vacio = pd.DataFrame({"a": []})
    assert ej.porcentaje_filas_perdidas(vacio, vacio) == 0.0, "Si el original no tiene filas, devuelve 0.0"
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert ej.porcentaje_filas_perdidas(df, df) == 0.0, "Si no se pierde ninguna fila, devuelve 0.0"
    assert ej.porcentaje_filas_perdidas(df, df.iloc[:0]) == 100.0, "Si se pierden todas, devuelve 100.0"


@pytest.mark.puntos(3)
def test_porcentaje_redondeo():
    r = ej.porcentaje_filas_perdidas(pd.DataFrame({"a": range(2015)}), pd.DataFrame({"a": range(1763)}))
    assert r == 12.51, f"(2015-1763)/2015*100 = 12.506... -> redondeado a 2 decimales 12.51; obtuviste {r}"


# ------------------------------------------------------------------ outliers_zscore (15)
@pytest.mark.puntos(6)
def test_outliers_zscore_mini():
    s = pd.Series([10] * 20 + [100])
    r = ej.outliers_zscore(s)
    assert isinstance(r, pd.Series) and r.dtype == bool, "Debe devolver una pd.Series booleana"
    assert r.tolist() == [False] * 20 + [True], (
        "Solo el 100 está a más de 3 desviaciones estándar (z ≈ 4.36)"
    )
    r2 = ej.outliers_zscore(s, umbral=5)
    assert r2.sum() == 0, "Con umbral=5 el 100 (z ≈ 4.36) ya NO es outlier"


@pytest.mark.puntos(4)
def test_outliers_zscore_indice_nan_y_constante():
    s = pd.Series([10.0] * 20 + [100.0, np.nan], index=[f"c{i}" for i in range(22)])
    r = ej.outliers_zscore(s)
    assert list(r.index) == list(s.index), "Debe conservar el índice de la serie"
    assert r["c21"] == False, "Un NaN no es outlier (debe dar False)"  # noqa: E712
    assert r["c20"] == True, "El 100 debe ser outlier"  # noqa: E712
    constante = pd.Series([7.0, 7.0, 7.0, 7.0])
    rc = ej.outliers_zscore(constante)
    assert rc.dtype == bool and rc.sum() == 0, (
        "Si la desviación estándar es 0 (todos iguales) ningún valor es outlier: devuelve todo False"
    )


@pytest.mark.puntos(5)
def test_outliers_zscore_dataset_real(viviendas):
    r = ej.outliers_zscore(viviendas["precio_usd"])
    assert int(r.sum()) == 30, (
        f"En precio_usd hay 30 valores con |z| > 3 (usando mean() y std() de pandas); obtuviste {int(r.sum())}"
    )
    # Comparación: la regla IQR marca 105 filas en precio_usd; la de 3 desviaciones, bastante menos.
    assert int(r.sum()) < 105, (
        "La regla de 3 desviaciones estándar debería marcar MENOS filas que la regla IQR (105) en precio_usd"
    )


# ------------------------------------------------------------------ distancia_mahalanobis (25)
@pytest.mark.puntos(10)
def test_mahalanobis_detecta_outlier_multivariado(altura_peso):
    copia = altura_peso.copy()
    d = ej.distancia_mahalanobis(altura_peso, ["altura_cm", "peso_kg"])
    pd.testing.assert_frame_equal(altura_peso, copia, obj="No modifiques el DataFrame de entrada")
    assert isinstance(d, pd.Series), "Debe devolver una pd.Series"
    assert list(d.index) == list(altura_peso.index), "La Serie debe tener el mismo índice que el DataFrame"
    # La idea de la clase: en cada eje por separado NO hay outliers (límites IQR:
    # altura 135.25-205.25, peso 31.25-97.25)... pero al cruzarlos, p15 (182 cm, 52 kg) es el más raro.
    assert d.idxmax() == "p15", (
        f"La persona p15 (182 cm, 52 kg) debe tener la mayor distancia; la mayor es {d.idxmax()}"
    )
    assert d["p15"] > 3, f"La distancia de p15 debe ser mayor que 3 (≈ 3.59); obtuviste {d['p15']:.3f}"


@pytest.mark.puntos(5)
def test_mahalanobis_valores_exactos(altura_peso):
    d = ej.distancia_mahalanobis(altura_peso, ["altura_cm", "peso_kg"])
    esperado = [1.7949, 1.3456, 1.0784, 0.8865, 0.6094, 0.4423, 0.1755, 0.1736,
                0.4244, 0.6170, 0.9124, 1.1110, 1.4099, 1.6928, 3.5930]
    assert d.to_numpy(dtype=float) == pytest.approx(esperado, abs=1e-3), (
        "Las distancias no coinciden. Revisa: covarianza con np.cov(X, rowvar=False) (ddof=1), "
        "su inversa, y la raíz cuadrada al final.\n"
        f"Esperado: {esperado}\nObtenido: {np.round(d.to_numpy(dtype=float), 4).tolist()}"
    )


@pytest.mark.puntos(5)
def test_mahalanobis_nan(altura_peso):
    df = altura_peso.copy()
    df.loc["p16"] = [170.0, np.nan]  # fila con peso desconocido
    d = ej.distancia_mahalanobis(df, ["altura_cm", "peso_kg"])
    assert list(d.index) == list(df.index), "El resultado debe incluir también las filas con NaN"
    assert np.isnan(d["p16"]), "Una fila con NaN en alguna columna debe tener distancia NaN"
    assert d.drop("p16").to_numpy(dtype=float) == pytest.approx(
        ej.distancia_mahalanobis(altura_peso, ["altura_cm", "peso_kg"]).to_numpy(dtype=float), abs=1e-9
    ), "La media y la covarianza se calculan SOLO con las filas sin NaN (el resto no debe cambiar)"


@pytest.mark.puntos(5)
def test_mahalanobis_dataset_real(viviendas):
    d = ej.distancia_mahalanobis(viviendas, ["area_m2", "precio_usd"])
    assert len(d) == len(viviendas), "Debe haber una distancia por cada fila del dataset"
    assert d.idxmax() == 1902, (
        f"La fila más lejana debe ser la 1902 (2746 m² con precio normal: error ×10); es {d.idxmax()}"
    )
    top5 = d.nlargest(5).index.tolist()
    assert 1089 in top5, (
        "La fila 1089 (219 m², más de 2 millones USD: la 'mansión') debe estar entre las 5 más lejanas; "
        f"top 5 obtenido: {top5}"
    )
