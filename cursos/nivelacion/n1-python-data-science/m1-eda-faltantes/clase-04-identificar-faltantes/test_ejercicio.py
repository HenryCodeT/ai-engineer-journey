"""Evaluador automático — Clase 04 · Cómo identifico datos faltantes (ejercicio).  NO MODIFICAR.

Cada test tiene un peso (@pytest.mark.puntos). La suma es 100.
"""
import numpy as np
import pandas as pd
import pytest

from framework.testing import cargar_ejercicio, leer_dataset

ej = cargar_ejercicio(__file__)

COLS_NUM = ["edad", "ingreso_mensual", "antiguedad_anios", "num_productos", "reclamos_ultimo_anio"]


@pytest.fixture
def mini():
    # a y b faltan en las MISMAS filas (0 y 1)  -> correlación de nulidad 1  (patrón MAR)
    # c falta en filas 0 y 2: sin relación con a/b -> correlación 0
    # d falta 3 de 4 (75 %)                       -> descartable con umbral 50; corr 0.58 con a, b y c
    # e nunca falta                               -> fuera de la matriz de nulidad
    return pd.DataFrame({
        "a": [np.nan, np.nan, 1.0, 1.0],
        "b": [np.nan, np.nan, 2.0, 2.0],
        "c": [np.nan, 3.0, np.nan, 3.0],
        "d": [np.nan, np.nan, np.nan, 4.0],
        "e": [1, 2, 3, 4],
    })


@pytest.fixture(scope="module")
def clientes():
    return leer_dataset("clientes_banco.csv")


@pytest.fixture(scope="module")
def clientes_nulos(clientes):
    """clientes_banco con nulos simulados igual que el profesor (sin usar tu función 3)."""
    d = clientes.copy()
    filas = d.sample(frac=0.6, random_state=123).index      # 60 % en las MISMAS filas
    d.loc[filas, ["edad", "num_productos"]] = np.nan
    for col, frac, semilla in [("ingreso_mensual", 0.10, 1), ("antiguedad_anios", 0.05, 2), ("segmento", 0.03, 3)]:
        d.loc[d.sample(frac=frac, random_state=semilla).index, col] = np.nan
    return d


# ---------------------------------------------------------------- reporte_nulos (15)
@pytest.mark.puntos(8)
def test_reporte_nulos_mini(mini):
    copia = mini.copy()
    r = ej.reporte_nulos(mini)
    pd.testing.assert_frame_equal(mini, copia, obj="No modifiques el DataFrame de entrada")
    assert isinstance(r, pd.DataFrame), "reporte_nulos debe devolver un DataFrame"
    assert list(r.columns) == ["variable", "nulos", "no_nulos", "porcentaje"], (
        f"Las columnas deben ser ['variable', 'nulos', 'no_nulos', 'porcentaje'] en ese orden. Obtuve {list(r.columns)}"
    )
    assert r["variable"].tolist() == ["d", "a", "b", "c", "e"], (
        "Orden esperado: de más a menos nulos y, en empate, el orden original -> ['d', 'a', 'b', 'c', 'e']. "
        f"Obtuve {r['variable'].tolist()} (¿usaste kind='stable'?)"
    )
    assert list(r.index) == [0, 1, 2, 3, 4], "El índice debe ser 0..n-1 (usa reset_index(drop=True))"
    assert r["nulos"].tolist() == [3, 2, 2, 2, 0], f"Nulos esperados [3, 2, 2, 2, 0], obtuve {r['nulos'].tolist()}"
    assert r["no_nulos"].tolist() == [1, 2, 2, 2, 4], (
        f"no_nulos = len(df) - nulos -> [1, 2, 2, 2, 4]; obtuve {r['no_nulos'].tolist()}"
    )
    assert r["porcentaje"].tolist() == pytest.approx([75.0, 50.0, 50.0, 50.0, 0.0]), (
        f"porcentaje = nulos / len(df) * 100 -> [75, 50, 50, 50, 0]; obtuve {r['porcentaje'].tolist()}"
    )


@pytest.mark.puntos(7)
def test_reporte_nulos_redondeo_y_dataset(clientes_nulos):
    tres = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    p = ej.reporte_nulos(tres).loc[0, "porcentaje"]
    assert p == pytest.approx(33.333), f"1 de 3 nulos = 33.333 % (redondeado a 3 decimales); obtuve {p}"

    r = ej.reporte_nulos(clientes_nulos)
    assert len(r) == clientes_nulos.shape[1] == 9, "Debe haber una fila por CADA columna (también las de 0 nulos)"
    assert r["variable"].tolist()[:5] == ["edad", "num_productos", "ingreso_mensual", "antiguedad_anios", "segmento"], (
        f"Las 5 columnas con más nulos, en orden, son edad, num_productos, ingreso_mensual, antiguedad_anios, segmento. "
        f"Obtuve {r['variable'].tolist()[:5]}"
    )
    assert r["nulos"].tolist()[:5] == [3000, 3000, 500, 250, 150], (
        f"Nulos esperados [3000, 3000, 500, 250, 150]; obtuve {r['nulos'].tolist()[:5]}"
    )
    assert (r["nulos"] + r["no_nulos"] == 5000).all(), "En cada fila, nulos + no_nulos debe ser 5000 (filas del dataset)"
    assert r.loc[0, "porcentaje"] == pytest.approx(60.0), "edad tiene 60 % de nulos"


# ---------------------------------------------------------------- columnas_descartables (10)
@pytest.mark.puntos(6)
def test_descartables_mini(mini):
    copia = mini.copy()
    r = ej.columnas_descartables(mini)
    pd.testing.assert_frame_equal(mini, copia, obj="No modifiques el DataFrame de entrada")
    assert isinstance(r, list), f"Debe devolver una lista, devolvió {type(r).__name__}"
    assert r == ["d"], (
        f"Con umbral=50 solo 'd' (75 %) se descarta; 'a', 'b', 'c' tienen exactamente 50 % y NO cuentan (> estricto). Obtuve {r}"
    )
    r40 = ej.columnas_descartables(mini, umbral=40)
    assert r40 == ["a", "b", "c", "d"], f"Con umbral=40 se descartan a, b, c, d en el orden original. Obtuve {r40}"
    assert ej.columnas_descartables(mini, umbral=80) == [], "Con umbral=80 ninguna supera: debe devolver []"


@pytest.mark.puntos(4)
def test_descartables_dataset(clientes_nulos):
    r = ej.columnas_descartables(clientes_nulos)
    assert r == ["edad", "num_productos"], (
        f"En clientes con nulos simulados solo edad y num_productos (60 %) superan el 50 %. Obtuve {r}"
    )
    r5 = ej.columnas_descartables(clientes_nulos, umbral=5)
    assert r5 == ["edad", "ingreso_mensual", "num_productos"], (
        f"Con umbral=5: edad, ingreso_mensual, num_productos (antiguedad tiene 5 % exacto y no cuenta). Obtuve {r5}"
    )


# ---------------------------------------------------------------- simular_faltantes (20)
@pytest.mark.puntos(10)
def test_simular_mini():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "c": [9, 9, 9, 9]})
    copia = df.copy()
    r = ej.simular_faltantes(df, ["a", "b"], fraccion=0.5, semilla=0)
    pd.testing.assert_frame_equal(df, copia, obj="No modifiques el DataFrame de entrada (usa df.copy())")
    assert list(r.columns) == ["a", "b", "c"] and len(r) == 4, "Debe conservar las columnas y la cantidad de filas"
    assert r["a"].isna().tolist() == [False, False, True, True], (
        "Con df.sample(frac=0.5, random_state=0) las filas elegidas son 2 y 3: 'a' debe ser NaN solo ahí"
    )
    assert r["a"].isna().equals(r["b"].isna()), "'a' y 'b' deben tener NaN en las MISMAS filas"
    assert r["c"].tolist() == [9, 9, 9, 9], "La columna 'c' no está en `columnas`: no debe cambiar"
    assert r.loc[0, "a"] == 1 and r.loc[1, "b"] == 6, "Los valores de las filas no elegidas no deben cambiar"


@pytest.mark.puntos(10)
def test_simular_dataset(clientes):
    r = ej.simular_faltantes(clientes, ["edad", "num_productos"], fraccion=0.6, semilla=123)
    assert clientes["edad"].isna().sum() == 0, "No modifiques el DataFrame original"
    assert r["edad"].isna().sum() == 3000 and r["num_productos"].isna().sum() == 3000, (
        "El 60 % de 5000 filas son 3000 NaN en edad y en num_productos"
    )
    assert r["edad"].isna().equals(r["num_productos"].isna()), "edad y num_productos deben faltar en las MISMAS filas"
    esperadas = set(clientes.sample(frac=0.6, random_state=123).index)
    assert set(r.index[r["edad"].isna()]) == esperadas, (
        "Las filas vaciadas deben ser exactamente df.sample(frac=fraccion, random_state=semilla).index"
    )
    otra = ej.simular_faltantes(clientes, ["edad"], fraccion=0.6, semilla=123)
    assert otra["edad"].isna().equals(r["edad"].isna()), "Con la misma semilla deben salir siempre las mismas filas"
    assert r["ingreso_mensual"].isna().sum() == 0, "Las columnas que no están en `columnas` no deben tener NaN"


# ---------------------------------------------------------------- correlacion_nulidad (15)
@pytest.mark.puntos(8)
def test_correlacion_mini(mini):
    copia = mini.copy()
    m = ej.correlacion_nulidad(mini)
    pd.testing.assert_frame_equal(mini, copia, obj="No modifiques el DataFrame de entrada")
    assert isinstance(m, pd.DataFrame), "Debe devolver un DataFrame"
    assert list(m.columns) == ["a", "b", "c", "d"] and list(m.index) == ["a", "b", "c", "d"], (
        "Solo entran las columnas con algún nulo (a, b, c, d), en el orden original; 'e' nunca falta y queda fuera. "
        f"Obtuve columnas {list(m.columns)}"
    )
    assert m.loc["a", "b"] == pytest.approx(1.0), "a y b faltan en las mismas filas: su correlación de nulidad debe ser 1.0"
    assert m.loc["a", "c"] == pytest.approx(0.0), "a y c no tienen relación: correlación 0.0"
    assert m.loc["a", "d"] == pytest.approx(0.58), "corr(a, d) = 0.577... redondeado a 2 decimales = 0.58"
    assert all(m.loc[c, c] == pytest.approx(1.0) for c in m.columns), "La diagonal debe valer 1.0"


@pytest.mark.puntos(7)
def test_correlacion_casos_borde_y_dataset(clientes_nulos):
    sin_nulos = pd.DataFrame({"x": [1, 2, 3]})
    r = ej.correlacion_nulidad(sin_nulos)
    assert isinstance(r, pd.DataFrame) and r.empty, "Si ninguna columna tiene nulos, devuelve un DataFrame vacío"

    todo_nulo = pd.DataFrame({"x": [np.nan, np.nan, np.nan], "y": [np.nan, 1.0, 2.0], "z": [1.0, np.nan, 2.0]})
    r = ej.correlacion_nulidad(todo_nulo)
    assert list(r.columns) == ["y", "z"], "Una columna 100 % nula no entra en la matriz (su indicador es constante)"

    m = ej.correlacion_nulidad(clientes_nulos)
    assert list(m.columns) == ["edad", "ingreso_mensual", "antiguedad_anios", "num_productos", "segmento"], (
        f"En clientes con nulos entran las 5 columnas con nulos en orden original. Obtuve {list(m.columns)}"
    )
    assert m.loc["edad", "num_productos"] == pytest.approx(1.0), (
        "edad y num_productos se vaciaron en las mismas filas: correlación de nulidad 1.0 (como en la clase)"
    )
    assert m.loc["edad", "ingreso_mensual"] == pytest.approx(0.02), (
        "edad e ingreso se vaciaron con semillas distintas: correlación casi 0 (0.02)"
    )


# ---------------------------------------------------------------- diagnostico_nulidad (15)
@pytest.mark.puntos(8)
def test_diagnostico_mini(mini):
    copia = mini.copy()
    r = ej.diagnostico_nulidad(mini)
    pd.testing.assert_frame_equal(mini, copia, obj="No modifiques el DataFrame de entrada")
    assert isinstance(r, dict), "Debe devolver un dict"
    assert list(r) == ["a", "b", "c", "d"], f"Las claves deben ser las columnas con nulos en orden: a, b, c, d. Obtuve {list(r)}"
    # Con umbral 0.5: a~b = 1.0, y d se correlaciona 0.58 con a, b y c -> las cuatro son "posible MAR"
    esperado = {"a": "posible MAR", "b": "posible MAR", "c": "posible MAR", "d": "posible MAR"}
    assert r == esperado, (
        f"Con umbral=0.5 esperaba {esperado}: a~b = 1.0 y d tiene 0.58 con a, b y c. Obtuve {r}"
    )
    estricto = ej.diagnostico_nulidad(mini, umbral=0.9)
    assert estricto == {"a": "posible MAR", "b": "posible MAR", "c": "potencial MCAR", "d": "potencial MCAR"}, (
        "Con umbral=0.9 solo a y b (correlación 1.0) son 'posible MAR'; c y d (máximo 0.58) son 'potencial MCAR'. "
        "Recuerda NO contar la diagonal (cada columna consigo misma vale 1). "
        f"Obtuve {estricto}"
    )
    assert ej.diagnostico_nulidad(pd.DataFrame({"x": [1, 2]})) == {}, "Sin columnas con nulos debe devolver {}"
    una = ej.diagnostico_nulidad(pd.DataFrame({"x": [1.0, np.nan], "y": [1, 2]}))
    assert una == {"x": "potencial MCAR"}, (
        f"Con una sola columna con nulos no hay 'otra' con quien correlacionar: 'potencial MCAR'. Obtuve {una}"
    )
    opuestas = pd.DataFrame({"p": [np.nan, np.nan, 1.0, 1.0], "q": [1.0, 1.0, np.nan, np.nan]})
    r_op = ej.diagnostico_nulidad(opuestas)
    assert r_op == {"p": "posible MAR", "q": "posible MAR"}, (
        "p falta justo donde q NO falta: correlación -1.0. Eso también es un patrón fuerte -> usa el valor "
        f"ABSOLUTO (.abs()). Obtuve {r_op}"
    )


@pytest.mark.puntos(7)
def test_diagnostico_dataset(clientes_nulos):
    r = ej.diagnostico_nulidad(clientes_nulos)
    esperado = {
        "edad": "posible MAR", "ingreso_mensual": "potencial MCAR", "antiguedad_anios": "potencial MCAR",
        "num_productos": "posible MAR", "segmento": "potencial MCAR",
    }
    assert r == esperado, (
        "Como en la clase: las dos variables vaciadas juntas (edad, num_productos) son 'posible MAR' y las "
        f"vaciadas al azar por separado son 'potencial MCAR'. Esperaba {esperado}, obtuve {r}"
    )


# ---------------------------------------------------------------- imputar_knn (25)
@pytest.mark.puntos(8)
def test_knn_a_mano():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0], "y": [10.0, 20.0, np.nan, 100.0], "zona": ["A", "A", "B", "B"]})
    copia = df.copy()
    r = ej.imputar_knn(df, ["x", "y"], k=2)
    pd.testing.assert_frame_equal(df, copia, obj="No modifiques el DataFrame de entrada (usa df.copy())")
    assert list(r.columns) == ["x", "y", "zona"], f"Deben quedar las mismas columnas en el mismo orden. Obtuve {list(r.columns)}"
    assert r.loc[2, "y"] == pytest.approx(15.0), (
        f"Con k=2 las vecinas más cercanas de x=3 son x=2 (y=20) y x=1 (y=10): promedio 15.0. Obtuve {r.loc[2, 'y']}"
    )
    assert r["y"].tolist()[:2] == [10.0, 20.0] and r.loc[3, "y"] == 100.0, "Los valores que ya existían no deben cambiar"
    assert r["zona"].tolist() == ["A", "A", "B", "B"], "La columna 'zona' no está en `columnas`: no debe cambiar"
    r1 = ej.imputar_knn(df, ["x", "y"], k=1)
    assert r1.loc[2, "y"] == pytest.approx(20.0), "Con k=1 solo cuenta la vecina más cercana (x=2, y=20): 20.0"


@pytest.mark.puntos(5)
def test_knn_conserva_indice():
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [5.0, np.nan, 7.0, 8.0], "otra": [np.nan, 1.0, 2.0, 3.0]},
        index=[10, 20, 30, 40],
    )
    r = ej.imputar_knn(df, ["x", "y"], k=2)
    assert list(r.index) == [10, 20, 30, 40], (
        f"El índice debe seguir siendo [10, 20, 30, 40]; obtuve {list(r.index)}. "
        "Si creas un DataFrame desde el array, pásale index=df.index"
    )
    assert r.loc[20, "y"] == pytest.approx(6.0), "Vecinas de x=2: x=1 (y=5) y x=3 (y=7) -> 6.0"
    assert r["otra"].isna().sum() == 1, "'otra' no está en `columnas`: debe conservar su NaN"


@pytest.mark.puntos(12)
def test_knn_dataset(clientes_nulos):
    from sklearn.impute import KNNImputer

    d = clientes_nulos.drop(columns=["edad", "num_productos"])   # el profe: con 60 % se descartan
    cols = ["ingreso_mensual", "antiguedad_anios", "reclamos_ultimo_anio", "abandono"]
    r = ej.imputar_knn(d, cols, k=10)

    assert r.shape == d.shape and list(r.index) == list(d.index), "Debe conservar la forma y el índice"
    assert r[cols].isna().sum().sum() == 0, "Las columnas imputadas no deben tener NaN"
    assert r["segmento"].isna().sum() == 150, "'segmento' no se imputa: debe conservar sus 150 NaN"
    habia = d["ingreso_mensual"].notna()
    assert (r.loc[habia, "ingreso_mensual"] == d.loc[habia, "ingreso_mensual"]).all(), (
        "Los ingresos que ya existían no deben cambiar"
    )
    esperado = KNNImputer(n_neighbors=10, weights="uniform").fit_transform(d[cols])
    np.testing.assert_allclose(
        r[cols].to_numpy(dtype=float), esperado, rtol=1e-9,
        err_msg="Los valores imputados no coinciden con KNNImputer(n_neighbors=10, weights='uniform')",
    )
    # El "pico" del que habló el profe: con la mediana, ~500 filas comparten el MISMO valor.
    con_mediana = d["ingreso_mensual"].fillna(d["ingreso_mensual"].median())
    pico_mediana = int(con_mediana.value_counts().iloc[0])
    pico_knn = int(r["ingreso_mensual"].value_counts().iloc[0])
    assert pico_knn < pico_mediana / 5, (
        f"Con KNN no debería aparecer un pico: el valor más repetido sale {pico_knn} veces "
        f"(con la mediana sale {pico_mediana})"
    )
