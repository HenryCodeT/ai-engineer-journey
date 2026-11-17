"""Evaluador automático — Clase 14 · Eliminación de variables con baja varianza.  NO MODIFICAR.

Cada test tiene un peso (@pytest.mark.puntos). La suma es 100.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from framework.testing import cargar_ejercicio

ej = cargar_ejercicio(__file__)

BINARIAS = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]
# Las 13 variables que conserva VarianceThreshold(0.5) en el X_train sintético
# (las mismas que nombra el profesor: battery_power, clock_speed, fc, int_memory, ...)
CONSERVADAS_05 = ["battery_power", "clock_speed", "fc", "int_memory", "mobile_wt", "n_cores",
                  "pc", "px_height", "px_width", "ram", "sc_h", "sc_w", "talk_time"]


def generar_moviles(n: int = 2000, semilla: int = 13) -> pd.DataFrame:
    """Versión sintética del dataset 'mobile price classification' (la MISMA de la clase 13).

    20 predictores numéricos; solo 4 (ram, battery_power, px_height, px_width) influyen en
    el rango de precio (0-3), el resto es ruido puro. 4 clases balanceadas (500 de cada una).
    """
    rng = np.random.default_rng(semilla)
    d = pd.DataFrame({
        "battery_power": rng.integers(500, 2000, n), "blue": rng.integers(0, 2, n),
        "clock_speed": rng.uniform(0.5, 3.0, n).round(1), "dual_sim": rng.integers(0, 2, n),
        "fc": rng.integers(0, 20, n), "four_g": rng.integers(0, 2, n),
        "int_memory": rng.integers(2, 65, n), "m_dep": rng.uniform(0.1, 1.0, n).round(1),
        "mobile_wt": rng.integers(80, 201, n), "n_cores": rng.integers(1, 9, n),
        "pc": rng.integers(0, 21, n), "px_height": rng.integers(0, 1961, n),
        "px_width": rng.integers(500, 2000, n), "ram": rng.integers(256, 4000, n),
        "sc_h": rng.integers(5, 20, n), "sc_w": rng.integers(0, 19, n),
        "talk_time": rng.integers(2, 21, n), "three_g": rng.integers(0, 2, n),
        "touch_screen": rng.integers(0, 2, n), "wifi": rng.integers(0, 2, n),
    })
    z = lambda c: (d[c] - d[c].mean()) / d[c].std()  # noqa: E731
    puntaje = (3 * z("ram") + z("battery_power") + 0.7 * z("px_height") + 0.7 * z("px_width")
               + rng.normal(0, 0.5, n))
    d["price_range"] = pd.qcut(puntaje, 4, labels=False).astype(int)
    return d


@pytest.fixture
def moviles_divididos():
    d = generar_moviles()
    X, y = d.drop(columns="price_range"), d["price_range"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)


@pytest.fixture
def pizarra():
    # El ejemplo de la pizarra: x1 y x3 cambian de rango según la clase; x2 casi no varía
    return pd.DataFrame({"x1": [2, 10, 3, 15], "x2": [1, 2, 1, 1], "x3": [100, 50, 130, 45]})


def _rf(X_tr, y_tr, X_te, y_te):
    modelo = RandomForestClassifier(max_depth=5, max_leaf_nodes=20, random_state=1)
    modelo.fit(X_tr, y_tr)
    return accuracy_score(y_te, modelo.predict(X_te))


# ---------------------------------------------------------------- 1 varianza_manual (15)
@pytest.mark.puntos(6)
def test_varianza_ejemplo_pizarra():
    r5 = ej.varianza_manual([-5, 5])
    r10 = ej.varianza_manual([-10, 10])
    assert isinstance(r5, float), f"Debe devolver un float; devolvió {type(r5).__name__}"
    assert r5 == pytest.approx(25.0), (
        f"[-5, 5]: media 0, ((-5)^2 + 5^2) / 2 = 25; obtuve {r5}"
    )
    assert r10 == pytest.approx(100.0), (
        f"[-10, 10]: misma media 0 pero más dispersos -> (100 + 100) / 2 = 100; obtuve {r10}"
    )
    r_x2 = ej.varianza_manual([1, 2, 1, 1])
    assert r_x2 == pytest.approx(0.1875), (
        f"x2 de la pizarra [1, 2, 1, 1]: media 1.25 -> varianza 0.75 / 4 = 0.1875; obtuve {r_x2}"
    )


@pytest.mark.puntos(4)
def test_varianza_ddof_y_vacio():
    x1 = [2, 10, 3, 15]
    r0 = ej.varianza_manual(x1)
    r1 = ej.varianza_manual(x1, ddof=1)
    assert r0 == pytest.approx(28.25), f"Con ddof=0 se divide entre n=4: 113 / 4 = 28.25; obtuve {r0}"
    assert r1 == pytest.approx(113 / 3), (
        f"Con ddof=1 se divide entre n-1=3: 113 / 3 = {113 / 3:.4f}; obtuve {r1}"
    )
    with pytest.raises(ValueError):
        ej.varianza_manual([])


@pytest.mark.puntos(5)
def test_varianza_sin_atajos_en_moviles(moviles_divididos, monkeypatch):
    X_tr = moviles_divididos[0]
    ram = X_tr["ram"]  # Series con índice desordenado y valores enteros
    esperado = float(np.var(ram.to_numpy().astype(float)))

    def _prohibido(*args, **kwargs):
        raise AssertionError("varianza_manual NO debe usar .var(), np.var(), .std() ni np.std(): "
                             "calcula la fórmula tú mismo")

    monkeypatch.setattr(pd.Series, "var", _prohibido)
    monkeypatch.setattr(pd.Series, "std", _prohibido)
    monkeypatch.setattr(np, "var", _prohibido)
    monkeypatch.setattr(np, "std", _prohibido)
    r = ej.varianza_manual(ram)
    assert r == pytest.approx(esperado, rel=1e-9), (
        f"La varianza (ddof=0) de 'ram' en X_train debería ser {esperado:.2f}; obtuve {r}"
    )


# ---------------------------------------------------------------- 2 ranking_varianza (12)
@pytest.mark.puntos(6)
def test_ranking_pizarra(pizarra):
    r = ej.ranking_varianza(pizarra)
    assert isinstance(r, pd.Series), "Debe devolver una Series"
    assert list(r.index) == ["x3", "x1", "x2"], (
        f"De mayor a menor varianza el orden es x3, x1, x2; obtuve {list(r.index)}"
    )
    assert np.allclose(r.to_numpy(), [1254.6875, 28.25, 0.1875]), (
        f"Varianzas esperadas (ddof=0) [1254.6875, 28.25, 0.1875]; obtuve {list(r.to_numpy())}"
    )
    r1 = ej.ranking_varianza(pizarra, ddof=1)
    assert r1["x2"] == pytest.approx(0.25), "Con ddof=1, var(x2) = 0.75 / 3 = 0.25: respeta el argumento ddof"


@pytest.mark.puntos(6)
def test_ranking_moviles(moviles_divididos):
    X_tr = moviles_divididos[0]
    X_c = X_tr.copy()
    r = ej.ranking_varianza(X_tr)
    pd.testing.assert_frame_equal(X_tr, X_c, obj="No modifiques X_train")
    esperado = X_tr.var(ddof=0).sort_values(ascending=False)
    assert len(r) == 20, f"Debe haber una varianza por cada una de las 20 columnas; hay {len(r)}"
    assert r.index[0] == "ram" and r.index[-1] == "m_dep", (
        f"En X_train la más dispersa es 'ram' (está en MB) y la menos 'm_dep' (0.1 a 1.0); "
        f"obtuve primera={r.index[0]}, última={r.index[-1]}"
    )
    assert r.is_monotonic_decreasing, "La Series debe estar ordenada de MAYOR a MENOR"
    assert np.allclose(r.to_numpy(), esperado.to_numpy()), (
        "Los valores no coinciden con X_train.var(ddof=0) ordenado de mayor a menor"
    )


# ---------------------------------------------------------------- 3 seleccionar_por_varianza (25)
@pytest.mark.puntos(8)
def test_seleccionar_mini():
    X_tr = pd.DataFrame({"a": [1, 2, 1, 1], "b": [0, 10, 0, 10], "c": [5, 5, 5, 5]},
                        index=[10, 11, 12, 13])
    X_te = pd.DataFrame({"a": [3, 1], "b": [10, 0], "c": [5, 5]}, index=[50, 51])
    r = ej.seleccionar_por_varianza(X_tr, X_te, umbral=0.5)
    assert isinstance(r, tuple) and len(r) == 3, (
        "Debe devolver una tupla (X_train_sel, X_test_sel, columnas_conservadas)"
    )
    X_tr_sel, X_te_sel, cols = r
    assert list(cols) == ["b"], (
        f"var(a)=0.1875, var(b)=25, var(c)=0 -> con umbral 0.5 solo sobrevive 'b'; obtuve {list(cols)}"
    )
    assert isinstance(cols, list), "columnas_conservadas debe ser una list (usa list(...))"
    assert isinstance(X_tr_sel, pd.DataFrame) and isinstance(X_te_sel, pd.DataFrame), (
        "X_train_sel y X_test_sel deben ser DataFrames (VarianceThreshold devuelve arrays: "
        "reconstruye el DataFrame con los nombres)"
    )
    assert list(X_tr_sel.columns) == ["b"] and list(X_te_sel.columns) == ["b"], (
        "Ambos DataFrames deben tener solo la columna 'b'"
    )
    assert list(X_tr_sel.index) == [10, 11, 12, 13] and list(X_te_sel.index) == [50, 51], (
        "Conserva el índice original de X_train y X_test"
    )
    assert X_te_sel["b"].tolist() == [10, 0], "Los valores de X_test deben ser los originales"


@pytest.mark.puntos(7)
def test_seleccionar_fit_solo_en_train():
    # 'k' es constante en TRAIN pero varía en TEST: la decisión se toma SOLO con train
    X_tr = pd.DataFrame({"k": [7, 7, 7, 7], "m": [0, 4, 0, 4]})
    X_te = pd.DataFrame({"k": [0, 100], "m": [1, 1]})
    X_tr_sel, X_te_sel, cols = ej.seleccionar_por_varianza(X_tr, X_te, umbral=0.0)
    assert cols == ["m"], (
        f"'k' no varía en train (varianza 0) -> se elimina aunque en test sí varíe. "
        f"Se esperaba ['m']; obtuve {cols}. ¿Hiciste fit con X_test o con todo?"
    )
    assert list(X_te_sel.columns) == list(X_tr_sel.columns) == ["m"], (
        "Train y test deben quedar con EXACTAMENTE las mismas columnas (las decididas en train); "
        "'m' es constante en test y aun así se conserva"
    )


@pytest.mark.puntos(10)
def test_seleccionar_moviles_umbral_05(moviles_divididos):
    X_tr, X_te, _, _ = moviles_divididos
    X_tr_c, X_te_c = X_tr.copy(), X_te.copy()
    X_tr_sel, X_te_sel, cols = ej.seleccionar_por_varianza(X_tr, X_te, umbral=0.5)
    pd.testing.assert_frame_equal(X_tr, X_tr_c, obj="No modifiques X_train")
    pd.testing.assert_frame_equal(X_te, X_te_c, obj="No modifiques X_test")
    assert cols == CONSERVADAS_05, (
        f"Con umbral 0.5 se pasa de 20 a 13 variables (como en la clase). Se esperaba\n"
        f"{CONSERVADAS_05}\nobtuve\n{cols}"
    )
    assert X_tr_sel.shape == (1600, 13) and X_te_sel.shape == (400, 13), (
        f"Formas esperadas (1600, 13) y (400, 13); obtuve {X_tr_sel.shape} y {X_te_sel.shape}"
    )
    assert not set(BINARIAS) & set(cols), (
        "Las 6 binarias (varianza máxima 0.25) deben eliminarse con umbral 0.5"
    )
    pd.testing.assert_frame_equal(X_te_sel, X_te[CONSERVADAS_05], check_dtype=False,
                                  obj="X_test_sel debe ser X_test con solo las 13 columnas (mismo índice y valores)")


# ---------------------------------------------------------------- 4 curva_umbral (16)
@pytest.mark.puntos(6)
def test_curva_pizarra(pizarra):
    r = ej.curva_umbral(pizarra, [0, 0.1, 0.2, 1, 30, 2000])
    assert isinstance(r, pd.DataFrame), "Debe devolver un DataFrame"
    assert list(r.columns) == ["umbral", "n_variables"], (
        f"Columnas esperadas ['umbral', 'n_variables']; obtuve {list(r.columns)}"
    )
    assert r["umbral"].tolist() == pytest.approx([0, 0.1, 0.2, 1, 30, 2000]), (
        "La columna 'umbral' debe tener los umbrales en el mismo orden recibido"
    )
    assert r["n_variables"].tolist() == [3, 3, 2, 2, 1, 0], (
        "Varianzas: x1=28.25, x2=0.1875, x3=1254.69 (sobrevive si varianza > umbral). "
        f"Se esperaba [3, 3, 2, 2, 1, 0]; obtuve {r['n_variables'].tolist()}. "
        "Con umbral 2000 ninguna sobrevive: debe dar 0, no un error"
    )
    assert np.issubdtype(r["n_variables"].dtype, np.integer), "'n_variables' debe ser entero"


@pytest.mark.puntos(10)
def test_curva_moviles_como_en_clase(moviles_divididos):
    X_tr = moviles_divididos[0]
    umbrales = np.round(np.arange(0, 2.01, 0.1), 1)   # de 0 a 2, de 0.1 en 0.1
    r = ej.curva_umbral(X_tr, umbrales)
    esperado = [int(VarianceThreshold(u).fit(X_tr).get_support().sum()) for u in umbrales]
    assert len(r) == 21, f"Hay 21 umbrales (0.0, 0.1, ..., 2.0); obtuve {len(r)} filas"
    obtenido = r["n_variables"].tolist()
    assert obtenido == esperado, (
        f"Conteos esperados (como VarianceThreshold):\n{esperado}\nobtuve:\n{obtenido}"
    )
    assert obtenido[0] == 20 and obtenido[5] == 13 and obtenido[-1] == 12, (
        "Umbral 0 -> 20 variables; 0.5 -> 13; 2.0 -> 12 (el 'codo')"
    )
    assert list(r.index) == list(range(21)), "El índice debe ser 0, 1, 2, ..."


# ---------------------------------------------------------------- 5 varianza_por_escala (14)
@pytest.mark.puntos(6)
def test_escala_mini():
    df = pd.DataFrame({"binaria": [0, 1, 0, 1], "miles": [1000, 3000, 2000, 4000]})
    r = ej.varianza_por_escala(df)
    assert isinstance(r, pd.DataFrame), "Debe devolver un DataFrame"
    assert list(r.columns) == ["original", "minmax", "estandar"], (
        f"Columnas esperadas ['original', 'minmax', 'estandar']; obtuve {list(r.columns)}"
    )
    assert list(r.index) == ["binaria", "miles"], "El índice deben ser las columnas de X_train, en su orden"
    assert r.loc["miles", "original"] == pytest.approx(1_250_000), (
        f"var(miles) original = 1 250 000; obtuve {r.loc['miles', 'original']}"
    )
    assert r.loc["binaria", "original"] == pytest.approx(0.25), "var de una binaria 50/50 = 0.25"
    assert r.loc["miles", "minmax"] == pytest.approx(5 / 36), (
        "Con MinMax, 'miles' pasa a [0, 2/3, 1/3, 1] -> varianza 5/36 ≈ 0.1389 (¡menor que la binaria!)"
    )
    assert r.loc["binaria", "minmax"] == pytest.approx(0.25), "Una binaria 0/1 no cambia con MinMax"
    assert np.allclose(r["estandar"].to_numpy(), 1.0), "Tras StandardScaler toda varianza (ddof=0) vale 1"


@pytest.mark.puntos(8)
def test_escala_moviles(moviles_divididos):
    X_tr = moviles_divididos[0]
    X_c = X_tr.copy()
    r = ej.varianza_por_escala(X_tr)
    pd.testing.assert_frame_equal(X_tr, X_c, obj="No modifiques X_train")
    assert list(r.index) == list(X_tr.columns), "Una fila por columna de X_train, en el mismo orden"
    mm = pd.Series(np.var(MinMaxScaler().fit_transform(X_tr), axis=0), index=X_tr.columns)
    assert np.allclose(r["original"].to_numpy(), X_tr.var(ddof=0).to_numpy()), (
        "'original' debe ser la varianza (ddof=0) de los datos sin escalar"
    )
    assert np.allclose(r["minmax"].to_numpy(), mm.to_numpy()), (
        "'minmax' no coincide con la varianza tras MinMaxScaler ajustado en X_train"
    )
    assert np.allclose(r["estandar"].to_numpy(), 1.0), "'estandar': todas las columnas deben valer 1"
    assert (r.loc[BINARIAS, "original"] < r.loc["ram", "original"]).all(), (
        "Sin escalar, ram (en MB) tiene muchísima más varianza que cualquier binaria"
    )
    assert (r.loc[BINARIAS, "minmax"] > r.loc["ram", "minmax"]).all(), (
        "Con MinMax el ranking se invierte: las binarias (~0.25) superan a ram (~0.08)"
    )


# ---------------------------------------------------------------- 6 comparar_accuracy_umbral (18)
@pytest.mark.puntos(12)
def test_comparar_umbral_05(moviles_divididos):
    X_tr, X_te, y_tr, y_te = moviles_divididos
    X_c = X_tr.copy()
    r = ej.comparar_accuracy_umbral(X_tr, X_te, y_tr, y_te, umbral=0.5)
    pd.testing.assert_frame_equal(X_tr, X_c, obj="No modifiques X_train")
    assert isinstance(r, dict) and set(r) == {"todas", "seleccion", "n_columnas"}, (
        "Debe devolver un dict con exactamente las claves 'todas', 'seleccion' y 'n_columnas'"
    )
    todas = _rf(X_tr, y_tr, X_te, y_te)
    sel = _rf(X_tr[CONSERVADAS_05], y_tr, X_te[CONSERVADAS_05], y_te)
    assert r["n_columnas"] == 13 and isinstance(r["n_columnas"], (int, np.integer)), (
        f"Con umbral 0.5 quedan 13 columnas (entero); obtuve {r['n_columnas']!r}"
    )
    assert r["todas"] == pytest.approx(todas), (
        f"'todas' = baseline de la clase 13 con 20 variables = {todas:.4f}; obtuve {r['todas']}. "
        "¿Usaste RandomForestClassifier(max_depth=5, max_leaf_nodes=20, random_state=1)?"
    )
    assert r["seleccion"] == pytest.approx(sel), (
        f"'seleccion' (13 variables) esperado {sel:.4f}; obtuve {r['seleccion']}"
    )


@pytest.mark.puntos(6)
def test_comparar_respeta_umbral(moviles_divididos):
    X_tr, X_te, y_tr, y_te = moviles_divididos
    r = ej.comparar_accuracy_umbral(X_tr, X_te, y_tr, y_te, umbral=1.0)
    cols = list(X_tr.columns[VarianceThreshold(1.0).fit(X_tr).get_support()])
    sel = _rf(X_tr[cols], y_tr, X_te[cols], y_te)
    assert r["n_columnas"] == 12, (
        f"Con umbral 1.0 también cae clock_speed (varianza ~0.52) -> 12 columnas; obtuve {r['n_columnas']}"
    )
    assert r["seleccion"] == pytest.approx(sel), (
        f"Con umbral 1.0 el accuracy esperado es {sel:.4f}; obtuve {r['seleccion']}. "
        "¿Usas el umbral que te pasan?"
    )
