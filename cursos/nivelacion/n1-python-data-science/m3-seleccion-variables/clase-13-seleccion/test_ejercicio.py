"""Evaluador automático — Clase 13 · Método de selección de variables (baseline).  NO MODIFICAR.

Cada test tiene un peso (@pytest.mark.puntos). La suma es 100.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from framework.testing import cargar_ejercicio, leer_dataset

ej = cargar_ejercicio(__file__)

UTILES = ["ram", "battery_power", "px_height", "px_width"]


def generar_moviles(n: int = 2000, semilla: int = 13) -> pd.DataFrame:
    """Versión sintética del dataset 'mobile price classification' de la clase.

    20 predictores numéricos; solo 4 (UTILES) influyen en el rango de precio (0-3),
    el resto es ruido puro. 4 clases balanceadas (500 de cada una).
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
def moviles():
    d = generar_moviles()
    return d.drop(columns="price_range"), d["price_range"]


@pytest.fixture
def moviles_divididos(moviles):
    X, y = moviles
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)


@pytest.fixture(scope="module")
def banco():
    return leer_dataset("clientes_banco.csv")


def _referencia(X_tr, y_tr, X_te, y_te, max_depth=5, max_leaf_nodes=20):
    modelo = RandomForestClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes,
                                    random_state=1)
    modelo.fit(X_tr, y_tr)
    return accuracy_score(y_te, modelo.predict(X_te))


# ---------------------------------------------------------------- 1 accuracy_manual (20)
@pytest.mark.puntos(8)
def test_accuracy_ejemplo_pizarra():
    # 12 puntos, 2 errores (como el dibujo de triángulos y equis del profesor)
    real = ["x"] * 6 + ["t"] * 6
    pred = ["x"] * 5 + ["t"] + ["t"] * 5 + ["x"]
    r = ej.accuracy_manual(real, pred)
    assert isinstance(r, float), f"Debe devolver un float; devolvió {type(r).__name__}"
    assert r == pytest.approx(10 / 12), f"10 aciertos de 12 -> se esperaba {10 / 12:.4f}; obtuve {r}"
    assert ej.accuracy_manual([0, 1, 1, 0], [0, 1, 0, 0]) == pytest.approx(0.75), (
        "Con [0,1,1,0] vs [0,1,0,0] hay 3 aciertos de 4 -> 0.75"
    )


@pytest.mark.puntos(4)
def test_accuracy_compara_por_posicion():
    # y_test típico: Series con índice desordenado; y_pred: array de numpy
    real = pd.Series([2, 0, 1, 3], index=[17, 3, 250, 8])
    pred = np.array([2, 0, 3, 3])
    r = ej.accuracy_manual(real, pred)
    assert r == pytest.approx(0.75), (
        f"Se compara posición por posición (no por índice): 3 de 4 aciertos -> 0.75; obtuve {r}"
    )
    pred_serie = pd.Series([2, 0, 3, 3], index=[0, 1, 2, 3])
    r2 = ej.accuracy_manual(real, pred_serie)
    assert r2 == pytest.approx(0.75), (
        "Dos Series con índices distintos también deben compararse por posición "
        "(pista: np.asarray). Se esperaba 0.75"
    )


@pytest.mark.puntos(3)
def test_accuracy_longitudes_distintas():
    with pytest.raises(ValueError):
        ej.accuracy_manual([0, 1, 1], [0, 1])


@pytest.mark.puntos(5)
def test_accuracy_engana_con_desbalance(banco):
    # Un "modelo" que SIEMPRE dice "no abandona" (0) en clientes_banco.csv
    y = banco["abandono"]
    pred = np.zeros(len(y), dtype=int)
    esperado = float((y == 0).mean())
    r = ej.accuracy_manual(y, pred)
    assert r == pytest.approx(esperado), (
        f"Predecir siempre 0 en clientes_banco debería dar accuracy = {esperado:.4f} "
        f"(¡alto sin haber aprendido nada!); obtuve {r}"
    )


# ---------------------------------------------------------------- 2 matriz_confusion_manual (18)
@pytest.mark.puntos(8)
def test_matriz_mini():
    r = ej.matriz_confusion_manual([0, 0, 1, 1, 1], [0, 1, 1, 1, 0], clases=[0, 1])
    assert isinstance(r, pd.DataFrame), "Debe devolver un DataFrame"
    esperado = pd.DataFrame([[1, 1], [1, 2]], index=[0, 1], columns=[0, 1])
    assert r.shape == (2, 2), f"Con 2 clases la matriz es 2x2; obtuve {r.shape}"
    assert list(r.index) == [0, 1] and list(r.columns) == [0, 1], (
        "index y columns deben ser exactamente las clases [0, 1] (filas = real, columnas = predicho)"
    )
    assert (r.to_numpy() == esperado.to_numpy()).all(), (
        f"Se esperaba (filas=real, columnas=predicho):\n{esperado}\nobtuve:\n{r}"
    )
    assert np.issubdtype(r.to_numpy().dtype, np.integer), "Los conteos deben ser enteros (int)"


@pytest.mark.puntos(5)
def test_matriz_clase_nunca_predicha_y_orden():
    real = ["bajo", "medio", "alto", "alto", "medio"]
    pred = ["bajo", "bajo", "alto", "medio", "medio"]  # 'muy_alto' no aparece nunca
    clases = ["muy_alto", "alto", "medio", "bajo"]
    r = ej.matriz_confusion_manual(real, pred, clases=clases)
    esperado = confusion_matrix(real, pred, labels=clases)
    assert list(r.index) == clases and list(r.columns) == clases, (
        f"Respeta el orden de `clases` {clases} en filas y columnas; obtuve "
        f"index={list(r.index)}, columns={list(r.columns)}"
    )
    assert (r.to_numpy() == esperado).all(), (
        "Una clase que nunca aparece debe tener su fila y columna en 0. "
        f"Se esperaba:\n{esperado}\nobtuve:\n{r.to_numpy()}"
    )


@pytest.mark.puntos(5)
def test_matriz_igual_a_sklearn_en_moviles(moviles_divididos):
    X_tr, X_te, y_tr, y_te = moviles_divididos
    modelo = RandomForestClassifier(max_depth=5, max_leaf_nodes=20, random_state=1).fit(X_tr, y_tr)
    pred = modelo.predict(X_te)
    r = ej.matriz_confusion_manual(y_te, pred, clases=[0, 1, 2, 3])
    esperado = confusion_matrix(y_te, pred, labels=[0, 1, 2, 3])
    assert r.shape == (4, 4), f"4 rangos de precio -> matriz 4x4; obtuve {r.shape}"
    assert (r.to_numpy() == esperado).all(), (
        f"Debe coincidir con sklearn confusion_matrix:\n{esperado}\nobtuve:\n{r.to_numpy()}"
    )
    assert int(np.trace(r.to_numpy())) == int((np.asarray(y_te) == pred).sum()), (
        "La suma de la diagonal debe ser el número de aciertos"
    )


# ---------------------------------------------------------------- 3 proporciones_por_clase (10)
@pytest.mark.puntos(6)
def test_proporciones_mini():
    r = ej.proporciones_por_clase([3, 1, 1, 1, 2, 2, 1, 2])
    assert isinstance(r, pd.Series), "Debe devolver una Series"
    assert list(r.index) == [1, 2, 3], f"El índice deben ser las clases ordenadas [1, 2, 3]; obtuve {list(r.index)}"
    assert np.allclose(r.to_numpy(), [0.5, 0.375, 0.125]), (
        f"Proporciones esperadas [0.5, 0.375, 0.125] (4/8, 3/8, 1/8); obtuve {list(r.to_numpy())}"
    )


@pytest.mark.puntos(4)
def test_proporciones_banco(banco):
    r = ej.proporciones_por_clase(banco["abandono"])
    esperado = banco["abandono"].value_counts(normalize=True).sort_index()
    assert list(r.index) == [0, 1], f"Clases esperadas [0, 1]; obtuve {list(r.index)}"
    assert r.sum() == pytest.approx(1.0), "Las proporciones deben sumar 1"
    assert np.allclose(r.to_numpy(), esperado.to_numpy()), (
        f"En clientes_banco se esperaba {esperado.round(4).to_dict()}; obtuve {r.round(4).to_dict()}"
    )


# ---------------------------------------------------------------- 4 dividir_estratificado (18)
@pytest.mark.puntos(8)
def test_dividir_igual_a_sklearn(moviles):
    X, y = moviles
    X_c, y_c = X.copy(), y.copy()
    r = ej.dividir_estratificado(X, y)
    assert isinstance(r, tuple) and len(r) == 4, "Debe devolver una tupla (X_train, X_test, y_train, y_test)"
    pd.testing.assert_frame_equal(X, X_c, obj="No modifiques X")
    pd.testing.assert_series_equal(y, y_c, obj="No modifiques y")
    X_tr, X_te, y_tr, y_te = r
    ref = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    assert len(X_tr) == 1600 and len(X_te) == 400, (
        f"Con 2000 filas y test_size=0.2 -> 1600 train / 400 test; obtuve {len(X_tr)} / {len(X_te)}"
    )
    assert list(X_te.index) == list(ref[1].index) and list(y_te.index) == list(ref[3].index), (
        "El test debe ser idéntico al de train_test_split(X, y, test_size=0.2, stratify=y, random_state=1). "
        "¿Olvidaste stratify=y o el random_state?"
    )
    assert list(X_tr.index) == list(ref[0].index), "X_train no coincide: revisa el ORDEN de la tupla devuelta"


@pytest.mark.puntos(6)
def test_dividir_mantiene_proporciones_banco(banco):
    X = banco.drop(columns=["abandono", "cliente_id", "segmento"])
    y = banco["abandono"]
    _, X_te, y_tr, y_te = ej.dividir_estratificado(X, y, test_size=0.2, random_state=1)
    p_total = y.mean()
    assert abs(y_tr.mean() - p_total) < 0.002 and abs(y_te.mean() - p_total) < 0.002, (
        f"Con stratify, la tasa de abandono debe ser casi igual en total ({p_total:.4f}), "
        f"train ({y_tr.mean():.4f}) y test ({y_te.mean():.4f})"
    )
    assert int(y_te.sum()) == int(round(0.2 * y.sum())), (
        f"Test debería tener ~20 % de los {int(y.sum())} positivos ({round(0.2 * y.sum())}); "
        f"tiene {int(y_te.sum())}"
    )


@pytest.mark.puntos(4)
def test_dividir_otro_tamano(moviles):
    X, y = moviles
    X_tr, X_te, y_tr, y_te = ej.dividir_estratificado(X, y, test_size=0.3, random_state=7)
    ref = train_test_split(X, y, test_size=0.3, stratify=y, random_state=7)
    assert len(X_te) == 600, f"Con test_size=0.3 el test debe tener 600 filas; tiene {len(X_te)}"
    assert list(X_te.index) == list(ref[1].index), (
        "Respeta los argumentos test_size y random_state que te pasan (no los dejes fijos)"
    )
    assert (y_te.value_counts() == 150).all(), "Estratificado: 150 muestras de cada uno de los 4 rangos en test"


# ---------------------------------------------------------------- 5 entrenar_baseline (14)
@pytest.mark.puntos(10)
def test_baseline_todas_las_variables(moviles_divididos):
    X_tr, X_te, y_tr, y_te = moviles_divididos
    r = ej.entrenar_baseline(X_tr, y_tr, X_te, y_te)
    esperado = _referencia(X_tr, y_tr, X_te, y_te)
    assert isinstance(r, float), f"Debe devolver un float; devolvió {type(r).__name__}"
    assert r == pytest.approx(esperado), (
        f"Con RandomForestClassifier(max_depth=5, max_leaf_nodes=20, random_state=1) se esperaba "
        f"accuracy {esperado:.4f}; obtuve {r:.4f}"
    )


@pytest.mark.puntos(4)
def test_baseline_respeta_hiperparametros(moviles_divididos):
    X_tr, X_te, y_tr, y_te = moviles_divididos
    r = ej.entrenar_baseline(X_tr, y_tr, X_te, y_te, max_depth=2, max_leaf_nodes=4)
    esperado = _referencia(X_tr, y_tr, X_te, y_te, max_depth=2, max_leaf_nodes=4)
    assert r == pytest.approx(esperado), (
        f"Con max_depth=2 y max_leaf_nodes=4 (árboles más podados) se esperaba {esperado:.4f}; "
        f"obtuve {r:.4f}. ¿Usas los argumentos que te pasan?"
    )


# ---------------------------------------------------------------- 6 comparar_con_subconjunto (20)
@pytest.mark.puntos(12)
def test_comparar_valores(moviles_divididos):
    X_tr, X_te, y_tr, y_te = moviles_divididos
    X_c = X_tr.copy()
    r = ej.comparar_con_subconjunto(X_tr, X_te, y_tr, y_te, UTILES)
    pd.testing.assert_frame_equal(X_tr, X_c, obj="No modifiques X_train")
    assert isinstance(r, dict) and set(r) == {"todas", "subconjunto", "diferencia"}, (
        "Debe devolver un dict con exactamente las claves 'todas', 'subconjunto' y 'diferencia'"
    )
    todas = _referencia(X_tr, y_tr, X_te, y_te)
    sub = _referencia(X_tr[UTILES], y_tr, X_te[UTILES], y_te)
    assert r["todas"] == pytest.approx(todas), f"'todas' esperado {todas:.4f}; obtuve {r['todas']}"
    assert r["subconjunto"] == pytest.approx(sub), (
        f"'subconjunto' (solo {UTILES}) esperado {sub:.4f}; obtuve {r['subconjunto']}"
    )
    assert r["diferencia"] == pytest.approx(sub - todas), "'diferencia' = subconjunto - todas"


@pytest.mark.puntos(5)
def test_comparar_ruido_vs_utiles(moviles_divididos):
    X_tr, X_te, y_tr, y_te = moviles_divididos
    utiles = ej.comparar_con_subconjunto(X_tr, X_te, y_tr, y_te, UTILES)
    ruido = ej.comparar_con_subconjunto(X_tr, X_te, y_tr, y_te, ["blue", "wifi", "sc_h", "talk_time"])
    assert utiles["diferencia"] > 0, (
        "Con solo las 4 variables útiles (de 20) el baseline debería MEJORAR: las otras 16 son ruido. "
        f"Obtuve diferencia = {utiles['diferencia']:.4f}"
    )
    assert ruido["subconjunto"] < 0.4, (
        "Con 4 variables de puro ruido el modelo debería acertar poco más que al azar (~0.25 con 4 clases); "
        f"obtuve {ruido['subconjunto']:.4f}"
    )


@pytest.mark.puntos(3)
def test_comparar_columna_inexistente(moviles_divididos):
    X_tr, X_te, y_tr, y_te = moviles_divididos
    with pytest.raises(KeyError):
        ej.comparar_con_subconjunto(X_tr, X_te, y_tr, y_te, ["ram", "no_existe"])
