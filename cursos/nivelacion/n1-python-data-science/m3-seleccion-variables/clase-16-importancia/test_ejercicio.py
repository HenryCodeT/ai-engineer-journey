"""Evaluador automático — Clase 16 · Métodos basados en la importancia.  NO MODIFICAR.

Cada test tiene un peso (@pytest.mark.puntos). La suma es 100.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from framework.testing import cargar_ejercicio, leer_dataset

ej = cargar_ejercicio(__file__)

UTILES = ["ram", "battery_power", "px_height", "px_width"]


def generar_moviles(n: int = 2000, semilla: int = 13) -> pd.DataFrame:
    """Versión sintética del dataset 'mobile price classification' (la misma de la clase 13).

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
def moviles_divididos():
    d = generar_moviles()
    X, y = d.drop(columns="price_range"), d["price_range"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)


@pytest.fixture
def banco():
    return leer_dataset("clientes_banco.csv")


def _modelo_importancia(random_state=1):
    return RandomForestClassifier(n_estimators=100, max_depth=15, max_leaf_nodes=20,
                                  random_state=random_state)


def _baseline():
    return RandomForestClassifier(max_depth=5, max_leaf_nodes=20, random_state=1)


def _columnas_referencia(X_tr, y_tr, threshold):
    sel = SelectFromModel(_modelo_importancia(), threshold=threshold).fit(X_tr, y_tr)
    return list(X_tr.columns[sel.get_support()])


def _acc(X_tr, y_tr, X_te, y_te):
    return accuracy_score(y_te, _baseline().fit(X_tr, y_tr).predict(X_te))


def _revisar_serie_ordenada(r, esperado: pd.Series, que: str):
    """Compara dos Series de importancias sin depender del orden de los empates."""
    assert isinstance(r, pd.Series), f"{que}: debe devolver una pandas Series"
    assert set(r.index) == set(esperado.index) and len(r) == len(esperado), (
        f"{que}: el índice deben ser los nombres de TODAS las columnas ({len(esperado)}); "
        f"obtuve {list(r.index)}"
    )
    valores = r.to_numpy(dtype=float)
    assert np.all(np.diff(valores) <= 1e-12), f"{que}: la Series debe estar ordenada de MAYOR a MENOR"
    assert np.allclose(r.reindex(esperado.index).to_numpy(dtype=float), esperado.to_numpy()), (
        f"{que}: los valores no coinciden con los esperados.\n"
        f"Esperado (top 5):\n{esperado.head().round(4)}\nObtuve (top 5):\n{r.head().round(4)}"
    )


# ---------------------------------------------------------------- 1 gini_manual (15)
@pytest.mark.puntos(9)
def test_gini_ejemplos_pizarra():
    casos = [([5, 5], 0.5), ([10, 0], 0.0), ([3, 1], 0.375), ([2, 2, 2, 2], 0.75), ([7], 0.0)]
    for conteos, esperado in casos:
        r = ej.gini_manual(conteos)
        assert isinstance(r, float), f"Debe devolver un float; con {conteos} devolvió {type(r).__name__}"
        assert r == pytest.approx(esperado), (
            f"Gini de {conteos} = 1 - suma(p^2) = {esperado}; obtuve {r}"
        )
    r = ej.gini_manual(np.array([6, 3, 1]))
    assert r == pytest.approx(1 - (0.36 + 0.09 + 0.01)), (
        f"Debe aceptar arrays de numpy: Gini de [6, 3, 1] = 0.54; obtuve {r}"
    )


@pytest.mark.puntos(3)
def test_gini_mas_mezcla_mas_gini():
    puro, poco, mitad = ej.gini_manual([9, 1]), ej.gini_manual([8, 2]), ej.gini_manual([5, 5])
    assert puro < poco < mitad, (
        "Mientras más mezcladas las clases, mayor el Gini: se esperaba "
        f"Gini([9,1]) < Gini([8,2]) < Gini([5,5]); obtuve {puro}, {poco}, {mitad}"
    )


@pytest.mark.puntos(3)
def test_gini_errores():
    with pytest.raises(ValueError):
        ej.gini_manual([0, 0])
    with pytest.raises(ValueError):
        ej.gini_manual([3, -1])


# ---------------------------------------------------------------- 2 ranking_importancia (20)
@pytest.mark.puntos(10)
def test_ranking_igual_a_referencia(moviles_divididos):
    X_tr, _, y_tr, _ = moviles_divididos
    X_c, y_c = X_tr.copy(), y_tr.copy()
    r = ej.ranking_importancia(X_tr, y_tr)
    pd.testing.assert_frame_equal(X_tr, X_c, obj="No modifiques X_train")
    pd.testing.assert_series_equal(y_tr, y_c, obj="No modifiques y_train")
    modelo = _modelo_importancia().fit(X_tr, y_tr)
    esperado = pd.Series(modelo.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    _revisar_serie_ordenada(r, esperado, "ranking_importancia")
    assert r.sum() == pytest.approx(1.0), f"Las importancias de un Random Forest suman 1; suman {r.sum():.4f}"


@pytest.mark.puntos(6)
def test_ranking_utiles_arriba(moviles_divididos):
    X_tr, _, y_tr, _ = moviles_divididos
    r = ej.ranking_importancia(X_tr, y_tr)
    assert set(r.index[:4]) == set(UTILES), (
        f"Las 4 variables que de verdad definen el precio {UTILES} deberían ser las 4 más "
        f"importantes; el top 4 fue {list(r.index[:4])}"
    )
    assert r.index[0] == "ram", f"'ram' es la que más pesa en el precio; quedó primera {r.index[0]!r}"


@pytest.mark.puntos(4)
def test_ranking_banco_real(banco):
    X = banco.drop(columns=["abandono", "cliente_id", "segmento"])
    y = banco["abandono"]
    r = ej.ranking_importancia(X, y, random_state=7)
    esperado = pd.Series(_modelo_importancia(7).fit(X, y).feature_importances_,
                         index=X.columns).sort_values(ascending=False)
    _revisar_serie_ordenada(r, esperado, "ranking_importancia en clientes_banco (random_state=7)")


# ---------------------------------------------------------------- 3 seleccionar_desde_modelo (20)
@pytest.mark.puntos(9)
def test_select_umbral_clase(moviles_divididos):
    X_tr, X_te, y_tr, _ = moviles_divididos
    X_c = X_tr.copy()
    r = ej.seleccionar_desde_modelo(X_tr, X_te, y_tr, threshold=0.01)
    pd.testing.assert_frame_equal(X_tr, X_c, obj="No modifiques X_train")
    assert isinstance(r, tuple) and len(r) == 3, "Debe devolver una tupla (X_train_sel, X_test_sel, columnas)"
    X_tr_sel, X_te_sel, cols = r
    esperado = _columnas_referencia(X_tr, y_tr, 0.01)
    assert list(cols) == esperado, (
        f"Con threshold=0.01 se esperaban {len(esperado)} columnas {esperado} (en el orden original); "
        f"obtuve {list(cols)}"
    )
    assert isinstance(X_tr_sel, pd.DataFrame) and isinstance(X_te_sel, pd.DataFrame), (
        "X_train_sel y X_test_sel deben ser DataFrames (selector.transform da un array sin nombres)"
    )
    assert list(X_tr_sel.columns) == esperado and list(X_te_sel.columns) == esperado, (
        "X_train_sel y X_test_sel deben tener exactamente las columnas elegidas"
    )
    assert X_tr_sel.index.equals(X_tr.index) and X_te_sel.index.equals(X_te.index), (
        "Conserva el índice original de X_train y X_test (solo se filtran columnas, no filas)"
    )
    pd.testing.assert_frame_equal(X_te_sel, X_te[esperado], obj="X_test_sel debe ser X_test[columnas]")


@pytest.mark.puntos(7)
def test_select_umbral_por_defecto(moviles_divididos):
    X_tr, X_te, y_tr, _ = moviles_divididos
    _, _, cols = ej.seleccionar_desde_modelo(X_tr, X_te, y_tr, threshold=None)
    assert sorted(cols) == sorted(UTILES), (
        "Con el umbral por defecto (la media de las importancias = 1/20 = 0.05) deberían quedar "
        f"solo las 4 variables útiles {UTILES}; obtuve {list(cols)}"
    )
    assert len(cols) < len(_columnas_referencia(X_tr, y_tr, 0.01)), (
        "Un umbral más alto (0.05 > 0.01) debe dejar MENOS columnas"
    )


@pytest.mark.puntos(4)
def test_select_coincide_con_ranking(moviles_divididos):
    X_tr, X_te, y_tr, _ = moviles_divididos
    _, _, cols = ej.seleccionar_desde_modelo(X_tr, X_te, y_tr, threshold=0.03)
    imp = pd.Series(_modelo_importancia().fit(X_tr, y_tr).feature_importances_, index=X_tr.columns)
    esperado = [c for c in X_tr.columns if imp[c] >= 0.03]
    assert list(cols) == esperado, (
        "SelectFromModel se queda con las columnas cuya importancia es >= threshold. "
        f"Con threshold=0.03 se esperaba {esperado}; obtuve {list(cols)}. ¿Usas el threshold que te pasan?"
    )


# ---------------------------------------------------------------- 4 importancia_permutacion (15)
@pytest.mark.puntos(8)
def test_permutacion_mini():
    rng = np.random.default_rng(0)
    senal = np.arange(40) % 10
    X = pd.DataFrame({
        "constante": np.full(40, 7),
        "senal": senal,
        "ruido": rng.integers(0, 100, 40),
    })
    y = pd.Series((senal >= 5).astype(int))
    modelo = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    r = ej.importancia_permutacion(modelo, X, y, random_state=0)
    assert isinstance(r, pd.Series), "Debe devolver una pandas Series"
    assert set(r.index) == {"constante", "senal", "ruido"}, f"Una entrada por columna; obtuve {list(r.index)}"
    assert r.index[0] == "senal", (
        "El árbol solo usa 'senal' (y = senal >= 5): al barajarla el accuracy cae, así que debe "
        f"quedar primera. Obtuve el orden {list(r.index)}"
    )
    assert r["senal"] > 0.2, f"Barajar 'senal' debería bajar mucho el accuracy; obtuve {r['senal']:.3f}"
    assert r["constante"] == pytest.approx(0.0) and r["ruido"] == pytest.approx(0.0), (
        "Barajar una columna que el modelo no usa no cambia nada: su importancia debe ser 0. "
        f"Obtuve constante={r['constante']}, ruido={r['ruido']}"
    )


@pytest.mark.puntos(7)
def test_permutacion_moviles(moviles_divididos):
    X_tr, X_te, y_tr, y_te = moviles_divididos
    modelo = _baseline().fit(X_tr, y_tr)
    X_c = X_te.copy()
    r = ej.importancia_permutacion(modelo, X_te, y_te)
    pd.testing.assert_frame_equal(X_te, X_c, obj="No modifiques X_test")
    ref = permutation_importance(modelo, X_te, y_te, n_repeats=5, random_state=1)
    esperado = pd.Series(ref.importances_mean, index=X_te.columns).sort_values(ascending=False)
    _revisar_serie_ordenada(r, esperado, "importancia_permutacion (n_repeats=5, random_state=1)")
    assert r.index[0] == "ram", f"'ram' debería ser la más importante también por permutación; fue {r.index[0]!r}"


# ---------------------------------------------------------------- 5 comparar_metodos (12)
@pytest.mark.puntos(8)
def test_comparar_resultados_de_la_clase():
    resultados = {
        "todas": (20, 0.83),
        "varianza": (13, 0.8475),
        "correlacion": (13, 0.8475),
        "importancia": (10, 0.8775),
        "importancia_otra_config": (10, 0.9025),
        "solo_ram": (1, 0.8475),
    }
    r = ej.comparar_metodos(resultados)
    assert isinstance(r, pd.DataFrame), "Debe devolver un DataFrame"
    assert list(r.columns) == ["metodo", "n_columnas", "accuracy"], (
        f"Columnas esperadas ['metodo', 'n_columnas', 'accuracy']; obtuve {list(r.columns)}"
    )
    orden = ["importancia_otra_config", "importancia", "solo_ram", "correlacion", "varianza", "todas"]
    assert list(r["metodo"]) == orden, (
        "Orden: accuracy de mayor a menor; si empatan, menos columnas primero; si empatan en todo, "
        f"orden alfabético.\nSe esperaba {orden}\nobtuve    {list(r['metodo'])}"
    )
    assert list(r.index) == list(range(6)), "Reinicia el índice: 0, 1, 2, ... (reset_index(drop=True))"
    assert list(r["n_columnas"]) == [10, 10, 1, 13, 13, 20], "n_columnas no corresponde a cada método"
    assert np.allclose(r["accuracy"].to_numpy(dtype=float), [0.9025, 0.8775, 0.8475, 0.8475, 0.8475, 0.83])
    assert np.issubdtype(r["n_columnas"].dtype, np.integer), "n_columnas debe ser entero"


@pytest.mark.puntos(4)
def test_comparar_vacio_y_uno():
    with pytest.raises(ValueError):
        ej.comparar_metodos({})
    r = ej.comparar_metodos({"todas": (20, 0.75)})
    assert r.shape == (1, 3) and r.loc[0, "metodo"] == "todas", (
        f"Con un solo método la tabla tiene 1 fila y 3 columnas; obtuve forma {r.shape}"
    )


# ---------------------------------------------------------------- 6 evaluar_seleccion (18)
@pytest.mark.puntos(10)
def test_evaluar_umbral_clase(moviles_divididos):
    X_tr, X_te, y_tr, y_te = moviles_divididos
    r = ej.evaluar_seleccion(X_tr, X_te, y_tr, y_te, threshold=0.01)
    assert isinstance(r, dict) and set(r) == {"todas", "seleccion", "n_columnas", "columnas"}, (
        "Debe devolver un dict con exactamente las claves 'todas', 'seleccion', 'n_columnas' y 'columnas'"
    )
    cols = _columnas_referencia(X_tr, y_tr, 0.01)
    todas = _acc(X_tr, y_tr, X_te, y_te)
    sel = _acc(X_tr[cols], y_tr, X_te[cols], y_te)
    assert list(r["columnas"]) == cols, f"'columnas' esperado {cols}; obtuve {list(r['columnas'])}"
    assert r["n_columnas"] == len(cols) and isinstance(r["n_columnas"], (int, np.integer)), (
        f"'n_columnas' debe ser el entero {len(cols)}; obtuve {r['n_columnas']!r}"
    )
    assert r["todas"] == pytest.approx(todas), (
        f"'todas' con el baseline (max_depth=5, max_leaf_nodes=20, random_state=1) = {todas:.4f}; "
        f"obtuve {r['todas']}"
    )
    assert r["seleccion"] == pytest.approx(sel), (
        f"'seleccion' (mismo baseline, solo {len(cols)} columnas) = {sel:.4f}; obtuve {r['seleccion']}"
    )


@pytest.mark.puntos(8)
def test_evaluar_menos_dimensiones_mas_desempeno(moviles_divididos):
    X_tr, X_te, y_tr, y_te = moviles_divididos
    r = ej.evaluar_seleccion(X_tr, X_te, y_tr, y_te, threshold=None)
    esperado = _acc(X_tr[sorted(UTILES, key=list(X_tr.columns).index)], y_tr,
                    X_te[sorted(UTILES, key=list(X_tr.columns).index)], y_te)
    assert r["n_columnas"] == 4, f"Con el umbral por defecto quedan 4 columnas; obtuve {r['n_columnas']}"
    assert r["seleccion"] == pytest.approx(esperado), (
        f"Accuracy con las 4 útiles esperado {esperado:.4f}; obtuve {r['seleccion']}"
    )
    assert r["seleccion"] > r["todas"], (
        "Como en la clase: con menos dimensiones el modelo debería rendir MÁS que con las 20 "
        f"(todas={r['todas']:.4f}, seleccion={r['seleccion']:.4f})"
    )
