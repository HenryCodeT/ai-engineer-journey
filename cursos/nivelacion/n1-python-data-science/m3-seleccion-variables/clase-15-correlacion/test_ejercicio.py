"""Evaluador automático — Clase 15 · Eliminar características altamente correlacionadas.  NO MODIFICAR.

Cada test tiene un peso (@pytest.mark.puntos). La suma es 100.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from framework.testing import cargar_ejercicio, leer_dataset

ej = cargar_ejercicio(__file__)

PARES_KAGGLE = {("fc", "pc"), ("px_height", "px_width"), ("sc_h", "sc_w"), ("four_g", "three_g")}
ELIMINADAS_KAGGLE = ["pc", "px_width", "sc_w", "three_g"]


def generar_moviles(n: int = 2000, semilla: int = 13) -> pd.DataFrame:
    """Dataset sintético de la clase 13 (copia). Sus 20 columnas son INDEPENDIENTES entre sí."""
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


def generar_moviles_correlado(n: int = 2000, semilla: int = 13) -> pd.DataFrame:
    """Variante de la clase 15: parte del dataset de la clase 13 y, como en el Kaggle real,
    vuelve correlacionados 4 pares: fc~pc, px_height~px_width, sc_h~sc_w y four_g~three_g.
    El precio se recalcula con la misma fórmula (ram, battery_power, px_height, px_width)."""
    d = generar_moviles(n, semilla).drop(columns="price_range")
    rng = np.random.default_rng(semilla + 2)
    d["pc"] = np.clip(np.round(0.6 * d["fc"] + rng.uniform(0, 12, n)), 0, 20).astype(int)
    d["px_height"] = np.clip(np.round(0.9 * (d["px_width"] - 500) + rng.normal(0, 500, n)),
                             0, 1960).astype(int)
    d["sc_w"] = np.clip(np.round(0.8 * (d["sc_h"] - 5) + rng.normal(0, 3, n)), 0, 18).astype(int)
    d["three_g"] = np.where(d["four_g"] == 1, 1, rng.integers(0, 2, n))
    z = lambda c: (d[c] - d[c].mean()) / d[c].std()  # noqa: E731
    puntaje = (3 * z("ram") + z("battery_power") + 0.7 * z("px_height") + 0.7 * z("px_width")
               + rng.normal(0, 0.5, n))
    d["price_range"] = pd.qcut(puntaje, 4, labels=False).astype(int)
    return d


@pytest.fixture
def moviles_divididos():
    d = generar_moviles_correlado()
    X, y = d.drop(columns="price_range"), d["price_range"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)


@pytest.fixture
def viviendas_num():
    v = leer_dataset("viviendas_lima.csv")
    return v.select_dtypes("number").drop(columns=["id", "precio_usd"])


def _acc_rf(X_tr, y_tr, X_te, y_te):
    m = RandomForestClassifier(max_depth=5, max_leaf_nodes=20, random_state=1).fit(X_tr, y_tr)
    return accuracy_score(y_te, m.predict(X_te))


def _pares(r):
    return [(a, b) for a, b in zip(r["col_a"], r["col_b"])]


# ---------------------------------------------------------------- 1 correlacion_pearson_manual (20)
@pytest.mark.puntos(7)
def test_pearson_casos_perfectos():
    x = [1, 2, 3, 4, 5]
    r = ej.correlacion_pearson_manual(x, [2 * v + 1 for v in x])
    assert isinstance(r, float), f"Debe devolver un float; devolvió {type(r).__name__}"
    assert r == pytest.approx(1.0), f"y = 2x + 1 es una recta creciente -> r = 1; obtuve {r}"
    r2 = ej.correlacion_pearson_manual(x, [-3 * v for v in x])
    assert r2 == pytest.approx(-1.0), f"y = -3x es una recta decreciente -> r = -1; obtuve {r2}"
    r3 = ej.correlacion_pearson_manual([1, 2, 3, 4], [1, 3, 2, 4])
    assert r3 == pytest.approx(0.8), f"corr([1,2,3,4], [1,3,2,4]) = 0.8; obtuve {r3}"


@pytest.mark.puntos(7)
def test_pearson_igual_a_numpy(moviles_divididos):
    X_tr = moviles_divididos[0]
    for a, b in [("fc", "pc"), ("px_height", "px_width"), ("ram", "battery_power")]:
        r = ej.correlacion_pearson_manual(X_tr[a], X_tr[b])
        esperado = np.corrcoef(X_tr[a], X_tr[b])[0, 1]
        assert r == pytest.approx(esperado), (
            f"corr({a}, {b}) en X_train debe coincidir con np.corrcoef: {esperado:.4f}; obtuve {r}"
        )


@pytest.mark.puntos(3)
def test_pearson_por_posicion():
    x = pd.Series([1.0, 2.0, 3.0, 4.0], index=[10, 20, 30, 40])
    y = pd.Series([2.0, 4.0, 6.0, 8.0], index=[40, 30, 20, 10])
    r = ej.correlacion_pearson_manual(x, y)
    assert r == pytest.approx(1.0), (
        "Se compara POSICIÓN por posición, no por índice (pista: np.asarray). "
        f"[1,2,3,4] vs [2,4,6,8] -> 1.0; obtuve {r}"
    )


@pytest.mark.puntos(3)
def test_pearson_longitudes_distintas():
    with pytest.raises(ValueError):
        ej.correlacion_pearson_manual([1, 2, 3], [1, 2])


# ---------------------------------------------------------------- 2 pares_correlacionados (22)
def _df_mini():
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "b": [2.0, 4.1, 5.9, 8.2, 9.8, 12.1],   # ~ 2a  (corr ≈ +1)
        "c": [3.0, 1.0, 4.0, 1.0, 5.0, 2.0],    # ruido (corr con a ≈ 0.04)
        "d": [6.0, 5.0, 3.0, 4.0, 2.0, 0.0],    # ~ -a  (corr ≈ -0.96)
    })


@pytest.mark.puntos(9)
def test_pares_mini():
    df = _df_mini()
    df_c = df.copy()
    r = ej.pares_correlacionados(df, umbral=0.5)
    pd.testing.assert_frame_equal(df, df_c, obj="No modifiques df")
    assert isinstance(r, pd.DataFrame), "Debe devolver un DataFrame"
    assert list(r.columns) == ["col_a", "col_b", "corr"], (
        f"Columnas esperadas ['col_a', 'col_b', 'corr']; obtuve {list(r.columns)}"
    )
    corr = df.corr()
    esperado = [("a", "b"), ("b", "d"), ("a", "d")]
    esperado.sort(key=lambda p: -abs(corr.loc[p[0], p[1]]))
    assert _pares(r) == esperado, (
        f"Con umbral 0.5 se esperaban los pares {esperado} (ordenados por |corr| de mayor a menor, "
        f"col_a = la columna que va antes); obtuve {_pares(r)}"
    )
    assert list(r.index) == list(range(len(r))), "El índice debe ser 0, 1, 2, ... (reset_index(drop=True))"
    assert r.loc[r["col_b"] == "d", "corr"].lt(0).all(), (
        "'corr' guarda el valor CON signo: d baja cuando a sube, así que su correlación es negativa"
    )
    assert np.allclose(r["corr"].to_numpy(), [corr.loc[a, b] for a, b in esperado]), (
        "Los valores de 'corr' deben ser los de df.corr() (Pearson)"
    )


@pytest.mark.puntos(4)
def test_pares_solo_triangulo_superior_y_umbral_estricto():
    df = _df_mini()
    r = ej.pares_correlacionados(df, umbral=0.0)
    assert len(r) == 6, (
        "Con 4 columnas hay 4*3/2 = 6 pares distintos (sin la diagonal y sin repetir a~b / b~a); "
        f"obtuve {len(r)} filas"
    )
    assert not (r["col_a"] == r["col_b"]).any(), "No incluyas la diagonal (una columna consigo misma)"
    vacio = ej.pares_correlacionados(df, umbral=1.0)
    assert len(vacio) == 0 and list(vacio.columns) == ["col_a", "col_b", "corr"], (
        "Con umbral 1.0 ningún par cumple |corr| > 1.0: devuelve un DataFrame vacío con las 3 columnas"
    )


@pytest.mark.puntos(5)
def test_pares_moviles(moviles_divididos):
    X_tr = moviles_divididos[0]
    r = ej.pares_correlacionados(X_tr, umbral=0.4)
    assert set(_pares(r)) == PARES_KAGGLE, (
        f"En el dataset de móviles con pares correlacionados, con umbral 0.4 se esperaban "
        f"exactamente {sorted(PARES_KAGGLE)}; obtuve {_pares(r)}"
    )
    independiente = generar_moviles().drop(columns="price_range")
    r13 = ej.pares_correlacionados(independiente, umbral=0.4)
    assert len(r13) == 0, (
        "En el dataset de la clase 13 (columnas independientes) no hay ningún par con |corr| > 0.4; "
        f"obtuve {_pares(r13)}"
    )


@pytest.mark.puntos(4)
def test_pares_viviendas(viviendas_num):
    r = ej.pares_correlacionados(viviendas_num, umbral=0.4)
    corr = viviendas_num.corr()
    cols = list(corr.columns)
    esperado = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))
                if abs(corr.iloc[i, j]) > 0.4]
    assert set(_pares(r)) == set(esperado), (
        f"En viviendas_lima.csv (columnas numéricas sin id ni precio) se esperaban {esperado}; "
        f"obtuve {_pares(r)}"
    )
    assert r["corr"].abs().is_monotonic_decreasing, "Ordena por |corr| de mayor a menor"


# ---------------------------------------------------------------- 3 columnas_a_eliminar (22)
@pytest.mark.puntos(8)
def test_eliminar_mini():
    df = _df_mini()
    df_c = df.copy()
    r = ej.columnas_a_eliminar(df, umbral=0.5)
    pd.testing.assert_frame_equal(df, df_c, obj="No modifiques X_train")
    assert isinstance(r, list), f"Debe devolver una lista; devolvió {type(r).__name__}"
    assert r == ["b", "d"], (
        "Pares sobre 0.5: a~b, a~d, b~d. b tiene una columna anterior (a) muy correlacionada -> "
        "se elimina; d también (con a y con b). a no tiene columnas anteriores -> se queda; "
        f"c no supera el umbral con nadie. Se esperaba ['b', 'd']; obtuve {r}"
    )


@pytest.mark.puntos(5)
def test_eliminar_valor_absoluto_y_nunca_ambas():
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [5.0, 4.0, 3.0, 2.0, 1.0],     # corr(x, y) = -1
        "z": [2.0, 2.0, 1.0, 3.0, 2.0],     # corr(x, z) = 0.0
    })
    r = ej.columnas_a_eliminar(df, umbral=0.4)
    assert r == ["y"], (
        "Una correlación de -1 es una relación lineal PERFECTA: usa el valor absoluto. "
        "Del par (x, y) se elimina solo y (la que va después), nunca ambas. "
        f"Se esperaba ['y']; obtuve {r}"
    )
    assert ej.columnas_a_eliminar(df, umbral=1.0) == [], (
        "|corr| debe ser ESTRICTAMENTE mayor que el umbral: con umbral 1.0 no se elimina nada"
    )


@pytest.mark.puntos(6)
def test_eliminar_moviles(moviles_divididos):
    X_tr = moviles_divididos[0]
    r = ej.columnas_a_eliminar(X_tr, umbral=0.4)
    assert r == ELIMINADAS_KAGGLE, (
        f"Con umbral 0.4 se elimina una columna de cada par (la segunda): {ELIMINADAS_KAGGLE} "
        f"(las mismas que listó el profesor). Obtuve {r}"
    )


@pytest.mark.puntos(3)
def test_eliminar_respeta_umbral(moviles_divididos):
    X_tr = moviles_divididos[0]
    corr = X_tr.corr().abs()
    sup = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    esperado = [c for c in sup.columns if (sup[c] > 0.65).any()]
    r = ej.columnas_a_eliminar(X_tr, umbral=0.65)
    assert r == esperado, (
        f"Con umbral 0.65 (más exigente) solo sobreviven los pares más fuertes; se esperaba {esperado}; "
        f"obtuve {r}. ¿Usas el argumento umbral?"
    )


# ---------------------------------------------------------------- 4 comparar_pearson_spearman (16)
@pytest.mark.puntos(6)
def test_spearman_monotona_no_lineal():
    x = np.arange(1, 11, dtype=float)
    r = ej.comparar_pearson_spearman(x, np.exp(x))
    assert isinstance(r, dict) and set(r) == {"pearson", "spearman"}, (
        "Debe devolver un dict con las claves 'pearson' y 'spearman'"
    )
    assert r["spearman"] == pytest.approx(1.0), (
        f"y = e^x siempre sube: los rankings coinciden -> Spearman = 1; obtuve {r['spearman']}"
    )
    esperado = np.corrcoef(x, np.exp(x))[0, 1]
    assert r["pearson"] == pytest.approx(esperado) and r["pearson"] < 0.8, (
        f"Pearson de y = e^x debería ser {esperado:.4f} (< 1: no es una recta); obtuve {r['pearson']}"
    )


@pytest.mark.puntos(5)
def test_spearman_parabola_falla():
    x = np.arange(-3, 4, dtype=float)       # -3..3, simétrico
    r = ej.comparar_pearson_spearman(x, x ** 2)
    assert abs(r["pearson"]) < 1e-9, (
        "y = x**2 con x simétrico: relación PERFECTA pero NO lineal -> Pearson = 0. "
        f"Obtuve {r['pearson']}"
    )
    assert abs(r["spearman"]) < 1e-9, (
        "Tampoco es monótona (baja y luego sube), así que Spearman también da 0 "
        f"(con empates: ranking promedio). Obtuve {r['spearman']}"
    )


@pytest.mark.puntos(5)
def test_spearman_con_empates_igual_a_pandas():
    rng = np.random.default_rng(15)
    x = rng.integers(0, 6, 40)
    y = x + rng.integers(0, 4, 40)
    r = ej.comparar_pearson_spearman(list(x), list(y))
    sx, sy = pd.Series(x), pd.Series(y)
    assert r["pearson"] == pytest.approx(sx.corr(sy)), "Pearson no coincide con pandas"
    assert r["spearman"] == pytest.approx(sx.corr(sy, method="spearman")), (
        f"Spearman (con empates -> ranking promedio) debe ser {sx.corr(sy, method='spearman'):.4f}; "
        f"obtuve {r['spearman']}"
    )
    with pytest.raises(ValueError):
        ej.comparar_pearson_spearman([1, 2, 3], [1, 2])


# ---------------------------------------------------------------- 5 evaluar_eliminacion (20)
@pytest.mark.puntos(12)
def test_evaluar_moviles(moviles_divididos):
    X_tr, X_te, y_tr, y_te = moviles_divididos
    X_tr_c, X_te_c = X_tr.copy(), X_te.copy()
    r = ej.evaluar_eliminacion(X_tr, X_te, y_tr, y_te, umbral=0.4)
    pd.testing.assert_frame_equal(X_tr, X_tr_c, obj="No modifiques X_train")
    pd.testing.assert_frame_equal(X_te, X_te_c, obj="No modifiques X_test")
    assert isinstance(r, dict) and set(r) == {"todas", "reducido", "n_columnas", "eliminadas"}, (
        "Debe devolver un dict con las claves 'todas', 'reducido', 'n_columnas' y 'eliminadas'"
    )
    assert r["eliminadas"] == ELIMINADAS_KAGGLE, (
        f"'eliminadas' esperado {ELIMINADAS_KAGGLE}; obtuve {r['eliminadas']}"
    )
    assert r["n_columnas"] == 16 and isinstance(r["n_columnas"], (int, np.integer)), (
        f"De 20 columnas se eliminan 4 -> quedan 16; obtuve {r['n_columnas']}"
    )
    todas = _acc_rf(X_tr, y_tr, X_te, y_te)
    red = _acc_rf(X_tr.drop(columns=ELIMINADAS_KAGGLE), y_tr, X_te.drop(columns=ELIMINADAS_KAGGLE), y_te)
    assert r["todas"] == pytest.approx(todas), (
        f"'todas' con RandomForestClassifier(max_depth=5, max_leaf_nodes=20, random_state=1) "
        f"esperado {todas:.4f}; obtuve {r['todas']}"
    )
    assert r["reducido"] == pytest.approx(red), f"'reducido' esperado {red:.4f}; obtuve {r['reducido']}"


@pytest.mark.puntos(8)
def test_evaluar_decide_solo_con_x_train(moviles_divididos):
    X_tr, X_te, y_tr, y_te = moviles_divididos
    # En X_test (y SOLO ahí) talk_time pasa a ser una copia escalada de int_memory.
    # Si la decisión mira X_test (o X_train + X_test juntos), talk_time aparecería como
    # "muy correlacionada" y se eliminaría. Con solo X_train no debe pasar.
    X_te = X_te.copy()
    X_te["talk_time"] = X_te["int_memory"] * 1000
    r = ej.evaluar_eliminacion(X_tr, X_te, y_tr, y_te, umbral=0.4)
    assert r["eliminadas"] == ELIMINADAS_KAGGLE, (
        "La selección de columnas se decide SOLO con X_train (\"toda la experimentación... "
        "tiene que hacerse con este dato\") y luego se aplica a X_test. "
        f"Se esperaba {ELIMINADAS_KAGGLE}; obtuve {r['eliminadas']}"
    )
    red = _acc_rf(X_tr.drop(columns=ELIMINADAS_KAGGLE), y_tr, X_te.drop(columns=ELIMINADAS_KAGGLE), y_te)
    assert r["reducido"] == pytest.approx(red), (
        f"'reducido' debe quitar de X_test las MISMAS columnas que de X_train; esperado {red:.4f}; "
        f"obtuve {r['reducido']}"
    )
