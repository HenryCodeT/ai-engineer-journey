"""Evaluador automático — Clase 17 · Reducción de la dimensionalidad (PCA).  NO MODIFICAR.

Cada test tiene un peso (@pytest.mark.puntos). La suma es 100.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from framework.testing import cargar_ejercicio, leer_dataset

ej = cargar_ejercicio(__file__)

NUMERICAS_BANCO = ["edad", "ingreso_mensual", "antiguedad_anios", "num_productos",
                   "reclamos_ultimo_anio", "es_activo"]


def generar_moviles(n: int = 2000, semilla: int = 13) -> pd.DataFrame:
    """Versión sintética del dataset 'mobile price classification' (la misma de la clase 13).

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
def mini():
    X_tr = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6],
                         "b": [2.0, 4, 5, 8, 10, 13],
                         "c": [5.0, 3, 4, 4, 6, 5]}, index=[10, 11, 12, 13, 14, 15])
    X_te = pd.DataFrame({"a": [2.5, 7.0], "b": [5.0, 12.0], "c": [4.0, 6.0]}, index=[20, 21])
    return X_tr, X_te


def _ref_pca(X_tr, X_te, n_components, escalar):
    if escalar:
        sc = StandardScaler().fit(X_tr)
        A, B = sc.transform(X_tr), sc.transform(X_te)
    else:
        A, B = np.asarray(X_tr, dtype=float), np.asarray(X_te, dtype=float)
    p = PCA(n_components=n_components).fit(A)
    return p.transform(A), p.transform(B), p


def _ref_rf(Z_tr, y_tr, Z_te, y_te):
    m = RandomForestClassifier(max_depth=5, max_leaf_nodes=20, random_state=1).fit(Z_tr, y_tr)
    return accuracy_score(y_te, m.predict(Z_te))


def _igual_salvo_signo(a, b):
    """Cada componente puede salir multiplicada por -1 (es la misma dirección): se compara |valor|."""
    return np.allclose(np.abs(np.asarray(a, dtype=float)), np.abs(np.asarray(b, dtype=float)), atol=1e-6)


# ---------------------------------------------------------------- 1 aplicar_pca (25)
@pytest.mark.puntos(8)
def test_aplicar_pca_mini_sin_escala(mini):
    X_tr, X_te = mini
    r = ej.aplicar_pca(X_tr, X_te, n_components=2, escalar=False)
    assert isinstance(r, tuple) and len(r) == 3, "Debe devolver una tupla (Z_train, Z_test, pca)"
    Z_tr, Z_te, pca = r
    assert isinstance(Z_tr, pd.DataFrame) and isinstance(Z_te, pd.DataFrame), (
        "Z_train y Z_test deben ser DataFrames"
    )
    assert list(Z_tr.columns) == ["PC1", "PC2"] and list(Z_te.columns) == ["PC1", "PC2"], (
        f"Con n_components=2 las columnas deben ser ['PC1', 'PC2']; obtuve {list(Z_tr.columns)}"
    )
    assert list(Z_tr.index) == list(X_tr.index) and list(Z_te.index) == list(X_te.index), (
        "Conserva el índice original de X_train / X_test (pista: index=X_train.index)"
    )
    ref_tr, ref_te, _ = _ref_pca(X_tr, X_te, 2, escalar=False)
    assert _igual_salvo_signo(Z_tr, ref_tr) and _igual_salvo_signo(Z_te, ref_te), (
        "Los valores proyectados no coinciden con PCA(n_components=2).fit(X_train) + transform. "
        f"Se esperaba Z_train≈\n{np.round(ref_tr, 4)}\nobtuve\n{np.round(Z_tr.to_numpy(), 4)}"
    )


@pytest.mark.puntos(8)
def test_aplicar_pca_moviles_con_escala(moviles_divididos):
    X_tr, X_te, _, _ = moviles_divididos
    Z_tr, Z_te, pca = ej.aplicar_pca(X_tr, X_te, n_components=3, escalar=True)
    assert isinstance(pca, PCA), f"El tercer elemento debe ser el objeto PCA ajustado; es {type(pca).__name__}"
    assert Z_tr.shape == (1600, 3) and Z_te.shape == (400, 3), (
        f"De R^20 a R^3: se esperaba Z_train (1600, 3) y Z_test (400, 3); obtuve {Z_tr.shape} y {Z_te.shape}"
    )
    ref_tr, ref_te, ref = _ref_pca(X_tr, X_te, 3, escalar=True)
    assert np.allclose(pca.explained_variance_ratio_, ref.explained_variance_ratio_, atol=1e-6), (
        "Con escalar=True hay que aplicar StandardScaler (fit con X_train) ANTES del PCA. "
        f"Varianza explicada esperada {ref.explained_variance_ratio_.round(4)}; "
        f"obtuve {np.round(pca.explained_variance_ratio_, 4)}"
    )
    assert _igual_salvo_signo(Z_tr, ref_tr) and _igual_salvo_signo(Z_te, ref_te), (
        "Los puntos proyectados no coinciden con StandardScaler + PCA(3) ajustados con X_train"
    )


@pytest.mark.puntos(6)
def test_aplicar_pca_no_ve_el_test(moviles_divididos):
    X_tr, X_te, _, _ = moviles_divididos
    X_te_raro = X_te * 1000 + 5  # un test MUY distinto: si se usa en algún fit, todo cambia
    Z_tr_a, _, pca_a = ej.aplicar_pca(X_tr, X_te, n_components=3, escalar=True)
    Z_tr_b, Z_te_b, pca_b = ej.aplicar_pca(X_tr, X_te_raro, n_components=3, escalar=True)
    assert getattr(pca_b, "n_samples_", None) == len(X_tr), (
        f"El PCA debe ajustarse SOLO con X_train ({len(X_tr)} filas); se ajustó con "
        f"{getattr(pca_b, 'n_samples_', '?')} filas"
    )
    assert np.allclose(Z_tr_a.to_numpy(), Z_tr_b.to_numpy()), (
        "Z_train cambió al cambiar X_test: el scaler o el PCA están aprendiendo de X_test. "
        "A X_test solo se le aplica transform (la pregunta de Samuel)"
    )
    assert np.allclose(pca_b.mean_, 0, atol=1e-8), (
        "Con escalar=True el PCA debe recibir el train ESTANDARIZADO (media 0); "
        f"pca.mean_ = {np.round(pca_b.mean_[:4], 4)}…"
    )
    _, ref_te, _ = _ref_pca(X_tr, X_te_raro, 3, escalar=True)
    assert _igual_salvo_signo(Z_te_b, ref_te), (
        "Z_test debe ser scaler.transform + pca.transform (ajustados con train) aplicados a X_test"
    )


@pytest.mark.puntos(3)
def test_aplicar_pca_porcentaje_y_no_modifica(moviles_divididos):
    X_tr, X_te, _, _ = moviles_divididos
    X_tr_c, X_te_c = X_tr.copy(), X_te.copy()
    Z_tr, Z_te, pca = ej.aplicar_pca(X_tr, X_te, n_components=0.90, escalar=False)
    pd.testing.assert_frame_equal(X_tr, X_tr_c, obj="No modifiques X_train")
    pd.testing.assert_frame_equal(X_te, X_te_c, obj="No modifiques X_test")
    assert list(Z_tr.columns) == ["PC1", "PC2", "PC3"], (
        "Con n_components=0.90 y sin escalar, PCA elige 3 componentes en los móviles: las "
        f"columnas deben ser PC1..PC3 (usa pca.n_components_); obtuve {list(Z_tr.columns)}"
    )


# ---------------------------------------------------------------- 2 varianza_explicada (15)
@pytest.mark.puntos(8)
def test_varianza_explicada_mini():
    falso = SimpleNamespace(explained_variance_ratio_=np.array([0.5, 0.3, 0.2]))
    r = ej.varianza_explicada(falso)
    assert isinstance(r, pd.DataFrame), "Debe devolver un DataFrame"
    assert list(r.columns) == ["componente", "varianza", "acumulada"], (
        f"Columnas esperadas ['componente', 'varianza', 'acumulada'] (en ese orden); obtuve {list(r.columns)}"
    )
    assert list(r["componente"]) == ["PC1", "PC2", "PC3"], (
        f"'componente' debe ser ['PC1', 'PC2', 'PC3']; obtuve {list(r['componente'])}"
    )
    assert list(r.index) == [0, 1, 2], f"El índice debe ser 0, 1, 2; obtuve {list(r.index)}"
    assert np.allclose(r["varianza"], [0.5, 0.3, 0.2]), "'varianza' es explained_variance_ratio_ tal cual"
    assert np.allclose(r["acumulada"], [0.5, 0.8, 1.0]), (
        f"'acumulada' debe ser [0.5, 0.8, 1.0] (0.5, 0.5+0.3, 0.5+0.3+0.2); obtuve {list(r['acumulada'])}"
    )


@pytest.mark.puntos(7)
def test_varianza_explicada_moviles(moviles_divididos):
    X_tr, _, _, _ = moviles_divididos
    pca = PCA(n_components=3).fit(X_tr)
    r = ej.varianza_explicada(pca)
    esperado = pca.explained_variance_ratio_
    assert len(r) == 3, f"PCA(3) -> 3 filas; obtuve {len(r)}"
    assert np.allclose(r["varianza"], esperado), (
        f"Varianza esperada {esperado.round(4)} (sin escalar, como en la clase: una componente "
        f"domina); obtuve {np.round(r['varianza'].to_numpy(), 4)}"
    )
    assert r["acumulada"].iloc[-1] == pytest.approx(esperado.sum()), (
        f"Las 3 componentes juntas conservan {esperado.sum():.4f} de la información; "
        f"la última 'acumulada' es {r['acumulada'].iloc[-1]}"
    )


# ---------------------------------------------------------------- 3 componentes_para (15)
@pytest.mark.puntos(6)
def test_componentes_para_sin_escala(moviles_divididos):
    X_tr, _, _, _ = moviles_divididos
    r = ej.componentes_para(X_tr, umbral=0.9, escalar=False)
    assert isinstance(r, (int, np.integer)) and not isinstance(r, bool), (
        f"Debe devolver un entero; devolvió {type(r).__name__}"
    )
    esperado = int(PCA(n_components=0.9).fit(X_tr).n_components_)
    assert r == esperado, (
        f"Sin escalar, PCA(n_components=0.9) elige {esperado} componentes (como al profe: 3); obtuve {r}"
    )


@pytest.mark.puntos(5)
def test_componentes_para_con_escala(moviles_divididos):
    X_tr, _, _, _ = moviles_divididos
    r = ej.componentes_para(X_tr, umbral=0.9, escalar=True)
    esperado = int(PCA(n_components=0.9).fit(StandardScaler().fit_transform(X_tr)).n_components_)
    assert r == esperado, (
        f"Con StandardScaler, para llegar al 90 % hacen falta {esperado} componentes "
        f"(ninguna variable domina por su escala); obtuve {r}"
    )


@pytest.mark.puntos(4)
def test_componentes_para_respeta_umbral(moviles_divididos):
    X_tr, _, _, _ = moviles_divididos
    r = ej.componentes_para(X_tr, umbral=0.95, escalar=False)
    esperado = int(PCA(n_components=0.95).fit(X_tr).n_components_)
    assert r == esperado, (
        f"Con umbral=0.95 (sin escalar) se esperaban {esperado} componentes; obtuve {r}. "
        "¿Usas el umbral que te pasan?"
    )


# ---------------------------------------------------------------- 4 comparar_con_y_sin_escala (15)
@pytest.mark.puntos(8)
def test_escala_moviles(moviles_divididos):
    X_tr, _, _, _ = moviles_divididos
    X_c = X_tr.copy()
    r = ej.comparar_con_y_sin_escala(X_tr)
    pd.testing.assert_frame_equal(X_tr, X_c, obj="No modifiques X_train")
    assert isinstance(r, dict) and set(r) == {"sin_escala", "con_escala"}, (
        "Debe devolver un dict con exactamente las claves 'sin_escala' y 'con_escala'"
    )
    sin = PCA().fit(X_tr).explained_variance_ratio_[0]
    con = PCA().fit(StandardScaler().fit_transform(X_tr)).explained_variance_ratio_[0]
    assert r["sin_escala"] == pytest.approx(sin, abs=1e-6), (
        f"Sin escalar, PC1 explica {sin:.4f} (casi todo es la columna 'ram', la de mayor escala); "
        f"obtuve {r['sin_escala']}"
    )
    assert r["con_escala"] == pytest.approx(con, abs=1e-6), (
        f"Con StandardScaler, PC1 explica solo {con:.4f}; obtuve {r['con_escala']}"
    )


@pytest.mark.puntos(7)
def test_escala_banco_real():
    banco = leer_dataset("clientes_banco.csv")
    X = banco[NUMERICAS_BANCO]
    r = ej.comparar_con_y_sin_escala(X)
    sin = PCA().fit(X).explained_variance_ratio_[0]
    con = PCA().fit(StandardScaler().fit_transform(X)).explained_variance_ratio_[0]
    assert r["sin_escala"] == pytest.approx(sin, abs=1e-6) and r["sin_escala"] > 0.99, (
        f"En clientes_banco, sin escalar, PC1 es prácticamente 'ingreso_mensual' (en miles): "
        f"explica {sin:.4f}; obtuve {r['sin_escala']}"
    )
    assert r["con_escala"] == pytest.approx(con, abs=1e-6), (
        f"Con StandardScaler, PC1 explica {con:.4f}; obtuve {r['con_escala']}"
    )


# ---------------------------------------------------------------- 5 evaluar_pca (15)
@pytest.mark.puntos(6)
def test_evaluar_pca_sin_escala(moviles_divididos):
    X_tr, X_te, y_tr, y_te = moviles_divididos
    r = ej.evaluar_pca(X_tr, X_te, y_tr, y_te, n_components=3, escalar=False)
    Z_tr, Z_te, _ = _ref_pca(X_tr, X_te, 3, escalar=False)
    esperado = _ref_rf(Z_tr, y_tr, Z_te, y_te)
    assert isinstance(r, float), f"Debe devolver un float; devolvió {type(r).__name__}"
    assert r == pytest.approx(esperado), (
        f"PCA(3) sin escalar + RandomForest(max_depth=5, max_leaf_nodes=20, random_state=1) "
        f"debería dar {esperado:.4f}; obtuve {r:.4f}"
    )


@pytest.mark.puntos(5)
def test_evaluar_pca_con_escala(moviles_divididos):
    X_tr, X_te, y_tr, y_te = moviles_divididos
    r = ej.evaluar_pca(X_tr, X_te, y_tr, y_te, n_components=3, escalar=True)
    Z_tr, Z_te, _ = _ref_pca(X_tr, X_te, 3, escalar=True)
    esperado = _ref_rf(Z_tr, y_tr, Z_te, y_te)
    assert r == pytest.approx(esperado), (
        f"Con StandardScaler + PCA(3) se esperaba {esperado:.4f}; obtuve {r:.4f}. "
        "¿Pasas `escalar` a aplicar_pca?"
    )


@pytest.mark.puntos(4)
def test_evaluar_pca_respeta_n_components(moviles_divididos):
    X_tr, X_te, y_tr, y_te = moviles_divididos
    r = ej.evaluar_pca(X_tr, X_te, y_tr, y_te, n_components=4, escalar=False)
    Z_tr, Z_te, _ = _ref_pca(X_tr, X_te, 4, escalar=False)
    esperado = _ref_rf(Z_tr, y_tr, Z_te, y_te)
    assert r == pytest.approx(esperado), (
        f"Con n_components=4 (sin escalar) se esperaba {esperado:.4f}; obtuve {r:.4f}. "
        "¿Usas el n_components que te pasan?"
    )


# ---------------------------------------------------------------- 6 cargas_principales (15)
@pytest.mark.puntos(8)
def test_cargas_mini():
    falso = SimpleNamespace(components_=np.array([[0.1, -0.9, 0.4, 0.05],
                                                  [0.7, 0.1, -0.2, 0.6]]))
    cols = ["a", "b", "c", "d"]
    r = ej.cargas_principales(falso, cols, componente=0, top=2)
    assert isinstance(r, pd.Series), "Debe devolver una Series"
    assert list(r.index) == ["b", "c"], (
        "En PC1 las cargas son a=0.1, b=-0.9, c=0.4, d=0.05: por |carga| van primero b y c; "
        f"obtuve {list(r.index)}"
    )
    assert np.allclose(r.to_numpy(), [-0.9, 0.4]), (
        f"Los valores deben conservar el SIGNO: [-0.9, 0.4]; obtuve {list(r.to_numpy())}"
    )
    r2 = ej.cargas_principales(falso, cols, componente=1, top=3)
    assert list(r2.index) == ["a", "d", "c"], (
        f"En PC2 (componente=1) el orden por |carga| es a (0.7), d (0.6), c (-0.2); obtuve {list(r2.index)}"
    )


@pytest.mark.puntos(7)
def test_cargas_moviles(moviles_divididos):
    X_tr, _, _, _ = moviles_divididos
    pca = PCA(n_components=3).fit(X_tr)
    pc1 = ej.cargas_principales(pca, list(X_tr.columns), componente=0, top=1)
    pc2 = ej.cargas_principales(pca, list(X_tr.columns), componente=1, top=1)
    pc3 = ej.cargas_principales(pca, list(X_tr.columns), componente=2, top=2)
    assert list(pc1.index) == ["ram"], (
        f"Sin escalar, PC1 es casi solo 'ram' (la columna de mayor varianza); obtuve {list(pc1.index)}"
    )
    assert list(pc2.index) == ["px_height"], f"PC2 debería ser casi solo 'px_height'; obtuve {list(pc2.index)}"
    assert set(pc3.index) == {"battery_power", "px_width"}, (
        f"PC3 mezcla 'battery_power' y 'px_width'; obtuve {list(pc3.index)}"
    )
