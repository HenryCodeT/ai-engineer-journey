"""Evaluador automático — Clase 18 · Desbalance de datos (el problema).  NO MODIFICAR.

Cada test tiene un peso (@pytest.mark.puntos). La suma es 100.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from framework.testing import cargar_ejercicio, leer_dataset

ej = cargar_ejercicio(__file__)

CENTROS = [(-1.0, -0.5), (1.0, -0.5), (0.0, 1.0)]
ESC_A = {0: 13, 1: 15, 2: 972}
ESC_C = {0: 202, 1: 104, 2: 694}
ESC_D = {0: 336, 1: 334, 2: 330}
NUMERICAS = ["edad", "ingreso_mensual", "antiguedad_anios", "num_productos",
             "reclamos_ultimo_anio", "es_activo"]


@pytest.fixture(scope="module")
def banco():
    return leer_dataset("clientes_banco.csv")


@pytest.fixture
def banco_dividido(banco):
    X = banco[NUMERICAS]
    y = banco["abandono"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)


def _escenario_ref(conteos, random_state=0):
    clases = sorted(conteos)
    X, y = make_blobs(n_samples=[conteos[c] for c in clases], centers=CENTROS[:len(clases)],
                      cluster_std=0.8, random_state=random_state)
    return X, np.array(clases)[y]


def _evaluar_ref(conteos, random_state=0):
    X, y = _escenario_ref(conteos, random_state)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y,
                                              random_state=random_state)
    pred = SVC(kernel="linear").fit(X_tr, y_tr).predict(X_te)
    minoritaria = min(conteos, key=lambda c: (conteos[c], c))
    return {
        "accuracy": accuracy_score(y_te, pred),
        "balanced_accuracy": balanced_accuracy_score(y_te, pred),
        "recall_minoritaria": float(np.mean(pred[y_te == minoritaria] == minoritaria)),
    }


# ---------------------------------------------------------------- 1 ratio_desbalance (10)
@pytest.mark.puntos(6)
def test_ratio_mini():
    r = ej.ratio_desbalance([0, 0, 0, 0, 0, 0, 1, 1])
    assert isinstance(r, float), f"Debe devolver un float; devolvió {type(r).__name__}"
    assert r == pytest.approx(3.0), f"6 de la clase 0 y 2 de la clase 1 -> 6/2 = 3.0; obtuve {r}"
    r2 = ej.ratio_desbalance(pd.Series(["a", "b", "b", "c", "c", "c"]))
    assert r2 == pytest.approx(3.0), f"Mayoritaria 'c' (3) / minoritaria 'a' (1) = 3.0; obtuve {r2}"
    assert ej.ratio_desbalance([1, 0, 1, 0]) == pytest.approx(1.0), (
        "Con clases del mismo tamaño el ratio debe ser 1.0 (balanceado)"
    )


@pytest.mark.puntos(4)
def test_ratio_banco(banco):
    conteos = banco["abandono"].value_counts()
    esperado = conteos.max() / conteos.min()
    r = ej.ratio_desbalance(banco["abandono"])
    assert r == pytest.approx(esperado), (
        f"En clientes_banco.csv el ratio 'no abandona' / 'abandona' es {esperado:.2f}; obtuve {r}"
    )


# ---------------------------------------------------------------- 2 generar_escenario (15)
@pytest.mark.puntos(8)
def test_escenario_conteos_y_formato():
    X, y = ej.generar_escenario(ESC_A, random_state=0)
    assert isinstance(X, pd.DataFrame), "X debe ser un DataFrame"
    assert isinstance(y, pd.Series), "y debe ser una Series"
    assert list(X.columns) == ["x1", "x2"], f"Las columnas de X deben ser ['x1', 'x2']; obtuve {list(X.columns)}"
    assert y.name == "clase", f"La Series y debe llamarse 'clase'; se llama {y.name!r}"
    assert X.shape == (1000, 2), f"13 + 15 + 972 = 1000 puntos en 2D; obtuve forma {X.shape}"
    assert list(X.index) == list(range(1000)) and list(y.index) == list(range(1000)), (
        "X e y deben tener índice 0..n-1"
    )
    obtenido = y.value_counts().sort_index().to_dict()
    assert obtenido == ESC_A, f"Los conteos por clase deben ser EXACTOS {ESC_A}; obtuve {obtenido}"


@pytest.mark.puntos(4)
def test_escenario_coincide_con_make_blobs_y_etiquetas():
    X, y = ej.generar_escenario(ESC_C, random_state=7)
    X_ref, y_ref = _escenario_ref(ESC_C, random_state=7)
    assert np.allclose(X.to_numpy(), X_ref), (
        "Los puntos no coinciden con make_blobs(n_samples=[...], centers=CENTROS[:k], "
        "cluster_std=DISPERSION, random_state=random_state). Revisa los argumentos."
    )
    assert (y.to_numpy() == y_ref).all(), "Las etiquetas no coinciden con las de make_blobs"
    # claves que no son 0, 1, 2
    conteos = {"cuadrado": 40, "triangulo": 10}
    X2, y2 = ej.generar_escenario(conteos, random_state=3)
    assert y2.value_counts().to_dict() == conteos, (
        f"Con claves de texto, las etiquetas deben ser esas mismas claves con sus conteos "
        f"{conteos}; obtuve {y2.value_counts().to_dict()}"
    )
    X3, _ = ej.generar_escenario(conteos, random_state=3)
    assert np.allclose(X2.to_numpy(), X3.to_numpy()), "Misma semilla -> mismos puntos"


@pytest.mark.puntos(3)
def test_escenario_mas_de_tres_clases():
    with pytest.raises(ValueError):
        ej.generar_escenario({0: 10, 1: 10, 2: 10, 3: 10})


# ---------------------------------------------------------------- 3 prediccion_mayoritaria (10)
@pytest.mark.puntos(5)
def test_mayoritaria_mini_y_empate():
    r = ej.prediccion_mayoritaria([0, 0, 0, 1], 3)
    assert isinstance(r, np.ndarray), f"Debe devolver un np.ndarray; devolvió {type(r).__name__}"
    assert r.tolist() == [0, 0, 0], f"La mayoritaria es 0 -> [0, 0, 0]; obtuve {r.tolist()}"
    r2 = ej.prediccion_mayoritaria(pd.Series([2, 1, 2, 1, 0]), 4)
    assert r2.tolist() == [1, 1, 1, 1], (
        f"Empate entre 1 y 2 (2 veces cada una): se usa la MENOR -> [1, 1, 1, 1]; obtuve {r2.tolist()}"
    )


@pytest.mark.puntos(5)
def test_mayoritaria_banco_accuracy_engana(banco_dividido):
    _, _, y_tr, y_te = banco_dividido
    pred = ej.prediccion_mayoritaria(y_tr, len(y_te))
    assert len(pred) == len(y_te), f"Debe devolver {len(y_te)} predicciones; devolvió {len(pred)}"
    assert set(np.unique(pred)) == {0}, "En clientes_banco la clase mayoritaria es 0 (no abandona)"
    acc = accuracy_score(y_te, pred)
    assert acc > 0.9, (
        f"El modelo tonto debería tener accuracy alto (~0.95) sin aprender nada; obtuve {acc:.3f}"
    )
    assert recall_score(y_te, pred) == 0.0, "...y aun así no encuentra a NINGÚN cliente que abandona"


# ---------------------------------------------------------------- 4 recall_por_clase (15)
@pytest.mark.puntos(8)
def test_recall_por_clase_mini():
    r = ej.recall_por_clase([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
    assert isinstance(r, pd.Series), f"Debe devolver una Series; devolvió {type(r).__name__}"
    assert list(r.index) == [0, 1], f"El índice deben ser las clases reales ordenadas [0, 1]; obtuve {list(r.index)}"
    assert r.loc[0] == pytest.approx(0.5), f"Clase 0: 1 acierto de 2 -> 0.5; obtuve {r.loc[0]}"
    assert r.loc[1] == pytest.approx(2 / 3), f"Clase 1: 2 aciertos de 3 -> 0.667; obtuve {r.loc[1]}"


@pytest.mark.puntos(4)
def test_recall_por_clase_vs_sklearn():
    rng = np.random.default_rng(18)
    real = rng.choice([2, 0, 1], size=300, p=[0.8, 0.15, 0.05])
    pred = np.where(rng.random(300) < 0.7, real, rng.choice([0, 1, 2], size=300))
    r = ej.recall_por_clase(real, pred)
    esperado = recall_score(real, pred, labels=[0, 1, 2], average=None)
    assert list(r.index) == [0, 1, 2], f"Índice esperado [0, 1, 2]; obtuve {list(r.index)}"
    assert np.allclose(r.to_numpy(), esperado), (
        f"Debe coincidir con sklearn recall_score(average=None) = {np.round(esperado, 4)}; "
        f"obtuve {np.round(r.to_numpy(), 4)}"
    )


@pytest.mark.puntos(3)
def test_recall_por_clase_compara_por_posicion():
    real = pd.Series([1, 0, 1, 0], index=[40, 3, 17, 8])
    pred = pd.Series([1, 0, 0, 0], index=[0, 1, 2, 3])
    r = ej.recall_por_clase(real, pred)
    assert r.loc[0] == pytest.approx(1.0) and r.loc[1] == pytest.approx(0.5), (
        "Se compara posición por posición (no por índice): clase 0 -> 1.0, clase 1 -> 0.5. "
        f"Obtuve {r.to_dict()} (pista: np.asarray)"
    )


# ---------------------------------------------------------------- 5 precision_recall_f1 (20)
@pytest.mark.puntos(8)
def test_prf1_mini():
    r = ej.precision_recall_f1([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0])
    assert isinstance(r, dict), "Debe devolver un dict"
    assert set(r) == {"precision", "recall", "f1"}, f"Claves esperadas precision, recall, f1; obtuve {set(r)}"
    for k in ("precision", "recall", "f1"):
        assert r[k] == pytest.approx(2 / 3), f"TP=2, FP=1, FN=1 -> {k} = 0.667; obtuve {r[k]}"
    r2 = ej.precision_recall_f1(["si", "si", "no", "no", "no"], ["si", "no", "no", "no", "si"],
                                clase_positiva="si")
    # TP=1, FP=1, FN=1
    assert r2["precision"] == pytest.approx(0.5) and r2["recall"] == pytest.approx(0.5), (
        f"Con clase_positiva='si': TP=1, FP=1, FN=1 -> precision = recall = 0.5; obtuve {r2}"
    )
    r3 = ej.precision_recall_f1([1, 1, 1, 1, 0], [1, 0, 0, 0, 0])
    # precision 1, recall 0.25 -> f1 = 2*0.25/1.25 = 0.4
    assert r3["f1"] == pytest.approx(0.4), (
        f"precision=1.0 y recall=0.25 -> F1 (media armónica) = 0.4, no el promedio 0.625; obtuve {r3['f1']}"
    )


@pytest.mark.puntos(4)
def test_prf1_modelo_tonto_cero():
    r = ej.precision_recall_f1([0, 0, 0, 1], [0, 0, 0, 0], clase_positiva=1)
    assert r == {"precision": 0.0, "recall": 0.0, "f1": 0.0} or all(
        v == pytest.approx(0.0) for v in r.values()
    ), (
        "Si el modelo nunca predice la clase positiva, precision, recall y F1 valen 0.0 "
        f"(no debe dividir entre 0); obtuve {r}"
    )


@pytest.mark.puntos(8)
def test_prf1_banco_vs_sklearn(banco_dividido):
    X_tr, X_te, y_tr, y_te = banco_dividido
    modelo = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_tr, y_tr)
    pred = modelo.predict(X_te)
    r = ej.precision_recall_f1(y_te, pred, clase_positiva=1)
    esperado = {
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "f1": f1_score(y_te, pred, zero_division=0),
    }
    for k, v in esperado.items():
        assert r[k] == pytest.approx(v), (
            f"En clientes_banco (clase positiva = abandona), {k} debería ser {v:.4f} como en "
            f"sklearn; obtuve {r[k]}"
        )


# ---------------------------------------------------------------- 6 balanced_accuracy_manual (10)
@pytest.mark.puntos(5)
def test_balanced_mini():
    r = ej.balanced_accuracy_manual([0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0])
    assert isinstance(r, float), f"Debe devolver un float; devolvió {type(r).__name__}"
    assert r == pytest.approx(0.5), (
        f"recall clase 0 = 1.0, recall clase 1 = 0.0 -> balanced accuracy 0.5 "
        f"(el accuracy sería 0.67); obtuve {r}"
    )


@pytest.mark.puntos(5)
def test_balanced_vs_sklearn():
    rng = np.random.default_rng(5)
    real = rng.choice([0, 1, 2], size=500, p=[0.05, 0.15, 0.8])
    pred = np.where(rng.random(500) < 0.6, real, 2)
    r = ej.balanced_accuracy_manual(real, pred)
    esperado = balanced_accuracy_score(real, pred)
    assert r == pytest.approx(esperado), (
        f"Debe coincidir con sklearn balanced_accuracy_score = {esperado:.4f}; obtuve {r}"
    )


# ---------------------------------------------------------------- 7 evaluar_escenario (20)
@pytest.mark.puntos(8)
def test_evaluar_escenario_a_accuracy_engana():
    r = ej.evaluar_escenario(ESC_A, random_state=0)
    assert isinstance(r, dict), "Debe devolver un dict"
    assert set(r) == {"accuracy", "balanced_accuracy", "recall_minoritaria"}, (
        f"Claves esperadas accuracy, balanced_accuracy, recall_minoritaria; obtuve {set(r)}"
    )
    assert r["accuracy"] > 0.95, (
        f"Escenario A (13/15/972): el accuracy debería salir ALTO (~0.97, como el 0.9722 del "
        f"profesor); obtuve {r['accuracy']:.3f}"
    )
    assert r["recall_minoritaria"] < 0.5, (
        f"...pero la clase minoritaria casi no se reconoce (recall bajo); obtuve {r['recall_minoritaria']:.3f}"
    )
    assert r["balanced_accuracy"] < 0.6, (
        f"...y la balanced accuracy lo delata (~0.33); obtuve {r['balanced_accuracy']:.3f}"
    )


@pytest.mark.puntos(6)
def test_evaluar_escenario_d_balanceado():
    r = ej.evaluar_escenario(ESC_D, random_state=0)
    assert r["accuracy"] > 0.7 and r["balanced_accuracy"] > 0.7, (
        f"Escenario D (336/334/330): accuracy y balanced accuracy deberían ser razonables "
        f"(> 0.7) y parecidas; obtuve {r}"
    )
    assert abs(r["accuracy"] - r["balanced_accuracy"]) < 0.05, (
        "Con clases balanceadas, accuracy y balanced accuracy casi coinciden"
    )
    assert r["recall_minoritaria"] > 0.6, (
        f"Con datos balanceados la clase 'minoritaria' sí se reconoce (recall > 0.6); obtuve {r}"
    )


@pytest.mark.puntos(6)
def test_evaluar_escenario_coincide_con_referencia():
    for conteos, semilla in ((ESC_C, 0), ({0: 13, 1: 55, 2: 932}, 1)):
        r = ej.evaluar_escenario(conteos, random_state=semilla)
        esperado = _evaluar_ref(conteos, semilla)
        for k, v in esperado.items():
            assert r[k] == pytest.approx(v), (
                f"Escenario {conteos} (semilla {semilla}): {k} debería ser {v:.4f}; obtuve {r[k]}. "
                "Revisa: test_size=0.3, stratify=y, SVC(kernel='linear') y la clase minoritaria."
            )
