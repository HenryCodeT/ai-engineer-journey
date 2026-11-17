"""Evaluador automático — Clase 09 · Codificación de variables y escalamiento.  NO MODIFICAR.

Cada test tiene un peso (@pytest.mark.puntos). La suma es 100.
"""
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (MinMaxScaler, Normalizer, OneHotEncoder, OrdinalEncoder,
                                   PowerTransformer, StandardScaler)

from framework.testing import cargar_ejercicio, leer_dataset

ej = cargar_ejercicio(__file__)


@pytest.fixture
def banco():
    # Mini versión del dataset "bank marketing" de la clase.
    return pd.DataFrame({
        "age": [30, 45, 52, 28, 39, 61],
        "job": ["admin", "blue-collar", "technician", "admin", "services", "retired"],
        "balance": [1200.0, -50.0, 300.0, 0.0, 15000.0, 800.0],
        "loan": ["no", "yes", "no", "no", np.nan, "no"],
        "marital": ["single", "married", "divorced", "single", "married", "married"],
        "y": ["no", "no", "yes", "no", "yes", "no"],
    })


@pytest.fixture(scope="module")
def viviendas():
    return leer_dataset("viviendas_lima.csv")


# ---------------------------------------------------------------- 1 separar_variables (10)
@pytest.mark.puntos(6)
def test_separar_variables_mini(banco):
    copia = banco.copy()
    r = ej.separar_variables(banco, objetivo="y")
    pd.testing.assert_frame_equal(banco, copia, obj="No modifiques el DataFrame de entrada")
    assert isinstance(r, dict) and set(r) == {"numericas", "categoricas"}, (
        "Debe devolver un dict con exactamente las claves 'numericas' y 'categoricas'"
    )
    assert list(r["numericas"]) == ["age", "balance"], (
        f"Numéricas esperadas ['age', 'balance'] (en el orden del DataFrame); obtuve {list(r['numericas'])}"
    )
    assert list(r["categoricas"]) == ["job", "loan", "marital"], (
        "Categóricas esperadas ['job', 'loan', 'marital']: la objetivo 'y' NO debe aparecer. "
        f"Obtuve {list(r['categoricas'])}"
    )
    r2 = ej.separar_variables(banco)
    assert list(r2["categoricas"]) == ["job", "loan", "marital", "y"], (
        "Sin objetivo (objetivo=None) la columna 'y' sí cuenta como categórica"
    )


@pytest.mark.puntos(4)
def test_separar_variables_viviendas(viviendas):
    r = ej.separar_variables(viviendas, objetivo="precio_usd")
    assert list(r["categoricas"]) == ["distrito", "tipo_inmueble", "cochera", "estado", "pais"], (
        f"En viviendas las categóricas son distrito, tipo_inmueble, cochera, estado y pais; obtuve {list(r['categoricas'])}"
    )
    assert "precio_usd" not in r["numericas"], "El objetivo precio_usd no se transforma: sácalo de las numéricas"
    assert len(r["numericas"]) == 8 and "banos" in r["numericas"], (
        f"Deben quedar 8 numéricas (id, area_m2, ..., distancia_centro_km); obtuve {list(r['numericas'])}"
    )


# ---------------------------------------------------------------- 2 elegir_encoder (10)
@pytest.mark.puntos(6)
def test_elegir_encoder_mini(banco):
    assert ej.elegir_encoder(banco, "loan") == "onehot", (
        "loan tiene 2 categorías (yes/no; el NaN NO cuenta) -> 'onehot'"
    )
    assert ej.elegir_encoder(banco, "marital") == "onehot", "marital tiene 3 categorías (<= 3) -> 'onehot'"
    assert ej.elegir_encoder(banco, "job") == "binary", "job tiene 5 categorías (> 3) -> 'binary'"
    assert ej.elegir_encoder(banco, "job", max_onehot=5) == "onehot", (
        "Con max_onehot=5, job (5 categorías) ya entra en OneHot: ¿usaste el parámetro max_onehot?"
    )
    assert ej.elegir_encoder(banco, "marital", max_onehot=2) == "binary", (
        "Con max_onehot=2, marital (3 categorías) debe ser 'binary'"
    )


@pytest.mark.puntos(4)
def test_elegir_encoder_viviendas(viviendas):
    esperado = {"distrito": "binary", "tipo_inmueble": "onehot", "pais": "onehot",
                "estado": "binary", "cochera": "binary"}
    obtenido = {c: ej.elegir_encoder(viviendas, c) for c in esperado}
    assert obtenido == esperado, (
        "En viviendas: distrito (15) y estado (4) -> binary; tipo_inmueble (3) y pais (2) -> onehot; "
        "cochera tiene 5 categorías por el texto sucio ('Si', 'si', 'SI', 'No', 'no ') -> binary. "
        f"Obtuve {obtenido}"
    )


# ---------------------------------------------------------------- 3 columnas_resultantes (15)
# (Un solo test con varios casos: así vale 10 puntos en total, no 10 por caso.)
CASOS_COLUMNAS = [
    (8, "onehot", 8), (8, "dummy", 7), (8, "binary", 3), (8, "ordinal", 1),
    (20, "binary", 5), (12, "binary", 4), (3, "binary", 2), (4, "binary", 2),
    (2, "binary", 1), (1, "binary", 1), (1, "dummy", 0), (2, "onehot", 2),
]


@pytest.mark.puntos(10)
def test_columnas_resultantes_valores():
    for n, metodo, esperado in CASOS_COLUMNAS:
        r = ej.columnas_resultantes(n, metodo)
        assert r == esperado, (
            f"Con {n} categorías y método '{metodo}' se esperaban {esperado} columnas; obtuve {r}. "
            "Recuerda: onehot n, dummy n-1, binary max(1, ceil(log2(n))), ordinal 1"
        )
        assert isinstance(r, int), f"Debe devolver un int de Python, no {type(r).__name__}"


@pytest.mark.puntos(5)
def test_columnas_resultantes_dimension_y_error():
    # El ejemplo de Carlos: 10 variables, 4 categóricas de 4 categorías cada una.
    numericas = 6
    dim_onehot = numericas + 4 * ej.columnas_resultantes(4, "onehot")
    dim_dummy = numericas + 4 * ej.columnas_resultantes(4, "dummy")
    dim_binary = numericas + 4 * ej.columnas_resultantes(4, "binary")
    assert (dim_onehot, dim_dummy, dim_binary) == (22, 18, 14), (
        "Ejemplo de la clase: de R^10 se pasa a R^22 con OneHot, R^18 con dummy y R^14 con binary; "
        f"obtuve {(dim_onehot, dim_dummy, dim_binary)}"
    )
    with pytest.raises(ValueError):
        ej.columnas_resultantes(4, "hash")


# ---------------------------------------------------------------- 4 codificar_binario (20)
@pytest.mark.puntos(10)
def test_codificar_binario_mini():
    df = pd.DataFrame({"precio": [1, 2, 3, 4, 5], "team": ["B", "A", "C", "B", "D"]},
                      index=[10, 11, 12, 13, 14])
    copia = df.copy()
    r = ej.codificar_binario(df, "team")
    pd.testing.assert_frame_equal(df, copia, obj="No modifiques el DataFrame de entrada")
    assert list(r.columns) == ["precio", "team_0", "team_1"], (
        "4 categorías -> 2 bits. La columna 'team' se elimina y se agregan al final team_0, team_1. "
        f"Obtuve columnas {list(r.columns)}"
    )
    assert list(r.index) == [10, 11, 12, 13, 14], "Conserva el índice original"
    # A=0=00, B=1=01, C=2=10, D=3=11  (team_0 es el bit de la izquierda)
    assert r["team_0"].tolist() == [0, 0, 1, 0, 1] and r["team_1"].tolist() == [1, 0, 0, 1, 1], (
        "Códigos alfabéticos A=0 (00), B=1 (01), C=2 (10), D=3 (11), con team_0 = bit MÁS significativo. "
        f"Obtuve team_0={r['team_0'].tolist()}, team_1={r['team_1'].tolist()}"
    )
    assert all(pd.api.types.is_integer_dtype(r[c]) for c in ["team_0", "team_1"]), (
        "Las columnas de bits deben ser enteras (0/1), no float ni bool"
    )
    r2 = ej.codificar_binario(pd.DataFrame({"s": ["si", "no", "si"]}), "s")
    assert list(r2.columns) == ["s_0"] and r2["s_0"].tolist() == [1, 0, 1], (
        "Con 2 categorías basta 1 bit: no=0, si=1 -> columna s_0 = [1, 0, 1]"
    )


@pytest.mark.puntos(10)
def test_codificar_binario_viviendas(viviendas):
    r = ej.codificar_binario(viviendas, "distrito")
    nuevas = ["distrito_0", "distrito_1", "distrito_2", "distrito_3"]
    assert list(r.columns) == [c for c in viviendas.columns if c != "distrito"] + nuevas, (
        "distrito tiene 15 categorías -> ceil(log2(15)) = 4 columnas distrito_0..distrito_3 al final "
        "(en lugar de 15 con OneHot o 14 con dummy)"
    )
    bits = r[nuevas].astype(int)
    codigo = bits["distrito_0"] * 8 + bits["distrito_1"] * 4 + bits["distrito_2"] * 2 + bits["distrito_3"]
    por_distrito = codigo.groupby(viviendas["distrito"]).nunique()
    assert (por_distrito == 1).all(), "Cada distrito debe tener SIEMPRE el mismo código binario"
    assert codigo.nunique() == 15, "Los 15 distritos deben tener 15 códigos distintos (sin choques)"
    mapa = codigo.groupby(viviendas["distrito"]).first()
    assert mapa["Ate"] == 0 and mapa["Surco"] == 14 and mapa["SJL"] == 10, (
        "Orden alfabético con sorted(): Ate es el primero (0 = 0000), SJL el 11.º (10 = 1010) y "
        f"Surco el último (14 = 1110). Obtuve Ate={mapa['Ate']}, SJL={mapa['SJL']}, Surco={mapa['Surco']}"
    )
    pd.testing.assert_series_equal(r["precio_usd"], viviendas["precio_usd"],
                                   obj="Las demás columnas no deben cambiar")


# ---------------------------------------------------------------- 5 escalar (20)
@pytest.mark.puntos(6)
def test_escalar_minmax_mini():
    df = pd.DataFrame({"area": [50.0, 100.0, np.nan, 150.0], "hab": [1, 3, 2, 5], "zona": list("abcd")})
    copia = df.copy()
    r = ej.escalar(df, ["area", "hab"], "minmax")
    pd.testing.assert_frame_equal(df, copia, obj="No modifiques el DataFrame de entrada")
    assert list(r.columns) == ["area", "hab", "zona"], "Mismas columnas y mismo orden"
    assert r["area"].iloc[[0, 1, 3]].tolist() == [0.0, 0.5, 1.0] and pd.isna(r["area"].iloc[2]), (
        "MinMax de area [50, 100, NaN, 150] debe dar [0, 0.5, NaN, 1]: el NaN se mantiene"
    )
    assert r["hab"].tolist() == [0.0, 0.5, 0.25, 1.0], "MinMax de hab [1, 3, 2, 5] debe dar [0, 0.5, 0.25, 1]"
    assert r["zona"].tolist() == list("abcd"), "Las columnas no indicadas no se tocan"
    assert ej.escalar(df, ["hab"])["hab"].max() == 1.0, "El método por defecto debe ser 'minmax'"


@pytest.mark.puntos(6)
def test_escalar_standard_mini():
    df = pd.DataFrame({"x": [2.0, 4.0, 6.0, 8.0], "y": [10, 10, 10, 50]})
    r = ej.escalar(df, ["x", "y"], "standard")
    esperado_x = [-1.341641, -0.447214, 0.447214, 1.341641]
    assert np.allclose(r["x"], esperado_x, atol=1e-5), (
        f"Standard de x [2, 4, 6, 8] (std poblacional, ddof=0) debe dar {esperado_x}; obtuve {r['x'].round(6).tolist()}. "
        "¿Usaste std() con ddof=1 (el defecto de pandas)?"
    )
    assert abs(r["y"].mean()) < 1e-9 and abs(r["y"].std(ddof=0) - 1) < 1e-9, (
        "Tras StandardScaler cada columna debe tener media 0 y varianza 1"
    )
    with pytest.raises(ValueError):
        ej.escalar(df, ["x"], "robusto")


@pytest.mark.puntos(8)
def test_escalar_viviendas_igual_sklearn_y_no_cambia_forma(viviendas):
    cols = ["area_m2", "precio_usd", "habitaciones"]
    for metodo, Scaler in [("minmax", MinMaxScaler), ("standard", StandardScaler)]:
        r = ej.escalar(viviendas, cols, metodo)
        ref = Scaler().fit_transform(viviendas[cols])
        assert np.allclose(r[cols].to_numpy(), ref), (
            f"Con metodo='{metodo}' el resultado debe coincidir con sklearn {Scaler.__name__}"
        )
        # Escalar NO cambia la forma: la asimetría (skew) es la misma.
        for c in cols:
            assert math.isclose(r[c].skew(), viviendas[c].skew(), rel_tol=1e-6), (
                f"Escalar no cambia la forma de la distribución: la asimetría de {c} debía seguir igual"
            )
    # Control (contraste con la clase): PowerTransformer SÍ cambia la forma.
    pt = PowerTransformer().fit_transform(viviendas[["precio_usd"]])[:, 0]
    assert abs(pd.Series(pt).skew()) < abs(viviendas["precio_usd"].skew())


# ---------------------------------------------------------------- 6 normalizar_filas_l2 (10)
@pytest.mark.puntos(5)
def test_normalizar_filas_mini():
    df = pd.DataFrame({"x": [3.0, 0.0, 1.0], "y": [4.0, 0.0, 1.0], "id": [1, 2, 3]})
    copia = df.copy()
    r = ej.normalizar_filas_l2(df, ["x", "y"])
    pd.testing.assert_frame_equal(df, copia, obj="No modifiques el DataFrame de entrada")
    assert list(r.columns) == ["x", "y", "id"] and r["id"].tolist() == [1, 2, 3], (
        "Deben quedar todas las columnas; 'id' no se toca"
    )
    assert np.allclose(r.loc[0, ["x", "y"]].astype(float), [0.6, 0.8]), "La fila (3, 4) tiene norma 5 -> (0.6, 0.8)"
    assert r.loc[1, ["x", "y"]].tolist() == [0.0, 0.0], "Una fila de ceros se queda en ceros (sin dividir entre 0)"
    assert np.allclose(r.loc[2, ["x", "y"]].astype(float), [1 / np.sqrt(2)] * 2), "La fila (1, 1) -> (0.7071, 0.7071)"


@pytest.mark.puntos(5)
def test_normalizar_filas_viviendas(viviendas):
    cols = ["area_m2", "habitaciones", "distancia_centro_km"]
    datos = viviendas.dropna(subset=cols)
    r = ej.normalizar_filas_l2(datos, cols)
    normas = np.sqrt((r[cols] ** 2).sum(axis=1))
    assert np.allclose(normas, 1.0), "Cada fila (punto) debe quedar con norma L2 = 1"
    assert np.allclose(r[cols].to_numpy(), Normalizer(norm="l2").fit_transform(datos[cols])), (
        "Debe coincidir con sklearn Normalizer(norm='l2')"
    )
    assert not math.isclose(r["area_m2"].skew(), datos["area_m2"].skew(), rel_tol=1e-3), (
        "(control) normalizar por fila SÍ cambia la distribución de la variable"
    )


# ---------------------------------------------------------------- 7 construir_preprocesador (15)
@pytest.mark.puntos(5)
def test_preprocesador_estructura():
    pre = ej.construir_preprocesador(["a"], ["b"], ["c", "d"])
    assert isinstance(pre, ColumnTransformer), "Debe devolver un ColumnTransformer"
    assert not hasattr(pre, "transformers_"), "Devuélvelo SIN entrenar (no llames a fit)"
    nombres = [t[0] for t in pre.transformers]
    assert nombres == ["onehot", "ordinal", "num"], f"Nombres esperados ['onehot', 'ordinal', 'num']; obtuve {nombres}"
    tr = {n: (t, cols) for n, t, cols in pre.transformers}
    assert isinstance(tr["onehot"][0], OneHotEncoder) and tr["onehot"][0].handle_unknown == "ignore", (
        "El transformador 'onehot' debe ser OneHotEncoder(handle_unknown='ignore')"
    )
    assert isinstance(tr["ordinal"][0], OrdinalEncoder), "El transformador 'ordinal' debe ser OrdinalEncoder()"
    assert isinstance(tr["num"][0], PowerTransformer), "Por defecto (escalador='power') 'num' es PowerTransformer()"
    assert list(tr["num"][1]) == ["c", "d"] and list(tr["onehot"][1]) == ["a"], "Cada transformador con SUS columnas"
    assert isinstance(ej.construir_preprocesador([], [], ["c"], "standard").transformers[2][1], StandardScaler)
    assert isinstance(ej.construir_preprocesador([], [], ["c"], "minmax").transformers[2][1], MinMaxScaler)
    with pytest.raises(ValueError):
        ej.construir_preprocesador(["a"], ["b"], ["c"], "robusto")


@pytest.mark.puntos(6)
def test_preprocesador_viviendas(viviendas):
    datos = viviendas.dropna(subset=["estado"])
    pre = ej.construir_preprocesador(["distrito", "tipo_inmueble"], ["estado"], ["area_m2", "habitaciones"])
    X = pre.fit_transform(datos)
    assert isinstance(X, np.ndarray), "La salida debe ser un array denso (usa sparse_threshold=0)"
    assert X.shape == (len(datos), 15 + 3 + 1 + 2), (
        "Columnas esperadas: 15 distritos + 3 tipos (onehot) + 1 (ordinal) + 2 numéricas = 21; "
        f"obtuve {X.shape[1]}. Las demás columnas (precio_usd, id...) se descartan"
    )
    nombres = list(pre.get_feature_names_out())
    assert nombres[0] == "onehot__distrito_Ate" and nombres[-1] == "num__habitaciones", (
        f"Nombres de salida inesperados: primero {nombres[0]}, último {nombres[-1]}"
    )
    assert "ordinal__estado" in nombres
    nuevo = datos.head(1).copy()
    nuevo["distrito"] = "Callao"
    fila = pre.transform(nuevo)
    assert fila[0, :15].sum() == 0, "Un distrito nunca visto (Callao) debe quedar con todos los OneHot en 0 (handle_unknown='ignore')"


@pytest.mark.puntos(4)
def test_preprocesador_cambiar_escalador_no_afecta_categoricas(viviendas):
    datos = viviendas.dropna(subset=["estado"])
    args = (["tipo_inmueble"], ["estado"], ["area_m2", "precio_usd"])
    X_pow = ej.construir_preprocesador(*args, escalador="power").fit_transform(datos)
    X_std = ej.construir_preprocesador(*args, escalador="standard").fit_transform(datos)
    X_mm = ej.construir_preprocesador(*args, escalador="minmax").fit_transform(datos)
    assert np.array_equal(X_pow[:, :4], X_std[:, :4]), (
        "Cambiar el transformador numérico NO debe cambiar las columnas categóricas (las 4 primeras)"
    )
    assert np.allclose(X_std[:, 4:].mean(axis=0), 0) and np.allclose(X_std[:, 4:].std(axis=0), 1), (
        "Con escalador='standard' las numéricas deben quedar con media 0 y desviación 1"
    )
    assert np.allclose(X_mm[:, 4:].min(axis=0), 0) and np.allclose(X_mm[:, 4:].max(axis=0), 1), (
        "Con escalador='minmax' las numéricas deben quedar entre 0 y 1"
    )
