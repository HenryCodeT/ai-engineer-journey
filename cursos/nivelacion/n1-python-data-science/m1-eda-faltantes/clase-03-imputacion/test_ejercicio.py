"""Evaluador automático — Clase 03 · Datos faltantes y métodos de imputación.  NO MODIFICAR.

Cada test tiene un peso (@pytest.mark.puntos). La suma es 100.
"""
import numpy as np
import pandas as pd
import pytest

from framework.testing import cargar_ejercicio, leer_dataset, ruta_dataset

ej = cargar_ejercicio(__file__)


@pytest.fixture
def clientes():
    # Tabla tipo "sistema de clientes": vacíos disfrazados en email y una categórica con huecos.
    return pd.DataFrame({
        "cliente": ["Ana", "Luis", "Rosa", "Juan", "Eva", "Pedro", "Lia"],
        "email": ["ana@x.com", "  ", "N/A", "No reporto", None, "-", "Lia@X.com"],
        "segmento": ["Masivo", np.nan, "Premium", "Masivo", np.nan, "Masivo", "Premium"],
        "edad": [70, 25, 81, 40, 66, 33, 29],
    })


@pytest.fixture
def mini_grupos():
    return pd.DataFrame({
        "zona": ["A", "A", "A", "B", "B", "B", "C", "C"],
        "km": [1.0, 3.0, np.nan, 10.0, 20.0, np.nan, np.nan, np.nan],
        "precio": [5, 6, 7, 8, 9, 10, 11, 12],
    })


@pytest.fixture(scope="module")
def viviendas():
    return leer_dataset("viviendas_lima.csv")


@pytest.fixture(scope="module")
def cochera_cruda():
    # Leída SIN la conversión automática de pandas: así "N/A" queda como texto (vacío disfrazado).
    return pd.read_csv(ruta_dataset("viviendas_lima.csv"), usecols=["id", "cochera"],
                       keep_default_na=False, dtype={"cochera": str})


# ---------------------------------------------------------------- normalizar_vacios (20)
@pytest.mark.puntos(12)
def test_normalizar_vacios_mini(clientes):
    copia = clientes.copy()
    r = ej.normalizar_vacios(clientes, "email")
    pd.testing.assert_frame_equal(clientes, copia, obj="No modifiques el DataFrame de entrada (usa df.copy())")
    assert list(r.columns) == list(clientes.columns), "No agregues ni quites columnas"
    esperado_nan = [False, True, True, True, True, True, False]
    assert r["email"].isna().tolist() == esperado_nan, (
        "Deben quedar como NaN: '  ' (fila 1), 'N/A' (2), 'No reporto' (3), None (4) y '-' (5). "
        f"Tu resultado marca NaN en: {r['email'].isna().tolist()}"
    )
    assert r.loc[0, "email"] == "ana@x.com" and r.loc[6, "email"] == "Lia@X.com", (
        "Los emails válidos no deben cambiar (¿los pasaste a minúsculas? Solo se usa lower() para COMPARAR)"
    )
    assert r["segmento"].isna().sum() == 2, "Solo se limpia la columna pedida: 'segmento' debe seguir igual"
    r2 = ej.normalizar_vacios(clientes, "email", marcadores=("n/a",))
    assert r2["email"].isna().tolist() == [False, False, True, False, True, False, False], (
        "Con marcadores=('n/a',) solo 'N/A' (y el None que ya existía) deben ser NaN: ¿usaste el parámetro marcadores?"
    )


@pytest.mark.puntos(8)
def test_normalizar_vacios_viviendas(cochera_cruda):
    assert cochera_cruda["cochera"].isna().sum() == 0, "(control) la cochera cruda no debería tener NaN"
    r = ej.normalizar_vacios(cochera_cruda, "cochera")
    assert r["cochera"].isna().sum() == 18, (
        f"En cochera hay 18 'N/A' escritos como texto: deben quedar 18 NaN. Obtuve {r['cochera'].isna().sum()}"
    )
    assert (r["cochera"] == "no ").sum() == 18, (
        "'no ' (con espacio) NO es un vacío: es un 'No' mal escrito y debe quedar tal cual"
    )
    assert (r["cochera"] == "Si").sum() == 1098, "Los valores válidos ('Si') no deben cambiar"


# ---------------------------------------------------------------- imputar_categorica (20)
@pytest.mark.puntos(12)
def test_imputar_categorica_mini(clientes):
    copia = clientes.copy()
    moda = ej.imputar_categorica(clientes, "segmento")
    pd.testing.assert_frame_equal(clientes, copia, obj="No modifiques el DataFrame de entrada (usa df.copy())")
    assert moda["segmento"].tolist() == ["Masivo", "Masivo", "Premium", "Masivo", "Masivo", "Masivo", "Premium"], (
        "Con estrategia='moda' los huecos (filas 1 y 4) deben valer 'Masivo' (aparece 3 veces, Premium 2)"
    )
    const = ej.imputar_categorica(clientes, "segmento", estrategia="constante")
    assert const.loc[[1, 4], "segmento"].tolist() == ["Desconocido", "Desconocido"], (
        "Con estrategia='constante' los huecos deben valer 'Desconocido' (el valor por defecto)"
    )
    otro = ej.imputar_categorica(clientes, "segmento", estrategia="constante", valor="No reportó")
    assert otro.loc[1, "segmento"] == "No reportó", "Con valor='No reportó' el hueco debe valer 'No reportó'"
    assert otro.loc[0, "segmento"] == "Masivo", "Los valores que ya existían no deben cambiar"
    with pytest.raises(ValueError):
        ej.imputar_categorica(clientes, "segmento", estrategia="media")


@pytest.mark.puntos(8)
def test_imputar_categorica_viviendas(viviendas):
    moda = ej.imputar_categorica(viviendas, "estado")
    assert moda["estado"].isna().sum() == 0, "'estado' no debe quedar con NaN"
    assert (moda["estado"] == "Bueno").sum() == 1071 + 73, (
        "La moda de 'estado' es 'Bueno' (1071): con los 73 huecos imputados debe haber 1144 'Bueno'"
    )
    const = ej.imputar_categorica(viviendas, "estado", estrategia="constante")
    assert (const["estado"] == "Desconocido").sum() == 73, "Deben aparecer exactamente 73 'Desconocido'"
    assert viviendas["estado"].isna().sum() == 73, "El DataFrame original debe conservar sus 73 NaN"
    assert const["cochera"].isna().sum() == viviendas["cochera"].isna().sum(), "No toques otras columnas"


# ---------------------------------------------------------------- imputar_mediana_por_grupo (25)
@pytest.mark.puntos(13)
def test_mediana_grupo_mini(mini_grupos):
    copia = mini_grupos.copy()
    r = ej.imputar_mediana_por_grupo(mini_grupos, "km", "zona")
    pd.testing.assert_frame_equal(mini_grupos, copia, obj="No modifiques el DataFrame de entrada (usa df.copy())")
    assert r["km"].isna().sum() == 0, "Después de imputar, 'km' no debe tener NaN (ni siquiera en la zona C)"
    assert r.loc[2, "km"] == pytest.approx(2.0), (
        f"Fila 2 (zona A): la mediana de A es 2.0 (de 1 y 3). Obtuve {r.loc[2, 'km']}"
    )
    assert r.loc[5, "km"] == pytest.approx(15.0), (
        f"Fila 5 (zona B): la mediana de B es 15.0 (de 10 y 20), NO la global. Obtuve {r.loc[5, 'km']}"
    )
    assert r.loc[6, "km"] == pytest.approx(6.5) and r.loc[7, "km"] == pytest.approx(6.5), (
        f"Zona C no tiene datos: usa la mediana GLOBAL de km = mediana([1, 3, 10, 20]) = 6.5. Obtuve {r.loc[6, 'km']}"
    )
    assert r.loc[[0, 1, 3, 4], "km"].tolist() == [1.0, 3.0, 10.0, 20.0], "Los valores que ya existían no deben cambiar"


@pytest.mark.puntos(12)
def test_mediana_grupo_viviendas(viviendas):
    faltaba = viviendas["distancia_centro_km"].isna()
    r = ej.imputar_mediana_por_grupo(viviendas, "distancia_centro_km", "distrito")
    assert r["distancia_centro_km"].isna().sum() == 0, "distancia_centro_km no debe quedar con NaN"
    ate = faltaba & (viviendas["distrito"] == "Ate")
    assert ate.sum() > 0 and np.allclose(r.loc[ate, "distancia_centro_km"], 14.805), (
        "Los huecos de Ate deben valer 14.805 (mediana de distancia en Ate)"
    )
    sjl = faltaba & (viviendas["distrito"] == "SJL")
    assert sjl.sum() > 0 and np.allclose(r.loc[sjl, "distancia_centro_km"], 13.05), "Los huecos de SJL deben valer 13.05"
    global_ = viviendas["distancia_centro_km"].median()
    assert r.loc[faltaba, "distancia_centro_km"].mean() > global_ + 2, (
        "Los huecos están en distritos lejanos: su valor imputado medio debe superar en más de 2 km "
        f"a la mediana global ({global_:.2f}). ¿Usaste la mediana global en lugar de la del grupo?"
    )
    assert viviendas["distancia_centro_km"].isna().sum() == 158, "El original debe conservar sus 158 NaN"


# ---------------------------------------------------------------- nulos_por_join (15)
@pytest.mark.puntos(7)
def test_join_mini():
    casas = pd.DataFrame({"distrito": ["Lince", "Ate", "Lince", "SJL", "SJL", "Surco"], "precio": range(6)})
    info = pd.DataFrame({"distrito": ["Lince", "Ate", "Barranco"], "zona": ["Centro", "Este", "Sur"]})
    copia = casas.copy()
    r = ej.nulos_por_join(casas, info, "distrito")
    pd.testing.assert_frame_equal(casas, copia, obj="No modifiques las tablas de entrada")
    assert r == {"filas_inner": 3, "filas_left": 6, "filas_sin_pareja": 3}, (
        "Lince(2) y Ate(1) tienen pareja -> inner 3; el left conserva las 6 filas y "
        f"SJL(2) + Surco(1) quedan sin pareja -> 3. Obtuve {r}"
    )
    assert all(type(v) is int for v in r.values()), "Los tres valores deben ser int de Python (usa int(...))"


@pytest.mark.puntos(8)
def test_join_viviendas(viviendas):
    # Tabla de distritos "incompleta": alguien olvidó cargar SJL y Ate.
    conocidos = sorted(set(viviendas["distrito"]) - {"SJL", "Ate"})
    info = pd.DataFrame({"distrito": conocidos, "codigo": range(len(conocidos))})
    r = ej.nulos_por_join(viviendas, info, "distrito")
    n_sjl_ate = int(viviendas["distrito"].isin(["SJL", "Ate"]).sum())
    assert r["filas_left"] == len(viviendas) == 2015, "El LEFT JOIN conserva todas las filas de la izquierda (2015)"
    assert r["filas_sin_pareja"] == n_sjl_ate == 292, (
        f"Las viviendas de SJL (167) y Ate (125) no tienen pareja: 292 filas con NaN fabricados. Obtuve {r['filas_sin_pareja']}"
    )
    assert r["filas_inner"] == 2015 - 292, "El INNER JOIN pierde esas 292 filas: deben quedar 1723"


# ---------------------------------------------------------------- impacto_imputacion (20)
@pytest.mark.puntos(10)
def test_impacto_mini():
    antes = pd.Series([2.0, 4.0, np.nan, 6.0])
    copia = antes.copy()
    r = ej.impacto_imputacion(antes, pd.Series([2.0, 4.0, 4.0, 6.0]))
    pd.testing.assert_series_equal(antes, copia, obj="No modifiques la Serie de entrada")
    assert set(r) == {"media_antes", "media_despues", "std_antes", "std_despues", "cambio_std_pct"}, (
        f"Claves esperadas: media_antes, media_despues, std_antes, std_despues, cambio_std_pct. Obtuve {sorted(r)}"
    )
    assert r["media_antes"] == 4.0 and r["media_despues"] == 4.0, "Imputar con la media no mueve la media (4.0)"
    assert r["std_antes"] == pytest.approx(2.0), f"std de [2, 4, 6] (pandas, ddof=1) es 2.0. Obtuve {r['std_antes']}"
    assert r["std_despues"] == pytest.approx(1.63), f"std de [2, 4, 4, 6] es 1.63. Obtuve {r['std_despues']}"
    assert r["cambio_std_pct"] == pytest.approx(-18.35), (
        "cambio_std_pct = (1.633 - 2.0) / 2.0 * 100 = -18.35 (calcúlalo con valores sin redondear). "
        f"Obtuve {r['cambio_std_pct']}"
    )
    assert all(isinstance(v, float) for v in r.values()), "Todos los valores deben ser float de Python"


@pytest.mark.puntos(10)
def test_impacto_viviendas(viviendas):
    ant = viviendas["antiguedad_anios"]
    con_mediana = ej.impacto_imputacion(ant, ant.fillna(ant.median()))
    assert con_mediana["std_antes"] == pytest.approx(10.09) and con_mediana["std_despues"] == pytest.approx(9.67), (
        f"antiguedad_anios: std antes 10.09 y después de imputar con la mediana 9.67. Obtuve {con_mediana}"
    )
    assert con_mediana["cambio_std_pct"] == pytest.approx(-4.11, abs=0.011), (
        "Imputar 170 huecos con el mismo valor aplana la variable: la std baja ≈ 4.11 %"
    )
    con_cero = ej.impacto_imputacion(ant, ant.fillna(0))
    assert con_cero["media_despues"] < con_cero["media_antes"] - 1, (
        "Rellenar con 0 (¡viviendas recién construidas!) hunde la media más de 1 año: eso es sesgo. "
        f"Obtuve antes={con_cero['media_antes']} y después={con_cero['media_despues']}"
    )
    assert con_cero["media_antes"] == pytest.approx(12.26), "La media de antiguedad_anios (sin NaN) es 12.26"
