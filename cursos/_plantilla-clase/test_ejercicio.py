"""Evaluador automático — Clase XX. NO MODIFICAR. Los puntos deben sumar 100."""
import pandas as pd
import pytest

from framework.testing import cargar_ejercicio, leer_dataset

ej = cargar_ejercicio(__file__)


@pytest.mark.puntos(100)
def test_funcion_ejemplo():
    assert ej.funcion_ejemplo(pd.DataFrame({"a": [1, 2]})) == 2, "Debe devolver el número de filas"
