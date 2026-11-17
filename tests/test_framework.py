"""Tests del propio framework (make test-framework)."""
import contextlib
import io
import textwrap

import pytest

from data.generar_datos import generar_viviendas
from framework import progreso
from framework.plugin import Calificador


def test_calificador_suma_puntos(tmp_path):
    archivo = tmp_path / "test_x.py"
    archivo.write_text(textwrap.dedent("""
        import pytest
        @pytest.mark.puntos(70)
        def test_ok(): assert True
        @pytest.mark.puntos(30)
        def test_falla(): raise NotImplementedError
    """))
    cal = Calificador()
    with contextlib.redirect_stdout(io.StringIO()):
        pytest.main([str(archivo), "-q", "-p", "no:cacheprovider", "--rootdir", str(tmp_path)], plugins=[cal])
    assert cal.nota == 70.0
    assert "no implementada" in cal.resultados[1].mensaje


def test_registrar_progreso(tmp_path, monkeypatch):
    monkeypatch.setattr(progreso, "ARCHIVO", tmp_path / "progreso.json")
    progreso.registrar("c/m/clase", 50, [{"nombre": "t1", "aprobado": False}])
    clase = progreso.registrar("c/m/clase", 80, [])
    assert clase["intentos"] == 2 and clase["mejor_nota"] == 80 and clase["aprobada"]


def test_dataset_reproducible():
    a, b = generar_viviendas(), generar_viviendas()
    assert a.equals(b)
    assert a.duplicated().sum() == 15
