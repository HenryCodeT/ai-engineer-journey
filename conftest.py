"""Configuración global de pytest: marcador de puntos y acceso a los datos."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))


def pytest_configure(config):
    config.addinivalue_line("markers", "puntos(n): peso del test en la nota (la suma de una clase = 100)")
