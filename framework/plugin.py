"""Plugin de pytest que calcula una nota 0-100 según el marcador @pytest.mark.puntos(n)."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

PUNTOS_POR_DEFECTO = 10


@dataclass
class ResultadoTest:
    nombre: str
    puntos: int
    aprobado: bool
    mensaje: str = ""


@dataclass(eq=False)  # eq=False mantiene el objeto hasheable: pytest>=9 lo exige a los plugins
class Calificador:
    resultados: list[ResultadoTest] = field(default_factory=list)

    @staticmethod
    def _puntos(item: pytest.Item) -> int:
        marca = item.get_closest_marker("puntos")
        return int(marca.args[0]) if marca and marca.args else PUNTOS_POR_DEFECTO

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        reporte = outcome.get_result()
        # Se cuenta en la fase "call", o en "setup" si falló ahí (p. ej. un fixture).
        if reporte.when == "call" or (reporte.when == "setup" and reporte.failed):
            mensaje = ""
            if reporte.failed:
                mensaje = _resumir_error(call.excinfo)
            self.resultados.append(
                ResultadoTest(item.name, self._puntos(item), reporte.passed, mensaje)
            )

    @property
    def total(self) -> int:
        return sum(r.puntos for r in self.resultados)

    @property
    def obtenidos(self) -> int:
        return sum(r.puntos for r in self.resultados if r.aprobado)

    @property
    def nota(self) -> float:
        return round(100 * self.obtenidos / self.total, 1) if self.total else 0.0


def _resumir_error(excinfo) -> str:
    if excinfo is None:
        return ""
    if excinfo.errisinstance(NotImplementedError):
        return "Función aún no implementada (NotImplementedError)."
    texto = str(excinfo.value).strip().splitlines()
    return (texto[0] if texto else excinfo.typename)[:200]
