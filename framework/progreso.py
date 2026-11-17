"""Lectura/escritura de progreso.json."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ARCHIVO = RAIZ / "progreso.json"
NOTA_APROBATORIA = 70


def cargar() -> dict:
    if ARCHIVO.exists():
        return json.loads(ARCHIVO.read_text(encoding="utf-8"))
    return {"clases": {}}


def guardar(datos: dict) -> None:
    ARCHIVO.write_text(json.dumps(datos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def registrar(clave: str, nota: float, detalle: list[dict]) -> dict:
    datos = cargar()
    clase = datos["clases"].setdefault(clave, {"intentos": 0, "mejor_nota": 0, "historial": []})
    clase["intentos"] += 1
    clase["mejor_nota"] = max(clase["mejor_nota"], nota)
    clase["ultima_nota"] = nota
    clase["aprobada"] = clase["mejor_nota"] >= NOTA_APROBATORIA
    clase["historial"].append({"fecha": datetime.now().isoformat(timespec="seconds"), "nota": nota})
    clase["historial"] = clase["historial"][-20:]
    clase["tests_fallidos"] = [d["nombre"] for d in detalle if not d["aprobado"]]
    guardar(datos)
    return clase


def clases_disponibles() -> list[Path]:
    """Todas las carpetas de clase que tienen un evaluador."""
    return sorted(
        p.parent for p in (RAIZ / "cursos").rglob("test_ejercicio.py")
        if "_plantilla-clase" not in p.parts
    )
