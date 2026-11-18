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


def generar_progreso_md() -> str:
    """Genera un PROGRESO.md visual desde progreso.json."""
    datos = cargar()
    clases_data = datos.get("clases", {})
    clases = clases_disponibles()

    lineas = ["# 📊 Mi Progreso · AI Engineer Journey\n"]
    lineas.append("Marca tu avance aquí. Actualizado automáticamente con `make progreso`.\n")

    # Agrupar por curso
    grupos = {}
    for ruta in clases:
        try:
            clave = ruta.relative_to(RAIZ / "cursos").as_posix()
            partes = clave.split("/")
            if len(partes) >= 2:
                grupo = "/".join(partes[:2])  # ej: nivelacion/n1-python-data-science
                clase = partes[-1]  # ej: clase-01-eda
                grupos.setdefault(grupo, []).append((clave, clase))
        except ValueError:
            continue

    # Estadísticas globales
    total = len(clases)
    aprobadas = sum(1 for c in clases if clases_data.get(
        c.relative_to(RAIZ / "cursos").as_posix(), {}
    ).get("aprobada", False))
    pendientes = total - aprobadas

    lineas.append(f"## 🎯 Resumen\n")
    lineas.append(f"- ✅ **Aprobadas:** {aprobadas}/{total}")
    lineas.append(f"- ⬜ **Pendientes:** {pendientes}")
    if aprobadas > 0:
        progreso_pct = int(aprobadas / total * 100)
        barra_llena = "█" * (progreso_pct // 5)
        barra_vacia = "░" * (20 - progreso_pct // 5)
        lineas.append(f"- 📈 **Progreso:** {barra_llena}{barra_vacia} {progreso_pct}%")
    lineas.append("")

    # Detallar por grupo
    for grupo in sorted(grupos.keys()):
        lineas.append(f"## 📚 {grupo.replace('/', ' › ')}\n")
        for clave, nombre_clase in grupos[grupo]:
            info = clases_data.get(clave, {})
            if info.get("aprobada"):
                nota = info["mejor_nota"]
                lineas.append(f"- [x] {nombre_clase} · nota {nota:.0f}/100")
            else:
                lineas.append(f"- [ ] {nombre_clase}")
        lineas.append("")

    # Notas personales
    lineas.append("## 📝 Mis notas y reflexiones\n")
    lineas.append("_(Agrega tus propias notas aquí. Esta sección no se sobreescribe)_\n")
    lineas.append("")

    return "\n".join(lineas)


def actualizar_progreso_md() -> None:
    """Actualiza PROGRESO.md preservando las notas personales."""
    archivo_md = RAIZ / "PROGRESO.md"

    nuevo_contenido = generar_progreso_md()

    # Si existe, preservar la sección de notas personales
    if archivo_md.exists():
        contenido_actual = archivo_md.read_text(encoding="utf-8")
        if "## 📝 Mis notas y reflexiones" in contenido_actual:
            partes = contenido_actual.split("## 📝 Mis notas y reflexiones", 1)
            if len(partes) == 2:
                # Preservar las notas personales del usuario
                notas_personales = partes[1].split("\n", 2)
                if len(notas_personales) > 2:
                    notas_usuario = notas_personales[2]
                    nuevo_contenido = nuevo_contenido.rsplit("_(Agrega tus propias notas aquí", 1)[0]
                    nuevo_contenido += notas_usuario

    archivo_md.write_text(nuevo_contenido, encoding="utf-8")
