"""CLI:  python -m framework evaluar <ruta-clase>  |  python -m framework progreso"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

import pytest

from . import progreso
from .plugin import Calificador

VERDE, ROJO, AMARILLO, GRIS, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[0m"


def _barra(nota: float, ancho: int = 25) -> str:
    llenos = int(round(nota / 100 * ancho))
    color = VERDE if nota >= progreso.NOTA_APROBATORIA else (AMARILLO if nota >= 40 else ROJO)
    return f"{color}{'█' * llenos}{GRIS}{'░' * (ancho - llenos)}{RESET}"


def _clave(ruta: Path) -> str:
    return ruta.resolve().relative_to(progreso.RAIZ / "cursos").as_posix()


def evaluar(ruta_txt: str, verbose: bool = False) -> int:
    ruta = Path(ruta_txt)
    test = ruta / "test_ejercicio.py"
    if not test.exists():
        print(f"{ROJO}No encontré {test}{RESET}")
        return 2

    calificador = Calificador()
    args = [str(test), "-p", "no:cacheprovider", "-q", "--no-header", "--tb=short" if verbose else "--tb=no"]
    if verbose:
        codigo = pytest.main(args, plugins=[calificador])
    else:
        with open(os.devnull, "w") as nulo, contextlib.redirect_stdout(nulo):
            codigo = pytest.main(args, plugins=[calificador])
    if not calificador.resultados:
        print(f"{ROJO}No se pudo ejecutar el evaluador (código pytest {int(codigo)}).{RESET}")
        print(f"Prueba con detalle: python -m framework evaluar {ruta_txt} -v")
        return 1

    print(f"\n📝 {_clave(ruta)}\n")
    for r in calificador.resultados:
        icono = f"{VERDE}✔{RESET}" if r.aprobado else f"{ROJO}✘{RESET}"
        linea = f"  {icono} {r.nombre:<48} {r.puntos:>3} pts"
        if not r.aprobado and r.mensaje:
            linea += f"\n      {GRIS}↳ {r.mensaje}{RESET}"
        print(linea)

    nota = calificador.nota
    detalle = [r.__dict__ for r in calificador.resultados]
    clase = progreso.registrar(_clave(ruta), nota, detalle)
    estado = f"{VERDE}APROBADO{RESET}" if nota >= progreso.NOTA_APROBATORIA else f"{AMARILLO}SIGUE INTENTANDO{RESET}"
    print(f"\n  Nota: {_barra(nota)} {nota:5.1f}/100  {estado}")
    print(f"  {GRIS}Intento #{clase['intentos']} · mejor nota {clase['mejor_nota']}{RESET}\n")
    if nota < progreso.NOTA_APROBATORIA:
        print(f"  💡 Atascado? En Claude Code: /tutor <tu duda>   ·   detalle: make evaluar-v CLASE={ruta_txt}\n")
    return 0


def mostrar_progreso() -> int:
    datos = progreso.cargar()["clases"]
    clases = progreso.clases_disponibles()
    print("\n📊 Progreso — AI Engineer Journey\n")
    grupo_actual = None
    for ruta in clases:
        clave = _clave(ruta)
        grupo, clase_nombre = clave.rsplit("/", 1)
        if grupo != grupo_actual:
            print(f"  {grupo}")
            grupo_actual = grupo
        info = datos.get(clave)
        if info:
            nota = info["mejor_nota"]
            print(f"    {_barra(nota, 15)} {nota:5.1f}  {clase_nombre}")
        else:
            print(f"    {GRIS}{'░' * 15}   ---  {clase_nombre} (pendiente){RESET}")
    aprobadas = sum(1 for c in clases if datos.get(_clave(c), {}).get("aprobada"))
    print(f"\n  ✅ {aprobadas}/{len(clases)} clases aprobadas (nota ≥ {progreso.NOTA_APROBATORIA})\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="framework", description="Evaluador del AI Engineer Journey")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_eval = sub.add_parser("evaluar", help="Calificar una clase")
    p_eval.add_argument("ruta")
    p_eval.add_argument("-v", "--verbose", action="store_true", help="Mostrar traceback de pytest")
    sub.add_parser("progreso", help="Ver avance")
    args = parser.parse_args(argv)
    if args.cmd == "evaluar":
        return evaluar(args.ruta, args.verbose)
    return mostrar_progreso()


if __name__ == "__main__":
    sys.exit(main())
