"""
Track SQL · Clase 01 · Consultas fundamentales
===============================================
Cada función devuelve un STRING con una consulta SQL (dialecto SQLite).
La tabla se llama `viviendas` y tiene las mismas columnas que data/raw/viviendas_lima.csv.

Evalúa con:
    make evaluar CLASE=cursos/tracks/sql/m1-fundamentos/clase-01-consultas

Prueba tus consultas antes (ver notas.md, sección "Practicar en vivo").
"""


def consulta_conteo_por_distrito() -> str:
    """Cantidad de inmuebles por distrito.

    Columnas: distrito, cantidad
    Orden: cantidad de mayor a menor; si empatan, distrito alfabético.
    """
    raise NotImplementedError


def consulta_precio_promedio_por_tipo() -> str:
    """Precio promedio por tipo de inmueble, redondeado a 0 decimales.

    Columnas: tipo_inmueble, precio_promedio
    Orden: precio_promedio de mayor a menor.
    💡 ROUND(valor, 0)
    """
    raise NotImplementedError


def consulta_faltantes() -> str:
    """Una sola fila con la cantidad de NULL en antiguedad_anios y en piso.

    Columnas: faltantes_antiguedad, faltantes_piso
    💡 COUNT(*) vs COUNT(columna)
    """
    raise NotImplementedError


def consulta_top_departamentos_caros(n: int = 5) -> str:
    """Los `n` DEPARTAMENTOS más caros.

    Columnas: id, distrito, area_m2, precio_usd
    Orden: precio_usd de mayor a menor; si empatan, id ascendente.
    (La consulta debe usar el valor de `n` que recibe la función.)
    """
    raise NotImplementedError


def consulta_distritos_caros(umbral: float = 200000) -> str:
    """Distritos cuyo precio PROMEDIO supera `umbral`, contando solo inmuebles con cochera = 'Si'.

    Columnas: distrito, precio_promedio (redondeado a 0 decimales), cantidad
    Orden: precio_promedio de mayor a menor.
    💡 ¿El filtro de cochera va en WHERE o en HAVING? ¿Y el del promedio?
    """
    raise NotImplementedError
