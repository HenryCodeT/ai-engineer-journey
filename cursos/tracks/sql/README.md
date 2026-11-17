# 🗄️ Track SQL
El profesor lo dijo en la clase 01: *"manejen su SQL"*. Además es requisito del programa, y se usa en el Curso 1 (M3 y M4) y en el Curso 3 (SQL tools para agentes).

Motor: **SQLite** (viene con Python, no hay que instalar nada). Los ejercicios cargan los CSV de `data/` como tablas.
Cada función de `ejercicio.py` **devuelve un string con la consulta SQL**, y el evaluador la ejecuta y compara el resultado.

| # | Clase | Carpeta | Estado |
|---|---|---|---|
| 01 | SELECT, WHERE, GROUP BY, HAVING, ORDER BY | `m1-fundamentos/clase-01-consultas` | ✅ |
| 02 | JOINs (viviendas + tabla de distritos) | ⏳ | |
| 03 | Subconsultas y CTEs (`WITH`) | ⏳ | |
| 04 | Funciones de ventana (`ROW_NUMBER`, `RANK`, `AVG() OVER`) | ⏳ | |
| 05 | SQL para preparar datos de ML (faltantes, outliers, features) | ⏳ | |
| 06 | PostgreSQL + Python (`psycopg`), en el Curso 1 | ⏳ | |

Pide las siguientes con: `/clase` diciendo "genera la clase 02 del track SQL: JOINs".
