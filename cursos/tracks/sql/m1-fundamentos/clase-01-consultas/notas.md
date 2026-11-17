# Track SQL · Clase 01 · Consultas fundamentales

## Orden en que se ESCRIBE y en que se EJECUTA una consulta
```sql
SELECT   distrito, AVG(precio_usd) AS precio_promedio   -- 5. qué columnas devuelvo
FROM     viviendas                                       -- 1. de qué tabla
WHERE    tipo_inmueble = 'Departamento'                  -- 2. filtro FILAS
GROUP BY distrito                                        -- 3. agrupo
HAVING   AVG(precio_usd) > 200000                        -- 4. filtro GRUPOS
ORDER BY precio_promedio DESC                            -- 6. ordeno
LIMIT    5;                                              -- 7. corto
```
🔑 `WHERE` filtra filas **antes** de agrupar y `HAVING` filtra grupos **después**.

## pandas ↔ SQL
| pandas | SQL |
|---|---|
| `df[df.tipo == "Casa"]` | `WHERE tipo = 'Casa'` |
| `df.groupby("d")["p"].mean()` | `SELECT d, AVG(p) ... GROUP BY d` |
| `df["x"].isna().sum()` | `SUM(CASE WHEN x IS NULL THEN 1 ELSE 0 END)` o `COUNT(*) - COUNT(x)` |
| `df.nlargest(5, "p")` | `ORDER BY p DESC LIMIT 5` |
| `df["d"].nunique()` | `COUNT(DISTINCT d)` |

⚠️ En SQL, `NULL = NULL` **no es verdadero**. Para buscar nulos se usa `IS NULL`.
⚠️ `COUNT(columna)` no cuenta los NULL, pero `COUNT(*)` sí.

## Practicar en vivo
```python
from framework.testing import conexion_sqlite
import pandas as pd
con = conexion_sqlite(viviendas="viviendas_lima.csv")
pd.read_sql("SELECT * FROM viviendas LIMIT 5", con)
```
