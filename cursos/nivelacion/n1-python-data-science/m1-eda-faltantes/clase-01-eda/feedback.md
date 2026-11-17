# Feedback · Clase 01 · Análisis Exploratorio de Datos (EDA)
**Nota automática:** 100/100 (12/12 tests)   ·   **Rúbrica:** C 4/4 · Cl 3/4 · BP 4/4 · Co 3/4

> Nota: la nota sale de correr `pytest` directamente (12 passed, suman 100 puntos). `python -m framework evaluar` falló por un error interno del framework (no de tu código), así que esta clase **todavía no quedó registrada en `progreso.json`**. Se registrará cuando se arregle el framework.

## ✅ Lo que hiciste bien
- **Todo vectorizado, sin un solo `for`.** `df.isna().sum().sum()` (`ejercicio.py:43`) y `df.isna().sum()` en `tabla_faltantes` (`ejercicio.py:137`) son justo cómo se trabaja en pandas: una operación sobre toda la tabla de una vez.
- **No mutas la entrada.** En `clasificar_variables` usaste `df.drop(columns=excluir)` (`ejercicio.py:70`), que devuelve un DataFrame **nuevo**. Te costó llegar a `columns=` (sin él, `drop` busca en las *filas*), pero quedó bien, y además manejaste el caso `excluir=None`.
- **Encadenaste pasos con criterio** en `tabla_faltantes` (`ejercicio.py:137-142`): contar, calcular porcentaje, armar la tabla, filtrar con máscara booleana y ordenar. Cada línea hace una sola cosa y se lee de arriba abajo.
- **`value_counts(dropna=False, normalize=normalizar)`** (`ejercicio.py:113`): pasaste el parámetro de la función directo al de pandas. Es limpio y muestra que entendiste que `dropna=False` significa "**no** descartes los NaN" (la lógica negativa que te costaba).
- **Leer vs asignar `.columns`**: en `ejercicio.py:72` *lees* `.columns` para sacar nombres, y en `ejercicio.py:93` y `:169` las *asignas* para renombrar. Usaste bien las dos caras.

## 🔧 A mejorar (con línea de código)
- `ejercicio.py:19-20` — `from pandas._libs.hashtable import value_count` y `from pandas.core import series` no se usan. Seguro los agregó el autocompletado del editor. Bórralos: además de sobrar, `_libs` es una parte **privada** de pandas (el `_` lo indica) y puede romperse entre versiones. Regla práctica: solo `import pandas as pd`, y todo se usa desde `pd.` o desde el `df`.
- `ejercicio.py:169-170` — quedan los comentarios de las pistas (`# paso 2: ¿qué 3 nombres...?`, `# paso 3: ...`). Eran andamio para resolverlo. Cámbialos por un comentario que diga el **por qué** con tus palabras (p. ej., por qué ordenar por la media ayuda a ver si la zona diferencia el precio) o quítalos.
- `ejercicio.py:92` y `:168` — la variable `table` está en inglés, mientras el resto del archivo usa español (`conteo`, `porcentaje`, `tabla`). En `tabla_faltantes` ya usas `tabla`: elige un idioma y mantenlo en todo el archivo. Lo mismo con `count` en `ejercicio.py:113`, que además se parece a los métodos `.count()` de pandas y confunde al leer.
- `ejercicio.py:42-43` — el docstring pide `int`, pero `.sum()` devuelve un entero de NumPy (`numpy.int64`). Los tests lo aceptan, pero si un día guardas este diccionario en JSON (por ejemplo, en una API del Curso 1) va a fallar. Pista: ¿qué función de Python convierte un número a `int` "normal"?
- `ejercicio.py:138` — piensa qué pasa si `df` tiene 0 filas: `len(df)` vale 0 y estarías dividiendo entre cero. Los tests no lo cubren, pero es un caso borde que vale la pena tener en mente.

## 🔁 Conceptos para repasar (te costaron, ya los resolviste, conviene que se asienten)
| Concepto | Regla corta | Dónde aparece en tu código |
|---|---|---|
| `( )` vs `[ ]` | `( )` **llama** a algo (método/función); `[ ]` **selecciona** (columna, filas, posición) | `df[columna].value_counts(...)` (`:113`) |
| método vs función suelta | `df.isna()` es del objeto; `len(df)` es una función de Python que recibe el objeto | `:137` vs `:138` |
| `pd` vs `df` | `pd` es la librería (crear cosas: `pd.DataFrame(...)`); `df` es *tus datos* (operar sobre ellos) | `:139` |
| `df.count()` vs `len(df)` | `len(df)` = total de filas; `df.count()` = valores **no nulos** por columna. Para un % de faltantes el denominador es `len(df)` | `:138` |
| groupby: entrada vs salida | `columna` y `objetivo` son nombres que **ya existen** en el df; `"cantidad"`, `"media_objetivo"`... son nombres **nuevos** que tú pones al resultado | `:168-169` |

## 🧠 Pregunta de comprensión
En `bivariante_categorica` usaste `agg(["count", "mean", "median"])`. En `viviendas_lima.csv` el precio tiene asimetría positiva (algunas casas carísimas). Si en una zona la **media** del precio es mucho mayor que la **mediana**, ¿qué te dice eso de esa zona? Y si una zona tiene `cantidad = 3`, ¿te fiarías de su media para decidir que esa variable "ayuda al modelo"? ¿Por qué?

## 📌 Siguiente paso
Dominio alcanzado (≥ 70). Según las dependencias, sigue con **N1 · M1 · clase 02 (Datos faltantes e imputación)**: tu `tabla_faltantes` es justo el punto de partida de ese tema, y M1–M2 son prioridad porque los usa el Curso 1 en su Módulo 4 (🔗). Cuando tengas el transcript, genérala con `/clase transcripts/n1-python/clase-02.txt`. Antes, limpia los imports y los comentarios señalados arriba (2 minutos) para que el archivo quede como referencia tuya.
