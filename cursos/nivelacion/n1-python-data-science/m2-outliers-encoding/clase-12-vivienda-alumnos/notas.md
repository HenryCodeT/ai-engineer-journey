# Clase 12 · Caso de uso Panorama de la vivienda (alumnos)
**Curso:** N1 · Python para Ciencia de Datos · **Módulo 2** (outliers y encoding) · Transcript: `transcripts/n1-python/clase-12.txt`
Título oficial: *Python para Ciencia de Datos - M2-7 - Caso de uso Panorama de la vivienda (Alumnos)*

> Sesión corta (~26 min) de **práctica en grupos y presentación de resultados**. No hay teoría nueva: el profesor reparte
> a los alumnos en salas, les da el dataset de viviendas de **Melbourne** y les pide repetir el análisis de datos faltantes
> de las clases 02–04. Luego dos grupos presentan y el profesor comenta sus hallazgos y responde preguntas.
>
> Esta carpeta tiene **solo notas**: la práctica real de esta clase es la **entrega (a) del proyecto N1** (ver el final).

---

## 0. Mapa de la sesión

| Momento | Qué pasa | Tema que repasa |
|---|---|---|
| 1 | El profe arma 5 salas con al menos una persona con Colab abierto | — |
| 2 | Consigna: analizar nulos del dataset de Melbourne, comparar imputaciones, ver la distribución, clasificar MCAR/MAR/MNAR | Clases 02–04 |
| 3 | **Grupo 1** presenta: % de nulos + correlación de nulidad | Reporte de nulos, `missingno.heatmap` |
| 4 | **Grupo 2** presenta: qué se puede imputar y con qué | Umbrales, KNN vs Iterative |
| 5 | Pregunta: ¿por qué hasta 30 %? → analogía con train/test 70/30 | Conexión con ML |
| 6 | Pregunta: ¿cómo leo la matriz de nulidad y la correlación? | `missingno.matrix`, `heatmap`, `dendrogram` |

---

## 1. La consigna del ejercicio

- Dataset: **Panorama de la vivienda de Melbourne**, un dataset "bastante famoso", con **18 396 registros**. El profe
  deja en Colab un código base (librerías + carga del CSV).
- Lo que pide, con sus palabras resumidas:
  1. Analizar cómo están los **valores nulos** y hacer "el reporte de lo que ya hicimos".
  2. **Comparar los métodos de imputación** y ver cómo queda cada uno.
  3. Mirar la **distribución** de los datos: que no se altere y que **no aparezcan esos picos** que salían al imputar con
     media, mediana o moda.
  4. Identificar qué variables son **MCAR**, cuáles **MAR** y cuáles **MNAR**.
- Lo que viene después (anunciado): primero imputación, luego **tratamiento de outliers**, y recién entonces ver **cómo
  el método de imputación y el de outliers impactan en la performance** de un modelo predictivo (regresión o clasificación).

> 🔗 Si no recuerdas qué es MCAR/MAR/MNAR, repasa la **clase 02**. El reporte de nulos, la correlación de nulidad y
> `KNNImputer`/`IterativeImputer` están en la **clase 04**. El impacto en el modelo (escenarios y MAE) lo ves en las
> **clases 10–11**.

---

## 2. Grupo 1: porcentaje de nulos y correlación de nulidad

**Lo que encontraron:**

| Variable | % de nulos (aprox.) |
|---|---|
| `BuildingArea` | ~57.5 % |
| `YearBuilt` | ~51 % |
| `CouncilArea` | ~33 % |

- En la **matriz de correlación de nulidad** aparecen correlaciones de **~0.8** entre algunas variables.
- Conclusión del profe: *"no es que sea todo totalmente al azar… hay cierto patrón que se está repitiendo en los datos
  faltantes"*. Si los nulos van juntos, **no es MCAR** → el problema de imputar *"ya no quedó tan simple como: tengo
  datos faltantes, mediana, mediana, mediana"*.

### La trampa de la correlación 1 con poquísimos nulos

- `Regionname`, `Distance` y `Propertycount` salían con **correlación de nulidad = 1** entre sí.
- ¿Por qué? Porque cada una tiene **un solo nulo** (≈ 0.005 %) **y es el mismo registro**. Coinciden al 100 % → correlación 1.
- Qué hacer, según el profe: ese único registro se puede **eliminar** "y no les afectaría nada", o imputar con mediana,
  moda o lo que quieran: con un solo valor **no hay efecto**.
- Posibles causas de nulos así: desde un **problema del script** que recolecta los datos hasta errores de **inserción en
  la base de datos**.

> **Idea clave:** una correlación de nulidad alta **no significa** que el problema sea grave. Siempre mírala **junto al
> número de nulos**. Correlación 1 con 1 nulo = irrelevante. Correlación 0.8 con 9 000 nulos = patrón serio (MAR).

**✏️ Mini-pregunta:** si dos columnas tienen 3 nulos cada una, en las mismas 3 filas, ¿qué correlación de nulidad
esperas? ¿Te preocupa?

---

## 3. Grupo 2: ¿qué se puede imputar?

### `BuildingArea` (57 %) y `YearBuilt` (51 %)
- A primera vista **no se pueden imputar**: el porcentaje de nulidad es demasiado alto.
- El grupo notó que `BuildingArea` tiene correlación de nulidad **~0.5** con otra variable y **~0.4** con
  latitud/longitud (que solo tienen **~18 %** de nulos).
- El profe: **si te obligan** a usar esa variable, hay que buscar **estrategias** (incluso revisar **papers** sobre
  imputación con alto % de faltantes), por ejemplo, aprovechar la relación con latitud/longitud.
- Advertencia importante: *"cada estrategia lleva error"*: el error de la estrategia **más** el error de la imputación.
  **Se va acumulando error** y luego hay que **medir su impacto en el modelo final**, porque podría sesgarlo.
- Lo ideal sería revisar el **script** que generó los datos (para entender por qué faltan), pero en este caso no se tiene.

### `CouncilArea` (33 %)
- El alumno lo veía riesgoso. El profe: **hasta ~30 % se puede imputar**, igual que en ML un split **70/30** "es pasable".
- También **depende de cuántos registros** hay: imputar ~6 000 de 18 396 es casi **un tercio**. Se puede intentar,
  pero con margen de error.
- Con un % alto, **cualquier** método simple sesga: media, mediana o moda generan **un pico gigantesco** en la distribución
  (lo que el profe mostró en clases anteriores).

### KNN vs Iterative
- El grupo usó `KNNImputer`, que imputa mirando a los **vecinos cercanos**, y **también puede generar picos**.
- El profe sugiere probar `IterativeImputer`: uno de los métodos **más avanzados**, que trata la imputación **como un
  problema de regresión** (predice la columna faltante a partir de las demás).

**✏️ Mini-pregunta:** ¿por qué imputar el 33 % de una columna con la mediana produce un "pico" en el histograma?

---

## 4. ¿Por qué el 30 %? La analogía con train/test

- En un problema de regresión se suele hacer un split **70/30**; *"ya no se recomienda un 60/40"*. Empíricamente se ha
  visto que con un 70 % de entrenamiento todavía hay suficiente **variabilidad** y "caracterización" de los individuos.
- En papers también se ven esquemas **80/20** y **90/10**, y **validación cruzada** (*k-fold*) con **5 o 10 folds**
  (entrenas con 9 partes y pruebas con 1, o con 4 y 1).
- Con pocos datos, 70/30 ya daría problemas y habría que subir a **80/20**. Con ~18 400 registros, 70/30 deja ~13 000
  para entrenar y ~5 500 para probar: suficiente.
- Aplicado a imputar: **hasta 30 % es bastante aceptable**; **50/50 ya es "jugársela"**; **más de 50 %, mejor retirar
  la variable**.

### Estrategia avanzada mencionada: autoencoders / *teacher-student*
- No es algo nuevo: en aprendizaje supervisado a veces hay datos **sin etiqueta** (técnicamente, "valores vacíos").
- Se entrena una red (un **autoencoder**) con los datos que **sí** están completos, poniéndolos como entrada y salida,
  para que aprenda un **vector representativo**. Luego los datos incompletos pasan por ahí y se **reconstruye** la salida.
- El profe lo llama esquema **teacher-student**: el *teacher* aprende a caracterizar (encoder), el *student* reconstruye.
  Se usó mucho en visión computacional y también sirve para imputación.

> 📌 Extra: esto lo verás de verdad en N3 (Deep Learning). Por ahora basta con saber que existe y que también acumula error.

### Tabla resumen: % de nulos → qué hacer (umbrales del profesor)

| % de nulos | Qué hacer | Ejemplo en Melbourne |
|---|---|---|
| Muy pocos (1 o unos cuantos registros) | Eliminar la fila o imputar con cualquier método: no hay efecto | `Regionname`, `Distance`, `Propertycount` (1 nulo) |
| Hasta ~30 % | Imputable; mejor métodos avanzados (`KNNImputer`, `IterativeImputer`) y revisar que no aparezcan picos | `CouncilArea` (~33 %, en el límite), lat/long (~18 %) |
| ~50 % | Muy arriesgado: "jugársela" | `YearBuilt` (~51 %) |
| > 50 % | Retirar la variable; solo si te obligan, buscar estrategias (papers) y medir el error en el modelo | `BuildingArea` (~57 %) |

> 📌 Extra: son **reglas prácticas**, no leyes. El profe insiste en que también depende del **número de registros** y,
> al final, de cómo le va al **modelo**.

---

## 5. Cómo leer los gráficos de missingno

### La matriz de nulidad (`missingno.matrix`)
- En el eje vertical están los **índices** de las filas (de 1 a 18 396, en el orden original). Cada línea blanca es un nulo.
- Si dos columnas tienen sus nulos en **bloques parejos** (las mismas filas), están correlacionadas.
- **Truco del profe** para comprobar si es MAR: **mezcla (reordena) las filas** y vuelve a graficar. Si los bloques
  correlacionados **siguen apareciendo juntos**, hay relación real (MAR). Un **MCAR**, por más que lo mezcles, **siempre
  se ve completamente aleatorio**.
- En este dataset **no hay un MCAR claro**: todas las variables con nulos muestran correlación. `Landsize` "hasta
  pareciera" MCAR, pero también se relaciona con otras.
- Abajo de la matriz se ve más intersección de nulos (más correlación); arriba, menos.

> 📌 Extra: el reordenamiento en pandas sería `df.sample(frac=1, random_state=42)` antes de `msno.matrix(...)`.

### El heatmap (`missingno.heatmap`) y el dendrograma (`missingno.dendrogram`)
- El heatmap muestra la **correlación entre los nulos** de cada par de columnas. El profe dice que es el que mejor se lee.
- El dendrograma **agrupa variables según su patrón de nulidad**: las que se unen abajo faltan juntas. Ejemplo:
  `Bathroom` y `Bedroom` aparecen juntas.

### Pregunta de un alumno: ¿9 000 nulos y 9 000 nulos = correlación 1?
- **No necesariamente.** Solo da 1 si los nulos están en **los mismos índices**.
- Si son **MCAR** (al azar), coincidirán en algunas filas por casualidad y la correlación saldrá **baja: ~0 a 0.3**,
  según cuántos registros haya en total.
- Si **coinciden mucho** (correlación de 1 o cercana), es señal de **MAR**.

**✏️ Mini-pregunta:** reordenas al azar las filas de Melbourne y los nulos de `BuildingArea` y `YearBuilt` siguen
apareciendo en las mismas filas. ¿Qué tipo de faltante sugiere eso?

---

## 6. Lo que tienes que quedarte

1. Imputar **no es** "pongo la mediana y listo": primero mira **cuánto** falta y **cómo** falta (patrón).
2. Correlación de nulidad alta **con muy pocos nulos** es engañosa; con muchos nulos indica **MAR**.
3. Umbrales del profe: **≤ 30 %** imputable, **~50 %** muy riesgoso, **> 50 %** retirar la variable.
4. Métodos simples con alto % de nulos generan **picos**; `KNNImputer` también puede generarlos; `IterativeImputer`
   (regresión) es más avanzado.
5. **Toda estrategia acumula error**: la decisión final se valida en el **modelo** (lo que hiciste en las clases 10–11 con MAE).

## 📌 Extra: conexión con ML / IA
- Un pico artificial en una variable hace que el modelo aprenda un valor "falso" muy repetido; en regresión esto puede
  sesgar las predicciones de precio.
- Si los nulos son MAR, la información de **otras columnas** ayuda a imputar mejor (por eso KNN e Iterative usan el resto
  de variables). Si fueran MNAR, ni eso basta: el hecho de faltar ya es información (a veces conviene una columna
  indicadora `fue_nulo`).

---

## ❓ Preguntas de repaso

1. En Melbourne, `Regionname`, `Distance` y `Propertycount` tienen correlación de nulidad 1. ¿Por qué no es preocupante y
   qué harías con ese registro?
2. Según los umbrales del profe, ¿qué harías con `BuildingArea` (~57 %), `YearBuilt` (~51 %) y `CouncilArea` (~33 %)?
   Justifica cada una.
3. ¿Qué analogía usa el profe para justificar que hasta un 30 % de nulos es imputable? Explícala con tus palabras.
4. ¿Cómo usarías la matriz de missingno (reordenando filas) para distinguir un MCAR de un MAR?
5. Dos columnas tienen 9 000 nulos cada una en un dataset de 18 000 filas. ¿Qué correlación de nulidad esperas si son
   MCAR? ¿Y si son MAR?

---

## 👉 Tu práctica: proyecto N1 (a)

Esta clase no tiene `ejercicio.py`: su práctica es la **entrega (a)** del proyecto integrador
`proyectos/n1-preprocesamiento-vivienda/` (*Pipeline de preprocesamiento: viviendas en Lima*), sobre
`data/raw/viviendas_lima.csv`.

**Entrega (a) = limpieza completa:**
- **Duplicados**: detectarlos y quitarlos.
- **Texto sucio**: categorías inconsistentes (mayúsculas, espacios, variantes del mismo valor).
- **Faltantes con imputación justificada**: reporte de nulos, tipo de faltante (MCAR/MAR/MNAR) y método elegido según %
  (usa la tabla de la sección 4).
- **Outliers**: detección y tratamiento (clases 06–08).
- **Encoding** de variables categóricas (clase 09).

**Entregables:** `src/preprocesamiento.py` con funciones puras (`limpiar(df)`, `imputar(df)`, `tratar_outliers(df)`,
`codificar(df)`…), tus **propios tests** en `tests/`, un `reporte_eda.ipynb` con hallazgos y la **justificación de cada
decisión**, y un `README.md` con el problema, las decisiones y el antes/después.

**Criterios de aceptación:** `python -m pytest proyectos/n1-preprocesamiento-vivienda` pasa; el dataset final no tiene
NaN, duplicados ni texto inconsistente; cada decisión está explicada (por qué mediana y no media, por qué IQR…); y lo
revisas con `/proyecto n1-preprocesamiento-vivienda revisar`.

Para empezar, planifícalo con:

```
/proyecto n1-preprocesamiento-vivienda planificar
```
