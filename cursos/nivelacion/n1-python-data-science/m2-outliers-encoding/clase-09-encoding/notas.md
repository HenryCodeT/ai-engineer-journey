# Clase 09 · Codificación de variables (encoding) y escalamiento
**Curso:** Python para Ciencia de Datos · **Módulo 2** (Outliers y encoding) · Título oficial: *Python para Ciencia de Datos - M2-4 - Codificación de Variables* · Transcript: `transcripts/n1-python/clase-09.txt`

> Clase larga (unas 1 h 48 min) y **central** del módulo. Primero explica en la pizarra **por qué** hay que codificar y escalar. Después, en Colab con el dataset *bank marketing*, separa variables, explora con tablas dinámicas, compara encoders (OneHot, dummy, ordinal, binario), distingue **transformar** (PowerTransformer) de **escalar** (MinMax / Standard) y lo junta todo en un `Pipeline` + `ColumnTransformer`.
>
> Los ejemplos con `viviendas_lima.csv` y `clientes_banco.csv` son 📌 Extra: el profesor usó *bank marketing*, pero los datasets del repo tienen las mismas situaciones. Las cifras de *bank marketing* (45 211 filas, etc.) sí son de la clase.

---

## 0. La idea en una frase
Un modelo solo entiende **puntos con coordenadas numéricas**. Por eso las **categóricas se codifican** (encoding) y las **numéricas se llevan a una escala parecida** (escalamiento) o se **transforman** hacia una normal cuando el algoritmo lo necesita.

---

## 1. ¿Por qué codificamos y escalamos? (la pizarra)
Los alumnos respondieron *"para que lo lean los algoritmos"* y *"las categóricas son difíciles de trabajar"*. El profesor lo explicó con geometría:

1. Un DataFrame tiene **filas y columnas**. Las columnas son las **variables** y cada fila es una **muestra u observación**.
2. Cada muestra se puede **proyectar como un punto** en un espacio $\mathbb{R}^n$, el **espacio de características**. Con 2 clases (puntos verdes y rojos), *"en teoría"* cada clase ocuparía su propia región. *"En la práctica, mientras más crece el $\mathbb{R}^n$, esos espacios están bastante intersectados"*, y la idea es encontrar cómo separarlos.
3. Un punto se describe con sus coordenadas $(x_1, x_2, \dots, x_n)$, así que **todo tiene que ser numérico**:
   - **Categóricas** → *"tenemos que codificar sí o sí las variables categóricas porque tienen que tener un valor numérico"*.
   - **Numéricas** → ya son números, pero el problema es la **escala**. Una columna puede tener valores de 100 a 500, otra de 1.0 a 2.3 y otra de $10^3$ a $10^6$. *"El nivel de escala en el que se encuentra cada variable numérica también afecta directamente a la distribución de estos puntos."* Por eso se llevan *"a un nivel de escala similar"*.

📌 Extra · en `viviendas_lima.csv`: `habitaciones` va de 0 a 7, `area_m2` llega a miles y `precio_usd` a millones. Si mides distancias entre viviendas, el precio decide casi todo solo porque sus números son más grandes.

🧠 **Mini-pregunta:** si una columna está en soles y otra en número de hijos, ¿cuál dominaría una distancia euclidiana sin escalar?

---

## 2. Escalar ≠ normalizar
*"Algunos dicen vamos a normalizar los datos… en realidad el término correcto sería **escalar**."*

| | **Escalar** | **Normalizar** |
|---|---|---|
| ¿Sobre qué actúa? | **Por columna** (variable): A → A escalado, B → B escalado… | **Por fila** (cada punto completo) |
| Ejemplo | MinMax, Standard | dividir el punto por su norma: $\vec{b} / \lVert\vec{b}\rVert$ con norma **L1** o **L2** |
| ¿Cambia la distribución de la variable? | **No**: *"no afecta su comportamiento"* | **Sí**: *"se afecta la distribución de la variable, porque lo está haciendo por filas"* |
| Se aplica a | Solo numéricas | El punto completo |

*"En la práctica los tomamos bastante como sinónimos, a pesar de que no es lo mismo."*

### ¿Cuándo se normaliza? (pregunta de Samuel)
- **Deep Learning:** en redes convolucionales existe la **batch normalization** (normalización por lotes). Se usa cuando *"lo que te importa es el punto"* y no el estudio de cada variable, sobre todo con datos **no estructurados**.
- **Regresión regularizada:** **Lasso (L1)**, **Ridge (L2)** y **ElasticNet** (la mezcla de las dos). Ayudan a que el modelo **converja mejor**. Bajan el peso de las variables **colineales** (redundantes, no independientes) y hacen que los **outliers pierdan influencia**. *"Más que una selección de variables, les baja la prioridad."* (Se verá con calma en Machine Learning).
- **Visión computacional** (el área del profesor): los píxeles van de 0 a 255. Se hace algo como $(x - 128)/255$, o se resta la **media** del canal y se divide por su **desviación estándar** (canales RGB). *"Aunque esta es una técnica de escalamiento, nosotros le decimos normalización."*
- Beneficio general: el modelo converge mejor y mejora la interpretación de los coeficientes.

📌 Extra · normalizar la fila (3, 4) con L2: la norma es $\sqrt{3^2+4^2}=5$, así que queda (0.6, 0.8), con longitud 1.

🧠 **Mini-pregunta:** si escalas `area_m2` con MinMax, ¿cambia su asimetría (*skew*)? ¿Y si normalizas cada fila con L2?

---

## 3. El caso práctico: *bank marketing*
- Dataset de campañas bancarias con **45 211** muestras. Variables: `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `day`, `duration`, `campaign`, `pdays` (de **−1 a 875**), `previous`…, y el objetivo **`y`** (yes/no).
- `y` tiene **5 289 "yes"** de 45 211 → *"la base de datos está desbalanceada así de frente"* (el desbalance se verá en el módulo 4).
- **Separar numéricas y categóricas automáticamente:**
  ```python
  num_vars = df.select_dtypes(include="number").columns
  cat_vars = df.select_dtypes(exclude="number").columns   # el complemento
  ```
  (En su ejemplo `y` quedó entre las categóricas. Luego aclara que **`y` no se transforma**).

### Exploración rápida con tablas dinámicas
- **pivottablejs**: *"tablas pivote de forma bastante rápida, es como si estuviera en un Excel, directamente en Python"*. Se arrastran variables y se cruzan (por ejemplo `education` × `y`, `marital` × `y`). Tiene barcharts, mapas de calor y agregaciones (conteo, promedio, mediana, suma, fracción…). **Se puede descargar como HTML** y abrir aparte aunque se cierre el Colab.
- También mencionó **pandas profiling** como alternativa (*"funciona bastante bien en modo local, hay que configurarlo un poquito"*).
- Recomendación: *"lo que yo recomendaría es que **siempre hagan un análisis con respecto a la variable objetivo**"*.
- Con histogramas vio que `balance`, `duration` y `campaign` tienen **muchos outliers** y que `pdays` tiene muchísimos valores en la misma posición.

📌 Extra · sin pivottablejs, lo mismo se hace con pandas:
```python
pd.crosstab(banco["segmento"], banco["abandono"], normalize="index")      # clientes_banco.csv
viv.pivot_table(index="estado", columns="tipo_inmueble", values="precio_usd", aggfunc="median")
```

### Orden del preprocesamiento
1. **Imputar nulos** (clases 02–04). En *bank marketing* no había nulos.
2. **Outliers** (clases 06–08).
3. **Encoding / transformación / escalamiento** (esta clase).

---

## 4. Separación **manual** de variables (mejor que la automática)
*"Cuando ustedes ya estén haciendo cosas avanzadas van a preferir hacer la separación de variables de forma **manual**."* Hay dos motivos:
- **Numéricas:** tras una **prueba de normalidad** sabes cuáles ya son normales. *"Me conviene solamente hacer esta transformación a las que no tienen distribución normal."*
- **Categóricas:** según **cuántas categorías** tenga cada una, conviene un encoder u otro. Revisando en pivottablejs:
  - `default`: 2 categorías (no 44 396 / yes 815). `housing` y `loan` también tienen 2. Con ellas *"puedo aplicar un OneHotEncoder con mucha tranquilidad"*.
  - `marital`: 3 categorías (single, married, divorced).
  - `job`: **12** categorías. *"No me conviene aplicar un OneHotEncoder, me va a generar muchas dimensiones."*

📌 Extra · en viviendas: `pais` (2) y `tipo_inmueble` (3) → OneHot; `distrito` (15) → no OneHot. `cochera` aparece con 5 por el texto sucio (`Si`, `si`, `SI`, `No`, `no `): **primero se limpia** y después se codifica.

---

## 5. Encoders para variables categóricas

### 5.1 OneHot vs dummy
- **OneHotEncoder:** *"por cada categoría se genera una nueva variable"*. Con *Team* = A, B, C salen 3 columnas (n).
- **Dummy:** genera **n − 1** columnas. Una categoría queda como **0 0** (la de referencia).

| team | OneHot: A B C | Dummy: B C |
|---|---|---|
| A | 1 0 0 | 0 0 |
| B | 0 1 0 | 1 0 |
| C | 0 0 1 | 0 1 |

### 5.2 La pregunta de Carlos (estadístico): ¿dummy u OneHot?
*"A mí me dijeron que es mejor aplicar el dummy… ¿cuándo conviene uno y cuándo el otro?"*
- El profesor: *"Yo preferiría **no usar ninguna de las dos**, salvo que tenga pocas categorías"* (2 o 3). Entre ellas la diferencia es solo una columna.
- El motivo es la **dimensionalidad**. Con 10 variables, de las cuales 4 son categóricas de 4 categorías: 6 numéricas + 16 columnas OneHot = $\mathbb{R}^{22}$ en el peor caso, y 6 + 12 con dummy = $\mathbb{R}^{18}$. Así pasas de $\mathbb{R}^{10}$ a $\mathbb{R}^{18}$–$\mathbb{R}^{22}$.
- **Maldición de la alta dimensionalidad:** *"si un modelo ya para dividir puntos en un $\mathbb{R}^2$ y un $\mathbb{R}^3$ se demora y ya tiene cierto sesgo, imagínate un $\mathbb{R}^{18}$"*. Y aunque después apliques selección de variables, el encoding ya está hecho.

### 5.3 Label / Ordinal encoding: una columna, pero con riesgo
- `OrdinalEncoder` da 0, 1, 2… en **una sola columna**. *"El modelo puede interpretar que la clase C pesa más que la clase A y B."*
- Pregunta de un alumno: los **distritos no tienen orden**. Si les pones 1, 2, 3, el modelo puede entender un peso que no existe.
- Aun así: *"ha habido casos donde he utilizado una codificación numérica y no me ha afectado para nada el rendimiento del modelo, pero hay otros casos donde sí"*. Hay que **probarlo experimentalmente**.
- El objetivo de un buen encoding: *"que no exista nivel de relevancia entre ellas… que se mantenga la independencia"* y que las categorías **no se intersecten** en el espacio.

### 5.4 Binary encoding: $\log_2 n$ columnas (el favorito del profesor)
- *"La binaria es la que más me gustaría usar."* Se cuenta en binario: 00, 01, 10, 11…
- *"Si tienes ocho categorías, el logaritmo de 8 en base 2 es 3: generas tres columnas. Con OneHot generas ocho y con dummy siete."*
- Pregunta de **Eduardo**: con una variable de **20 categorías** → $2^4 = 16 < 20 \le 32 = 2^5$, así que bastan **5 columnas**. Otra opción es **probar un ordinal** y medir si afecta al modelo.
- Librería **`category_encoders`** (`BinaryEncoder` y muchos otros; *"todos tienen su `inverse_transform`"*). En la demo, `education` (4 categorías) y `job` (12) quedaron con muy pocas columnas.
- Interpretación (pregunta de un alumno): con OneHot lees fácil "es estudiante sí/no", pero `0011` no se entiende a simple vista. El profesor responde que lo que importa es *"garantizar que no exista una intersección entre categorías"*: cada categoría tiene su propio código. La lectura individual del OneHot *"es fácil para ti, pero difícil para la máquina"* (por las dimensiones que añade). Hay que hacer pruebas experimentales para elegir.

| Método | Columnas para n categorías | n = 8 | n = 15 (distrito) | n = 20 |
|---|---|---|---|---|
| OneHot | n | 8 | 15 | 20 |
| Dummy | n − 1 | 7 | 14 | 19 |
| Binario | ⌈log₂ n⌉ | 3 | 4 | 5 |
| Ordinal | 1 | 1 | 1 | 1 |

📌 Extra · en este curso definimos binario = `max(1, ceil(log2(n)))` (con n = 1 o 2 basta 1 columna). Ojo: el `BinaryEncoder` real de `category_encoders` numera desde 1 y reserva espacio para valores desconocidos, así que a veces genera **una columna más** (por ejemplo 4 en vez de 3 con 8 categorías). La idea es la misma.

🧠 **Mini-pregunta:** ¿cuántas columnas genera `distrito` (15) con cada método? ¿Cuál elegirías para un modelo lineal y por qué?

---

## 6. Numéricas: **transformar** (PowerTransformer) vs **escalar**

### 6.1 PowerTransformer: cambia la forma
- Aplica una **transformación de potencia** para que los datos *"se aproximen a una distribución normal, pero no siempre ocurre"*.
- Dos métodos: **Box-Cox** (*"solo admite entradas positivas"*) y **Yeo-Johnson** (positivos y negativos). **Por defecto usa Yeo-Johnson**.
- Tiene `fit`, `transform`, `fit_transform` e **`inverse_transform`** para volver a los valores originales.
- En la demo, `age` pasó a valores de −3 a 3 con forma cercana a una normal, y `balance` quedó centrada en 0.
- **No se toca `y`**: *"y está en su espacio original"*. Solo se transforma la X.

**Pregunta de Arturo:** *"¿Eso es escalar?"* → *"No, no las está escalando, **las está transformando**. Es un x que llevo a un x′ de tal forma que se aproxime a una distribución normal."* **Sí cambia la distribución.**

### 6.2 Escaladores: NO cambian la forma
*"Un escalador no afecta la distribución de la variable, la variable simplemente se va a un rango menor de datos, pero nada más."*
- **MinMaxScaler:** lleva los datos a un nuevo mínimo y máximo, **por defecto [0, 1]**: $x' = \frac{x - \min}{\max - \min}$.
- **StandardScaler:** *"le quitan la media y la dividen sobre la desviación estándar"*: $x' = \frac{x - \bar{x}}{s}$. Los datos quedan centrados en 0 (más o menos entre −3 y 3) con **varianza 1**.
- *"El escalamiento no ha transformado la variable, no ha hecho que se aproxime a una normal. No hay que confundirse con eso."*
- Al final de la clase: después de un PowerTransformer los valores ya quedan en rangos pequeños (−4 a 4), así que *"ya no sería tan necesario escalarla"*, aunque se puede probar.

📌 Extra · comprobación con `precio_usd` de viviendas:

| | skew | rango |
|---|---|---|
| original | 4.12 | 20 300 – 2 121 535 |
| MinMax | 4.12 | 0 – 1 |
| Standard | 4.12 | −1.2 – 17.4 |
| PowerTransformer | ≈ 0 | −3.1 – 4.4 |

### 6.3 ¿Qué modelos lo necesitan?
- **Sensibles** a normalidad / escala: **regresión lineal**, **regresión logística** (modelos probabilísticos), **SVM** (*"una mejora de un modelo lineal"*), **kNN** (al que también le afectan las categóricas) y **PCA**, que *"sí o sí necesita"* esto porque trabaja con **autovalores y autovectores**.
- **No les afecta:** **árboles de decisión, Random Forest, XGBoost, LightGBM** (*"trabajan con otro tipo de reglas"*).
- Si tu PCA + SVM no funciona, *"el PCA no es el problema: el problema es que la variable no la has transformado a una distribución próxima a la normal"*. *"Nos volvemos muy técnicos en el código pero nos olvidamos de qué necesita un algoritmo."*

🧠 **Mini-pregunta:** vas a entrenar un Random Forest para predecir `precio_usd`. ¿Es obligatorio aplicar PowerTransformer? ¿Y para una regresión lineal?

---

## 7. Todo junto: `Pipeline` + `ColumnTransformer`
- **Pipeline:** *"una subrutina que me permite encapsular muchos procesos en un solo módulo"*.
- **ColumnTransformer:** aplica cada transformador **solo a sus columnas**. Recibe tuplas `(nombre, transformador, columnas)`.

Lo que construyó el profesor (adaptado a viviendas, 📌):
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer

cat_onehot  = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])
cat_ordinal = Pipeline(steps=[("encoder", OrdinalEncoder())])
num         = Pipeline(steps=[("transf",  PowerTransformer())])      # Yeo-Johnson

preprocessor = ColumnTransformer([
    ("onehot",  cat_onehot,  ["tipo_inmueble", "pais"]),
    ("ordinal", cat_ordinal, ["estado", "distrito"]),
    ("num",     num,         ["area_m2", "habitaciones"]),
])
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

pipeline.fit(datos)                         # aprende las transformaciones
X = pipeline.transform(datos)               # o directamente fit_transform(datos)
columnas = pipeline.get_feature_names_out() # 'onehot__tipo_inmueble_Casa', 'num__area_m2', ...
```
- `handle_unknown="ignore"`: si llega una categoría nunca vista, se ignora (todas sus columnas OneHot en 0) en lugar de dar error.
- **Ventaja clave:** *"le cambié el encoder por aquí nada más y no pasó nada: las numéricas siguen con su propia transformación"*. Probó cambiar el ordinal por un `BinaryEncoder` y luego separó las numéricas en dos grupos: unas con PowerTransformer y otras (`day`, `campaign`, `previous`…) solo con `StandardScaler`.
- Conclusión: *"ustedes definen y hacen un estudio de qué variables requieren OneHot, cuáles ordinal o binary, cuáles requieren ser transformadas para aproximarse a una normal y cuáles no lo necesitan y solo se escalan"*. Todo eso depende también del **algoritmo** que vas a usar.

---

## 8. Cierre: outliers con PyOD y lo que viene
- Faltaba eliminar outliers. Lo hizo con **PyOD**, el algoritmo **ABOD** con `contamination=0.1`: de **45 000** registros quedaron unos **40 000**. *"Si quisiera eliminar más outliers podría colocar 0.2."* Ver **clase 08** (kNN / IsolationForest y `contamination`).
- Repaso del camino: imputación → outliers → escalamiento o transformación → encoding (*"el encoding es para las categóricas, y esto es para las numéricas"*).
- **Próximas clases:** **selección de características** y **reducción de dimensionalidad** (*"no es lo mismo"*), y al final **balance de datos**.
- Eduardo preguntó por **agrupar variables**: *"Eso es parte de la ingeniería de características: se analiza qué variables son altamente correlacionadas (A, B, C) y se juntan en una sola."*

---

## 9. ¿Por qué importa para ML / IA? (📌 Extra)
- Sin encoding, **ningún modelo de scikit-learn acepta texto**: `fit` falla con `could not convert string to float`.
- Un OneHot sobre 15 distritos agrega 15 dimensiones. Con pocos datos (2 000 viviendas) eso favorece el **sobreajuste** (📌 Extra).
- kNN, SVM, regresión regularizada y PCA **miden distancias o varianzas**: sin escalar, la columna con números más grandes manda.
- El `ColumnTransformer` se **entrena solo con el train** y se aplica igual al test y a producción. Así se evita la **fuga de información** (data leakage) (📌 Extra: lo verás en el curso de ML).

---

## 📌 Extra (agregado, no dicho en clase)
- Ejemplos y tablas con `viviendas_lima.csv` / `clientes_banco.csv`.
- `pd.crosstab` / `pivot_table` como sustituto de pivottablejs.
- La regla exacta `max(1, ceil(log2(n)))` y la nota sobre la columna extra del `BinaryEncoder` real.
- `sparse_threshold=0` en el `ColumnTransformer`, para obtener una matriz normal y no "sparse" (lo usa el ejercicio).
- Comentarios sobre sobreajuste y data leakage.

---

## ❓ Preguntas de repaso
1. Explica con tus palabras por qué una fila es un "punto" y qué tiene que ver eso con codificar las categóricas.
2. ¿Qué diferencia hay entre **escalar** y **normalizar**? ¿Cuál cambia la distribución de una variable y por qué?
3. Tienes `tipo_inmueble` (3 categorías) y `distrito` (15). ¿Qué encoder usarías para cada una y cuántas columnas generaría cada opción?
4. ¿Por qué `StandardScaler` **no** convierte una variable asimétrica en normal? ¿Qué usarías si tu modelo sí necesita normalidad?
5. ¿Qué ventaja tiene armar el preprocesamiento con `ColumnTransformer` en lugar de transformar cada columna a mano?
