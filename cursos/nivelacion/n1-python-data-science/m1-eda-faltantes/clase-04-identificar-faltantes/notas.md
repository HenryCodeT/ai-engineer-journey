# Clase 04 · Cómo identifico datos faltantes (ejercicio)
**Curso:** N1 · Python para Ciencia de Datos · **Módulo 1** (EDA y datos faltantes) · Transcript: `transcripts/n1-python/clase-04.txt`
Título oficial: *Python para Ciencia de Datos - M1-4 - Como identifico datos faltantes (Ejercicio)*

> Es la clase **práctica** más importante del módulo (unos 53 minutos de código en Google Colab). El profesor junta todo lo
> de las clases 01–03 en un solo recorrido: **simula nulos → los reporta → debate qué se puede imputar → identifica el tipo
> (MCAR / MAR / MNAR) con gráficos → imputa con métodos simples y avanzados**. Si solo vas a leer una clase del módulo con calma, que sea esta.

---

## 0. Mapa de la clase en una tabla

| Paso | Qué hace el profesor | Herramienta | Función del ejercicio |
|---|---|---|---|
| 1 | Carga el dataset y quita 2 columnas sobrantes | `pd.read_csv`, `df.drop(columns=[...])`, `df.info()` | — |
| 2 | Descubre categorías "Unknown" = faltantes disfrazados | `value_counts()` | — (lo practicas en la clase 03) |
| 3 | Simula nulos (60 % en las **mismas filas** de dos columnas) | `df.sample(frac=...)`, `df.loc[filas, cols] = np.nan` | `simular_faltantes` |
| 4 | Arma un reporte de nulos | `isna().sum()`, `pd.concat`, porcentaje | `reporte_nulos` |
| 5 | Debate: ¿se pueden imputar todas? → **no** | criterio + umbral | `columnas_descartables` |
| 6 | Identifica el tipo de faltante | heatmap de correlación de nulidad, dendrograma (missingno) | `correlacion_nulidad`, `diagnostico_nulidad` |
| 7 | Imputa | `ffill/bfill`, media/mediana, `KNNImputer`, `IterativeImputer` | `imputar_knn` |
| 8 | Quita duplicados | `drop_duplicates(keep="first")` | — |

---

## 1. El entorno y el dataset

- Trabaja en **Google Colab** para que todos estén "en igualdad de condiciones" (sin problemas de versiones). Se puede usar
  Jupyter local o VS Code sin problema.
- Librerías: `numpy`, `pandas` y **`missingno`**, que el profesor llama *"la obligatoria"* para graficar datos faltantes.
- Comentario del profesor sobre ChatGPT / Copilot: *"pueden utilizarlo, pero la idea es que ustedes entiendan lo que están generando"*.
- **Dataset:** clientes de un banco que **cancelaron su tarjeta de crédito** (churn) y clientes que la mantienen.
  - Variable objetivo (target): `Attrition_Flag` (se fue / se quedó).
  - `df.info()` → **10 127 filas**, 23 columnas. Las **2 últimas sobran** y las elimina:
    *"ustedes ya de forma manual saben qué columnas no les va a servir… las pueden colocar aquí en una lista"* → `df.drop(columns=[...])`.
  - Inicialmente **no tiene ningún nulo**.

> En el repo usamos `data/raw/clientes_banco.csv` (5000 clientes, target `abandono`), que tampoco tiene nulos: es el análogo perfecto.

---

## 2. El faltante "disfrazado": la categoría Unknown

El dataset no tenía NaN, pero al hacer un conteo de `Marital_Status` aparece:

| Divorciado | Casado | Soltero | **Desconocido** |
|---|---|---|---|
| 748 | 4687 | 3943 | **749** |

- *"Muchas veces en la base de datos no me aparece ni siquiera desconocido, ni me aparece nulo: me aparece una categoría de más.
  Eso también es un problema."*
- Esos 749 **son datos faltantes**: no sabemos si son casados, solteros o divorciados. ¿Se reparten como el resto? ¿Son todos
  de una categoría? *"No lo sabemos."*
- Conclusión: la variable **ya tiene sesgo**. Lo mismo pasa con `Income_Category`, que también trae "Unknown".
- 🔑 *"Aparentemente no había datos nulos, pero si me doy cuenta comienzan a aparecer."* → siempre revisa `value_counts()` de las categóricas.

---

## 3. Simular nulos (con fines didácticos)

Como el dataset venía completo, el profesor **fabrica** nulos:

- **60 %** de NaN en `Dependent_count` (cantidad de dependientes) y `Marital_Status` (estado civil), **en las mismas filas**.
  Resultado: ambas quedan con **4051** valores no nulos de 10 127.
- Otros nulos en `Credit_Limit` (límite de crédito; queda con 9114 no nulos), `Total_Trans_Ct` y `Total_Trans_Amt`.

**Cómo se hace en pandas** (lo que implementas en `simular_faltantes`):

```python
filas = df.sample(frac=0.6, random_state=123).index   # elige al azar el 60 % de las filas (reproducible)
df.loc[filas, ["Dependent_count", "Marital_Status"]] = np.nan   # NaN en esas filas, en AMBAS columnas
```

- `random_state` = **semilla**: con la misma semilla salen siempre las mismas filas (los tests dependen de eso).
- Poner los NaN en **las mismas filas** es lo que luego hará aparecer una correlación de nulidad = 1 (sección 6).

---

## 4. El reporte de nulos

El profesor escribe una función que da un informe *"un poquito más organizado comparado con lo que devuelve `info()`"*:

1. Cuenta los nulos por columna (`isna().sum()`) y los ordena.
2. Los expresa en **porcentaje**: `× 100` y **redondeo a 3 decimales**.
3. Une conteo y porcentaje en un solo DataFrame con `pd.concat(..., axis=1)` y `reset_index()`.
4. Renombra columnas a **variable · cantidad de nulos · porcentaje** y agrega **cantidad de no nulos** = filas totales − nulos.
5. Reordena las columnas (`reindex`).

Resultado (forma):

| variable | nulos | no_nulos | porcentaje |
|---|---|---|---|
| Dependent_count | 6076 | 4051 | 59.998 |
| Marital_Status | 6076 | 4051 | 59.998 |
| Credit_Limit | 1013 | 9114 | 10.003 |
| … | … | … | … |

Luego genera los gráficos de **missingno** en una figura grande con subgráficos (`subfigures`): barras de % de nulos, matriz,
heatmap y dendrograma. Advertencia del profesor: *"dependiendo de la cantidad de variables y registros, estas gráficas se vuelven pesaditas"*;
la tabla propia es más rápida. Sugerencia: **filtrar antes** las columnas que sí tienen nulos y graficar solo esas.

---

## 5. El debate: ¿se pueden imputar todas las variables?

Pregunta del profesor a la clase: *"¿Se podrán imputar todas las variables o no?"* Respuesta: **NO.**

### Qué hacer con una variable con 60 % de nulos
1. **Paro y reviso el script SQL** de extracción (o le digo al **ingeniero de datos** que lo revise). *"Yo veo esto, paro el código, me voy a revisar el script SQL, así de frente."*
2. Si el script está bien y **sigue saliendo igual** → es un **problema de ingreso de datos**: *"no podemos hacer nada"*.
3. **Reportar** y **descartar**: *"estas dos variables tienen que salir del análisis"*, con un reporte diciendo que presentan una enorme cantidad de nulos.
4. Puede haber una causa humana: a la gente quizá no le gusta dar su estado civil o su número de dependientes (información personal). Eso ya *"escapa de las manos"* del analista.

**¿Por qué no imputar?** *"No pueden reconstruir la mitad de algo con la otra mitad… como si tuviesen la mitad de un mapa."*
Y es un **60 %**, más de la mitad. *"Pueden utilizar un modelo avanzado y realmente no le va a acertar."*
(A un alumno: si el jefe insiste, *"dile que no eres mago"*.)

### Si te OBLIGAN a usar la variable (todas las opciones son riesgosas)
| Opción | Problema |
|---|---|
| Trabajar solo con el **40 %** de filas que tienen dato | Sesgas **todas** las variables: les quitas el 60 % de su información. De 1000 registros te quedas con 400: ¿son representativos? |
| Rellenar con **0** | Si es **categórica**, creas una **categoría nueva**; si es **numérica**, **destruyes la distribución**. |
| **Bootstrap** (muestrear repetidamente el 40 %) | Posible, pero sigue siendo un 60 % faltante: muy riesgoso. Habría que comprobarlo hasta el modelo final y su despliegue. |

> 🔑 Regla práctica de esta clase → `columnas_descartables(df, umbral=50)`: si el % de nulos supera el umbral, la variable se reporta y sale.

---

## 6. ¿Qué tipo de faltante es? (MCAR / MAR / MNAR)

Repaso de la clase 02: **MCAR** = falta completamente al azar; **MAR** = la ausencia depende de **otra variable**; **MNAR** = no es al azar, pero no vemos de qué depende.

### 6.1 El heatmap de correlación de nulidad (`missingno.heatmap`)
*"La forma más rápida es verlo en la matriz de correlación de nulidad entre variables."*

- Toma **todas las variables que presentan nulos** y calcula la **correlación** entre sus patrones de ausencia.
- Por dentro es: convertir cada columna en *¿falta? True/False* y correlacionar esas columnas → `df[cols].isna().corr()`.

**Cómo leerlo:**

| Valor | Interpretación del profesor | Tipo |
|---|---|---|
| **1** entre `Marital_Status` y `Dependent_count` | *"Si en un registro le falta un dato a estado civil, en el mismo registro de cantidad de dependientes también le va a faltar."* | **MAR**: depende de otra característica |
| 0.6 – 0.8 | En datos reales es lo más común: *"hay cierta correlación entre datos nulos"* | posible MAR |
| ≈ 0 con todas | *"No tienen correlación con nadie más"* → **potenciales** MCAR | potencial MCAR |
| — | MNAR: *"los más difíciles"*; no son al azar pero tampoco muestran correlación | MNAR |

(El 1 sale perfecto porque él puso los nulos a mano en las mismas filas.)

**Consecuencia para imputar:** si es **MAR**, *"no es recomendable hacerlo con un método paramétrico"* (media/mediana): *"mejor hay que utilizar estrategias no paramétricas"*.

### 6.2 Confirmar MCAR
- *"Potenciales"*: hay que verificarlo. Una forma que menciona: **volver a generar el gráfico varias veces mezclando las filas al azar**
  (reorganizando el índice); *"con tres creo que ya es suficiente"*.
- Otra: **ordenar los datos por una columna** y mirar la matriz de nulidad para ver si los huecos se concentran en algún tramo.
  Al hacerlo, `Dependent_count` y `Marital_Status` mantienen la correlación de 1 y `Credit_Limit` sigue con su distribución diferente.
- Su diagnóstico final: `Credit_Limit` y `Total_Trans_Ct` → MCAR; `Dependent_count` + `Marital_Status` → MAR;
  `Total_Trans_Amt` → *"algo ocurre, hay que analizarlo"*.
- *"El análisis de datos nulos ya es un dolor de cabeza… hay que verificar bien que no se confunda un MCAR con un MNAR."*

> 📌 Extra (importante): la correlación **no cambia** al mezclar las filas (se calcula fila a fila); lo que cambia es el **dibujo** de la
> matriz. Y el heatmap **solo compara columnas con nulos entre sí**: si la ausencia depende de una columna **completa** (por ejemplo, el
> target), no lo verás ahí. Para eso sirven ordenar por esa columna o la `tasa_faltantes_por_grupo` de la clase 02. El notebook lo muestra
> con `antiguedad_anios`, que falta más en clientes que abandonaron.

### 6.3 El dendrograma (`missingno.dendrogram`)
- Agrupa las variables según su **patrón de nulidad**: las que no tienen nulos quedan en un grupo; `Dependent_count` y `Marital_Status`
  en otro; `Total_Trans_Ct` se junta con `Total_Trans_Amt`…
- Es **clustering jerárquico** (aprendizaje no supervisado), con variantes según la **función de distancia** y el **tipo de enlace**.
  *"Lo van a entender mejor cuando vean Machine Learning."*

---

## 7. ¿Cómo imputo?

Antes de imputar, el profesor elige **solo las columnas con al menos un nulo** (*"porque no quiero imputar todas"*) y mira su
**boxplot** y su **histograma (20 bins)** para conocer la distribución.

### 7.1 Forward fill / backward fill
- `ffill`: rellena con el valor **anterior**; `bfill`: con el **siguiente**. Si hay `4, NaN, 6`: ffill → 4, bfill → 6.
- Se usa en **series de tiempo**, donde hay **continuidad**. Aquí cada fila es un cliente distinto: *"no lo recomiendo"*.

### 7.2 Métodos paramétricos (estadísticos)
- Se puede imputar con **media, mediana, cuartil 1, cuartil 3**, por bloques… Válidos si corroboras que es **MCAR**.
- **Media:** *"lo último en lo que voy a pensar"*: los **outliers** la mueven.
- **Mediana:** no es sensible a outliers… **pero genera un PICO** en el histograma: muchas filas con el mismo valor exacto.
  *"Esto es lo que me preocupa."* *"Mientras más datos no los tengo, peor."*
- Referencia del profesor: un valor "aceptable" de faltantes sería **hasta ~30 %**.
- Por qué importa el pico (respuesta a un alumno): no es que la variable deba ser normal; es que imputar crea *"otra tendencia"* que la
  variable no tenía: *"la estás alterando más de lo que ya está"*. Y complica transformaciones posteriores (p. ej. logarítmica).
- Alternativa que sugirió un alumno: **regresión** → el problema de imputación se vuelve un problema de Machine Learning.

### 7.3 Métodos avanzados: KNNImputer
*"Los métodos que más se utilizan en la práctica"*. En scikit-learn (`sklearn.impute`) hay varios imputers.

**Idea de kNN (con el dibujo del profesor):** hay triángulos y círculos; llega una **X**. ¿Es triángulo o círculo? *Pregúntale a sus k vecinos más cercanos.*
- Con k = 2: los más cercanos son un triángulo y un círculo → gana el más cercano (círculo).
- Con k = 3: dos círculos y un triángulo → círculo.

**Para imputar:** se miran los vecinos más próximos del punto con hueco y se rellena con lo que ellos tienen.
`weights="uniform"` = *"todos los 10 vecinos tienen el mismo peso"* (se podrían dar pesos manuales distintos).

```python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=10, weights="uniform")
valores = imputer.fit_transform(df[cols])                      # "aprende" los vecinos y rellena
df_imp = pd.DataFrame(valores, columns=cols, index=df.index)   # la salida es un array: se vuelve DataFrame
```
Resultado: *"ya no se ven tanto esos picos, un poquito más suave, pero igual se mantiene la tendencia"*.

> 📌 Precisión: el profesor dijo *"el valor que se repite más en la vecindad"*. Eso es kNN para **clasificación**. `KNNImputer`
> rellena con el **promedio** de los k vecinos (por eso en el ejemplo del ejercicio, vecinas con 20 y 10 → 15).
> La distancia se calcula con las demás columnas que le pases.

### 7.4 Métodos avanzados: IterativeImputer
- Convierte la imputación en una **regresión multivariada**: *"más robusta pero a la vez más costosa"* computacionalmente.
- Paso a paso: se elige un **regresor**; se toma la primera columna con faltantes; las filas con dato son el **entrenamiento** y las que
  faltan, la "prueba"; el regresor usa **todas las demás variables como entradas** y esa columna como salida; predice los huecos;
  sigue con la siguiente columna. Una pasada completa = una **ronda**; se repite `max_iter` veces.

```python
from sklearn.experimental import enable_iterative_imputer   # obligatorio: es experimental
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

imp = IterativeImputer(estimator=ExtraTreesRegressor(), max_iter=10, random_state=123)
valores = imp.fit_transform(df[cols])
```
- Puedes usar otro regresor (XGBoost, LightGBM…). Con 10 127 × 4 tarda (muestra el progreso de cada ronda).
- *"No estamos exentos de sesgo o de errores."*

### 7.5 ¿Cuál es mejor? Se decide en el modelo
- *"En la práctica son los que funcionan mejor, pero siempre podemos analizar cuál fue la repercusión en el modelo final."*
- Compara KNN vs Iterative, o distintos regresores, en tu **protocolo experimental**: lo que manda es el **desempeño del modelo**.
- **Categóricas:** también funcionan, pero *"previamente tienes que aplicar un encoding"* (clase 09).

---

## 8. Duplicados (rapidito)
Copia el DataFrame, pregunta cuáles están duplicados, **conserva el primero** y elimina el resto: `df.drop_duplicates(keep="first")`.
*"No es nada complejo."*

---

## 9. Lo que tienes que quedarte
1. **Primero el reporte de nulos**: ¿cuánto falta en cada variable? (y revisa categorías "Unknown").
2. **Mucho faltante (≈60 %) no se imputa**: revisar script → reportar → descartar.
3. **Correlación de nulidad**: ≈1 = MAR (evitar media/mediana); ≈0 = *potencial* MCAR (confirmar); MNAR, el más difícil.
4. **La mediana deja un pico**; KNN e Iterative imputan "más suave".
5. **La última palabra la tiene el desempeño del modelo.**

---

## 📌 Extra: conexión con ML / IA
- Un modelo aprende la distribución que le das. Un pico artificial (mediana) le enseña que "muchísimos clientes ganan exactamente 2970",
  algo falso; y si los huecos eran MAR, rellenarlos con un número fijo **borra la señal** de "por qué faltaba".
- Las variables con 60 % de nulos meten más ruido que información: descartarlas (y documentarlo) suele dar modelos más estables.
- `fit_transform` es el patrón de **todo** scikit-learn: lo usarás con escaladores, encoders y modelos. ⚠️ Más adelante aprenderás que el
  imputer se ajusta (`fit`) **solo con los datos de entrenamiento**, para no "filtrar" información del test.

## 📌 Extra: cómo se conectan las funciones del ejercicio
| # | Función | Qué practica | Puntos |
|---|---|---|---|
| 1 | `reporte_nulos` | el informe del profesor (variable · nulos · no_nulos · porcentaje) | 15 |
| 2 | `columnas_descartables` | la regla del debate (más del umbral → fuera) | 10 |
| 3 | `simular_faltantes` | NaN en las mismas filas, reproducible con semilla | 20 |
| 4 | `correlacion_nulidad` | lo que dibuja el heatmap de missingno | 15 |
| 5 | `diagnostico_nulidad` | leer el heatmap: "posible MAR" / "potencial MCAR" | 15 |
| 6 | `imputar_knn` | KNNImputer conservando índice y columnas | 25 |

---

## ❓ Preguntas de repaso
1. El dataset del profesor "no tenía nulos", pero `Marital_Status` tenía 749 "Unknown". ¿Por qué eso ya es un problema de faltantes y qué sesgo introduce?
2. Una variable tiene 60 % de nulos y el script SQL está correcto. ¿Qué haces y por qué no la imputas? Nombra los dos riesgos de "trabajar con el 40 %" y de "rellenar con 0".
3. En el heatmap de nulidad, `A` y `B` tienen correlación 1 y `C` tiene ≈ 0 con todas. ¿Qué tipo de faltante sugiere cada caso y qué método de imputación evitarías en `A` y `B`?
4. ¿Por qué la imputación con mediana genera un pico en el histograma, y por qué empeora cuanto más faltantes hay?
5. ¿Qué diferencia hay entre `KNNImputer` e `IterativeImputer` (idea, costo) y cómo decides al final cuál usar?
