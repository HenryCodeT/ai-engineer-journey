# Clase 06 · Qué son los outliers (y respondiendo consultas)
**Curso:** Python para Ciencia de Datos · **Módulo 2** (Outliers y encoding) · Título oficial: *M2-1 · Que son Outliers y respondiendo consultas* · Transcript: `transcripts/n1-python/clase-06.txt`

> Clase **teórica** de unos 32 minutos, sin código. La primera mitad explica **qué es un outlier**, **por qué daña un modelo** y **las dos formas de tratarlo**. La segunda mitad son **preguntas de alumnos** (Carlos, Arturo) que llevan a temas útiles: puntos de interés, series de tiempo, `PowerTransformer` y la correlación entre variables categóricas.
>
> Los ejemplos con `viviendas_lima.csv` y el código son 📌 Extra: el profesor no los mostró, pero sirven para ver cada idea con datos reales del repo. Los métodos para **detectar** outliers (IQR, z-score…) se verán en la clase 07 (Tratamiento de outliers).

---

## 0. La idea en una frase
Un **outlier** es un dato que **se desvía mucho del resto**. Para describir un negocio puede ser un **hallazgo**. Para entrenar un modelo suele ser un **problema** que hay que tratar, aunque **sin eliminarlos todos a ciegas**.

---

## 1. ¿Qué es un outlier?
- *"Significan anomalías, valores anómalos, datos atípicos, que no deberían estar en cierto momento."*
- El ejemplo del profesor: *"tengo un conjunto de triángulos: triángulo, triángulo, triángulo… y de repente me aparece un **círculo**"*. Eso es un valor atípico: *"algo que no debería estar, pero apareció"*.
- Definición formal: *"un elemento u objeto de datos que **se desvía significativamente del resto** de los objetos"*.
- Detectar outliers es parte esencial del **EDA** (el análisis exploratorio de la clase 01). Ese análisis tiene nombre propio: **minería de outliers** (*outlier mining*).
- En **una dimensión** el outlier está "muy arriba o muy abajo". En **muchas dimensiones** puede estar *"simplemente fuera del espacio de distribución normal de los puntos"*, aunque en cada columna por separado parezca normal (outliers **multidimensionales**).

### ¿De dónde salen? (causas)
| Causa | Lo que dijo el profesor | Cómo se ve en `viviendas_lima.csv` (📌 Extra) |
|---|---|---|
| **Errores de medición** | El **instrumento** de recolección no se validó. Un buen instrumento *"mide siempre lo mismo"*: si mides en cuatro grupos, los resultados deben ser exactos o muy próximos | 12 viviendas con `area_m2` **multiplicada por 10** (un departamento de 689 m² que cuesta 138 700 USD) |
| **Errores de ejecución** | *"El script estuvo bastante fatal y tengo que corregirlo"* | 3 viviendas con `antiguedad_anios = 150`: un valor imposible, como la "edad de 200 años" |

🧠 **Mini-pregunta:** un termómetro marca 20 °C en una sala, y otro termómetro en la misma sala marca 35 °C. Según el profesor, ¿qué falla: la medición o la ejecución?

---

## 2. Las dos caras de un outlier
| Tipo de análisis | Qué es el outlier | Frase del profesor |
|---|---|---|
| **Descriptivo** (entender qué pasó) | Un **hallazgo valioso**, *"una oportunidad para mejorar el proceso de negocio"* | *"Desde la parte descriptiva muy bien, puede ser una oportunidad"* |
| **Inferencial** (ML, modelos predictivos) | Un **sesgo**: *"va a generar que mi modelo falle y no generalice"* | *"Representa error dentro del modelo y hay que eliminarlo"* |

- El tratamiento de outliers se hace sobre todo *"con miras a un modelo predictivo"*: **regresión**, **clasificación** o **clustering**. (Aprendizaje por refuerzo no se trata en el curso.)
- ¿Por qué daña al modelo? En ciencia de datos *"queremos hacer suposiciones sobre una población"*, y los valores extremos distorsionan esas suposiciones.

### Ejemplo: la recta de regresión
El profesor mostró dos gráficos de una **regresión lineal**: con outliers y sin ellos. *"Esta recta es el modelo [...] la tendencia es diferente: los outliers **jalan demasiado** el modelo."*

📌 Extra, con el dataset: si ajustamos `precio_usd ≈ a · area_m2 + b`:

| Datos usados | Pendiente `a` (USD por m²) |
|---|---|
| Todas las viviendas (con los 12 errores de área ×10) | **≈ 381** |
| Sin esos 12 errores | **≈ 1527** |

¡Solo **12 filas de 2015** hacen que el modelo crea que un m² vale **4 veces menos**! Eso es "no generalizar": el modelo aprendió de un error.

```python
import numpy as np
errores = df["area_m2"] > 400
np.polyfit(df["area_m2"], df["precio_usd"], 1)[0]                          # ≈ 381.5
np.polyfit(df.loc[~errores, "area_m2"], df.loc[~errores, "precio_usd"], 1)[0]  # ≈ 1527.3
```

### Modelos más robustos (adelanto de ML)
- *"Hay mejoras en las funciones de coste que evitan esto"*: a la regresión lineal se le agrega un **factor alfa** en la función de coste y se obtienen **Lasso** y **Ridge**, y su mezcla (**ElasticNet**).
- Con ese factor alfa el modelo es *"menos sensible a los outliers"* y además ayuda a reducir dimensionalidad.
- *"Eso ya lo hablaremos en Machine Learning con más calma."*

🧠 **Mini-pregunta:** una tienda tuvo un día con ventas 10 veces mayores de lo normal. ¿Ese dato es bueno o malo? Responde dos veces: para el gerente (descriptivo) y para un modelo que predice ventas (inferencial).

---

## 3. "No podemos eliminar todos los outliers"
*"Aquí viene una verdad muy cruda: no podemos eliminar todos los outliers."* Hay dos razones:

1. **Se pierden datos buenos.** *"Si tengo cinco variables y dos tienen outliers, y elimino esas filas, puedo comprometer a las otras tres."* Cada fila borrada se lleva sus valores sanos en las demás columnas.
2. **Hay outliers multidimensionales.** No siempre se ven columna por columna.

📌 Extra: en `viviendas_lima.csv`, borrar las 12 filas con área ×10 también borra 12 precios, 12 distritos y 12 antigüedades que estaban bien. Si además borras los precios extremos, las antigüedades imposibles… cada vez el dataset se encoge más.

---

## 4. Qué puede significar un outlier
| Significado | Explicación del profesor | En el dataset (📌 Extra) |
|---|---|---|
| **Error** | *"Tenemos una persona con 0 años: seguramente sea un error de carga de datos."* Detectarlo sirve para encontrar errores | `antiguedad_anios = 150` |
| **Límites** | Valores que *"se escapan del grupo medio, pero queremos **mantener el dato modificado** para que no perjudique el aprendizaje del modelo"* | Llevar esa antigüedad a un tope razonable (sección 5) |
| **Punto de interés** | *"Puede que sean los casos anómalos los que queremos detectar: nuestro **objetivo** y no nuestro enemigo"* (📌 Extra: detección de fraude) | Las pocas viviendas de lujo con precios muy por encima del resto (la mayor, 2.1 millones de USD): pueden ser caras **de verdad** |

---

## 5. Las dos formas de tratarlos
*"Los outliers se pueden tratar de dos formas: o **los eliminamos**, o los **llevamos a sus límites** superior e inferior."*

### Llevar a límites (capping)
Funciona cuando la variable tiene **límites conocidos**. El ejemplo de la **edad**:
- No existen edades **negativas**: todo valor < 0 se lleva a **0**. Caso real: unos alumnos tenían edades negativas porque una función que convertía **meses a años** falló. Eran **bebés de meses**: lo correcto era ponerles 0.
- *"Nadie supera los 120 años; con margen de error, nadie supera los 125."* Toda edad > 125 se baja al **límite superior** (o *"a un nivel de un cuartil"*).
- Primer paso práctico: *"identificar las columnas que sabemos que tienen límites que no pueden superar"*.

📌 Extra, en pandas es una línea con `clip` (no borra filas, y los NaN siguen siendo NaN):

```python
df_limpio = df.copy()
df_limpio["antiguedad_anios"] = df_limpio["antiguedad_anios"].clip(lower=0, upper=100)
# las 3 viviendas de 150 años pasan a 100; las otras 2012 filas no cambian
```

> ¿Por qué 100 y no 125? Es una **decisión de negocio**: en este dataset las antigüedades reales llegan a 60 años, así que 100 es un tope holgado. Lo importante es que el límite tenga una justificación, no que sea un número mágico.

### Caso real: los sensores de voltaje
- Unos alumnos (capacitación en una empresa) trabajaban con **sensores de voltaje** en torres. El rango válido era **0 a 100**, pero a veces llegaban a **150**.
- **Análisis descriptivo:** cruzaron los outliers con la **fecha y hora** de cada medición y descubrieron que coincidían con **lluvias y tormentas** (rayos). Además, de las marcas de sensor que usaban, **dos resistían mejor**: se alteraban menos y **se desgastaban menos**. Costaban un poco más, pero al hacer el análisis de costos **prefirieron pagar más** por un sensor que dura más y da datos más limpios.
- **Para el modelo predictivo:** *"simplemente los llevaron a sus límites, para **no dejar un hueco**"*. El sensor no debería medir fuera de 0–100, así que capping fue la solución.
- 🔑 *"Encuentran algo y eso repercute en todo."* El mismo outlier sirvió primero como **hallazgo** (descriptivo) y luego se **trató** (inferencial).

🧠 **Mini-pregunta:** si en vez de capping hubieran **eliminado** las mediciones de 150, ¿qué "hueco" habría quedado en una serie de mediciones cada minuto?

---

## 6. Preguntas de los alumnos

### Carlos: ¿y si el valor alto tiene lógica?
- Carlos mide **kilogramos cosechados por persona**. Lo normal es **40–50**, pero un grupo pequeño llega a **60–70** en las mismas condiciones.
- Respuesta: eso **no es un error**. Son *"personas fuera de lo común"* (mejor condición física, otro fenotipo/genotipo). Es un **caso excepcional**, un **punto de interés**, no un valor que haya que recortar.
- La diferencia clave: el capping se usa con **límites que sabemos que no se pueden superar** (como la edad). 70 kg cosechados es raro, pero **posible**.

📌 Extra: en el dataset, la mansión de 2.1 millones de USD en Jesús María es el "Carlos" de las viviendas: es rara, pero no imposible. En cambio, 150 años de antigüedad sí es imposible.

### Arturo: picos de ventas en una serie de tiempo
- Arturo predice **ventas diarias** y ve picos (Navidad, campañas, un día de 420 000 soles). ¿Debe quitarlos?
- Respuesta del profesor:
  - Si el pico **se repite** (toda Navidad, todo diciembre), es **estacionalidad**: *"no es un outlier, es un comportamiento normal de tu negocio"*.
  - Un outlier sería un pico **aleatorio que no se repite**: *"normalmente vendes 500 y ese día vendiste 2000, pero no se vuelve a repetir"*.
  - Se pueden usar **promedios móviles** y funciones de **suavizado** para que esos picos sueltos pesen menos.
  - Clave: ¿serie **univariante** o **multivariante**? Si el modelo recibe variables como **campaña (1/0)**, tráfico, **PBI**, **inflación**, entonces el pico ya no es un outlier: *"tú le estás alimentando con más información para que el modelo consiga entender el porqué"*.
  - Las series de tiempo *"tienen otro tratamiento"*: ventanas de días, mirar días anteriores para predecir el futuro. (Arturo usa **Prophet** de Facebook.)

### PowerTransformer: acercar una variable a la normal
- A Arturo le había ido mejor al **normalizar**. El profesor recomienda probar si sus variables *"se pueden aproximar a una normal"* con **`PowerTransformer`** de scikit-learn.
- Tiene dos transformaciones:

| Método | Acepta | Nota |
|---|---|---|
| **Box-Cox** | Solo valores **positivos** (> 0) | Falla si hay ceros o negativos |
| **Yeo-Johnson** | Positivos, cero y **negativos** | Es el método por defecto de sklearn |

- ¿Para qué? *"Hay modelos que necesitan, por definición, que la distribución de la variable sea normal o próxima a una normal"*:
  - **PCA** (reducción de dimensionalidad) necesita variables normales: si no, *"su reducción no es óptima"*.
  - **t-SNE** no lo necesita: por eso *"en la práctica funciona mejor que PCA en muchos casos"*.
  - **Ni PCA ni t-SNE funcionan bien con variables categóricas.** Para eso existe **MCA**.

📌 Extra, con el dataset: `precio_usd` tiene una cola larga a la derecha (asimetría ≈ 4.1). Con Box-Cox queda en ≈ 0, y la mansión de 2.1 millones pasa de estar a **~17 desviaciones** de la media a **~4.4**.

```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method="box-cox")             # o "yeo-johnson"
df_t = df.copy()
df_t[["precio_usd"]] = pt.fit_transform(df[["precio_usd"]])
df["precio_usd"].skew(), df_t["precio_usd"].skew()   # ≈ 4.12  ->  ≈ 0.00
```

### Correlación con variables binarias: V de Cramér
- Arturo pregunta cómo correlacionar una variable binaria (*"campaña: 1 el día que hay, 0 el que no"*).
- Respuesta: una binaria *"es una categórica de dos categorías"*. Entre categóricas se usa la **V de Cramér** (*"la función de Cramér"*).
- ⚠️ *"Correlación de categóricas entre ellas y correlación de numéricas entre ellas: **no lo mezclen**. Sí les va a salir algo, pero no algo bueno."*

📌 Extra: la V de Cramér va de **0** (no hay relación) a **1** (una variable determina a la otra). En el dataset, `tipo_inmueble` y "tiene 0 habitaciones" dan **V = 1** (solo las oficinas tienen 0 habitaciones), mientras que `tipo_inmueble` y `estado` dan **V ≈ 0.04** (casi no se relacionan). La fórmula está en el reto del ejercicio.

---

## 7. Lo que tienes que quedarte
1. Outlier = dato que se desvía mucho del resto. Puede venir de un **error de medición** o de **ejecución**.
2. **Descriptivo:** puede ser un hallazgo. **Predictivo:** es sesgo que impide generalizar.
3. **No se eliminan todos:** borrar filas destruye datos buenos de otras columnas.
4. Antes de tratarlo, decide qué significa: **error**, **límite** o **punto de interés**.
5. Dos tratamientos: **eliminar** o **llevar a límites** (capping), este último cuando la variable tiene límites conocidos.
6. En series de tiempo, un pico que **se repite** es estacionalidad, no outlier.
7. `PowerTransformer` (Box-Cox solo positivos / Yeo-Johnson acepta negativos) acerca variables a la normal. Correlación: categóricas con **V de Cramér**, sin mezclarlas con numéricas.

## 🔗 Conexión con ML / IA
- Los modelos lineales (regresión, y luego Lasso/Ridge) **aprenden promedios**: un outlier mueve la recta y empeora las predicciones para todos los demás.
- El capping deja el dataset **completo** (sin huecos), lo que importa cuando las filas son escasas o cuando son mediciones en el tiempo.
- `PowerTransformer` es un paso típico de **preprocesamiento** dentro de un `Pipeline` de sklearn, antes de PCA o de modelos que asumen normalidad.
- La V de Cramér se usará en el **Módulo 3** (selección de variables) para descartar categóricas redundantes.

---

## ❓ Preguntas de repaso
1. Da un ejemplo en el que un outlier sea un **hallazgo valioso** y otro en el que sea un **sesgo** para el modelo. ¿Qué cambia entre uno y otro?
2. ¿Por qué el profesor dice que **no se pueden eliminar todos** los outliers? Da las dos razones.
3. En `viviendas_lima.csv`, clasifica cada caso como error, límite o punto de interés: (a) antigüedad de 150 años, (b) una casa de 2.1 millones de USD en Jesús María, (c) un departamento de 689 m² que cuesta 138 700 USD.
4. ¿Cuándo tiene sentido **llevar a límites** (capping) en lugar de eliminar? ¿Por qué no aplicarías capping a los kilos cosechados de Carlos?
5. Tienes una columna con valores negativos y quieres acercarla a una normal. ¿Usarías Box-Cox o Yeo-Johnson? ¿Por qué?
