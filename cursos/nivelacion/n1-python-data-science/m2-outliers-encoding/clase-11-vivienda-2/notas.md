# Clase 11 · Caso Panorama de la vivienda, parte 2 (docente)
**Curso:** Python para Ciencia de Datos · **Módulo 2** (Outliers y encoding) · Título oficial: *M2-6 · Caso de uso Panorama de la vivienda Parte 2 (Docente)* · Transcript: `transcripts/n1-python/clase-11.txt`

> Clase práctica de unos 38 minutos. Continúa el laboratorio de la **clase 10** (Melbourne, objetivo `Price`). Tiene tres partes:
> 1. Un **debate**: ¿imputar y quitar outliers **antes** o **después** de dividir train/test?
> 2. Un **escenario avanzado**: imputar con `KNNImputer` → quitar outliers con PyOD `KNN` → dividir → medir el MAE.
> 3. **Preguntas de alumnos** (Arturo y otro compañero): cuándo se puede imputar, la mediana con datos no normales y `PowerTransformer`.
>
> El profesor trabajó con el dataset de **Melbourne** (18 396 filas). Aquí repetimos cada paso con `data/raw/viviendas_lima.csv` (2015 filas, objetivo `precio_usd`). Los números de Lima y el código sobre Lima son 📌 Extra.
>
> Requisito: la **clase 10** (`clase-10-vivienda-1`), que trae `dividir`, `score_dataset`, `imputar_train_test`… Esta clase **no** las repite.

---

## 0. La idea en una frase
Antes de entrenar hay que dejar **X sin nulos** y **sin filas anómalas**, sin tocar nunca la **y**. Los tamaños de X e y tienen que coincidir siempre. Y conviene saber **en qué momento** se ajusta cada transformación (antes o después de dividir), porque eso cambia la **honestidad** de tu MAE.

---

## 1. El debate: ¿antes o después de dividir?

### Lo que dijo el profesor
- *"Siempre hay una conversación sobre cuándo hacerla: antes de dividir o después de dividir los datos."*
- Él se inclina por **antes**: *"yo considero que este proceso debería hacerse mucho antes de hacer la división"*, *"esta imputación realmente debería hacerse a la data total, no a la data ya particionada"*.
- Da **tres razones**:
  1. **Preservar la integridad de los datos de prueba.**
  2. **Evitar la fuga de datos** (*data leakage*).
  3. **Tener mejor control del proceso.**
- Su idea de fondo: si imputo solo con el 80 % de train, *"estoy reduciendo la muestra [...] puedo dejar de lado ciertas cuestiones"*, porque el imputador aprende de menos datos.
- **El argumento contrario**, que él también explica: al dividir, *"los puntos que están aquí son puntos característicos relevantes de toda esta población, y los que quedan también"*. Es decir, train y test siguen siendo **representativos**, así que se puede imputar después.
- **Depende de cuántos datos tengas:** *"cuando uno tiene muchísima data, no afecta realmente, el performance se mantiene"*. *"Cuando hay poca data puede ser que sí veamos un efecto."*
- Lo que **no** se discute: *"el escalamiento debería hacerse después [...] la reducción de dimensionalidad, sobre todo si van a aplicar PCA, también debería hacerse después. Eso sí no hay debate."*

| Transformación | Postura del profesor |
|---|---|
| Imputación de nulos | Antes de dividir (es su preferencia y lo reconoce como debate) |
| Quitar outliers | Antes de dividir (también es debate) |
| Escalamiento (StandardScaler, MinMaxScaler…) | **Siempre después**: `fit` en train, `transform` en test |
| Reducción de dimensionalidad (PCA) | **Siempre después** |

### 📌 Extra: lo que hace la industria (y por qué)
Te lo cuento con honestidad para que conozcas **las dos posturas**. La práctica más extendida en la industria y la que recomienda la documentación de scikit-learn es **ajustar el imputador y el detector de outliers SOLO con train**. Se hace dentro de un `Pipeline` o de una validación cruzada, igual que el escalamiento. ¿Por qué?

- El test simula **datos del futuro** que el modelo nunca vio. Si imputas **antes** de dividir, la mediana o los vecinos del `KNNImputer` se calculan **también con las filas de test**. Así, un poco de información del test se "cuela" en lo que ve el modelo al entrenar. Eso es justamente **data leakage**.
- Ejemplo concreto con Lima: con `SimpleImputer(strategy="median")`, la mediana de `area_m2` aprendida de **todo** X es **94.6**. Aprendida **solo de train** es **95.15**. La diferencia es pequeña, pero el primer número **ya conoce el test**.
- En producción (la API del Curso 1) llegará **una sola casa nueva**: no podrás "re-imputar con toda la data" porque esa data no existe todavía. El imputador tiene que estar **ya ajustado** (con train) y solo hacer `transform`. Por eso se entrena así desde el principio.
- Con los outliers pasa algo parecido. Si los quitas **antes** de dividir, el test también queda "limpio" y el MAE parece mejor. Pero en la vida real **sí** llegarán casas raras al modelo.

**¿Entonces el profesor se equivoca?** No necesariamente. Él mismo dice que con muchos datos el resultado es casi igual, y eso suele ser cierto. Su preocupación (que el imputador aprenda de una muestra más pequeña) es real cuando hay pocos datos. Lo profesional es **medirlo**: mismo split, mismo modelo, y lo único que cambia es dónde haces el `fit`. Eso es la función 6 del ejercicio (`comparar_antes_vs_despues_split`). En Lima sale (📌 Extra):

| Imputador (mediana) ajustado con… | MAE (USD) |
|---|---|
| Todo X (**antes** de dividir) | ≈ 46 329 |
| Solo X_train (**después** de dividir) | ≈ 46 552 |

La versión "antes" sale **un poco mejor**. Es lo esperable: el imputador ya conocía el test, así que la nota es algo **optimista**. La diferencia es de solo ~220 USD, lo que confirma lo que dijo el profesor: con suficientes datos, el efecto es pequeño.

🧠 **Mini-pregunta:** si tu imputador por mediana se ajusta con **todo** X y luego divides, ¿el modelo "vio" de alguna forma las filas de test? ¿Qué parte de ellas?

---

## 2. El escenario avanzado, paso a paso

### Paso 0 · Separar X e y
- *"Primero quiero mantener separado mi X de mi Y."*
- Se queda solo con las **numéricas** (`select_dtypes`): *"el tratamiento de variables categóricas lo vamos a dejar para después"*.
- `X = df.drop("Price")`, `y = df["Price"]`.
- **El objetivo no debe tener nulos:** *"si hay nulos tenemos un problema, porque tendríamos que imputar la variable objetivo, lo cual ya es otro dolor de cabeza"*. En Melbourne no los había: *"por suerte no tengo que imputar"*. Los nulos quedan solo en los **predictores**.

```python
# 📌 Extra: lo mismo con Lima
df = pd.read_csv("data/raw/viviendas_lima.csv")
X = df.select_dtypes(include="number").drop(columns=["precio_usd", "id"])   # 7 columnas
y = df["precio_usd"]                                                         # sin nulos
```
> 📌 Extra: también quitamos `id`. Es numérico, pero no describe la casa. Un modelo podría aprender patrones falsos del orden de carga.

### Paso 1 · Imputar con `KNNImputer`
- Se ajusta con X y sale `X_imp`. En Melbourne: *"no tengo un valor faltante, tengo 18 396 registros con 12 columnas"*.
- La regla que repite: *"estos tamaños deberían coincidir porque no estoy ni agregando ni quitando filas, lo único que estoy haciendo es completando datos nulos [...] por favor no se olviden de que estos tamaños coincidan sí o sí"*.

```python
from sklearn.impute import KNNImputer
X_imp = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(X), columns=X.columns, index=X.index)
print(X_imp.shape, y.shape)   # (2015, 7) (2015,)  -> mismas filas
```

### Paso 2 · Quitar outliers con PyOD `KNN`
- *"Ahora el siguiente paso delicado es la eliminación de outliers, porque aquí sí van a ser filtrados."*
- `from pyod.models.knn import KNN`, `clf = KNN(contamination=0.1)`, `clf.fit(X_imp)`.
- Aviso del profesor: *"no confundir este KNN con el KNN de machine learning"*. Este KNN es el **detector de outliers** de PyOD, no el clasificador.
- **`fit` todavía no borra nada:** *"lo único que estamos haciendo es ajustarlo [...] ha analizado todo mi dataframe, pero no elimina nada"*.
- `y_pred = clf.labels_` → *"las etiquetas de cada fila"*: **0 = inlier, 1 = outlier**. Es una Serie de 18 396 valores, casi todos 0 y algunos 1.
- **Recién aquí se filtra**, con la misma máscara para X y para y: `X_r = X_imp[y_pred == 0]`, `y_r = y[y_pred == 0]`. Melbourne pasa de **18 396 → 16 842** filas.
- `X_r` sale como **array** de numpy, así que se reconvierte: `pd.DataFrame(X_r, columns=X.columns)`. *"Las columnas deberían ser las mismas, solamente he reducido filas."*
- *"En ningún momento he tocado a y"*: el detector trabaja solo con X.

**Por fila, no por variable** (pregunta de Samuel):
> *"No lo hace por variable, lo hace por registro, porque una fila es un vector de características en R^n. Cada uno de estos registros es un punto. Basta que uno de estos se altere y automáticamente se sale de la dimensión original."*

Es decir, una casa no es "un área rara" o "un baño raro". Es **un punto** en un espacio de 12 dimensiones (7 en Lima). KNN mide distancias entre esos puntos.

#### 📌 Extra: convenciones de etiquetas (¡ojo!)
| Librería | Inlier | Outlier |
|---|---|---|
| PyOD (`labels_`) | `0` | `1` |
| scikit-learn (`fit_predict`) | `1` | `-1` |
| Este ejercicio (`mascara_inliers`) | `True` | `False` |

PyOD no está instalado en este repo. Usamos los equivalentes de scikit-learn: `IsolationForest` (como en la clase 08) y `LocalOutlierFactor`, que, igual que el KNN de PyOD, se basa en **distancias a vecinos**. En la clase 08 ya convertiste -1/1 a 0/1 (`convertir_etiquetas_sklearn`). Aquí vamos directo a una **máscara booleana**, que es lo que de verdad necesitas para filtrar.

### Paso 3 · Dividir y medir
- `train_test_split(X_r, y_r, train_size=0.8, test_size=0.2, random_state=...)` y luego `score_dataset(X_train, X_test, y_train, y_test)` de la clase 10.
- Resultados en Melbourne:

| Escenario | MAE |
|---|---|
| Mediana, sin quitar outliers (escenario 1, "por defecto") | ≈ 235 423 |
| **KNNImputer + PyOD KNN** | **≈ 215 762** → *"bajó el error bastante"* |
| KNNImputer + PyOD ABOD | ≈ 257 62x → *"funcionó peor en este caso"* |

- Para probar otro detector *"simplemente es cambiar aquí el import"*.
- **Tarea:** *"prueben estos escenarios"*. Y añade: *"en un caso real se utilizan más estos métodos en la práctica"*.

📌 Extra, lo mismo con Lima (RandomForest 50 árboles, `max_depth=6`, split 80/20, `random_state=0`):

| Escenario | Filas | MAE (USD) |
|---|---|---|
| Mediana, sin quitar outliers | 2015 | ≈ 46 329 |
| KNNImputer, sin quitar outliers | 2015 | ≈ 49 951 |
| KNNImputer + IsolationForest (10 %) | 1813 | ≈ 45 915 |
| KNNImputer + LOF (10 %) | 1813 | ≈ 45 498 |
| KNNImputer + KNN "a lo PyOD" (distancia al 5.º vecino, 10 %; ver notebook) | 1813 | ≈ 44 963 |

> 📌 Extra, lectura crítica: cuando quitas outliers **antes** de dividir, el **test también pierde sus casas raras**. Por eso parte de la mejora es que el examen se volvió más fácil, y no solo que el modelo sea mejor. Para comparar de verdad, el test debería ser el mismo en todos los escenarios. Es la misma idea del punto 1.

🧠 **Mini-pregunta:** después de filtrar outliers, `X_r` tiene 1813 filas. ¿Cuántas debe tener `y_r`? ¿Qué pasaría si filtraras X pero olvidaras filtrar y?

---

## 3. Consideraciones para imputar (pregunta de Arturo)
1. **Mira el porcentaje de nulos.**
   - *"Si superan el 30 % ya es muy difícil."* Hasta un 30 % se puede, aunque *"igual siempre hay error"*.
   - Con 50–60 %: *"hay que eliminar esas variables"*.
   - *"Y si me fuerzan a utilizar la variable, tendríamos que ver estrategias, pero ya sabiendo que tenemos sesgo en el modelo."*
2. **La mediana falla si la distribución no es normal.** *"Por más que imputes con una mediana te va a aparecer un pico."* Es lo que viste en el módulo 1: todos los nulos caen en el mismo valor.
3. **Usa métodos que no dependan de la normalidad:**
   - **`KNNImputer`**: *"busca relaciones entre los puntos"*.
   - **`IterativeImputer` con un regresor de árbol** (ExtraTrees, que se usó en clase, XGBoost o LightGBM): *"no necesitan llevarlo tanto a la normalidad"*.
   - En cambio, `IterativeImputer` con una regresión **logística/lineal** *"obviamente necesita normalidad"*.
   - La alternativa es transformar la variable para acercarla a la normal, pero *"eso ya es convertir la variable"*.

📌 Extra, en Lima: `banos` tiene 89 nulos (4.4 %), `antiguedad_anios` 170 (8.4 %), `distancia_centro_km` 158 (7.8 %) y `piso` 454 (22.5 %). Ninguna pasa del 30 %. Pero **ojo con `piso`**: sus 454 nulos son todos **casas** (una casa no tiene "piso"). Imputarles un piso con KNN inventa un dato que no existe. Este es un nulo estructural (MNAR, módulo 1), y una opción mejor sería una marca `es_casa` o un valor fijo.

## 4. `PowerTransformer`, PCA y t-SNE (pregunta de un compañero)
- **¿Cuándo usar `PowerTransformer`?** Cuando el **método que viene después necesita normalidad**. El ejemplo del profesor es **PCA**: *"funciona con variables numéricas que tienen aproximación tipo normal"*.
- Tiene dos métodos. El profesor recordó **Box-Cox** (*"asume que los valores tienen que ser positivos"*), pero no el nombre del otro, que acepta negativos. 📌 Extra: el otro es **Yeo-Johnson**, que es el método por defecto de `PowerTransformer`.
- *"Pero mis datos no son normales; si los transformo los estoy forzando."* Para eso propone un **método alternativo: t-SNE**, que *"no necesita que sea normal"* y que *"funciona mucho mejor que PCA [...] en la mayoría de casos, no siempre"*.
- Si tu clasificador **requiere** variables normales, *"necesito transformar los datos sí o sí"*. `KNNImputer` no lo necesita.

📌 Extra: `PowerTransformer` también **aprende parámetros** (el λ de cada columna). Por eso va en el grupo de "después de dividir", igual que el escalamiento: `fit` en train y `transform` en test.

---

## 5. Lo que viene
*"Vamos a ver lo que es tratamiento de variables categóricas y lo vamos a incluir dentro de este laboratorio."* Hasta ahora solo usamos las numéricas. `distrito`, `tipo_inmueble` y `estado` esperan su turno (encoding, clase 09).

## 6. ¿Para qué sirve en ML/IA? (📌 Extra)
- **Imputar sin perder filas** te deja entrenar con todos los datos. La mayoría de modelos de scikit-learn no aceptan NaN.
- **Quitar outliers por fila** evita que unas pocas casas imposibles "jalen" el modelo (clase 06).
- **Saber dónde va cada `fit`** separa un MAE honesto de uno optimista. En el Curso 1 (sistema `mentor-ia`, capa v1) tu modelo vivirá detrás de una API y recibirá **filas nuevas de una en una**. Ahí solo existe el `transform` de lo que ajustaste con train.

---

## ❓ Preguntas de repaso
1. Nombra las tres razones del profesor para imputar **antes** de dividir, y el argumento a favor de hacerlo **después**.
2. ¿Qué dos transformaciones dijo que van **siempre** después de dividir? ¿Qué tienen en común con un imputador (📌 Extra: ¿qué "aprenden"?)
3. ¿Por qué `X_imp` y `y` deben tener las mismas filas después de imputar? ¿Y por qué tras quitar outliers hay que filtrar **las dos** con la misma máscara?
4. En PyOD, ¿qué significa `labels_ == 1`? ¿Y `fit_predict == -1` en scikit-learn?
5. Tienes una variable con 55 % de nulos y distribución muy sesgada. Según la clase, ¿qué harías? ¿Y si te obligan a usarla?
