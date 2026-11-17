# Clase 10 · Caso Panorama de la vivienda (1): comparar escenarios
**Curso:** Python para Ciencia de Datos · **Módulo 2** (Outliers y encoding) · Título oficial: *M2-5 · Caso de uso Panorama de la vivienda (Docente)* · Transcript: `transcripts/n1-python/clase-10.txt`

> Clase **práctica** corta (~16 min). Es el **planteamiento** de un caso que sigue en las clases 11 (parte 2) y 12 (versión de los alumnos, que alimenta la entrega (a) del proyecto N1). El profesor no resuelve todo: arma el **escenario 1** (el más básico) para mostrar la **dinámica** y deja la tarea de construir 4 escenarios más.
>
> El profesor usa el dataset de **Melbourne** (precios de viviendas, objetivo `Price`). En el repo practicamos con su análogo, `data/raw/viviendas_lima.csv` (objetivo `precio_usd`). Los ejemplos con datos de Lima y todo lo marcado 📌 Extra no lo dijo el profesor: son ayudas para entender.

---

## 0. La idea en una frase
Para saber si una técnica de preprocesamiento **realmente ayuda**, se deja **todo lo demás fijo** (la división de datos y el modelo), se cambia **solo** la técnica y se mira cómo se mueve **una sola métrica**. Cada combinación que pruebas es un **escenario**.

---

## 1. El encargo del día
- Mismo dataset que la práctica de imputación de la sesión anterior (*"ustedes reportaron resultados bastante interesantes"*).
- Aplicarle **dos procesos**, en este orden:
  1. **Imputación** de faltantes (ya lo hicieron la sesión pasada).
  2. **Eliminación de outliers** con **PyOD**, con un **máximo de 10 % de contaminación**. Algoritmos: el **kNN** (*"que es famoso"*) y **otros tres** a elección (*"los dejo a su libre criterio"*).
- Después, un **código de entrenamiento** pequeño. Es una tarea de **regresión** (predecir un número: el precio).
- *"No me voy a concentrar en el análisis de supuestos [...] vamos a concentrarnos únicamente en **una métrica** y ver cómo es afectada por la estrategia que utilicemos."*
- Generar **cinco escenarios**. El profesor arma el primero.

> Al inicio importó cosas de clasificación (`accuracy_score`, matriz de confusión) y él mismo aclaró: *"creo que no lo vamos a utilizar porque esta es una tarea de regresión"*.

🧠 **Mini-pregunta:** si quieres predecir **si** una casa se vende en menos de un mes (sí/no), ¿es regresión o clasificación? ¿Y si quieres predecir **en cuántos días** se vende?

---

## 2. Vocabulario nuevo (explicado simple)
| Término | Qué es | Ejemplo con viviendas |
|---|---|---|
| **Variable objetivo (`y`)** | Lo que el modelo tiene que predecir | `precio_usd` (en Melbourne, `Price`) |
| **Predictores (`X`)** | Las columnas que el modelo usa para predecir | `area_m2`, `habitaciones`, `banos`… |
| **Train / test** | Partes los datos en dos: con **train** el modelo aprende; con **test** lo examinas con casas que **nunca vio** | 80 % = 1612 casas para aprender, 20 % = 403 para examinar |
| **Random Forest** | Muchos **árboles de decisión** (reglas tipo "si área > 100 y distrito caro → precio alto"); cada uno predice y se **promedian** | 50 árboles opinan y se toma el promedio |
| **Hiperparámetro** | Una "perilla" del modelo que **tú** fijas antes de entrenar (no la aprende de los datos) | `n_estimators=50`, `max_depth=6` |
| **Overfitting (sobreajuste)** | El modelo **memoriza** el train (incluido su ruido) y falla con casas nuevas | Un árbol tan profundo que tiene una regla para cada casa |
| **MAE** (*mean absolute error*) | Promedio de los errores en valor absoluto. Mismas unidades que `y`. **Menor es mejor** | MAE = 46 000 → "en promedio me equivoco por 46 mil dólares" |
| **Escenario** | Una combinación concreta de preprocesamiento | "mediana + sin quitar outliers" |

Ejemplo de MAE a mano: precios reales 100, 200 y 300 mil; predicciones 110, 180 y 300 mil → errores 10, 20 y 0 → MAE = (10 + 20 + 0) / 3 = **10 mil**.

🧠 **Mini-pregunta:** si el MAE de un escenario es 45 600 y el de otro es 46 500, ¿cuál prefieres y por qué?

---

## 3. Escenario 1 (básico), paso a paso
### 3.1 Separar `y` y `X`
```python
y = df["Price"]                                   # objetivo
X = df.drop(columns=["Price"]).select_dtypes(exclude="object")   # solo numéricas
```
- *"Me voy a encargar de trabajar solamente con las variables numéricas, todavía no nos vamos a centrar en el tratamiento de variables tipo object"*. En Melbourne hay varias: `Method`, `SellerG`, `Date`, `CouncilArea`… (el encoding de texto llega en la clase 09 y en el caso completo).
- *"Incluso hay **dos columnas que deberían irse** porque sabemos que sus valores faltantes son demasiados"*. Esa limpieza es parte de **tu** tarea (el básico no la hace).
- En Lima (📌 Extra): `X` queda con 7 numéricas. Quitamos también `id`: es un número, pero no dice nada del precio.

### 3.2 Dividir en train y test (fuera de los escenarios)
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
```
- *"80/20 y un random state de 0, por favor mantengan esta configuración para que todos tengamos lo mismo"*.
- El profesor movió esta celda **arriba**: *"esta división no tiene nada que ver con ningún escenario"*. Se hace **una vez** y todos los escenarios la comparten. Si cada escenario barajara distinto, no sabrías si el MAE cambió por la técnica o por la suerte de la división.

### 3.3 El modelo y `score_dataset` (también fijos)
```python
def score_dataset(X_train, X_valid, y_train, y_valid):
    modelo = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=0)
    modelo.fit(X_train, y_train)
    preds = modelo.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
```
- **Random Forest Regressor**: *"un regresor basado en árboles de decisión [...] múltiples árboles de decisión"*. Eligió este en vez de uno lineal o Lasso *"por cuestiones didácticas"*.
- `n_estimators=50`: *"el número de árboles; cada árbol va a hacer una predicción [...] al final se va a promediar"*.
- `max_depth=6`: *"una profundidad máxima del árbol de seis para evitar que se sobreajuste demasiado"* (overfitting).
- *"Esto se llaman **hiperparámetros** y hay que calibrarlos"*, y no son los mismos para cada algoritmo. `random_state` también se puede ver como uno (la semilla).
- MAE: *"una métrica de error que me va a decir qué tan ajustados están los puntos con respecto al modelo"*. Más detalle en clases futuras.

### 3.4 Lo único que cambia: los tres pasos de cada escenario
1. **Definir el imputador** → en el básico, `SimpleImputer(strategy="median")` (*"lo más simple"*; si no le pones estrategia, por defecto usa la **media**; el profesor la escribe explícita para que se vea).
2. **Eliminar outliers** → en el básico **se omite**: *"me pasé por alto los outliers en este escenario, y vamos a ver qué efecto tiene"*.
3. **Entrenar y medir** → `error = score_dataset(...)` e imprimirlo.

> 📌 Extra (corrección): `SimpleImputer` sin `strategy` usa `"mean"` (media), no la mediana. Por eso conviene escribir siempre `strategy="median"`.

---

## 4. La regla clave: `fit_transform` en train, `transform` en test
```python
my_imputer = SimpleImputer(strategy="median")
X_train_imp = my_imputer.fit_transform(X_train)   # APRENDE las medianas del train y las aplica
X_test_imp  = my_imputer.transform(X_test)        # SOLO aplica lo aprendido
```
- *"`fit_transform` es para que aprendan los parámetros de los datos de **entrenamiento**, y luego los aplico a los datos de test: ahí ya **no** puedo utilizar `fit_transform`, voy a utilizar solamente `transform`"*.
- *"No se olviden de este detalle, que es muy importante"*.
- `fit` = **aprender** (calcular la mediana). `transform` = **usar** lo aprendido (rellenar).

### 📌 Extra · ¿Por qué? El *data leakage* con números
Train `banos` = [1, 2, 3, NaN] → mediana del train = **2**. Test `banos` = [NaN, 9, 9].
- ✅ Correcto: el NaN del test se rellena con **2** (lo que sabías al entrenar).
- ❌ Con `fit_transform` en el test: se rellena con **9** (la mediana del test).

El test simula **casas futuras** que todavía no has visto. Si usas información del test para preparar los datos, el modelo "espía el examen" (**data leakage**, fuga de datos): el MAE sale más optimista de lo que será en la realidad.

🧠 **Mini-pregunta:** si haces `imputer.fit(X)` con **todo** `X` antes de dividir, ¿hay leakage? ¿Por qué?

---

## 5. Lo que el escenario básico ignora (y tu tarea)
- *"Me pasé por alto todo el análisis de qué tipo de valor perdido tengo, si es un MCAR, un MAR o un MNAR [...] imputar todo con la mediana"*. (Recuerda la clase 02: *"no siempre se puede imputar mediante una mediana"*.)
- La tarea: probar imputadores más avanzados (**`KNNImputer`**, **`IterativeImputer`**) y eliminación de outliers (**kNN de PyOD** u otro), y reportar *"cuánto se afectó el mean absolute error, si ha subido o ha disminuido"*.
- Al final tendrás `X` **con** y **sin** remoción de outliers para comparar.

### 📌 Extra · Cómo quitar outliers sin romper nada
1. Detectar outliers **después de imputar** (estos algoritmos no aceptan NaN).
2. Eliminarlos **solo del train**: el test representa el mundo real y no se "limpia" (no puedes negarte a predecir el precio de una casa porque es rara).
3. Filtrar **`X_train` e `y_train` con la misma máscara**. Si solo filtras `X_train`, las filas de `X` e `y` quedan desalineadas (el modelo aprendería el precio de otra casa).
4. PyOD no está instalado en el repo: usamos **`IsolationForest`** de scikit-learn (también está en PyOD; lo viste en la clase 08 junto al puntaje kNN). Ojo con la convención: scikit-learn devuelve `1 = inlier`, `-1 = outlier`; PyOD usa `0 = inlier`, `1 = outlier`.

### 📌 Extra · Resultados en `viviendas_lima.csv` (como referencia)
| Escenario | MAE (USD) | Filas train |
|---|---|---|
| Mediana + sin outliers (IsolationForest 10 %) | ≈ 45 600 | 1450 |
| **1 · Básico: mediana** | ≈ 46 550 | 1612 |
| IterativeImputer | ≈ 46 760 | 1612 |
| IterativeImputer + sin outliers | ≈ 46 860 | 1450 |
| KNNImputer (sin escalar) | ≈ 50 380 | 1612 |

Lecciones: (1) lo "avanzado" **no siempre gana**: `KNNImputer` sin escalar las columnas mide distancias dominadas por `area_m2` y sale peor; (2) quitar outliers ayudó con la mediana pero no con `IterativeImputer`: **hay que medir, no suponer**; (3) las diferencias son de ~1–4 mil USD sobre ~46 mil: pequeñas, pero es justo lo que esta metodología permite ver.

---

## 6. Flujo completo (resumen)
```
cargar df
y, X  = objetivo, solo numéricas              ─┐
X_train, X_test, ... = split 80/20, rs=0        ├─ FIJO para todos
score_dataset = RandomForest(50, depth 6) → MAE ─┘
para cada escenario:
    1. imputador  → fit_transform(train), transform(test)
    2. outliers   → (opcional) quitar SOLO del train, X e y juntos
    3. MAE        → score_dataset
tabla de escenarios ordenada por MAE
```

---

## 7. Lo que tienes que quedarte
- Comparar escenarios = **cambiar una sola cosa** y dejar fijo lo demás (split, modelo, métrica).
- `fit_transform` en train, **solo** `transform` en test.
- Los outliers se quitan **solo del train**, filtrando `X` e `y` a la vez, con máximo 10 % de contaminación.
- El básico (mediana, sin outliers) es la **línea base**: todo escenario se compara contra él.

## 🔗 Conexión con ML / IA
- Esta es la forma real de decidir el preprocesamiento de un modelo: **experimentos controlados** contra una línea base. En el proyecto N1 (entrega a) tendrás que **justificar** cada decisión (imputación, outliers) y esta tabla de escenarios es tu evidencia.
- La regla `fit` en train / `transform` en test aplica a **todo**: escaladores, encoders, PCA… Más adelante, `Pipeline` de scikit-learn la automatiza para que no se te olvide.
- En sistemas de IA en producción, el *data leakage* es de los errores más caros: el modelo luce excelente en pruebas y falla con datos reales.

## Ejercicio (`ejercicio.py`)
| # | Función | Qué practica |
|---|---|---|
| 1 | `preparar_xy` | Separar `y` / `X` solo numéricas (3.1) |
| 2 | `dividir` | Split 80/20 fijo (3.2) |
| 3 | `score_dataset` | El modelo fijo y el MAE (3.3) |
| 4 | `imputar_train_test` | `fit_transform` en train, `transform` en test (4) |
| 5 | `quitar_outliers_train` | Outliers solo del train, X e y alineados (5) |
| 6 | `evaluar_escenarios` | Automatizar los escenarios y ordenarlos por MAE |

Se relaciona con: `imputar_knn` (clase 04) y `detectar_isolation_forest` / `filtrar_inliers` (clase 08). Aquí el foco no es el detector sino **dónde** y **cuándo** se aplica.

## ❓ Preguntas de repaso
1. ¿Por qué el profesor saca el `train_test_split` fuera de los escenarios y pide a todos `random_state=0`?
2. ¿Qué aprende `fit` en un `SimpleImputer(strategy="median")`, y por qué en el test solo se usa `transform`?
3. ¿Qué hacen `n_estimators=50` y `max_depth=6` en el Random Forest? ¿Por qué se llaman hiperparámetros?
4. Si quitas outliers de `X_train` pero olvidas filtrar `y_train`, ¿qué pasa? ¿Y si los quitas también del test?
5. El escenario con `KNNImputer` dio peor MAE que la mediana en Lima. Propón una razón y cómo lo comprobarías.
