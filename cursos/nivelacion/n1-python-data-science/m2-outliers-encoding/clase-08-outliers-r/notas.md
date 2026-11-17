# Clase 08 · Outliers multivariados con algoritmos de ML (PyOD)
**Curso:** Python para Ciencia de Datos · **Módulo 2** (Outliers y encoding) · Título oficial: *M2-3 · Outliers en R* · Transcript: `transcripts/n1-python/clase-08.txt`

> ⚠️ **Sobre el título:** el video se llama *"Outliers en R"*, pero **no hay nada de R**. Todo es **Python** y es la **continuación directa de la clase 07**, que terminó justo cuando el profesor iba a presentar la librería **PyOD**. Son unos 19 minutos: la primera mitad es PyOD con el dataset de Boston, y la segunda son preguntas de alumnos (cómo elegir el mejor algoritmo, pipelines, Polars).
>
> Los ejemplos con `viviendas_lima.csv` y los equivalentes en scikit-learn son 📌 Extra: PyOD no está instalado en este repo, así que el ejercicio usa scikit-learn (que hace lo mismo) y la sección 7 explica la correspondencia.

---

## 0. La idea en una frase
Cuando hay **muchas columnas**, no elimines outliers columna por columna: deja que un **algoritmo de ML** clasifique cada fila como **inlier** u **outlier** mirando todas las columnas a la vez, y decide **cuánto dataset aceptas sacrificar** con `contamination`.

---

## 1. ¿Por qué no basta con limpiar columna por columna?
- Al empezar, el profesor advierte: si eliminas outliers **de forma unidimensional** (una columna cada vez, como con IQR en la clase 07), *"puedo haber generado"* outliers a nivel **multidimensional**. Lo que recomienda: *"dejarlo en manos de un algoritmo tipo machine learning"*.
- Algo que conecta con todo el curso: *"tanto el análisis de valores faltantes como el caso de outliers al final se reducen a ser resueltos con modelos de Machine Learning"* (como la imputación con KNN o MICE de la clase 03).
- Además, el IQR por columnas era muy caro: en Boston se perdió **casi la mitad** de las filas (*"eliminé el 40 % de los datos, y eso que me faltó analizar a nivel multivariante"*).

---

## 2. Detectar outliers = clasificación binaria
- Lo que hacen estos algoritmos: *"cada punto se clasifica en dos clases: la clase inlier y la clase outlier"*.
- **Cada fila es un punto.** *"Cada registro que tienen en su dataset es un punto, solo que en una dimensión n: si tienen 20 variables, es un punto en la dimensión 20"* (en **R^20**).
- Un alumno preguntó si el algoritmo es univariado. Respuesta: **no, es multivariado**. Trabaja con *"puntos en la dimensión original"*: cada registro es *"un vector, un punto en el espacio"*.
- Otro alumno preguntó si eso es un **tensor**. El profesor dijo que **no**: es un *"vector de características"*. Los tensores son *"otro concepto"*, un tipo de dato de PyTorch pensado *"para mandarlo a la GPU"*.

Ejemplo con nuestro dataset (📌 Extra): si tomamos 6 columnas numéricas de `viviendas_lima.csv` (sin `precio_usd`, que es lo que queremos predecir), la primera vivienda es el punto

```
(area_m2=30, area_construida_m2=28.5, habitaciones=1, banos=1, antiguedad_anios=15, distancia_centro_km=3.05)
```

Es decir, un punto en **R^6**.

🧠 **Mini-pregunta:** `clientes_banco.csv` tiene, digamos, 12 columnas numéricas (sin la columna objetivo). ¿En qué espacio "vive" cada cliente?

---

## 3. La librería PyOD
- **PyOD** (*Python Outlier Detection*) reúne en un solo lugar muchos algoritmos de detección de outliers que *"antes se hacían manualmente, pero ahora ya está todo encapsulado"*. Tiene algoritmos **no supervisados, semisupervisados y supervisados**.
- No viene con Python: se instala con `pip install pyod`.
- Advertencia del profesor: *"no va a funcionar siempre el mismo algoritmo"*. Hay que ver cuál da mejor desempeño **para tu caso de uso**.

### El algoritmo kNN (k vecinos más cercanos)
*"Los vecinos que están más próximos a cierto punto son considerados inliers, y los puntitos más alejados, muy aislados, muy lejanos, son categorizados como outliers."* Es *"un algoritmo bastante básico pero poderoso"*.

Hay **tres formas** de convertir las distancias a los vecinos en un **puntaje de outlier**:

| Método en PyOD | Puntaje | Qué mide |
|---|---|---|
| `"largest"` (por defecto) | distancia al **k-ésimo** vecino | lo lejos que está el vecino más lejano del grupo |
| `"mean"` | **promedio** de las k distancias | lo aislado que está el punto, en promedio |
| `"median"` | **mediana** de las k distancias | lo mismo, pero robusta a un vecino raro |

Ejemplo pequeño (📌 Extra), con los puntos `0, 1, 2, 3, 10` en una recta y **k = 3**:

| Punto | Sus 3 vecinos | Distancias | largest | mean | median |
|---|---|---|---|---|---|
| 0 | 1, 2, 3 | 1, 2, 3 | 3 | 2 | 2 |
| 1 | 0, 2, 3 | 1, 1, 2 | 2 | 1.33 | 1 |
| 2 | 1, 3, 0 | 1, 1, 2 | 2 | 1.33 | 1 |
| 3 | 2, 1, 0 | 1, 2, 3 | 3 | 2 | 2 |
| **10** | 3, 2, 1 | 7, 8, 9 | **9** | **8** | **8** |

El 10 tiene el puntaje más alto con los tres métodos, así que es el outlier.

- El profesor nombró otros algoritmos: **ABOD** (basado en ángulos) y **CBLOF** (basado en clústeres).

🧠 **Mini-pregunta:** en la tabla, ¿por qué el punto 1 tiene `median = 1` pero `mean = 1.33`?

---

## 4. Código mostrado en clase (Boston, 506 filas)
```python
df_tmp = df2.copy()                         # 1. copia del DataFrame
df_tmp = df_tmp.drop(columns=["MEDV"])      # 2. quitar la variable objetivo -> 506 × 13

from pyod.models.knn import KNN             # 3. importar UNO de los algoritmos de PyOD
clf = KNN(contamination=0.2)                # 4. crear la instancia (programación orientada a objetos)
clf.fit(df_tmp)                             # 5. entrenar
pred = clf.predict(df_tmp)                  # 6. predecir: 0 = inlier, 1 = outlier

df_tmp[pred == 0].shape                     # (417, 13) -> inliers
df_tmp[pred == 1].shape                     # (89, 13)  -> outliers
```
- El nombre exacto de la columna objetivo es una suposición: en el audio se oye "inl" (📌 Extra: en Boston, el objetivo suele ser `MEDV`, el precio).
- ¿Por qué quitar la columna objetivo? Porque **queremos predecirla**. Los outliers se buscan en las **features**.
- kNN *"lo hace bastante rápido"*, aunque *"depende del tamaño del dataset"*.
- Resultado: *"me quedé con un 80 % de datos nada más, pero eliminé los puntos que estaban mucho más alejados del comportamiento normal"*.

---

## 5. `contamination`: el parámetro clave
*"Le dicen: compromete a lo mucho el 20 % de la base de datos, no elimines más que eso, y elimínate los outliers que son más perjudiciales, los que están más alejados de todo el grupo de puntos, los que te van a dar más error."*

- Es un **tope**: la proporción máxima del dataset que aceptas sacrificar.
- El profesor lo conectó con la gráfica de la clase 06: *"los puntos alejados hacían que toda la gráfica de regresión se vaya hacia arriba"*.
- Así lo explicó él: puede haber pocos outliers (por ejemplo, el 10 %) y *"ya los elimina todos"*, pero *"el máximo, el tope, es el 20 %"*.
- 📌 Extra (cómo funciona por dentro): PyOD calcula el puntaje de cada fila, toma el **percentil `(1 − contamination)`** como umbral y marca como outlier a quien lo **supera**. Por eso el resultado queda **en o por debajo** del tope: en Boston salieron 89 de 506 (≈ 17.6 %). Eso es lo que implementas en `marcar_por_contaminacion`.

| Método | Cuánto se perdió en Boston |
|---|---|
| IQR columna por columna (clase 07) | casi la mitad (506 → 268) |
| kNN de PyOD, `contamination=0.2` | 89 filas (506 → 417) |

📌 Extra con viviendas (6 columnas, 1622 filas sin NaN): kNN con `k=5` y `contamination=0.2` marca **325** outliers y conserva **1297** filas (**79.96 %**). La vivienda más aislada es la de `area_m2 = 2746` (un error de digitación ×10 de la clase 06).

🧠 **Mini-pregunta:** si pones `contamination=0.05` en un dataset de 2000 filas, ¿cuántas filas, como mucho, vas a perder?

---

## 6. Cada algoritmo marca puntos distintos
- Con puntos sintéticos en 2D y la **misma** contaminación (10 %), el profesor comparó: **ABOD detectó 19** outliers y **kNN 11**.
- En los gráficos, los puntos **blancos** son inliers y los **negros** outliers. Las zonas sombreadas son las regiones "normales" de cada algoritmo: kNN genera *"una región grandota"* y más regular, mientras que ABOD genera varias regiones *"muy globulares"*.
- Con otro dataset, **CBLOF** marcó como outliers *"un grupo de puntos que anteriormente eran inliers"*.
- **¿Cuál es mejor?** Un alumno dijo que kNN parecía más preciso. El profesor respondió que mirando la gráfica *"aquí no podemos saber nada"*: se decide por el **rendimiento del modelo final** (regresión, clasificación o clustering). Hay que hacer *"todo un protocolo de experimentación"*.
- *"El tratamiento de outliers no es algo sencillo."*

📌 Extra con viviendas, `contamination=0.2` en los dos casos:

| Comparación | Outliers A | Outliers B | Coinciden | Jaccard |
|---|---|---|---|---|
| kNN vs IsolationForest | 325 | 325 | 244 | 0.601 |
| kNN vs LocalOutlierFactor | 325 | 325 | 156 | 0.316 |

**Índice de Jaccard** = filas marcadas por **los dos** ÷ filas marcadas por **al menos uno**. Vale 1 si marcan exactamente las mismas filas y 0 si no coinciden en ninguna.

---

## 7. 📌 Extra · PyOD vs scikit-learn (lo que usarás en el ejercicio)
PyOD no está instalado aquí, pero scikit-learn trae equivalentes:

| PyOD | scikit-learn | Nota |
|---|---|---|
| `KNN(n_neighbors=k, method="largest")` | `NearestNeighbors(n_neighbors=k+1)` y la última columna de distancias | **+1** porque el vecino 0 de cada punto es **él mismo** |
| `IForest(contamination=c)` | `IsolationForest(contamination=c, random_state=42)` | aísla puntos con cortes al azar; los raros se aíslan rápido |
| `LOF(contamination=c)` | `LocalOutlierFactor(contamination=c)` | compara la densidad de un punto con la de sus vecinos |
| `predict` → **0 = inlier, 1 = outlier** | `fit_predict` → **1 = inlier, −1 = outlier** | por eso existe `convertir_etiquetas_sklearn` |

```python
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=5 + 1).fit(X)
dist, _ = nn.kneighbors(X)
puntaje = dist[:, -1]          # "largest" con k = 5 (la columna 0 es el propio punto)
```

⚠️ **Escala** (📌 Extra): kNN usa distancias. `area_m2` va de 20 a 2746 y `habitaciones` de 1 a 7, así que el área **domina** la distancia. Si escalas antes (`StandardScaler`), cambian los outliers: con viviendas, Jaccard ≈ 0.50 entre kNN sin escalar y escalado. Lo probarás en el notebook (Tu turno 2).

---

## 8. Preguntas de alumnos: pipelines y Polars
**¿Hay un método que diga cuál detector es mejor sin llegar al final?**
- Existe **Pipeline** (de scikit-learn): *"colocas en un pipeline todos los métodos"*. Pero si metes **todo** en una misma búsqueda, es una multiplicación: *"5 por 5, 25"* combinaciones, *"más todo lo demás"*, y con 10 000 registros *"te sale para más de un día"* de entrenamiento.
- Su recomendación: **hazlo por partes**, siempre con los **datos de entrenamiento**:
  1. Mejor método de **imputación** y de **detección de outliers** (sin calibrar parámetros todavía).
  2. Mejor método de **encoding** (clase 09).
  3. Recién entonces los **modelos** (por ejemplo **LightGBM** y **XGBoost**: *"especialmente LightGBM, que le gana a XGBoost"*). Calibrarlos demora mucho.
- Lo que encuentres mejor en cada paso lo metes al pipeline.

**¿El pipeline corre en local o en la nube?**
- En **local**: *"todo lo que están haciendo aquí lo pueden hacer en local"*, solo necesitas instalar las librerías.
- Si los datos llevan preprocesamiento, *"metes todo en el pipeline, y ese pipeline es lo que te llevas al despliegue"*.

**Muchos datos → Polars**: *"es un pandas pero con esteroides"*, y *"rapidísimo"* si tienes muchísima data.

El profesor anunció que en la siguiente sesión harán un ejercicio comparando el rendimiento de estos modelos para un caso de uso.

---

## 9. 📌 Extra · Por qué importa para el modelo final
- Un modelo (sobre todo una regresión lineal) aprende de **todas** las filas. Unas pocas filas absurdas, como un departamento de 2746 m², distorsionan lo que aprende.
- Con `contamination` controlas el equilibrio: si quitas **demasiado**, el modelo tiene pocos datos y no generaliza; si quitas **muy poco**, aprende de errores.
- Elegir el detector es un **hiperparámetro más**. Se compara con validación (más adelante en el programa) y no mirando gráficos.
- **Regla de oro:** detecta los outliers usando solo el conjunto de **entrenamiento**, nunca el de prueba. Si no, "espías" los datos con los que evalúas.

---

## 10. El ejercicio (`ejercicio.py`)
| # | Función | Qué practica |
|---|---|---|
| 1 | `puntaje_knn(X, k, metodo)` | puntaje kNN "largest" / "mean" / "median" con `NearestNeighbors` |
| 2 | `marcar_por_contaminacion(puntajes, contamination)` | convertir puntajes en 0/1 con un tope del % |
| 3 | `convertir_etiquetas_sklearn(pred)` | traducir −1/1 a 1/0 |
| 4 | `detectar_isolation_forest(df, columnas, ...)` | otro algoritmo, mismo idioma de etiquetas |
| 5 | `comparar_detectores(a, b)` | intersección y Jaccard: "cada algoritmo marca distinto" |
| 6 | `filtrar_inliers(df, etiquetas)` | quedarse con los inliers y medir el % conservado |

---

## ❓ Preguntas de repaso
1. ¿Por qué eliminar outliers columna por columna puede dejar outliers **multivariados** sin detectar? Da un ejemplo con viviendas (piensa en `area_m2` y `habitaciones`).
2. Explica con tus palabras qué significa `contamination=0.2`. ¿Es "exactamente el 20 %" o "como máximo el 20 %" según el profesor?
3. En kNN, ¿qué diferencia hay entre los métodos `"largest"`, `"mean"` y `"median"`? ¿Cuál es menos sensible a un vecino raro?
4. ABOD marcó 19 outliers y kNN 11 con la misma contaminación. ¿Cómo decidirías cuál usar? (Pista: el profesor dijo que no se decide mirando la gráfica.)
5. ¿Por qué el profesor recomienda armar el pipeline **por partes** en vez de probar todas las combinaciones de una vez?
