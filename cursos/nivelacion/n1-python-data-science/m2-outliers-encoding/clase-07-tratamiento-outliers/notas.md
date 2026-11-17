# Clase 07 · Tratamiento de outliers
**Curso:** Python para Ciencia de Datos (N1) · **Módulo 2** · Título oficial: *Python para Ciencia de Datos - M2-2 - Tratamiento de Outliers* · Transcript: `transcripts/n1-python/clase-07.txt`

> Clase práctica de unos 13 minutos, con un notebook de Colab sobre el dataset **Boston** (precio de casas). El profesor repasa por qué los outliers dañan las métricas y cómo se detectan. Luego aplica la **regla IQR** a una variable y después a **todas a la vez**, y muestra el problema: el dataset se queda con casi la mitad de las filas. Termina anunciando la librería **PyOD** para detectar outliers en muchas dimensiones, y **el transcript se corta ahí** (seguramente está incompleto). Lo que viene después de ese punto aparece aquí como 📌 Extra.
>
> La clase 06 ya vio qué es un outlier, el capping con límites conocidos, media vs mediana, PowerTransformer y Cramér. Aquí no se repite: esta clase trata de **cómo detectarlos con reglas estadísticas y qué pasa cuando los eliminas**.

---

## 0. La idea en una frase
Detectar outliers **columna por columna** (boxplot, IQR, ±3σ) es fácil, pero **borrarlos así** tiene dos problemas: **pierdes muchas filas** (y con ellas información de las otras variables) y **no ves los outliers que aparecen al combinar variables** (outliers multivariados).

---

## 1. El dataset de la clase: Boston
- Datos de viviendas de Boston, **506 registros**. El objetivo es **predecir el precio de una casa**.
- Algunas variables que menciona: `CRIM` (índice de criminalidad), `ZN`/`INDUS` (zona residencial / industrial), `RM` (número de habitaciones; el profesor duda un poco: *"si no me equivoco"*), `AGE` (antigüedad de la casa), `B`, `LSTAT`.
- Dice que es *"parecido al de Melbourne"*, el que vieron al inicio de esa sesión.
- En este repo trabajamos lo mismo con **`data/raw/viviendas_lima.csv`** (2015 filas, target `precio_usd`), que trae outliers sembrados a propósito (ver sección 7).

---

## 2. Outliers univariados vs multivariados

### 2.1 Lo que dijo el profesor
- *"Los outliers no solamente son univariados, también pueden ser **multivariados**. Pueden existir outliers en dimensiones superiores."*
- Su ejemplo con una imagen: la **altura** sola, en un boxplot, no tiene ningún outlier. El **peso** solo, tampoco. Pero *"cuando yo cruzo dos variables, peso con altura, ¡oh sorpresa!, me doy cuenta de que aquí y aquí aparecieron outliers"*.
- *"Hasta tres dimensiones lo puedo observar muy bien... pero si tengo 20 o 30 variables ya no puedo observar"*. Ese es el problema: **¿hay outliers en dimensiones superiores?** No los puedes ver con un gráfico.

### 2.2 Ejemplo concreto (📌 Extra, mismo caso que usan los tests)
14 personas en las que altura y peso suben juntos, más una persona (p15) **alta pero muy liviana**:

| persona | altura_cm | peso_kg |
|---|---|---|
| p01 … p14 | 150, 155, …, 185 | 50, 53, …, 82 (suben con la altura) |
| **p15** | **182** | **52** |

- **Altura sola:** límites IQR = 135.25 a 205.25. 182 está dentro → **no es outlier**.
- **Peso solo:** límites IQR = 31.25 a 97.25. 52 está dentro → **no es outlier**.
- **Juntas:** casi nadie de 182 cm pesa 52 kg. En un gráfico de dispersión, p15 queda **lejos de la "nube"** diagonal. Es un **outlier multivariado**.

```
peso
 82 |                          *
 76 |                     *  *
 69 |                *  *
 62 |          *  *
 55 |    *  *
 50 | *                     X  <- p15 (182 cm, 52 kg): normal en cada eje, rara en conjunto
    +-------------------------- altura
     150                    185
```

> 🔑 Un boxplot por columna **nunca** detectaría a p15. Hace falta una medida que mire **las variables juntas** (sección 6).

---

## 3. El impacto de los outliers en las métricas (repaso)
El profesor: *"impactan directamente las métricas. La más perjudicada, como ustedes ya saben, es la **media**"*.

Su ejemplo: una lista con media ≈ 5.45, a la que se le agrega un **300** al final. Reconstruido con números (📌 Extra: la lista exacta no aparece en el transcript):

| métrica | `[2,3,4,5,5,5,6,7,7,8,8]` | + 300 al final | ¿Qué pasó? |
|---|---|---|---|
| **media** | 5.45 | **30.00** | Se dispara: *"sube a 30"* |
| **mediana** | 5.0 | 5.5 | *"Se afecta, pero bien poquito"* |
| **moda** | 5 | 5 | *"Sigue igual"* |
| **desviación estándar** | 1.97 | **85.05** | Sube porque *"se calcula en función de la media"* |

> *"Solamente un outlier me malogró toda la distribución. Entonces hay que tratarlos, y tratarlos correctamente."*

---

## 4. Cómo se detectan (univariado)
El profesor da tres reglas para una sola variable y una idea para varias:

| Regla | Cómo funciona | Comentario |
|---|---|---|
| **Boxplot / IQR** | Outlier si está por debajo de `Q1 − 1.5·IQR` o por encima de `Q3 + 1.5·IQR` | *"La forma más rápida de ver outliers a nivel univariante es con un diagrama de caja"*. Los bigotes llegan justo a esos límites |
| **Percentiles 5 y 95** | Filtrar todo lo que está fuera del percentil 5 y del 95 | Siempre "recorta" ~10% de los datos, haya o no outliers reales (📌 Extra) |
| **±3 desviaciones estándar** | Outlier si está a 3 o más desviaciones estándar de la media (`|z| > 3`) | Usa media y desviación, que los propios outliers inflan (📌 Extra, ver 4.2) |
| **Bivariado / multivariado** | *"Se miden utilizando un índice de influencia o apalancamiento (leverage), o la distancia"* | *"Ya se complica un poquito la cosa"* |

### 4.1 La regla IQR paso a paso
El profesor la considera **la más clásica** *"debido a que no tengo que encontrar ningún hiperparámetro como en otros métodos"*.

```
Q1  = percentil 25
Q3  = percentil 75
IQR = Q3 − Q1                      (rango intercuartílico: dónde vive el 50% central)
límite inferior = Q1 − 1.5 · IQR
límite superior = Q3 + 1.5 · IQR
```
> *"Lo que está por debajo del cuartil 1 [− 1.5·IQR] es un outlier, lo que está sobre el cuartil 3 + 1.5·IQR también es otro outlier, y el resto está dentro del comportamiento normal de la variable."*

**Ejemplo en `viviendas_lima.csv`, columna `area_m2`:**
```python
q1 = df["area_m2"].quantile(0.25)   # 73.15
q3 = df["area_m2"].quantile(0.75)   # 126.05
iqr = q3 - q1                        # 52.90
inferior = q1 - 1.5 * iqr            # -6.2   (no hay áreas negativas: por abajo no cae nadie)
superior = q3 + 1.5 * iqr            # 205.4
outliers = df[(df["area_m2"] < inferior) | (df["area_m2"] > superior)]   # 147 filas
```
Igual que con `CRIM` en Boston: **por debajo no hay nada, por arriba sí hay**.

### 4.2 La regla de 3 desviaciones estándar (z-score)
```python
z = (df["precio_usd"] - df["precio_usd"].mean()) / df["precio_usd"].std()
outliers = df[z.abs() > 3]          # 30 filas en precio_usd (la regla IQR marca 105)
```
📌 Extra: esta regla marca **menos** que IQR cuando hay outliers enormes, porque esos outliers inflan la media y la desviación estándar (justo lo de la sección 3) y así "se esconden".

---

## 5. Eliminar outliers con IQR (lo que hizo en el notebook)

### 5.1 Una sola variable (CRIM)
- Calcula Q1, Q3 e IQR de `CRIM`, filtra los que están sobre el límite superior y los **elimina**.
- Observación clave: *"Profesor, todavía hay muchos outliers"*. Sí: el máximo bajó de ~80 a ~8, pero en el nuevo boxplot **siguen apareciendo puntitos**, *"ya más próximos"* al límite.
- 📌 Extra, por qué pasa: al borrar filas, **los cuartiles se recalculan sobre los datos nuevos**, la caja se achica y nuevos puntos quedan fuera. En `viviendas_lima.csv`: tras quitar los 147 outliers de `area_m2`, si aplicas la regla otra vez caen **76 más**. Si repites el proceso, siempre encontrarás "nuevos outliers".

### 5.2 Todas las variables a la vez (la "automatización")
- *"Si quisiera tratar todas las variables de forma directa, se puede automatizar"*: hace una copia (`df_temp`), calcula percentil 25 y 75 **de todas las columnas**, el IQR de todas y filtra **de una sola vez**.
- Resultado visual: `INDUS` quedó *"bastante limpio"*, `RM` con muy pocos puntos, otras con *"algún puntito, pero dentro de un margen aceptable"*.

Esquema del código (📌 Extra: reconstrucción, no es su código literal):
```python
q1 = df.quantile(0.25)
q3 = df.quantile(0.75)
iqr = q3 - q1
fuera = (df < q1 - 1.5 * iqr) | (df > q3 + 1.5 * iqr)   # True donde hay outlier
df_limpio = df[~fuera.any(axis=1)]                       # fuera la fila si falla en ALGUNA columna
```

### 5.3 El problema: se perdió casi la mitad
- *"Mi dataset se ha reducido a **268 registros**; originalmente eran **506**."* Eso es **47%** de pérdida (*"entre 40 y 50%"*).
- Por qué es grave, según el profesor:
  1. **Tamaño:** *"me afecta el tamaño del dataset"*.
  2. **Otras variables:** al borrar una fila por un outlier en `CRIM` también borras sus valores normales de `RM`, `AGE`… *"pude haber comprometido otras variables"*.
  3. **Sesgo:** *"la distribución de las otras variables me la puede sesgar un poquito"*.
  4. **Multivariado:** *"Esto solamente es a nivel univariante, y nada me garantiza"* que no queden outliers en dimensiones superiores; incluso *"nada me garantiza que yo no haya generado outliers por culpa de eso"*.

**Lo mismo en `viviendas_lima.csv`** (📌 Extra), aplicando IQR a `area_m2`, `precio_usd`, `antiguedad_anios` y `distancia_centro_km` a la vez:
- Quedan **1763 de 2015** filas → se pierde **12.51%**.
- Pero mira el sesgo: las **casas** eran el **22.5%** del dataset y pasan a ser el **15.2%**. ¿Por qué? Porque las casas son grandes (≈180 m²) y la regla IQR de `area_m2` marcó **139 casas** como "outliers", cuando solo **12** filas eran errores de digitación reales. El modelo vería muchas menos casas de las que existen: **aprendería mal los precios de casas**.

> 🔑 La regla IQR no sabe si un valor extremo es un **error** (área ×10) o un **caso real pero poco común** (una casa grande). Borrar sin mirar puede eliminar justo la información que el modelo necesita.

---

## 6. ¿Entonces qué hacemos? Algoritmos multivariados y PyOD
- *"No es una tarea tan sencilla. Se requieren algoritmos un poquito más superiores, no solamente para la dimensión uno, sino para la dimensión n."*
- *"Esos algoritmos están empaquetados por suerte en Python en una librería llamada **PyOD**... nos va a ahorrar un dolor de cabeza terrible, porque empaqueta los algoritmos que tienen mejor desempeño para detectar outliers, no solo a nivel unidimensional, sino multidimensional."*
- **Aquí termina el transcript.**

### 📌 Extra: lo que probablemente venía después
| Herramienta | Idea | Dónde está |
|---|---|---|
| **Distancia de Mahalanobis** | Distancia al centro de los datos que **tiene en cuenta cómo se relacionan las variables** (covarianza). p15 (182 cm, 52 kg) da ≈ 3.59, el resto ≤ 1.8 | Se calcula con numpy (**función 6 del ejercicio**) |
| **Isolation Forest** | Construye árboles al azar que van "partiendo" los datos. Los outliers se **aíslan con pocos cortes** | `sklearn.ensemble.IsolationForest` y PyOD (`pyod.models.iforest`) |
| **LOF (Local Outlier Factor)** | Compara la **densidad** de un punto con la de sus vecinos: si está en una zona mucho más vacía, es raro | `sklearn.neighbors.LocalOutlierFactor` y PyOD (`pyod.models.lof`) |
| **PyOD** | Librería que agrupa más de 30 detectores (IForest, LOF, KNN, ECOD, autoencoders…) con la misma interfaz `fit` / `predict` | `pip install pyod` (en este repo no está instalada; el notebook usa scikit-learn) |

Mahalanobis, explicada sin fórmulas: la distancia normal (euclidiana) mide en "centímetros y kilos". Mahalanobis mide en **"cuántas desviaciones típicas" en la dirección en que se mueven los datos**. Como altura y peso suben juntos, alejarse **a lo largo** de la diagonal es normal, pero salirse **de** la diagonal (alto y liviano) cuesta mucho: la distancia sube rápido.

En `viviendas_lima.csv`, la distancia de Mahalanobis sobre (`area_m2`, `precio_usd`) pone primero la fila 1902 (**2746 m²** con precio normal: área ×10) y entre las primeras la fila 1089 (**219 m²** y **2.1 millones USD**: la "mansión" sembrada). Son dos combinaciones imposibles que el modelo de precios no debería aprender.

---

## 7. Outliers sembrados en `viviendas_lima.csv` (para practicar)
Según `data/generar_datos.py`:
- **12 filas con `area_m2` × 10** (errores de digitación: áreas de cientos o miles de m², hasta 2746, con precio normal).
- **5 "mansiones"**: `precio_usd` multiplicado por 5–9 con área normal.
- **3 filas con `antiguedad_anios = 150`** (imposible).
- Además, la variable `area_m2` tiene **colas naturales**: las casas son mucho más grandes que los departamentos. Por eso la regla IQR marca 147 filas aunque solo 12 son errores.

---

## 8. ¿Por qué importa para ML/IA?
- **Regresión lineal y redes neuronales** se dejan "jalar" por outliers (minimizan errores al cuadrado): unos pocos errores pueden torcer todo el modelo.
- Pero **borrar a ciegas** reduce datos y **sesga** el entrenamiento (las casas en el ejemplo). Un modelo entrenado sin casas grandes predice mal justo esas casas en producción.
- Los detectores multivariados (Isolation Forest, LOF, Mahalanobis) también se usan **en producción** para detectar fraude, fallos de sensores o datos de entrada "raros" antes de pasárselos a un modelo.

---

## 9. Lo que tienes que quedarte
1. Hay outliers **univariados** (se ven en un boxplot) y **multivariados** (solo aparecen al cruzar variables; con más de 3 no se pueden graficar).
2. Un outlier dispara la **media** y la **desviación estándar**; la mediana y la moda casi no cambian.
3. Regla IQR: `IQR = Q3 − Q1`, límites `Q1 − 1.5·IQR` y `Q3 + 1.5·IQR`. Es la clásica porque **no tiene hiperparámetros**. Alternativas: percentiles 5–95 y ±3 desviaciones estándar.
4. Tras eliminar, **siguen apareciendo outliers** porque los cuartiles se recalculan sobre los datos nuevos.
5. Aplicar IQR a todas las columnas a la vez puede borrar **casi la mitad** del dataset (Boston: 506 → 268), sesgar otras variables y **no garantiza** nada en varias dimensiones.
6. Para n dimensiones hay algoritmos especializados, agrupados en **PyOD** (📌 Extra: Isolation Forest, LOF; y Mahalanobis como medida de distancia).

---

## ❓ Preguntas de repaso
1. Una persona mide 182 cm y pesa 52 kg. Ni la altura ni el peso salen de su boxplot. ¿Puede ser un outlier? ¿Cómo se llama ese tipo de outlier y por qué un boxplot no lo detecta?
2. En la lista `[2,3,4,5,5,5,6,7,7,8,8]` agregas un 300. ¿Qué métricas cambian mucho, cuáles poco y por qué la desviación estándar también sube?
3. Calcula a mano los límites IQR de `[1, 2, 3, 4, 5, 6, 7, 8, 100]` (Q1 = 3, Q3 = 7). ¿Qué valores quedan fuera?
4. El profesor eliminó los outliers de `CRIM` y en el nuevo boxplot seguían apareciendo puntos. ¿Por qué pasa eso?
5. Al aplicar IQR a todas las columnas de `viviendas_lima.csv`, las casas pasan del 22.5% al 15.2% del dataset. ¿Qué problema le causa eso a un modelo que predice precios? ¿Qué harías en lugar de borrar todo?
