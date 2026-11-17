# Clase 13 · Método de selección de variables (motivación y baseline)
**Curso:** Python para Ciencia de Datos · **Módulo 3** (Selección de variables) · Título oficial: *Python para Ciencia de Datos - M3-1 - Metodo de selección de Variables* · Transcript: `transcripts/n1-python/clase-13.txt`

> Clase corta (~23 min) y **primera del Módulo 3**. El profesor explica **por qué** sobran variables (la dimensión), presenta el caso *mobile price classification* de Kaggle y entrena un **modelo base (baseline)** con las 20 variables: un Random Forest "podado" que llega a **accuracy ≈ 0.83**. Cierra con la pregunta que guía el módulo: *"¿lo puedo mejorar o no?"*.
>
> Los **cómo** (baja varianza, correlación, importancia, PCA) llegan en las clases 14–17. Aquí solo se construye el punto de partida contra el que se compararán.
>
> El repo no tiene el dataset de móviles. Los tests y la práctica usan una **versión sintética** con las mismas 20 columnas y 4 rangos de precio, y también `clientes_banco.csv`. Todo eso es 📌 Extra. Las cifras de la clase (20 variables, 0.83…) sí son del profesor.

---

## 0. La idea en una frase
No todas las columnas aportan información. Si trabajas **solo con las que importan**, el modelo es **más simple, más rápido y generaliza mejor**. Pero para saber si "mejoró" primero necesitas un **baseline**: el resultado con **todas** las variables.

---

## 1. Cada columna es una dimensión
- *"Cada una de estas columnas me representa una dimensión de este vector de características."* Una muestra (fila) es **un punto** en el **espacio de características** $\mathbb{R}^n$, donde $n$ = número de columnas.
- *"A veces la dimensión es tan elevada que el modelo estimado se vuelve complejo"*: le cuesta más **converger**.
  - En **regresión**, le cuesta más encontrar el **hiperplano** que ajuste los puntos.
  - En **clasificación**, le cuesta más encontrar el hiperplano que **separe** las clases.
- *"Mientras más alta la dimensión, más difícil"* es ajustar o separar los puntos.

📌 Extra · en el caso de la clase, un móvil es un punto con 20 coordenadas: (batería, bluetooth, velocidad de reloj, …, wifi). Es un punto de $\mathbb{R}^{20}$.

### Selección de variables ≠ reducción de dimensionalidad
El profesor lo corrige en vivo: *"reducción de dimensionalidad o selección de variables, mejor dicho, no, no es lo mismo"*.

| | **Selección de variables** | **Reducción de dimensionalidad** |
|---|---|---|
| Qué hace | **Elige** algunas columnas originales y descarta las demás | **Crea** columnas nuevas que combinan las originales |
| Ejemplo | quedarse con `ram`, `battery_power`, `px_height`, `px_width` | PCA (clase 17) |
| ¿Se entiende cada columna? | Sí, son las mismas de antes | Menos: son mezclas |

(La tabla es 📌 Extra. La distinción la dijo el profesor y PCA llega en la clase 17).

### No todas las variables aportan
*"Yo tengo $\mathbb{R}^n$, pero no necesariamente las $n$ variables aportan toda la información necesaria. Puede ser que de estas $n$ variables solamente cuatro o cinco sean las que más aportan"* para caracterizar la variable objetivo, sea en clasificación o en regresión.
- En vez de un $\mathbb{R}^{20}$ se puede trabajar con un $\mathbb{R}^{8}$ o un $\mathbb{R}^{10}$ → *"un modelo menos complejo y por lo tanto que generalice un poquito mejor"*.

🧠 **Mini-pregunta:** si un dataset tiene 20 columnas y te quedas con 8 de ellas tal cual, ¿hiciste selección de variables o reducción de dimensionalidad?

---

## 2. ¿Para qué seleccionar variables? (las razones de la clase)
1. **Mejorar la precisión** del modelo.
2. **Reducir el sobreajuste** (overfitting).
3. **Entrenar más rápido.**
4. **Reducir la complejidad** del modelo.
5. Hacer un modelo **más fácil de interpretar**.

📌 Extra · por qué el ruido estorba: una columna de puro ruido (por ejemplo, si el móvil tiene wifi, cuando eso no influye en el precio) igual le ofrece al modelo cortes "casuales" que parecen útiles en train. Con pocos datos, el modelo puede aprender esas casualidades. Eso es sobreajuste.

---

## 3. El caso: *mobile price classification* (Kaggle)
- Una **empresa nueva de móviles** quiere competir con grandes como **Apple o Samsung**. Quiere **estimar el precio** de sus equipos a partir de sus características: RAM, memoria interna, procesador…
- No estiman el precio exacto: lo **categorizan en 4 niveles** (`price_range` = 0, 1, 2, 3). Eso es **clasificación de 4 clases**.
- Variables mencionadas: poder de la **batería**, **bluetooth**, **velocidad de reloj**, **dual SIM**, **memoria interna**, **número de núcleos**, **RAM**…
- `data.info()` → **todas numéricas** y **sin nulos**. Por eso no hace imputación ni tratamiento de outliers (*"ustedes si gustan luego pueden ir mejorando los resultados"*).
- 21 columnas: **20 predictores (X)** + `price_range` (**y**).
- **No escala los datos** a propósito: *"solamente me quiero concentrar en el efecto que se genera con todas las variables o con una cantidad seleccionada de variables"*.
- Consejo: *"lo importante es siempre cambiar de dataset para que ustedes vean un poco más la variabilidad"*.

```python
X = data.drop(columns="price_range")   # predictores
y = data["price_range"]                # objetivo
```

---

## 4. División train/test con `stratify`
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1)
```
- **`stratify=y`**: *"para decirle que no sea tan aleatorio… sino que trate de forma proporcionada de extraer de cada categoría cierta cantidad de muestras"*.
- El dibujo del profesor: sin `stratify`, el test podría llevarse casi nada de la clase 0 y muchísimo de otra. Con `stratify` *"garantizo que siempre hay muestras de todas las clases, y de forma proporcionada"*.
- Tiene sentido en **clasificación** (la y es categórica). En regresión la y es continua y no se estratifica así.
- **`random_state=1`**: *"para que ustedes puedan replicar los resultados en sus computadores"*.
- El profesor dijo *"20 % para entrenamiento"*, pero `test_size=0.2` significa **20 % para test** (y 80 % para train). 📌 Extra: aclaración.

### Ejemplo numérico (📌 Extra)
En `clientes_banco.csv` hay 5 000 clientes y **255 abandonan** (5.1 %). Con `stratify=y` y `test_size=0.2`, el test tiene 1 000 clientes y **exactamente 51** que abandonan (5.1 %). Sin `stratify`, podrían caer 40 o 65 por azar. Con un desbalance así, eso cambia mucho la evaluación.

🧠 **Mini-pregunta:** con 4 clases de 500 muestras cada una y `test_size=0.2, stratify=y`, ¿cuántas muestras de la clase 3 habrá en el test?

---

## 5. El modelo: Random Forest "podado"
```python
from sklearn.ensemble import RandomForestClassifier
modelo = RandomForestClassifier(max_depth=5, max_leaf_nodes=20, random_state=1)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
```
(En vivo primero dijo profundidad 10 y luego la dejó en **5**. Se usa sin calibrar hiperparámetros de forma fina: *"una poda bastante básica, rudimentaria, con fines didácticos"*.)

### ¿Qué es un árbol de decisión?
- Trabaja con **reglas** sobre las variables. Analiza cada variable ($x_1, x_2, x_3$…) y genera **puntos de corte**.
- 📌 Extra · ejemplo de un árbol de profundidad 2:
  ```
                 ¿ram <= 2000?
                /             \
             sí                no
     ¿battery <= 1200?     ¿px_width <= 1000?
       /        \             /         \
   precio 0  precio 1    precio 2    precio 3     <- 4 nodos HOJA
  ```
- **Profundidad** (`max_depth`): cuántos **niveles** crece el árbol hacia abajo. El ejemplo tiene 2 niveles, así que su profundidad es 2. Con `max_depth=5` el árbol *"puede crecer hasta una profundidad de cinco, no puede crecer más"*.
- **Nodo hoja** (`max_leaf_nodes`): *"es aquel punto que ya no crece más, donde finaliza el árbol"*: de ahí *"ya no sale nada"*. Es donde se da la predicción. `max_leaf_nodes=20` significa como máximo 20 hojas por árbol. (En vivo dijo *"máximo 20 muestras por hoja"*, pero el parámetro limita la **cantidad de hojas**, no las muestras por hoja. 📌 Extra: aclaración).

### ¿Por qué "podar"?
- *"Un solo árbol de decisión no es muy fiable"* y *"tienden al sobreajuste rápido"*.
- Si no se controla, *"el árbol crece de forma desproporcionada y se llega a un sobreajuste"*. Por eso se **poda** con `max_depth` y `max_leaf_nodes`.
- **Random Forest** = *"un modelo que internamente trabaja con muchos árboles de decisión"* (por defecto, 100). Cada árbol vota y gana la clase más votada (📌 Extra: el detalle de la votación).
- Hay muchos más hiperparámetros, pero *"lo vamos a ver en el curso de Machine Learning"*.

🧠 **Mini-pregunta:** si un árbol tiene profundidad 1, ¿cuántos nodos hoja puede tener como máximo?

---

## 6. Accuracy (exactitud)
- *"De cada 100 muestras, ¿cuántas se han clasificado correctamente?"* Si son 80, el accuracy es 80 %.
- El dibujo de la pizarra: 12 puntos (triángulos y equis) con 2 mal clasificados → **accuracy = 10/12 ≈ 0.83**.

$$\text{accuracy} = \frac{\text{predicciones correctas}}{\text{total de muestras}}$$

- ⚠️ *"Hay que tener cuidado con ella cuando trabajemos con datos desbalanceados"* (se verá en el módulo 4).

### 📌 Extra · por qué engaña con desbalance
En `clientes_banco.csv` el 94.9 % **no** abandona. Un "modelo" que **siempre** responde "no abandona" tiene **accuracy = 0.949** sin haber aprendido nada, y no detecta a **ninguno** de los clientes que se van, que son justo los que le importan al banco. Por eso, antes de celebrar un accuracy, mira las **proporciones de cada clase**.

### Resultado del baseline en la clase
**accuracy ≈ 0.83**: *"de cada 100 se equivoca clasificando 17 y 83 los clasifica correctamente"*, sumando las 4 clases.

🧠 **Mini-pregunta:** en un test de 400 móviles, el modelo acierta 332. ¿Cuál es su accuracy?

---

## 7. Matriz de confusión
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()     # gráfico (necesita matplotlib)
```
- **Filas = clase real, columnas = clase predicha** (convención de scikit-learn).
- *"La diagonal es lo que indica la correcta clasificación de las clases"*. Lo que queda **fuera de la diagonal son errores**. En su resultado señaló errores de 14, 2, 22, 9 y 14 en distintas celdas.
- *"No se preocupen mucho por entenderlo todavía, en Machine Learning lo van a ver con calma."*

### 📌 Extra · ejemplo mini
Real: `[0, 0, 1, 1, 1]`, predicho: `[0, 1, 1, 1, 0]`

| real \ predicho | 0 | 1 |
|---|---|---|
| **0** | **1** ✔ | 1 ✘ |
| **1** | 1 ✘ | **2** ✔ |

- Diagonal: 1 + 2 = 3 aciertos. Total: 5. Accuracy = 3/5 = 0.6.
- La celda (fila 0, columna 1) dice "**era** clase 0 y el modelo **dijo** 1". Pasó 1 vez.
- La matriz dice **qué clases se confunden entre sí**. El accuracy solo da un número global.

🧠 **Mini-pregunta:** si sumas una **fila** completa de la matriz, ¿qué obtienes?

---

## 8. Cierre: ¿se puede mejorar?
- *"Si este fue mi primer resultado, ¿será que yo lo puedo mejorar o no?"*
- *"En este punto tengo un $\mathbb{R}^{20}$: cada muestra tiene una dimensión de 20, y eso hace que el modelo se vuelva un poquito más complejo."*
- Desde aquí, el módulo prueba técnicas para elegir variables y **compara siempre contra este 0.83**.

📌 Extra · en el dataset sintético del ejercicio, el baseline con 20 variables da **0.7525**. Quedándose solo con las 4 variables que realmente influyen (`ram`, `battery_power`, `px_height`, `px_width`) sube a **0.8125**. Menos variables, mejor modelo. Pero ahí las útiles **ya las conocíamos** porque el dataset es inventado. En datos reales hay que descubrirlas, y eso es lo que enseñan las clases 14–17.

---

## 9. ¿Por qué importa para ML / IA? (📌 Extra)
- **Baseline primero, siempre.** Sin un número de referencia no sabes si un cambio (seleccionar variables, escalar, otro modelo) ayudó o empeoró.
- **Misma división, misma semilla.** Para comparar dos modelos de forma justa, ambos deben evaluarse sobre **el mismo test** (`random_state` fijo).
- En producción, menos variables significa menos datos que recolectar, limpiar y mantener, y predicciones más rápidas.
- Esto se repite en todo el programa: cada dimensión de más cuesta cómputo y puede meter ruido. Pasa también con los vectores de texto (embeddings) que verás más adelante.

---

## 📌 Extra (agregado, no dicho en clase)
- Dataset sintético de móviles (mismas 20 columnas que Kaggle, solo 4 útiles) y ejemplos con `clientes_banco.csv`.
- Tabla selección vs. reducción, dibujo del árbol, detalle de la votación del Random Forest.
- Aclaraciones: `test_size=0.2` es 20 % para **test**; `max_leaf_nodes` limita la **cantidad de hojas**.
- Ejemplo de por qué el accuracy engaña con desbalance (0.949 prediciendo siempre 0).
- Ejemplo mini de matriz de confusión.

---

## ❓ Preguntas de repaso
1. Explica con tus palabras por qué una dimensión alta hace más difícil que un modelo **separe** las clases o **ajuste** los puntos.
2. ¿Qué diferencia hay entre **seleccionar variables** y **reducir la dimensionalidad**? Da un ejemplo de cada una.
3. ¿Qué hace `stratify=y` en `train_test_split` y por qué es aún más importante en `clientes_banco.csv` que en el dataset de móviles?
4. ¿Qué controlan `max_depth` y `max_leaf_nodes` en un árbol y por qué ayudan a evitar el sobreajuste?
5. Un modelo tiene accuracy 0.95 en `clientes_banco.csv`. ¿Por qué eso **no** basta para decir que es bueno, y qué mirarías en la matriz de confusión?
