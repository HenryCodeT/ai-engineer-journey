# Clase 05 · Resolviendo consultas: imputación en series de tiempo
**Curso:** Python para Ciencia de Datos · **Módulo 1** · Transcript: `transcripts/n1-python/clase-05.txt`

> Sesión corta de **preguntas y respuestas** (unos 9 minutos, sin código). Un alumno, Arturo, cuenta un caso real: imputó una columna con faltantes y **su error casi no cambió**. El profesor explica por qué: su caso era una **serie de tiempo**, y ahí las reglas cambian.
> Esta clase **solo tiene notas**: no hay ejercicio, tests ni notebook.

---

## 0. Antes de empezar: palabras que vas a necesitar

| Palabra | Qué significa, en simple |
|---|---|
| **Serie de tiempo** | Datos ordenados por fecha, donde **el orden importa**. Ejemplo: visitantes a un centro comercial **cada día**. El dato del martes se parece al del lunes; si mezclas las filas, pierdes información. |
| **Serie de tiempo multivariable** | Varias columnas que cambian en el tiempo a la vez (visitantes, clima, promociones…) y que juntas ayudan a predecir un objetivo. |
| **MAPE** | *Mean Absolute Percentage Error*: en promedio, **en qué porcentaje se equivoca** el modelo. MAPE = 7% significa "en promedio me equivoco un 7% respecto al valor real". Más bajo, mejor. |
| **Interpolación** | Rellenar un hueco **usando los valores de antes y de después**, siguiendo la línea (o curva) que forman. |
| **Ventana de tiempo** | Cuántos días previos le muestras al modelo para predecir el siguiente. Ejemplo: "con los últimos 7 días, predigo el día 8". |

📌 Extra: el MAPE de un solo dato. Si el tráfico real fue 1 000 personas y el modelo predijo 930, el error es |1000 − 930| / 1000 = 7%. El MAPE es el promedio de ese porcentaje en todos los días.

---

## 1. La pregunta de Arturo

- **Problema:** proyectar el **tráfico diario** de personas a los centros comerciales de una empresa retail, en todas sus cadenas.
- **Datos:** desde **junio de 2021** hasta más o menos **septiembre**.
- Una columna tenía **20% de datos faltantes**, y esa columna estaba **muy correlacionada** con el objetivo.
- Lo que hizo:
  1. **Primero** rellenó los faltantes con **0** (valor por defecto) y entrenó → **MAPE ≈ 7%**.
  2. **Después** imputó con **interpolación** dentro de ese rango de tiempo ("seguía la línea solo en ese rango de tiempo") → **MAPE ≈ 7%**, con una variación pequeñita.
- Su duda: *"si es una columna con correlación alta, cualquier variación en ella debería cambiar el resultado… pero no se movía nada"*. Además comentó que en papers hay mucha discusión sobre cómo tratar los faltantes.

---

## 2. La respuesta del profesor

### 2.1 "Era una serie de tiempo, entonces la cuestión cambia"
- El profesor se da cuenta a mitad de la conversación: *"Ah, claro, pero tenías serie de tiempo… aquí la cuestión cambió"*.
- En una serie de tiempo **puedes visualizar el hueco en el tiempo**: graficas la variable y ves dónde falta el tramo.
- *"Cuando hablamos de serie de tiempo son otro tipo de imputación."* Hay varias, pero **la más famosa es la interpolación**:
  - **lineal** (une los puntos con una recta),
  - **cúbica** / **polinomial de grado 3** (une los puntos con una curva).
- Hay otras estrategias también, pero con la interpolación *"lo puedes imputar"*.

### 2.2 La correlación de una sola variable no basta
- Arturo tenía una **serie de tiempo multivariable** (más de una variable).
- 🔑 *"El que una variable tenga alta correlación con otra… tiene que verse también qué pasa con las demás variables."* Las otras variables también pueden influir directamente en el objetivo.
- Idea clave: **las variables en conjunto tienen que mapear tu variable objetivo en el espacio**. No se trata de una columna estrella: es la combinación de todas la que explica el objetivo.

### 2.3 Lo que probablemente pesó más: la ventana de tiempo
- Según el profesor, el análisis tenía que ir **más allá de esa variable**: *"tenías que haber analizado de repente la ventana, la ventana móvil, o cuántos días previos"*.
- *"Creo que eso también pudo haber influido bastante en tus resultados."*
- Reconoce que tendría que ver el código completo para estar seguro.

### 2.4 El caso contrario: bitcoin
- Otro alumno predice el **precio de bitcoin** con varias variables juntas y **ventanas de tiempo**, usando redes **LSTM** y **Transformer**.
- En su caso **la ventana de tiempo** es lo que más afecta el rendimiento del modelo.
- Tuvo que imputar datos y **sí le mejoró directamente**, porque esa variable imputada, junto con las otras cuatro, en conjunto explican mejor el objetivo.
- Moraleja: **a veces imputar ayuda mucho y a veces casi nada**; depende del conjunto de variables y de cómo armes el problema.

---

## 3. Cierre del profesor: los datos secuenciales son otro mundo

- 🔑 *"A veces la imputación no afecta mucho."* Pero las series de tiempo son un **caso diferente**.
- Las series de tiempo son **datos secuenciales**: hay que extraer **información espacio-temporal**.
  - *"Ya no va simplemente de una variable, sino de cómo se comporta cierta variable en el tiempo"*, y de cómo el comportamiento de **todas** las variables en el tiempo influye en el objetivo.
- El análisis se puede hacer **por ventanas de tiempo**.
- Existen estrategias de **normalización** que permiten **alinear tendencias** entre secuencias, como **Dynamic Time Warping** (DTW), que *"funcionan bastante bien"*.
  - 📌 Extra: DTW compara dos series aunque estén "desfasadas". Ejemplo: dos centros comerciales con el mismo patrón semanal, pero uno con el pico el sábado y otro el domingo. DTW los reconoce como parecidos.

---

## 4. Conexión con clases anteriores

- **Clase 01 (EDA):** el primer paso siempre es **mirar los datos**. Aquí el profesor insiste en lo mismo: en una serie de tiempo, **grafica la variable** para ver dónde está el hueco antes de decidir cómo rellenarlo.
- **Clase 04 (identificar faltantes):** ahí viste que `ffill`/`bfill` tienen sentido **solo en series de tiempo** (copian el valor anterior o siguiente porque el orden importa), y que en **datos tabulares** se usan imputadores como **KNN** o **Iterative**. Esta clase suma la **interpolación** como herramienta típica de series de tiempo.

### Tabla resumen: ¿cómo imputar según el tipo de datos?

| | **Datos tabulares** (ej. `viviendas_lima.csv`) | **Series de tiempo** (ej. tráfico diario) |
|---|---|---|
| ¿Importa el orden de las filas? | No: cada fila es independiente | **Sí**: cada fila depende de las anteriores |
| Imputación simple | Media / mediana / moda | `ffill` / `bfill` (copiar el anterior o el siguiente) |
| Imputación "inteligente" | **KNN**, **Iterative** (usan las otras columnas) | **Interpolación** lineal, cúbica o polinomial de grado 3 (usa los vecinos en el tiempo) |
| Cómo ver el problema | Conteo de nulos, patrones entre columnas (MCAR/MAR/MNAR) | **Graficar en el tiempo** y ver el hueco |
| Qué más influye en el modelo | Las features y su relación con el objetivo | Además: **ventana de tiempo**, comportamiento conjunto de las variables en el tiempo |

---

## 5. 📌 Extra: interpolación en pandas, con un mini ejemplo

Tráfico de 4 días; faltan el día 2 y el día 3.

```python
import pandas as pd

trafico = pd.Series([1000, None, None, 1600],
                    index=pd.date_range("2021-06-01", periods=4, freq="D"))

trafico.fillna(0)
# 2021-06-01    1000.0
# 2021-06-02       0.0   <- ¡como si el centro comercial hubiera cerrado!
# 2021-06-03       0.0
# 2021-06-04    1600.0

trafico.interpolate()            # lineal por defecto
# 2021-06-01    1000.0
# 2021-06-02    1200.0   <- sigue la recta de 1000 a 1600
# 2021-06-03    1400.0
# 2021-06-04    1600.0
```

- Con **0**, el modelo "aprende" que hubo dos días sin nadie, algo que nunca pasó. Eso es **ruido falso**.
- Con **interpolación lineal**, de 1000 a 1600 en 3 pasos se sube 200 por día: 1200 y 1400. Tiene sentido.
- Variantes: `trafico.interpolate(method="time")` tiene en cuenta las fechas reales, y `method="polynomial", order=3` o `method="cubic"` usan curvas (requieren `scipy`).
- Ojo: `interpolate()` **asume que las filas están ordenadas por fecha**. Antes, `df.sort_values("fecha")`.

---

## 6. La lección práctica: compara la métrica, no supongas

Lo que hizo Arturo es justo el experimento correcto, aunque el resultado le sorprendiera:

1. Entrena con los faltantes rellenos con **0** (o la estrategia más simple) → anota la métrica (MAPE, RMSE…).
2. Entrena con una **imputación razonable** (interpolación si es serie de tiempo; KNN/Iterative si es tabular) → anota la métrica.
3. **Compara.** Si casi no cambia, la imputación no era el cuello de botella: mira **otras cosas** (en series de tiempo, la **ventana**; en general, el **conjunto de variables**).

**Por qué importa para ML/IA:** el modelo solo aprende de lo que le das. Rellenar mal mete patrones falsos, pero obsesionarse con la imputación puede ocultar el verdadero problema. Medir con una métrica es lo que te dice dónde poner el esfuerzo.

---

## Preguntas de repaso

1. ¿Por qué rellenar con 0 el tráfico de un centro comercial puede confundir al modelo? ¿Qué le estás "diciendo" que pasó esos días?
2. En la serie `[50, NaN, 70]` (un dato por día), ¿qué valor pondría `interpolate()` lineal? ¿Y `ffill`?
3. Si una columna tiene alta correlación con el objetivo, ¿por qué imputarla bien no garantiza que el modelo mejore? (Pista: piensa en "el conjunto de variables".)
4. ¿Qué es una ventana de tiempo y por qué el profesor cree que pudo influir más que la imputación en el caso de Arturo?
5. ¿Usarías `interpolate()` para rellenar el `precio_usd` faltante en `viviendas_lima.csv`? ¿Por qué sí o por qué no?
