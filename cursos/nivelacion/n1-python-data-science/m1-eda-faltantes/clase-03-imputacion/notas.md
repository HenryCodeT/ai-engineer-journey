# Clase 03 · Datos faltantes y métodos de imputación
**Curso:** Python para Ciencia de Datos · **Módulo 1** · Título oficial: *M1-3 · Métodos de imputación* · Transcript: `transcripts/n1-python/clase-03.txt`

> Clase corta y **teórica** (unos 6 minutos, sin código). El profesor explica **de dónde salen** los datos faltantes, **por qué son peligrosos** para un modelo y cuenta una anécdota sobre un **LEFT JOIN** mal puesto. Todavía **no** muestra métodos concretos de imputación. Esos métodos aparecen aquí como 📌 Extra y se profundizan en las clases siguientes.

---

## 0. La idea en una frase
**Imputar** = aplicar una estrategia para **completar lo que falta**. Pero antes de rellenar hay que saber **por qué** falta, porque si se rellena mal el modelo aprende algo falso (**sesgo**).

---

## 1. Lo que dijo el profesor

### 1.1 Un problema antiguo
- Los datos faltantes *"no son un tema nuevo"*. Existen desde los primeros sistemas de almacenamiento, *"incluso cuando guardábamos en archivos txt"*.
- Datos faltantes e imputación van juntos: la imputación agrupa *"las estrategias que vamos a aplicar para poder completar lo que nos falta"*.

### 1.2 Las tres causas (según su experiencia)
| # | Causa | Ejemplo del profesor | ¿Cómo se vería en `viviendas_lima.csv`? (📌 Extra) |
|---|---|---|---|
| 1 | **Errores de programación**: el sistema no validó bien un campo | *"No se validó bien cierta cuestión"* | Un formulario que acepta `cochera = "N/A"` como texto en lugar de obligar a elegir Sí/No |
| 2 | **Datos no reportados por el cliente** | Un **adulto mayor** que no maneja el correo y no digita su email ni su teléfono. Clientes que **no quieren dar** cierta información | Dueños de inmuebles caros que no declaran el número de `banos` |
| 3 | **Manejo inapropiado** de los datos | (lo menciona sin detallar) | Una consulta que une tablas mal y "fabrica" nulos (ver 1.5) |

> *"Eso ya genera un vacío, un valor no, por efecto un cero, le podríamos decir un valor en blanco"*: el hueco puede guardarse como blanco, como nulo o hasta como cero.

### 1.3 La "sexta categoría"
- En el sistema estos errores *"tal vez no se vean mucho"*. Se notan **cuando haces analítica**.
- Su ejemplo: *"tengo una variable categórica con cinco categorías, pero me aparece una sexta que dice **blanco, vacío, no reportó**"*.
- 🔑 *"Ahí se está perdiendo información. No hay cómo saber exactamente el comportamiento de esa variable, sea categórica o numérica."*

### 1.4 Por qué hay que tratarlos (las tres consecuencias)
1. **Sesgo:** *"si no lo manejan de forma buena les va a aparecer esta palabrita: sesgo"*. Sesgo en el modelo produce **conclusiones engañosas**.
2. **Muchos modelos de ML no aceptan valores faltantes.**
3. **Limita la generalización:** *"un modelo tiene que ser generalizable, es decir, hay que entrenarlo con datos que representen el **comportamiento normal** de mi información, y hay que quitarle valores atípicos"*.

### 1.5 Anécdota: el culpable era el script SQL
- Un alumno (Kelvin) resume los nulos como *"dolor de cabeza"*.
- El profesor cuenta que pide el script con el que sacaron la información y encuentra que *"en vez de un **INNER JOIN** hizo un **LEFT JOIN**, que son cosas diferentes"*, y así *"me sacaste nulos"*.
- *"Pasaron todo un día con dolor de cabeza y la culpa era del script."*
- **Recomendación:** *"tampoco les pido que sean especialistas en SQL, pero **sepan entender el código SQL**"*, para poder comprobar si la consulta devuelve lo que pediste.

---

## 2. Lo que tienes que quedarte
1. Un faltante tiene una **causa**: sistema, cliente o manejo. La causa decide qué hacer con él.
2. No todos los vacíos son `NaN`: a veces aparecen como **texto** ("blanco", "N/A", "no reportó") y pandas no los cuenta.
3. Tratar mal los faltantes produce **sesgo**, y además el modelo **ni siquiera entrena** con NaN.
4. Antes de imputar, revisa **la consulta**: un `LEFT JOIN` puede fabricar nulos que no existen en la realidad.

🧠 **Mini-pregunta:** un cliente de 78 años no puso su email. ¿Cuál de las tres causas es? ¿Y si el campo email aceptó el texto "ninguno" porque nadie lo validó?

---

## 📌 Extra 1: los vacíos disfrazados en pandas
Para pandas, `NaN` es "no hay dato", pero el texto `"N/A"` o `" "` es **un valor más**. `df.isna()` no lo cuenta.

Ojo: `pd.read_csv` ya convierte por defecto algunos textos (`"N/A"`, `"NA"`, `""`, `"null"`…) a NaN. Los que **no** conoce (`"-"`, `"no reporto"`, `"sin dato"`, `"   "`) pasan como texto. En `viviendas_lima.csv`, la columna `cochera` trae 18 `"N/A"`:

```python
cruda = pd.read_csv("data/raw/viviendas_lima.csv", keep_default_na=False)  # sin conversión automática
cruda["cochera"].isna().sum()        # 0  <- pandas cree que no falta nada
cruda["cochera"].value_counts()      # Si 1098, No 853, N/A 18, no  18, SI 14, si 14
```

La receta: normaliza el texto **solo para comparar** y marca como NaN lo que esté en una lista de marcadores.

```python
texto = df[col].str.strip().str.lower()          # "  N/A " -> "n/a"
mascara = texto.isin(["", "n/a", "-", "no reporto"])
df.loc[mascara, col] = np.nan
```

🧠 **Mini-pregunta:** ¿`"no "` (con espacio) es un vacío disfrazado o un "No" mal escrito? ¿Por qué no debería convertirse en NaN?

---

## 📌 Extra 2: métodos básicos de imputación (adelanto)
| Método | Para qué tipo | Código | Ventaja | Riesgo |
|---|---|---|---|---|
| **Borrar filas** | cualquiera | `df.dropna()` | simple | pierdes datos: en viviendas pasas de 2015 a 1248 filas |
| **Media / mediana** | numérica | `s.fillna(s.median())` | rápido; la mediana resiste outliers | aplana la variable (baja la std) |
| **Moda** | categórica | `s.fillna(s.mode().iloc[0])` | rápido | infla la categoría más común |
| **Constante** ("Desconocido") | categórica | `s.fillna("Desconocido")` | conserva la señal de "no reportó" | añade una categoría más |
| **Por grupo** | numérica | `s.fillna(df.groupby(g)[c].transform("median"))` | respeta diferencias entre grupos | el grupo puede no tener datos |
| **Indicador de faltante** | cualquiera | `df[c + "_falta"] = df[c].isna()` (clase 02) | el modelo "ve" que faltaba | una columna más |

**Ejemplo con viviendas: mediana global vs. mediana por distrito.** `distancia_centro_km` falta sobre todo en distritos periféricos (lo viste en la clase 02: es MAR).

| Distrito | Mediana del distrito | Si imputas con la mediana global (10.60 km)… |
|---|---|---|
| Ate | 14.81 km | "acercas" la vivienda 4 km al centro |
| Chorrillos | 14.93 km | igual |
| SJL | 13.05 km | la acercas 2.5 km |

Con la mediana **por distrito**, cada hueco recibe un valor creíble para su zona. La clave es `transform("median")`: devuelve una mediana **por fila** (la de su grupo), del mismo largo que el DataFrame.

🧠 **Mini-pregunta:** ¿por qué `groupby("distrito")["distancia_centro_km"].median()` (sin `transform`) no sirve directamente dentro de `fillna`?

---

## 📌 Extra 3: el LEFT JOIN en pandas
`pd.merge` es el JOIN de pandas. Imagina una tabla de distritos donde alguien **olvidó cargar** SJL y Ate:

| | INNER JOIN (`how="inner"`) | LEFT JOIN (`how="left"`) |
|---|---|---|
| Filas | 1723 (se pierden 292 **sin avisar**) | 2015 (todas) |
| NaN nuevos | 0 | 292 filas con NaN en las columnas de la tabla de distritos |

```python
left = pd.merge(viviendas, distritos, on="distrito", how="left", indicator=True)
left["_merge"].value_counts()    # both 1723, left_only 292  <- left_only = sin pareja
```

Ninguno de los dos está "mal": dependen de lo que pediste. Lo peligroso es **no saber cuál usaste**. Esos 292 NaN no son clientes que no reportaron: los fabricó la consulta, y lo correcto es arreglar la consulta, no imputarlos.

🧠 **Mini-pregunta:** si haces el INNER JOIN, ¿qué distritos desaparecen del análisis? ¿Qué le pasaría a un modelo de precios entrenado así cuando le pidas predecir en SJL?

---

## 📌 Extra 4: medir el sesgo de una imputación
Rellenar muchos huecos con el **mismo número** cambia la forma de la variable. Con `antiguedad_anios` (170 huecos):

| Versión | Media | Std |
|---|---|---|
| original (sin NaN) | 12.26 | 10.09 |
| imputada con la mediana (10) | 12.07 | 9.67 (baja ≈ 4 %) |
| imputada con 0 | 11.22 | 10.24 |

- Con la **mediana** la media casi no se mueve, pero la variable se **aplana** (aparece un pico artificial en 10 años).
- Con **0** dices que 170 viviendas son nuevas: la media cae un año entero. Eso es sesgo puro.

Fórmula del ejercicio: `cambio_std_pct = (std_despues - std_antes) / std_antes * 100`. Si sale negativo, la variable se aplanó.

---

## 📌 Extra: conexión con ML/IA
- **scikit-learn no entrena con NaN:** `LinearRegression().fit(X, y)` lanza `ValueError: Input X contains NaN`. Imputar es un paso obligatorio del preprocesamiento.
- **Más adelante** usarás `SimpleImputer(strategy="median" | "most_frequent" | "constant")` dentro de un `Pipeline`. Las funciones de este ejercicio son esa misma lógica hecha a mano.
- **Sesgo y generalización:** si el set de entrenamiento tiene valores inventados que no se parecen a la realidad (todas las distancias de Ate = 10 km), el modelo aprende una relación falsa y **generaliza mal**, que es justo lo que advirtió el profesor.
- 🔗 Lo del LEFT JOIN conecta con el **track SQL** (`cursos/tracks/sql/`): *"sepan entender el código SQL"*.

---

## 🧪 Ejercicio de esta clase (`ejercicio.py`)
| # | Función | Idea de la clase | Puntos |
|---|---|---|---|
| 1 | `normalizar_vacios` | la "sexta categoría": blanco / N/A / no reportó → NaN | 20 |
| 2 | `imputar_categorica` | completar una categórica con la moda o con "Desconocido" | 20 |
| 3 | `imputar_mediana_por_grupo` | imputar sin meter sesgo: la mediana de su grupo | 25 |
| 4 | `nulos_por_join` | la anécdota: el LEFT JOIN que fabrica nulos | 15 |
| 5 | `impacto_imputacion` | medir el sesgo: media y std antes y después | 20 |

Evalúa con: `python -m framework evaluar cursos/nivelacion/n1-python-data-science/m1-eda-faltantes/clase-03-imputacion`

---

## ❓ Preguntas de repaso
1. ¿Cuáles son las tres causas de datos faltantes que menciona el profesor? Da un ejemplo de cada una.
2. ¿Qué es la "sexta categoría" y por qué hace que *"se pierda información"* sobre la variable?
3. Nombra las tres consecuencias de no tratar bien los faltantes. ¿Qué tiene que ver la **generalización** con esto?
4. ¿En qué se diferencian un INNER JOIN y un LEFT JOIN, y por qué el LEFT JOIN puede "sacarte nulos"? ¿Imputarías esos nulos?
5. (📌 Extra) ¿Por qué imputar `distancia_centro_km` con la mediana por distrito mete menos sesgo que con la mediana global?
