# Clase 02 · Estrategias para datos faltantes (MCAR, MAR, MNAR)
**Curso:** Python para Ciencia de Datos · **Módulo 1** · Transcript: `transcripts/n1-python/clase-02.txt`

> Clase corta y **teórica** (unos 7 minutos, sin código). El profesor presenta los **tres tipos de datos faltantes** con un ejemplo de IQ y desempeño laboral. El código de esta carpeta es práctica 📌 Extra para llevar la teoría a `viviendas_lima.csv`.

---

## 1. Punto de partida: el script ya está revisado
- El profesor parte de este supuesto: *"imaginemos que revisé el script y está todo bien"*. Si el script de extracción está bien y aun así faltan datos, hay que averiguar **por qué faltan**.
- Toca ponerse "un poquito teóricos", y por dos razones:
  1. **Reporte descriptivo:** entender la naturaleza de los faltantes ayuda a hacer mejores reportes.
  2. **Síntoma del sistema:** *"saber qué tipo de dato faltante tienen […] hasta puede ser síntoma de que algo está pasando en el sistema"*.

---

## 2. Los tres tipos de datos faltantes

| Tipo | Nombre | ¿De qué depende que falte? | ¿Qué hacer? (según el profesor) |
|---|---|---|---|
| **MCAR** | *Missing Completely At Random*: faltan completamente al azar | De nada: no hay correlación con otras variables | Se puede imputar con **métodos paramétricos o estadísticos** (p. ej. la mediana), mejor si son **pocos** |
| **MAR** | *Missing At Random*: faltan al azar, pero dentro de grupos | De **otra variable observada** | Cuidado. **Reportarlo** en el EDA y revisar si el script o el registro tienen sesgo |
| **MNAR** | *Missing Not At Random*: no faltan al azar | De algo que **no vemos** en las otras variables (factores externos) | Cuidado. Requiere **un estudio**, revisar la base de datos y el script |

### MCAR: faltan completamente al azar
- Suelen tener **muchos menos nulos** que los otros tipos y **no tienen correlación con otras variables**.
- *"Por más que analizan no van a encontrar que existe cierta tendencia"*: no hay agrupación ni patrón.
- Son los que se pueden imputar con métodos paramétricos o estadísticos, por ejemplo con la **mediana**.
- Matiz del profesor: *"tampoco es tan bueno imputar con la mediana realmente, pero podrían aplicarlo"*. Depende de **cuántos** faltantes haya: *"si tienen una pequeña cantidad, una imputación con mediana en esa variable no les va a afectar mucho"*.

### MAR: la ausencia depende de otra variable
- Es un caso **más amplio** que MCAR. A primera vista los faltantes parecen aleatorios, pero **la aleatoriedad ocurre solo en grupos específicos** de datos.
- **Ejemplo del profesor:** se quiere ver si el **IQ** de un trabajador se correlaciona con su **score de desempeño** laboral.
  - En la columna MAR faltan scores **cuando el IQ es bajo**: *"con IQ menores a 92 faltan los datos"*, y si se siguen extrayendo datos, a todos los que tengan IQ < 92 les va a faltar el score.
  - *"Esto ya no es completamente al azar, algo ha ocurrido."* ¿De quién es la culpa? Puede ser **el script** o **el registro en la base de datos**. No lo sabemos, hay que analizarlo.
- 🔑 *"Esto, como resultado a nivel de análisis exploratorio, **tienen que reportarlo**. Es muy importante, algo está pasando."* Sirve para corregir el script o para ver que **la forma de registrar los datos tiene cierto sesgo**.

### MNAR: no es al azar, pero tampoco vemos la relación
- *"Hay datos faltantes, sí, pero no entendemos el porqué."* No hay relación visible con las otras variables, y tampoco es completamente al azar.
- En el ejemplo faltan scores con IQ bajo **y también** con IQ más alto, así que no hay relación directa con el IQ. Faltan *"por algo"*: **factores externos**.
- Puede llevar **un estudio posterior**: revisar la base de datos o el script. A veces el sistema está bien, pero **la forma de registrar** está mal.
- Igual que MAR, *"todo esto sirve como reporte"*.

### Sobre los gráficos pequeños de la diapositiva (pregunta de un alumno)
Representan relaciones entre variables (X, Y, R): en MCAR falta X o falta Z sin relación con nada; en MAR **hay** una relación entre la ausencia y otra variable; en MNAR algo está pasando, pero no se ve con qué.

---

## 3. Lo que tienes que quedarte
1. **Identificar el tipo de faltante es en sí un resultado del EDA**, aunque no imputes nada todavía.
2. **MCAR + pocos faltantes →** imputar con la mediana afecta poco.
3. **MAR / MNAR →** cuidado: imputar a ciegas puede esconder un problema del sistema o meter sesgo.

---

## 📌 Extra: cómo se ve esto en pandas (para el ejercicio)
El profesor no mostró código. Estas son las herramientas para llevar la teoría a datos reales:

| Pregunta | Código | Función del ejercicio |
|---|---|---|
| ¿Dónde hay hueco? | `df[col].isna()` → True/False | `agregar_indicador_faltante` |
| ¿Faltan más en algún grupo? (señal de MAR) | `df.groupby(grupo)[col_falta].mean() * 100` | `tasa_faltantes_por_grupo` |
| ¿Las filas con hueco son "distintas"? | `df.loc[mascara, var].mean()` vs `df.loc[~mascara, var].mean()` | `comparar_media_por_faltante` |
| Reporte rápido MCAR / posible MAR | rango de tasas por grupo vs un umbral | `clasificar_mecanismo` |
| Imputar un MCAR con pocos faltantes | `df[col].fillna(df[col].median())` | `imputar_mediana` |

**Truco clave:** la media de una columna True/False es la **proporción de True**. `[True, False, False, False].mean()` da `0.25`, es decir, falta el 25 %.

### 📌 Extra: lo que esconde `viviendas_lima.csv`
El dataset se generó con faltantes de los tres tipos **a propósito**. Descubrirlos es parte del ejercicio:
- `antiguedad_anios`: ¿la tasa de faltantes cambia según `tipo_inmueble`?
- `distancia_centro_km`: mira la tasa por `distrito`. ¿Se parece a la historia del IQ < 92?
- `piso`: mira la tasa por `tipo_inmueble`. Aquí el faltante **tiene sentido** (es estructural). ¿Lo imputarías con la mediana?
- `banos`: compara el `precio_usd` medio de las filas con y sin dato. ¿Qué podría estar pasando en el registro?

### 📌 Extra: por qué importa para el modelo (ML)
- La mayoría de modelos de scikit-learn **no aceptan NaN**: algo hay que hacer con ellos, y la decisión depende del tipo.
- Si un faltante es **MAR**, el hecho de que falte **trae información**. En `distancia_centro_km`, que falte "delata" un distrito periférico, más barato. Por eso a veces se agrega la columna indicadora `<col>_falta` como **feature** para el modelo (scikit-learn lo hace con `SimpleImputer(add_indicator=True)`).
- Imputar con la mediana un MAR o un MNAR puede **sesgar** el modelo. Si a los inmuebles caros les falta `banos` y le pones la mediana (un valor "típico", bajo), el modelo aprende una relación falsa entre baños y precio.
- Lo de la clase 01 sigue valiendo: **la estrategia de preprocesamiento cambia el rendimiento del modelo**. La clase 03 (métodos de imputación) sigue a partir de aquí.

---

## ❓ Preguntas de repaso
1. ¿Cuál es la diferencia entre MCAR y MAR? Explícala con el ejemplo del IQ y el score de desempeño.
2. Según el profesor, ¿en qué tipo de faltante es razonable imputar con la mediana, y de qué depende que "no afecte mucho"?
3. Encuentras que a todos los registros de un distrito les falta la distancia. ¿Qué tipo de faltante sospechas y qué harías antes de imputar?
4. ¿Por qué el profesor dice que el tipo de faltante puede ser "síntoma de que algo está pasando en el sistema"? Da dos posibles culpables.
5. ¿Por qué MNAR no se puede detectar mirando la tasa de faltantes por grupo?
