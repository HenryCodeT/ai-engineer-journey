# Clase 01 · Análisis Exploratorio de Datos (EDA)
**Curso:** Python para Ciencia de Datos · **Módulo 1** · Transcript: `transcripts/n1-python/clase-01.txt`

> Clase introductoria. Todavía sin código: el profesor muestra **el mapa completo** de un proyecto de datos y dónde encaja cada tema del curso.

---

## 1. Todo empieza en los datos y en SQL
- El punto de partida siempre es **la carga de datos**.
- Recomendación fuerte (del profesor y de un compañero): **aprender SQL** y lo básico de modelado de bases de datos.
  - SQL procesa millones de filas en minutos. Un error típico es traer todo "crudo" y procesarlo en Python, lo que vuelve lento el sistema.
  - Lo que se puede resolver en el gestor de base de datos (consultas, procedimientos almacenados) le quita carga al código.
  - SQL es un estándar, pero **cada gestor** (SQL Server, Oracle, PostgreSQL…) tiene su propio dialecto y agrega funciones propias.

### ¿Cómo te llegan los datos en el trabajo?
| Escenario | Qué haces tú |
|---|---|
| Empresa pequeña | Escribes el script SQL y extraes los datos |
| Te dan un script | Lo ejecutas, y puedes editarlo u optimizarlo |
| Empresa grande | Das los requerimientos y el **ingeniero de datos** te entrega un CSV o Excel |

---

## 2. El flujo de un proyecto de datos

```
Carga de datos ─► EDA ─┬─► Analítica DESCRIPTIVA ──────────────► Dashboard
                       │
                       └─► FEATURE ENGINEERING ─► Modelo PREDICTIVO (ML) ─► Despliegue
```

- **Dos destinos posibles:** quedarte en el análisis descriptivo (un dashboard) o llegar a un **modelo predictivo/inferencial** con Machine Learning.
- **El puente entre los dos es el Feature Engineering (ingeniería de características):**
  - balance de datos
  - selección de variables
  - reducción de dimensionalidad
  - técnicas de encoding
  - técnicas de escalamiento
- Según el profesor, **lo más trabajoso es el análisis previo al modelado** (la "flecha roja" de su diagrama). Lo comprueba con sus tesistas que trabajan con datos reales de empresas.
- *"Hay librerías que hacen bastantes cosas, pero hay que entender lo que hacemos para poder interpretar los resultados."*

---

## 3. Los pasos del EDA (asumiendo que queremos llegar a un modelo predictivo)
1. Definir la **variable objetivo** (target). La excepción es clustering, que se ve en el curso de ML.
2. Identificar los **tipos de variables**: numéricas y categóricas.
3. Estudiar cómo se comportan:
   - **numéricas**: su distribución.
   - **categóricas**: conteo de cada categoría.
   - **bivariante**: cruzar cada variable con el target para ver si ayuda a diferenciar.
4. ⭐ **Detección de datos faltantes** (perdidos).
5. ⭐ **Detección de outliers** (valores atípicos).
6. Feature engineering, que se ve en los módulos 2 a 4.

> 🔑 **Los pasos 4 y 5 son los más críticos y los que más trabajo dan.** Por eso el curso empieza por ellos.

---

## 4. La idea más importante de la clase
> **La estrategia de preprocesamiento que elijas repercute directamente en el rendimiento del modelo.**
> Con la estrategia A puedes tener 90% de performance y con la B, 95%.

En el curso de Machine Learning se aprende a medir qué decisión ayudó y cuál no. Y todo depende de la base de datos con la que trabajes.

---

## 📌 Extra (no fue parte de la clase: es práctica para el ejercicio)
Herramientas de pandas para los pasos 1 a 3:

| Pregunta | Código |
|---|---|
| ¿Qué tamaño tiene? | `df.shape`, `df.info()` |
| ¿Hay duplicados? | `df.duplicated().sum()` |
| ¿Qué tipos hay? | `df.dtypes`, `df.select_dtypes(include="number")` |
| ¿Cómo se distribuyen las numéricas? | `df.describe()`, `df.skew()`, histogramas |
| ¿Cómo se reparten las categóricas? | `df[col].value_counts(dropna=False)` |
| ¿Esta variable diferencia al target? | `df.groupby(col)[target].agg(["mean", "median"])` |
| ¿Cuántos faltantes hay? | `df.isna().sum()` |

El dataset de práctica `data/raw/viviendas_lima.csv` tiene **problemas puestos a propósito**: duplicados, nulos, texto sucio, outliers y una columna casi constante. Encontrarlos es parte del juego.

---

## ❓ Preguntas de repaso
1. ¿Qué une el análisis descriptivo con el modelado predictivo?
2. ¿Por qué conviene resolver parte del procesamiento en SQL y no en Python?
3. ¿Cuáles son los dos pasos más críticos del EDA según el profesor?
4. ¿Por qué dos estrategias de preprocesamiento distintas pueden dar modelos con distinto rendimiento?
5. En `viviendas_lima.csv`, ¿por qué crees que `piso` tiene tantos faltantes? ¿Es un "error" o tiene sentido?
