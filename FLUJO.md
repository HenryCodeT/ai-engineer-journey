# 🔁 FLUJO DE TRABAJO + PROMPTS
¿No sabes qué usar? → `/ayuda` o `/ayuda <tu situación>`

## Las 3 reglas
1. **Primero ves el video, después pasas el transcript.** El transcript ayuda a que las notas y los ejercicios salgan fieles a la clase. No reemplaza ver la clase.
2. **El ejercicio lo escribes tú, sin IA.** La IA te da pistas (`/tutor`) y te evalúa (`/evaluar`). Si no lo puedes explicar con tus palabras, todavía no lo aprendiste.
3. **Proyectos por hito, no por clase.** En N1, al cerrar las clases 12, 17 y 24. En los cursos en vivo, en cada taller (una capa de Mentor IA).

## Flujo por clase
```
1. Ver el video (anota 2-3 ideas propias)
2. Pegar transcript + título → /clase
3. Leer notas.md + correr practica.ipynb
4. Implementar ejercicio.py SIN IA ──atascado (~15 min)──► /tutor
5. make evaluar CLASE=... hasta ≥ 70
6. /evaluar → responder la pregunta de comprensión
7. diario.md (3 líneas) + git commit
   └─ ¿cerraste un bloque? → /proyecto
```

---

## Prompts listos para copiar

### 1 · Nueva clase (pegando el transcript)
```
/clase Título: "Python para Ciencia de Datos - M1-2 - Datos Faltantes y Métodos de Imputación"
Mis notas: el profe enfatizó MCAR vs MNAR; no entendí bien cuándo usar KNN.

<pega aquí el transcript completo>
```
Claude lo guarda en `transcripts/<curso>/clase-XX.txt` y genera la carpeta de la clase. Si no puede deducir el número de clase, te pregunta.

### 1b · Nueva clase (si ya guardaste el archivo)
```
/clase transcripts/n1-python/clase-02.txt "M1-2 - Datos Faltantes y Métodos de Imputación"
```

### 1c · Clase sin ejercicio (dudas, resultados de alumnos)
```
/clase "M1-5 - Resolviendo consultas" — es una clase de dudas: solo notas.md con preguntas y respuestas, sin ejercicio.
<transcript>
```

### 2 · Atascado
```
/tutor en clase-02, función imputar_por_grupo: quiero rellenar con la mediana de cada distrito.
Intenté df.fillna(df.median()) y rellena con la mediana global.
```
👉 Siempre di **qué intentaste**.

### 3 · Ver el error detallado
```bash
make evaluar-v CLASE=cursos/nivelacion/n1-python-data-science/m1-eda-faltantes/clase-02-faltantes
```

### 4 · Terminé el ejercicio
```
/evaluar cursos/nivelacion/n1-python-data-science/m1-eda-faltantes/clase-02-faltantes
```
Después **responde la pregunta de comprensión** en el chat.

### 5 · Necesito más práctica
```
Usa el generador-ejercicios en modo refuerzo para la clase 02: me costó la imputación por grupos.
```

### 6 · ¿Dónde estoy? ¿Qué sigue?
```
/progreso
```

### 7 · Proyectos
```
/proyecto n1-preprocesamiento-vivienda planificar
/proyecto n1-preprocesamiento-vivienda revisar
/proyecto mentor-ia planificar v1
/proyecto mentor-ia revisar v1
```

### 8 · Tracks
```
/clase genera la clase 02 del track SQL: JOINs
```

### 9 · Clases en vivo (agrega la conexión con Mentor IA)
```
/clase Título: "Curso 1 - Módulo 3 - APIs y servicios preparados para IA"
¿Qué parte de esta clase aplico en la capa v1 de Mentor IA?
<transcript>
```

---

## ¿Qué pasa si no uso el agente "correcto"?
Nada grave: `CLAUDE.md` aplica siempre, y la nota solo la pone `make evaluar`. Los comandos son atajos; si describes lo que necesitas, Claude elige el agente. El único riesgo real es pedirle con insistencia que haga el ejercicio por ti.
