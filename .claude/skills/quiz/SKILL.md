---
name: quiz
description: Genera un quiz de 5 preguntas (opción múltiple y casos prácticos) sobre un tema del programa AI Engineer y corrige las respuestas del estudiante.
---
# Quiz

Argumento: tema o módulo (ej. "trade-offs", "outliers", "RAG", "todo el curso 1").

1. Identifica el contexto del tema:
   - Nivelación: Python, ML, DL
   - Curso 1: roles, sistemas de IA, software engineering
   - Curso 2: LLMs, RAG, fine-tuning
   - Curso 3: agentes, MCP
   - Curso 4: LLMOps

2. Lee las notas correspondientes en la ruta del curso/módulo.

3. Haz 5 preguntas, UNA a la vez, mezclando:
   - 2 conceptuales (definiciones, comparaciones)
   - 2 de escenario ("eres AI Engineer trabajando en X y necesitas...")
   - 1 práctica (cálculo de costos, análisis de datos, interpretación de métricas)

4. Tras cada respuesta:
   - ✓ o ✗
   - Por qué (explicación breve)
   - El dato clave a recordar

5. Al final:
   - Puntaje (X/5)
   - Temas a repasar
   - Clase recomendada para practicar

**Ejemplo de uso:**
- `/quiz outliers`
- `/quiz todo curso 1`
- `/quiz RAG`
