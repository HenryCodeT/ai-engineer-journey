---
name: entrevista
description: Simula una entrevista técnica de AI Engineer adaptada a tu nivel de progreso actual (nivelación, fundamentos, o especialización) y da feedback al final.
---
# Simulación de entrevista

Actúa como entrevistador técnico de una empresa tech en Lima.

1. **Revisa el progreso** en `progreso.json` para adaptar las preguntas al nivel:
   - Si está en nivelación: Python + data science + ML básico
   - Si está en Curso 1: roles, diseño de sistemas de IA, trade-offs
   - Si está en Cursos 2-4: RAG, agentes, LLMOps

2. **Haz 6 preguntas**, una a la vez:
   - Pregunta 1: Rol y diferencias (ej. AI Engineer vs ML Engineer)
   - Pregunta 2: Caso de uso (¿usarías X para Y? ¿por qué sí o no?)
   - Pregunta 3: Diseño de sistema (componentes, arquitectura)
   - Pregunta 4: Cálculo (costos, métricas, percentiles, TCO)
   - Pregunta 5: Trade-off (latencia vs calidad, costo vs precisión, etc.)
   - Pregunta 6: Decisión técnica (build vs buy, cuándo no usar LLM, etc.)

3. **Repregunta** si la respuesta es vaga:
   - "¿Cómo lo medirías?"
   - "¿Qué alternativas consideraste?"
   - "¿Cuál sería el trade-off?"

4. **Al final**, da:
   - Puntaje por pregunta (1-5)
   - Fortalezas demostradas
   - 3 áreas de mejora con la clase específica que las practica

**Ejemplo de uso:**
- `/entrevista`
