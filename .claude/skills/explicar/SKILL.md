---
name: explicar
description: Explica un concepto del programa AI Engineer (roles, determinista vs probabilístico, componentes de sistemas de IA, RAG, tools, memoria, evaluación, observabilidad, trade-offs, build vs buy, cuándo no usar LLM, etc.) con analogía, ejemplo en Python y pregunta de comprobación.
---
# Explicar un concepto

Argumento: el tema (ej. "retrieval", "p95", "build vs buy", "data leakage", "overfitting").

1. Identifica si el tema es de:
   - Nivelación (Python, ML, DL)
   - Curso 1 (fundamentos de software e IA, roles, sistemas, trade-offs)
   - Cursos posteriores (RAG, agentes, LLMOps)

2. Busca el tema en las `notas.md` de las clases correspondientes o en la teoría relevante para ser consistente con el curso.

3. Responde con esta estructura (breve, en español):
   - **En una frase**: definición concisa
   - **Analogía** de la vida cotidiana (preferiblemente con contexto peruano cuando ayude)
   - **Ejemplo en Python** (≤ 15 líneas):
     - Para temas de LLM: usa `from framework.llm import FakeLLM` si aplica
     - Para temas de ML/datos: usa datasets de `data/raw/`
     - Ejecútalo con Bash para mostrar la salida real
   - **En producción / ML**: caso real y un error común
   - **Comprueba**: una pregunta para verificar comprensión

4. Indica qué clase(s) practica ese concepto con su ruta completa.

**Ejemplo de uso:**
- `/explicar retrieval`
- `/explicar data leakage`
- `/explicar temperature`
