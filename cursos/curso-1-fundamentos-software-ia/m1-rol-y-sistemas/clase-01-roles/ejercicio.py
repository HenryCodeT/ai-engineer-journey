"""Clase 01 · El rol del AI Engineer frente a otros perfiles

Evalúa con:  make evaluar CLASE=cursos/curso-1-fundamentos-software-ia/m1-rol-y-sistemas/clase-01-roles
"""
from __future__ import annotations

import re

ROLES = {
    "AI Engineer": "Diseña e integra sistemas completos con IA en producción: arquitectura, "
                   "RAG, tools, evaluación, observabilidad, costos y confiabilidad.",
    "LLM Developer": "Construye features sobre APIs/SDKs de LLMs: integra el modelo en la app, "
                     "maneja streaming, function calling y frameworks.",
    "Prompt Engineer": "Diseña, prueba e itera instrucciones y ejemplos para obtener el "
                       "comportamiento deseado del modelo.",
    "Data Scientist": "Analiza datos, formula hipótesis, hace experimentos y estadística "
                      "para responder preguntas de negocio.",
    "ML Engineer": "Entrena, optimiza, despliega y mantiene modelos de ML; pipelines de "
                   "features, MLOps, reentrenamiento.",
}

PALABRAS_CLAVE = {
    "AI Engineer": ["arquitectura", "producción", "observabilidad", "evaluación", "rag",
                    "integrar", "sistema", "costo", "latencia", "guardrails"],
    "LLM Developer": ["api", "sdk", "streaming", "function calling", "langchain", "endpoint",
                      "chatbot", "integración"],
    "Prompt Engineer": ["prompt", "few-shot", "instrucciones", "plantilla", "tono", "ejemplos"],
    "Data Scientist": ["análisis", "hipótesis", "estadística", "dashboard", "a/b", "correlación",
                       "exploratorio", "insight"],
    "ML Engineer": ["entrenar", "fine-tuning", "mlops", "reentrenamiento", "features", "gpu",
                    "desplegar un modelo", "pipeline de entrenamiento"],
}


def asignar_rol(tarea: str) -> str:
    """Devuelve el rol con más palabras clave presentes en la tarea.

    Implementa un clasificador por REGLAS (sí, determinista: no necesitas un LLM
    para esto, ¡y esa es parte de la lección!). Cuenta cuántas palabras clave
    de cada rol aparecen en la tarea (en minúsculas) y devuelve el rol con más
    coincidencias. Si hay empate o ninguna coincidencia, devuelve "AI Engineer"
    (el perfil generalista que integra).

    Pistas:
    - Convierte la tarea a minúsculas
    - Para cada rol, cuenta cuántas palabras clave aparecen
    - Usa re.search() con r"\b{palabra}\b" para buscar palabras completas
    - Devuelve el rol con más coincidencias (en caso de empate, "AI Engineer")

    Ejemplo:
        >>> asignar_rol("Entrenar y desplegar un modelo de detección de fraude con MLOps")
        'ML Engineer'
    """
    raise NotImplementedError("Implementa asignar_rol()")


# PARTE B — QUIZ
# 1. Un equipo tiene un chatbot que funciona en demo pero en producción es lento,
#    caro y nadie sabe por qué falla. ¿Quién lidera la solución?
#    a) Prompt Engineer  b) Data Scientist  c) AI Engineer  d) ML Engineer
#
# 2. Hay que crear un modelo propio de detección de fraude con datos tabulares
#    y reentrenarlo cada semana. ¿Perfil principal?
#    a) ML Engineer  b) Prompt Engineer  c) LLM Developer  d) AI Engineer
#
# 3. ¿Qué diferencia MEJOR a un AI Engineer de un LLM Developer?
#    a) El AI Engineer solo escribe prompts
#    b) El AI Engineer es responsable del sistema end-to-end (evaluación,
#       observabilidad, costo, confiabilidad), no solo de llamar al modelo
#    c) El LLM Developer entrena modelos desde cero
#    d) No hay ninguna diferencia
#
# 4. Marketing quiere saber si la nueva campaña aumentó las ventas. ¿Quién?
#    a) AI Engineer  b) Data Scientist  c) Prompt Engineer  d) LLM Developer
#
# 5. Un AI Engineer típico de 2026 pasa MÁS tiempo en...
#    a) Entrenar LLMs desde cero en clusters de GPUs
#    b) Escribir papers de investigación
#    c) Integrar modelos existentes con datos, tools y evaluación en un producto
#    d) Etiquetar datos manualmente

RESPUESTAS_QUIZ = {
    # TODO: completa los valores con las letras correctas ("a", "b", "c" o "d")
    # Revisa ../teoria/01_roles.md para las respuestas
    1: None,
    2: None,
    3: None,
    4: None,
    5: None,
}


if __name__ == "__main__":
    # Demo: prueba tu implementación
    for tarea in [
        "Entrenar y desplegar un modelo de detección de fraude con MLOps",
        "Mejorar el prompt con ejemplos few-shot para un tono más amable",
        "Diseñar la arquitectura RAG y su observabilidad en producción",
    ]:
        print(f"{tarea!r} -> {asignar_rol(tarea)}")
