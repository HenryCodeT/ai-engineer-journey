"""Tests para Clase 01: Roles del AI Engineer"""
import pytest
from ejercicio import asignar_rol, RESPUESTAS_QUIZ


@pytest.mark.puntos(10)
def test_ml_engineer():
    """Identifica tarea de ML Engineer (entrenamiento + MLOps)"""
    assert asignar_rol("Entrenar y desplegar un modelo de detección de fraude con MLOps") == "ML Engineer"


@pytest.mark.puntos(10)
def test_prompt_engineer():
    """Identifica tarea de Prompt Engineer (prompts + few-shot)"""
    assert asignar_rol("Mejorar el prompt con ejemplos few-shot") == "Prompt Engineer"


@pytest.mark.puntos(10)
def test_data_scientist():
    """Identifica tarea de Data Scientist (análisis + estadística)"""
    assert asignar_rol("Análisis exploratorio y prueba de hipótesis con estadística") == "Data Scientist"


@pytest.mark.puntos(10)
def test_llm_developer():
    """Identifica tarea de LLM Developer (API + streaming)"""
    assert asignar_rol("Conectar el chatbot a la API con streaming") == "LLM Developer"


@pytest.mark.puntos(15)
def test_ai_engineer_por_defecto():
    """AI Engineer como default y cuando combina múltiples áreas"""
    assert asignar_rol("Hacer algo interesante") == "AI Engineer"
    assert asignar_rol("Diseñar la arquitectura RAG con observabilidad en producción") == "AI Engineer"


@pytest.mark.puntos(10)
def test_mayusculas():
    """Maneja texto en mayúsculas correctamente"""
    assert asignar_rol("ENTRENAR CON GPU Y FINE-TUNING") == "ML Engineer"


@pytest.mark.puntos(5)
def test_quiz_completo():
    """Quiz: todas las respuestas completadas"""
    assert None not in RESPUESTAS_QUIZ.values(), "Completa todas las respuestas del quiz en RESPUESTAS_QUIZ"


@pytest.mark.puntos(30)
def test_quiz_correctitud():
    """Quiz: respuestas correctas según la teoría"""
    esperado = {1: "c", 2: "a", 3: "b", 4: "b", 5: "c"}
    for pregunta, respuesta_correcta in esperado.items():
        assert RESPUESTAS_QUIZ[pregunta] == respuesta_correcta, (
            f"Pregunta {pregunta}: esperaba '{respuesta_correcta}', obtuve '{RESPUESTAS_QUIZ[pregunta]}'. "
            f"Revisa ../teoria/01_roles.md"
        )
