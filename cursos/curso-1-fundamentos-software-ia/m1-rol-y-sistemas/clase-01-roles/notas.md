# Clase 01: El rol del AI Engineer frente a otros perfiles

> **Teoría completa:** `../teoria/01_roles.md`
> **Objetivo:** Identificar qué hace un AI Engineer y en qué se diferencia de LLM Developer, Prompt Engineer, Data Scientist y ML Engineer.

## La idea en una frase

El **AI Engineer** es un ingeniero de software que construye **sistemas completos en producción** donde un modelo de IA es *uno* de los componentes. Su pregunta no es "¿qué responde el modelo?", sino "¿este sistema funciona bien, rápido, barato, seguro y de forma medible para usuarios reales?".

## Mapa de perfiles

| Perfil | Pregunta central | Entregable típico |
|---|---|---|
| **Data Scientist** | ¿Qué nos dicen los datos? | Análisis, experimento A/B, insight, dashboard |
| **ML Engineer** | ¿Cómo entreno y sirvo *mi* modelo de forma fiable? | Modelo entrenado + pipeline de entrenamiento/serving |
| **Prompt Engineer** | ¿Qué instrucciones hacen que el modelo se comporte como quiero? | Prompts, ejemplos few-shot, guías de estilo |
| **LLM Developer** | ¿Cómo integro un LLM en esta feature? | Feature con API del LLM, streaming, function calling |
| **AI Engineer** | ¿Cómo construyo un **sistema** de IA confiable de punta a punta? | Producto en producción con RAG/tools, evaluación, observabilidad, control de costo |

## Analogía: construir un restaurante

- **Data Scientist**: estudia qué platos se venden más y por qué
- **ML Engineer**: cría y entrena al chef (crea el modelo)
- **Prompt Engineer**: escribe las recetas e instrucciones exactas para el chef
- **LLM Developer**: conecta al chef con la cocina y el sistema de pedidos
- **AI Engineer**: diseña el restaurante entero: cocina, proveedores (datos), meseros (tools), control de calidad (evaluación), cámaras y métricas (observabilidad), costos y seguridad alimentaria

## Cosas importantes

1. **Los límites son difusos.** En una startup una sola persona hace los 5 roles. En una empresa grande están separados.
2. **El AI Engineer casi nunca entrena LLMs desde cero.** Usa modelos existentes y crea valor con datos propios, contexto, tools, evaluación y producto.
3. **El AI Engineer es dueño de las métricas del sistema:** calidad (accuracy, groundedness), latencia (p95), costo por request, tasa de error.
4. **El Prompt Engineering es una habilidad, no siempre un puesto.** Todo AI Engineer escribe prompts, pero también sabe que el prompt es solo una pieza.

## Ejercicio de esta clase

Vas a construir un **clasificador de roles** determinista (sin LLM) que cuenta palabras clave y asigna la tarea al perfil correcto.

**Lección implícita:** No necesitas un LLM para todo. A veces reglas simples son más rápidas, baratas y confiables.

## Para profundizar

- Lee `../teoria/01_roles.md` completo
- Ejecuta `practica.ipynb` para ver ejemplos interactivos
- Implementa `ejercicio.py` SIN ayuda de IA
- ¿Atascado? → `/tutor` o `/pista-gradual`
