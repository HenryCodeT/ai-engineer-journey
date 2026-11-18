# 1. El rol del AI Engineer frente a otros perfiles

> **Objetivo:** identificar qué hace un AI Engineer y en qué se diferencia de LLM Developer, Prompt Engineer, Data Scientist y ML Engineer.
> **Ejercicio:** `ej01_roles.py`

## La idea en una frase

El **AI Engineer** es un ingeniero de software que construye **sistemas completos en producción** donde un modelo de IA es *uno* de los componentes. Su pregunta no es "¿qué responde el modelo?", sino "¿este sistema funciona bien, rápido, barato, seguro y de forma medible para usuarios reales?".

## Mapa de perfiles

| Perfil | Pregunta central | Entregable típico | Herramientas típicas |
|---|---|---|---|
| **Data Scientist** | ¿Qué nos dicen los datos? | Análisis, experimento A/B, insight, dashboard | pandas, SQL, estadística, notebooks |
| **ML Engineer** | ¿Cómo entreno y sirvo *mi* modelo de forma fiable? | Modelo entrenado + pipeline de entrenamiento/serving | PyTorch, scikit-learn, MLflow, GPUs, MLOps |
| **Prompt Engineer** | ¿Qué instrucciones hacen que el modelo se comporte como quiero? | Prompts, ejemplos few-shot, guías de estilo | Playground, plantillas, datasets pequeños |
| **LLM Developer** | ¿Cómo integro un LLM en esta feature? | Feature con API del LLM, streaming, function calling | SDKs (Anthropic, OpenAI), frameworks |
| **AI Engineer** | ¿Cómo construyo un **sistema** de IA confiable de punta a punta? | Producto en producción con RAG/tools, evaluación, observabilidad, control de costo | Todo lo anterior + arquitectura, evals, trazas, infra |

## Analogía: construir un restaurante

- **Data Scientist**: estudia qué platos se venden más y por qué.
- **ML Engineer**: cría y entrena al chef (crea el modelo).
- **Prompt Engineer**: escribe las recetas e instrucciones exactas para el chef.
- **LLM Developer**: conecta al chef con la cocina y el sistema de pedidos.
- **AI Engineer**: diseña el restaurante entero: cocina, proveedores (datos), meseros (tools), control de calidad (evaluación), cámaras y métricas (observabilidad), costos y seguridad alimentaria.

## Cosas importantes que debes saber

1. **Los límites son difusos.** En una startup una sola persona hace los 5 roles. En una empresa grande están separados. Lo importante es saber *qué responsabilidad* estás cubriendo.
2. **El AI Engineer casi nunca entrena LLMs desde cero.** Usa modelos existentes (vía API o self-hosted) y crea valor con **datos propios, contexto, tools, evaluación y producto**.
3. **El AI Engineer es dueño de las métricas del sistema:** calidad (accuracy, groundedness), latencia (p95), costo por request, tasa de error.
4. **El Prompt Engineering es una habilidad, no siempre un puesto.** Todo AI Engineer escribe prompts; pero también sabe que el prompt es solo una pieza.

## Ejemplo: mismo proyecto, distintos roles

Proyecto: *"Asistente que responde preguntas de clientes sobre las políticas de la tienda."*

- Data Scientist: analiza los tickets históricos → "el 40% son sobre devoluciones".
- Prompt Engineer: escribe el prompt de sistema con tono y ejemplos.
- LLM Developer: integra la API del modelo en el chat web con streaming.
- ML Engineer: (si hiciera falta) hace fine-tuning de un clasificador de intención.
- **AI Engineer**: diseña el RAG sobre las políticas, define el dataset de evaluación, pone guardrails de PII, mide p95 y costo, añade fallback si el proveedor cae y decide si conviene usar un LLM o reglas para cada tipo de consulta.

## Autoevaluación

1. ¿Por qué un chatbot que "funciona en la demo" puede ser un problema del AI Engineer y no del Prompt Engineer?
2. Nombra 3 métricas de las que es responsable un AI Engineer.
3. ¿En qué caso un ML Engineer es más necesario que un AI Engineer?
