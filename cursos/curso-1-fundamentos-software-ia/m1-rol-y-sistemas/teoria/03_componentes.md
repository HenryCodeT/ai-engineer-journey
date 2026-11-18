# 3. Componentes de una aplicación moderna de IA

> **Objetivo:** reconocer las 9 piezas y qué problema resuelve cada una.
> **Ejercicios:** `ej03` (mapa), `ej05` (retrieval + contexto), `ej06` (tools), `ej07` (memoria), `ej08` (evaluación), `ej09` (observabilidad)

```
                ┌──────────────── OBSERVABILIDAD (trazas, costo, latencia) ───────────────┐
 usuario ──►  [ guardrails ] ──► ┌───────────── CONTEXTO ─────────────┐ ──► MODELO ──► [ validación ] ──► respuesta
                                 │ prompt sistema + MEMORIA +          │        │
                                 │ fragmentos de RETRIEVAL (de DATOS)  │        └──► TOOLS (APIs, BD, calculadora)
                                 └─────────────────────────────────────┘
                └──────────── EVALUACIÓN (offline y online) · INFRAESTRUCTURA (deploy, escalado) ────────────┘
```

| Componente | Qué es | Problema que resuelve | Ejemplo |
|---|---|---|---|
| **Modelo** | El LLM (u otro modelo) | Entender y generar lenguaje | Claude Haiku vía API, un modelo de 8B local |
| **Datos** | Tu información propia, limpia y versionada | El modelo no conoce tu empresa | PDFs de políticas, catálogo, tickets |
| **Contexto** | Todo lo que pones en el prompt en esa llamada | El modelo solo "sabe" lo que ve | Instrucciones + historial + fragmentos |
| **Tools** | Funciones que el modelo puede *pedir* ejecutar | El modelo no puede actuar ni calcular exacto | Consultar pedido, calculadora, enviar correo |
| **Memoria** | Lo que persiste entre turnos o sesiones | Los LLMs no recuerdan nada entre llamadas | Últimos N mensajes, resumen, preferencias |
| **Retrieval** | Buscar la información relevante en tus datos | No cabe todo en el contexto (y sería caro) | BM25, embeddings + base vectorial (RAG) |
| **Evaluación** | Dataset + métricas para medir calidad | "Parece que funciona" no es una métrica | 200 preguntas con respuesta esperada |
| **Observabilidad** | Logs, trazas y métricas de cada llamada | No puedes depurar ni optimizar a ciegas | p95 de latencia, costo por request, errores |
| **Infraestructura** | Donde y cómo corre todo | Disponibilidad, escalado, seguridad | API en contenedor, colas, caché, secretos |

## Detalles que separan a un junior de un AI Engineer

- **Contexto ≠ memoria ≠ retrieval.** Memoria y retrieval son *fuentes*; el contexto es *lo que finalmente entra en el prompt*. El arte está en elegir qué entra (cada token cuesta y diluye la atención).
- **El LLM no ejecuta tools.** Propone una llamada (JSON); **tu código valida y ejecuta**. Nunca uses `eval()` sobre salida del modelo. Lista blanca de tools permitidas.
- **RAG reduce alucinaciones, no las elimina.** Si el retrieval trae el documento equivocado, el modelo responderá con seguridad algo equivocado. Por eso se evalúa el retrieval por separado.
- **Evaluación desde el día 1.** Antes de cambiar de modelo o de prompt, corre tu dataset: ¿mejoró o empeoró?
- **Observabilidad = poder responder:** "¿por qué el usuario X recibió esa respuesta el martes a las 3pm y cuánto nos costó?"

## Autoevaluación

1. ¿Qué componente agregarías si los usuarios se quejan de que "el bot olvida lo que le dije"?
2. ¿Y si el bot inventa el plazo de devolución?
3. ¿Y si la factura del proveedor se duplicó este mes y nadie sabe por qué?
