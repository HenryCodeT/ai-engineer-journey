# 🗺️ ROADMAP: Programa de Especialización AI Engineer (DSRP)

> Sin fechas. **Avanzas cuando dominas**: una clase está hecha con nota ≥ 70 y un curso, cuando su parte del sistema funciona y pasó revisión.
> El orden importa solo por **dependencias**, marcadas con 🔗.

```
NIVELACIÓN (grabada)                 CURSOS EN VIVO: cada uno agrega una capa a Mentor IA
┌─────────────────────────┐          ┌──────────────────────────────────────────────────────────┐
│ N1 Python Ciencia Datos │──┐       │ C1 Backend AI-ready ─► C2 RAG + LoRA ─► C3 Agentes + MCP │
│ N2 Machine Learning     │──┼──────►│        ─► C4 LLMOps (evals, observabilidad, cloud)       │
│ N3 Deep Learning        │──┘       │        ─► C5 Proyecto final: demo y defensa              │
└─────────────────────────┘          └──────────────────────────────────────────────────────────┘
  + tracks de apoyo: SQL · Git y terminal              + bonus asíncrono: Agentes Multimodales
```

---

## 🎓 El sistema único: **Mentor IA**
Un asistente de estudio que **usa tu propio material del programa**: notas, transcripts, ejercicios y progreso.
Lo construyes curso a curso, como pide el programa (*"cada curso suma una nueva parte al proyecto final"*), y lo usas mientras estudias.
Detalle de las capas en [`sistema/mentor-ia/README.md`](sistema/mentor-ia/README.md).

| Capa | Curso | Qué le agregas a Mentor IA |
|---|---|---|
| v0 | Nivelación | Las piezas: datos limpios, modelos de ML y DL (proyectos modulares) |
| v1 | Curso 1 | API FastAPI tipada: clases, progreso y evaluación + Postgres/Redis + tests + Docker + **modelo de ML de recomendación de repaso** |
| v2 | Curso 2 | **RAG** sobre tus notas y transcripts con citas, búsqueda híbrida y evaluación de retrieval · LoRA opcional |
| v3 | Curso 3 | **Agentes** tutor, evaluador y generador con LangGraph · **servidor MCP** · memoria de tus temas débiles · aprobación humana |
| v4 | Curso 4 | **Golden set + LLM-as-judge**, trazas (OpenTelemetry/Langfuse), seguridad (prompt injection), CI/CD con quality gate, deploy en Azure, FinOps |
| v5 | Curso 5 | SLO, threat model, ADRs, demo y **defensa técnica** · extensión avanzada |

---

## 🟦 Nivelación (sesiones grabadas · evaluación en la plataforma)

### N1 · Python para Ciencia de Datos (`cursos/nivelacion/n1-python-data-science`, 25 clases)
| Módulo | Clases | Temas |
|---|---|---|
| M1 · EDA e imputación | 01–05 | Flujo de un proyecto, tipos de variables, distribuciones, datos faltantes, métodos de imputación |
| M2 · Outliers y encoding | 06–12 | Detección y tratamiento de outliers, codificación, caso *Panorama de la vivienda* |
| M3 · Selección de características | 13–17 | Baja varianza, correlación, importancia, reducción de dimensionalidad |
| M4 · Balance de datos | 18–25 | Desbalance, métodos a nivel de datos, imágenes, desafío de clasificación, Market Basket |

🧩 **Proyecto modular N1:** `proyectos/n1-preprocesamiento-vivienda`

### N2 · Machine Learning (`cursos/nivelacion/n2-machine-learning`)
Algoritmos de regresión · algoritmos de clasificación · clustering · métodos avanzados de ML.
🧩 **Proyecto modular N2:** modelo de precio de vivienda + clasificador de abandono, comparando estrategias de preprocesamiento.
🔗 *Se usa en:* Curso 1, Módulo 4 (train/validation/test, métricas, overfitting, data leakage).

### N3 · Deep Learning (`cursos/nivelacion/n3-deep-learning`)
Fundamentos de procesamiento de señales e imágenes · redes convolucionales · modelos de secuencia.
🧩 **Proyecto modular N3:** clasificador de imágenes, con manejo de desbalance (conecta con N1, clase 21).
🔗 *Se usa en:* Curso 1, Módulo 4 (PyTorch, tensores) y Curso 2, Módulo 1 (transformers, attention).

---

## 🟩 Cursos en vivo

### Curso 1 · Fundamentos de Ingeniería de Software e IA (`cursos/curso-1-fundamentos-software-ia`)
| Módulo | Temas oficiales |
|---|---|
| M1 · Rol del AI Engineer y sistemas de IA | AI Engineer vs. LLM Dev, Prompt Eng., DS y ML Eng. · software determinista vs. probabilístico · componentes de una app de IA · trade-offs de calidad, latencia, costo y privacidad · Build vs. Buy · cuándo NO usar un LLM |
| M2 · Ingeniería de software para IA | Typing, excepciones, async/await, configuración y dependencias · Pydantic y JSON Schema · SOLID y Clean Architecture · logging estructurado · pytest (unitarias, integración, contrato) · Git/GitHub, PRs y code review |
| M3 · APIs y servicios preparados para IA | HTTP, REST, OpenAPI y FastAPI · sync, async y streaming · API keys, secretos y OAuth · rate limits, retries, timeouts, idempotencia, circuit breakers · PostgreSQL + Redis |
| M4 · Datos y ML para AI Engineers | Train/val/test, overfitting, leakage · métricas · embeddings y cosine similarity · PyTorch · ciclo de vida de ML · SQL para servicios de IA |
| M5 · Desarrollo asistido por IA | Claude Code, Codex y Copilot · spec-driven development · tests y refactor asistidos · delegar a coding agents y validar sus cambios |

🛠️ **Taller:** AI-ready Backend Service → **Mentor IA v1**
🔗 *Antes conviene:* track Git · track SQL · N2 (métricas, leakage).
💡 Este mismo repo es práctica del M5: lo usas con Claude Code, agentes y specs.

### Curso 2 · LLMs, RAG y Fine-Tuning (`cursos/curso-2-llms-rag-finetuning`)
| Módulo | Temas oficiales |
|---|---|
| M1 · Foundation models y selección | Transformers, attention, tokens, context window · pretraining, post-training, reasoning, LLM vs. SLM · hosted vs. open source · latencia, TTFT, costo por token · routing de modelos |
| M2 · Prompt, contexto y salidas estructuradas | Jerarquía system/developer/user · JSON Schema + Pydantic · selección, compresión y caché de contexto · versionado de prompts y pruebas de regresión · prompt injection |
| M3 · RAG e ingeniería de conocimiento | Ingesta (PDF, DOCX, web, SQL, audio, imágenes) · parsing y OCR · embeddings y vector DBs · BM25 e híbrido · query rewriting · chunking y parent-child · reranking · citas, freshness, permisos · Recall@K, MRR, faithfulness · GraphRAG |
| M4 · Fine-tuning con LoRA/PEFT | Prompting vs. RAG vs. FT vs. tools · curación de datasets · SFT, LoRA, QLoRA · adapters y serving · comparar el modelo base con el adaptado |

🛠️ **Taller:** Enterprise RAG + LoRA/PEFT Lab → **Mentor IA v2**
🔗 *Antes conviene:* N3 (modelos de secuencia) · Curso 1, M4 (embeddings).

### Curso 3 · Agentes de IA, MCP y Sistemas Multiagente (`cursos/curso-3-agentes-mcp-multiagente`)
| Módulo | Temas oficiales |
|---|---|
| M1 · Tools, APIs e integración | Function/tool calling con JSON Schema · tools de API, SQL, archivos y búsqueda · validación estricta · OAuth y secretos · retries e idempotencia · allowlists, least privilege, aprobación humana |
| M2 · Agentes y orquestación | Agent loop · workflow vs. agente · planning, routing, reflection · Human-in-the-Loop · single, router, supervisor y multi-agent · primero sin framework, luego OpenAI Agents SDK / LangGraph / Foundry |
| M3 · MCP y A2A | MCP client/server, tools, resources y prompts · remote MCP y transports · autorización y scopes · seguridad en MCP · A2A |
| M4 · Memoria, estado y agentes durables | Estado conversacional · memoria semántica y episódica · checkpointing, colas y eventos · tareas largas con recovery · approval gates y multi-tenancy |

🛠️ **Taller:** Stateful AI Agent + MCP → **Mentor IA v3**
💡 Los agentes de `.claude/agents/` son el prototipo: aquí los construyes "de verdad".

### Curso 4 · LLMOps: IA en Producción (`cursos/curso-4-llmops`)
| Módulo | Temas oficiales |
|---|---|
| M1 · Desarrollo guiado por evaluación | Golden y synthetic datasets · checks deterministas, LLM-as-judge, pairwise · evaluar RAG y agentes (task success, trajectory) · regression tests y quality gates · red teaming, A/B y evaluación humana |
| M2 · Observabilidad y confiabilidad | Versionar prompts, modelos y datasets · tracing y OpenTelemetry · latencia, tokens, errores y costo · SLIs, SLOs y error budgets · fallbacks, routing y rollback |
| M3 · Seguridad y gobernanza | Prompt injection directo e indirecto, jailbreaks · PII y RAG poisoning · tool abuse y excessive agency · RBAC, sandboxing y audit logs · threat modeling en MCP · OWASP GenAI y NIST AI RMF |
| M4 · Cloud, CI/CD y FinOps | FastAPI serving, Docker, serverless, K8s · API Gateway y managed identity · GitHub Actions con eval gate · blue/green y canary · IaC · token budgeting, caching y model routing · Azure/Foundry (equivalencias en AWS y GCP) |

🛠️ **Taller:** Production AI Pipeline → **Mentor IA v4**
🎁 **Bonus asíncrono · Agentes Multimodales:** STT (Whisper), TTS y visión. En Mentor IA: preguntarle por voz y que lea diagramas.
💡 El evaluador de pytest de este repo pasa a ser un sistema de **evals de LLM**: practicas LLMOps sobre tu propio framework.

### Curso 5 · Proyecto Final: Sistema de IA en Producción (`cursos/curso-5-proyecto-final`)
- **M1:** API tipada y testeada · componente LLM y/o ML · RAG · tools, SQL y MCP · evals con golden set · tracing, métricas y un SLO · threat model y auth · CI/CD y endpoint en cloud · FinOps · diagrama, README y ADRs · demo
- **M2:** una extensión avanzada (LoRA, multimodal, **recommender y feedback loops**, multi-agente/A2A o GraphRAG) y defensa de los trade-offs

🛠️ **Taller:** Demo y defensa técnica → **Mentor IA v5**

---

## 🧰 Tracks de apoyo (requisitos del programa)
| Track | Por qué | Se usa en |
|---|---|---|
| `cursos/tracks/sql` | El profesor insistió en SQL en la clase 01 · requisito del programa | N1, Curso 1 M3–M4, Curso 3 M1 (SQL tools) |
| `cursos/tracks/git` | Requisito del programa (Git y terminal) | Curso 1 M2 (PRs y code review) y todo el sistema |

## ✅ Criterio de "terminado"
- **Clase:** nota automática ≥ 70 (`make progreso`).
- **Curso:** sus clases aprobadas + su capa de Mentor IA funcionando, con tests, y revisada con `/proyecto ... revisar`.
- **Diario:** una entrada por sesión en `diario.md` (qué aprendí, qué me costó, qué repasar).
