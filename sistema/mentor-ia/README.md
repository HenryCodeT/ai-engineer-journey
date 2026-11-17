# 🎓 Mentor IA: el sistema que construyes durante todo el programa
Un asistente de estudio que conoce **tu** material (notas, transcripts, ejercicios y progreso), te explica, te evalúa y te recomienda qué repasar.
Cada curso en vivo le agrega una capa. Al final es tu **proyecto integrador** y lo defiendes en el Curso 5.

> Avanza capa por capa: una capa está lista cuando cumple sus criterios de aceptación, no por fecha.

## Arquitectura objetivo (v5)
```
             ┌────────────── Cliente (CLI / web / voz) ──────────────┐
                                     │
                           FastAPI  (auth, rate limit)
              ┌──────────────────────┼─────────────────────────┐
         Agentes (LangGraph)     RAG service             Evaluador
     tutor · evaluador · gen.   híbrido + rerank      tests + LLM-as-judge
              │                      │                         │
          MCP server ◄────── Postgres + pgvector/Qdrant ──── Redis (caché)
     (progreso, ejercicios)          │
              └──── Modelo ML: recomendador de repaso (sklearn/MLflow)
                                     │
                OpenTelemetry → Langfuse/Phoenix · GitHub Actions (eval gate) · Azure
```

## Capas
### v1 · Curso 1: AI-ready Backend Service
- [ ] `GET /clases`, `GET /progreso`, `POST /evaluar/{clase}` (envuelve `framework/`)
- [ ] Modelos Pydantic, errores tipados, logging estructurado, config por entorno
- [ ] Persistencia en PostgreSQL (progreso, intentos) y Redis (caché)
- [ ] Tests de unidad, integración y contrato · Dockerfile · docker compose
- [ ] **ML:** modelo que estima la probabilidad de reprobar un tema a partir de intentos y notas → `GET /recomendaciones`
- [ ] ADR-001: por qué FastAPI + Postgres

### v2 · Curso 2: Enterprise RAG (+ LoRA/PEFT Lab)
- [ ] Ingesta de `cursos/**/notas.md`, `transcripts/` y PDFs del curso, con metadata (curso, clase, tema)
- [ ] Chunking + embeddings + pgvector o Qdrant · BM25 + híbrido + reranking
- [ ] `POST /preguntar` con respuesta estructurada **y citas** a la clase de origen
- [ ] Golden set inicial: 30 preguntas con su respuesta y su clase fuente · Recall@K, MRR
- [ ] (Lab) LoRA: adaptar un modelo pequeño para generar ejercicios en el formato del repo, comparado con el modelo base

### v3 · Curso 3: Stateful AI Agent + MCP
- [ ] Agente tutor (pistas graduales) y evaluador (rúbrica): primero sin framework, luego con LangGraph
- [ ] Tools: RAG, ejecutar evaluador, leer progreso, SQL de solo lectura
- [ ] **Servidor MCP** que expone progreso, ejercicios y evaluación (úsalo desde Claude Code)
- [ ] Memoria de temas débiles · checkpointing · aprobación humana antes de escribir archivos

### v4 · Curso 4: Production AI Pipeline
- [ ] Evals: golden set + LLM-as-judge (calidad del feedback del tutor) + trayectoria de agentes
- [ ] Tracing con OpenTelemetry → Langfuse o Phoenix · métricas de latencia, tokens y costo
- [ ] Seguridad: un transcript con prompt injection escondido como caso de prueba · PII · tool abuse
- [ ] GitHub Actions: tests → evals → build → staging → **quality gate** → producción (Azure)
- [ ] FinOps: costo por pregunta, caché, routing entre modelo pequeño y grande
- [ ] (Bonus multimodal) preguntar por voz (Whisper) y entender imágenes o diagramas

### v5 · Curso 5: Demo y defensa
- [ ] Un SLO definido y medido · threat model · auth y authz
- [ ] Diagrama de arquitectura, README profesional y ADRs
- [ ] **Extensión avanzada:** recommender con feedback loop (el modelo de v1 aprende de tus nuevos intentos), o multi-agente/A2A
- [ ] Demo y defensa de trade-offs: calidad, costo, latencia y seguridad

## Estructura (se crea al empezar v1)
```
sistema/mentor-ia/
├── src/mentor_ia/{api,domain,rag,agents,mcp,ml,evals}/
├── tests/
├── docs/adr/        # Architecture Decision Records
├── evals/golden/    # golden sets
└── docker-compose.yml
```
Arranca con `/proyecto mentor-ia planificar v1` cuando empiece el Curso 1.
