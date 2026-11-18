# 🧠 AI Engineer Journey · Henry

Repositorio de aprendizaje para el **Programa de Especialización AI Engineer (DSRP)**.
Cada clase se convierte en **notas → práctica → ejercicio evaluado → feedback**.
Cada curso en vivo agrega una capa a **Mentor IA**, el sistema único que defiendes en el proyecto final.

> Sin fechas: avanzas cuando dominas (nota ≥ 70 por clase). El orden lo dan las dependencias del `ROADMAP.md`.

```
Transcript de la clase ──► /clase ──► notas.md + practica.ipynb + ejercicio.py + tests
                                              │
                         tú implementas  ◄────┘
                                              │
                    make evaluar ──► tests automáticos (nota 0-100) ──► progreso.json
                                              │
                    /evaluar ──► agente evaluador: rúbrica + feedback.md
                                              │
                    dudas ──► agente tutor (método socrático, no te da la solución)
```

## 🚀 Inicio rápido

```bash
# 1. Entorno
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Generar los datasets de práctica (sintéticos, sin internet)
make datos

# 3. Ver la ruta de aprendizaje y tu progreso
make progreso

# 4. Trabajar la clase 01
#    - Lee cursos/nivelacion/n1-python-data-science/m1-eda-faltantes/clase-01-eda/notas.md
#    - Explora practica.ipynb
#    - Implementa las funciones de ejercicio.py
make evaluar CLASE=cursos/nivelacion/n1-python-data-science/m1-eda-faltantes/clase-01-eda

# ¿Sin `make` (Windows)? Usa directamente:
python -m framework evaluar cursos/nivelacion/n1-python-data-science/m1-eda-faltantes/clase-01-eda
python -m framework progreso
```

## 🤖 Usarlo con Claude Code

📘 **Flujo de trabajo y prompts listos para copiar: [`FLUJO.md`](FLUJO.md)** · ¿No sabes qué usar? → `/ayuda`

Abre la carpeta con `claude` y tendrás disponibles:

| Comando | Qué hace |
|---|---|
| `/ayuda [situación]` | Lista comandos y agentes, y te dice cuál usar con el prompt listo |
| `/clase <transcript pegado o ruta>` | Guarda el transcript en `transcripts/` y genera la carpeta de la clase (notas, notebook, ejercicio, tests) |
| `/evaluar <ruta-clase>` | Corre los tests + el agente **evaluador** revisa tu código con rúbrica y deja `feedback.md` |
| `/tutor <duda>` | El agente **tutor** te explica con preguntas guía, analogías y pistas graduales |
| `/progreso` | Resumen de avance, puntos débiles y qué estudiar después |
| `/proyecto <n1-…\|mentor-ia> [planificar\|revisar] [capa]` | Planifica o revisa un proyecto modular o una capa de Mentor IA |

**Skills interactivos** (nuevos):
- `/explicar <tema>` - Explica un concepto con analogía, código y ejemplo
- `/quiz <tema>` - Quiz de 5 preguntas con corrección inmediata
- `/entrevista` - Simulación de entrevista técnica con feedback
- `/pista-gradual <clase o problema>` - Pistas graduales nivel 1→2→3
- `/siguiente` - Qué estudiar a continuación según tu progreso

Agentes (en `.claude/agents/`):

- **tutor** — explica conceptos, nunca entrega la solución completa de un ejercicio a la primera.
- **evaluador** — corre los tests, aplica la rúbrica (correctitud, claridad, buenas prácticas, comprensión) y escribe feedback.
- **revisor-codigo** — code review estilo industria (para proyectos).
- **generador-ejercicios** — crea ejercicios + tests nuevos a partir de notas/transcripts, o ejercicios de refuerzo sobre tus puntos débiles.

## 📁 Estructura

```
├── CLAUDE.md                 # Reglas del repo para Claude Code
├── FLUJO.md                  # Flujo por clase + prompts para copiar
├── ROADMAP.md                # Temario oficial, dependencias y capas de Mentor IA
├── PROGRESO.md               # Visualización amigable del progreso (auto-generado)
├── progreso.json             # Lo actualiza el evaluador (no editar a mano)
├── framework/                # Motor de evaluación y progreso (Python)
│   └── llm/                  # FakeLLM y ClaudeLLM para ejercicios con LLMs
├── data/                     # Generadores de datasets de práctica
├── transcripts/              # n1-python/, n2-ml/, n3-dl/, curso-1/ … curso-5/
├── cursos/
│   ├── nivelacion/
│   │   ├── n1-python-data-science/   # 25 clases (la clase 01 ya está lista)
│   │   ├── n2-machine-learning/
│   │   └── n3-deep-learning/
│   ├── curso-1-fundamentos-software-ia/
│   ├── curso-2-llms-rag-finetuning/
│   ├── curso-3-agentes-mcp-multiagente/
│   ├── curso-4-llmops/
│   ├── curso-5-proyecto-final/
│   └── tracks/
│       ├── sql/              # con ejercicio evaluado (clase 01 lista)
│       └── git/              # retos con checklist
├── proyectos/                # Proyectos modulares de la nivelación (n1, n2, n3)
└── sistema/mentor-ia/        # 🎓 El sistema único que crece en los cursos 1 a 5
```

Cada clase sigue la misma plantilla:

```
clase-XX-tema/
├── notas.md            # Resumen de la clase
├── practica.ipynb      # Notebook guiado (ejecutar celda por celda)
├── ejercicio.py        # Funciones que TÚ implementas
├── test_ejercicio.py   # El evaluador automático (no modificar)
└── feedback.md         # Lo genera /evaluar
```

## 🔁 Flujo recomendado por clase

1. Ve el video.
2. `/clase` pegando el título y el transcript (ver `FLUJO.md`).
3. Lee `notas.md` y ejecuta `practica.ipynb`.
4. Implementa `ejercicio.py` **sin IA**. Atascado > 15 min → `/tutor`.
5. `make evaluar CLASE=...` hasta pasar ≥ 70.
6. `/evaluar` para feedback cualitativo. Haz commit: `git commit -m "clase 01: EDA (nota NN)"`.

## 🧰 Primer commit sugerido

```bash
git init && git add . && git commit -m "Inicio: AI Engineer Journey"
gh repo create ai-engineer-journey --private --source=. --push   # opcional
```

## 🧪 ¿Cómo funciona la nota?
Cada test de `test_ejercicio.py` tiene un peso (`@pytest.mark.puntos(n)`) y la nota es la suma de los tests aprobados, de 0 a 100. Se aprueba con **≥ 70**. Cada intento queda registrado en `progreso.json`, que los agentes leen para saber en qué temas estás flojo.
