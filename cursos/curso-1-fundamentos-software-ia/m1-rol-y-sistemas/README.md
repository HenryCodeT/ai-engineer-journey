# Módulo 1: Rol del AI Engineer y Sistemas de IA

Este módulo cubre los fundamentos del rol de AI Engineer y el diseño de sistemas de IA en producción.

## 🎯 Objetivos del módulo

1. Identificar el rol del AI Engineer frente a LLM Developer, Prompt Engineer, Data Scientist y ML Engineer
2. Distinguir software determinista vs sistemas probabilísticos y cuándo usar cada uno
3. Reconocer componentes: modelo, datos, contexto, tools, memoria, retrieval, evaluación, observabilidad, infraestructura
4. Analizar trade-offs: calidad, latencia, costo, privacidad, seguridad, confiabilidad
5. Aplicar Build vs Buy y reconocer cuándo NO usar un LLM

## 📚 Estructura

### Teoría (`teoria/`)
5 lecciones fundamentales:
- `01_roles.md` - Roles en AI Engineering
- `02_determinista_vs_probabilistico.md` - Software determin��stico vs probabilístico
- `03_componentes.md` - Los 9 componentes de una app de IA
- `04_tradeoffs.md` - Trade-offs en sistemas de IA
- `05_build_vs_buy.md` - Build vs Buy y cuándo NO usar LLM

### Clases (13 ejercicios evaluados)

#### Nivel Fácil (clases 01-04)
- `clase-01-roles/` - Clasificador de roles + quiz conceptual
- `clase-02-determinista-probabilistico/` - Temperature, normalización de salidas
- `clase-03-componentes/` - Los 9 componentes de un sistema de IA
- `clase-04-costos-tokens/` - Cálculo de costos por llamada y mensual, caché

#### Nivel Medio (clases 05-09)
- `clase-05-retrieval-contexto/` - Mini-RAG y manejo de alucinaciones
- `clase-06-tools/` - Function calling robusto y seguro
- `clase-07-memoria/` - Gestión de memoria con presupuesto de tokens
- `clase-08-evaluacion/` - Evaluación con dataset: accuracy y consistencia
- `clase-09-observabilidad/` - Trazas, costo, p50/p95

#### Nivel Difícil (clases 10-13)
- `clase-10-selector-modelos/` - Selector de modelos con restricciones y Pareto
- `clase-11-build-vs-buy/` - Análisis de TCO y punto de equilibrio
- `clase-12-cuando-no-llm/` - Router híbrido y análisis de ahorro
- `clase-13-confiabilidad-seguridad/` - PII, prompt injection, retries y fallback

### Laboratorio (`laboratorio/`)
- `lab_claude_real.py` - Experimentos con Claude real (requiere API key)

## 🤖 LLM simulado vs real

Los ejercicios usan **FakeLLM** (de `framework.llm`) para:
- Practicar sin costos
- Tests reproducibles y deterministas
- Simular comportamientos reales (temperature, latencia, fallas, alucinaciones)

Para usar Claude real:
```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
python cursos/curso-1-fundamentos-software-ia/m1-rol-y-sistemas/laboratorio/lab_claude_real.py
```

## 🔁 Flujo por clase

1. Lee la teoría correspondiente en `teoria/`
2. Lee `notas.md` de la clase
3. Explora `practica.ipynb`
4. Implementa `ejercicio.py` **sin IA**
5. `make evaluar CLASE=cursos/curso-1-fundamentos-software-ia/m1-rol-y-sistemas/clase-XX-tema`
6. ¿Atascado? → `/tutor` o `/pista-gradual`
7. Aprobado (≥70) → `/evaluar` para feedback cualitativo

## 📈 Progreso

Verifica tu avance:
```bash
make progreso
```

Cada clase suma puntos al total del Curso 1.
