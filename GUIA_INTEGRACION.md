# 🔄 Guía de Integración: ai-engineer-modulo1 → ai-engineer-journey

Este documento explica cómo se integró el contenido de `ai-engineer-modulo1` en `ai-engineer-journey` y qué falta por completar.

## ✅ Cambios completados

### 1. Infraestructura base
- ✅ **FakeLLM integrado** en `framework/llm/`
  - `fake_llm.py` - LLM simulado para tests reproducibles
  - `claude_llm.py` - Adaptador a Claude real
  - `stub_llm.py` - LLM de prueba para tests
  - Importable con: `from framework.llm import FakeLLM, ClaudeLLM, crear_llm`

- ✅ **Skills nuevos** en `.claude/skills/`
  - `explicar/` - Explica conceptos con analogías y ejemplos
  - `quiz/` - Quiz interactivo sobre temas
  - `entrevista/` - Simulación de entrevista técnica
  - `pista-gradual/` - Pistas graduales sin revelar la solución
  - `siguiente/` - Recomienda qué estudiar a continuación

- ✅ **Visualización mejorada**
  - `PROGRESO.md` auto-generado desde `progreso.json`
  - Emojis y barras de progreso visuales
  - `make progreso` actualiza ambos archivos

### 2. Contenido del Módulo 1 (Curso 1)
- ✅ Estructura base creada en `cursos/curso-1-fundamentos-software-ia/m1-rol-y-sistemas/`
- ✅ **Teoría** copiada (5 archivos .md)
- ✅ **Laboratorio** copiado (`lab_claude_real.py`)
- ✅ **README del módulo** explicando estructura y flujo
- ✅ **Clase 01 (roles)** completamente adaptada como ejemplo:
  - `notas.md` - Resumen de la teoría
  - `practica.ipynb` - Notebook guiado
  - `ejercicio.py` - Con plantilla (NotImplementedError)
  - `test_ejercicio.py` - Con decoradores @pytest.mark.puntos
  - Tests funcionando: 100 puntos totales

## 🔧 Proceso de adaptación (usar clase-01 como plantilla)

Para adaptar los 12 ejercicios restantes (02-13), sigue este proceso:

### Paso 1: Crear estructura
```bash
mkdir -p "cursos/curso-1-fundamentos-software-ia/m1-rol-y-sistemas/clase-XX-tema"
```

### Paso 2: Generar notas.md
1. Lee `teoria/XX_tema.md` correspondiente
2. Crea `notas.md` con:
   - Referencia a la teoría completa
   - Resumen de conceptos clave
   - Qué se va a practicar en el ejercicio

**Plantilla:**
```markdown
# Clase XX: [Tema]

> **Teoría completa:** `../teoria/XX_tema.md`
> **Objetivo:** [Objetivo del módulo correspondiente]

## Concepto clave

[Resumen de 2-3 párrafos]

## Ejercicio de esta clase

[Qué va a implementar Henry]

## Para profundizar

- Lee `../teoria/XX_tema.md` completo
- Ejecuta `practica.ipynb`
- Implementa `ejercicio.py` SIN ayuda de IA
```

### Paso 3: Adaptar ejercicio.py
1. Copia de `../ai-engineer-modulo1/ejercicios/nivelX_dificultad/ejXX_tema.py`
2. Modifica el header:
   ```python
   """Clase XX · [Tema]

   Evalúa con:  make evaluar CLASE=cursos/curso-1-fundamentos-software-ia/m1-rol-y-sistemas/clase-XX-tema
   """
   from __future__ import annotations
   ```
3. Reemplaza las implementaciones con `raise NotImplementedError("Implementa [función]()")`
4. Agrega docstrings detallados con pistas

### Paso 4: Adaptar test_ejercicio.py
1. Copia de `../ai-engineer-modulo1/tests/test_ejXX.py`
2. Modifica imports:
   ```python
   """Tests para Clase XX: [Tema]"""
   import pytest
   from ejercicio import [funciones_a_testear]
   ```
3. Agrega decoradores `@pytest.mark.puntos(N)` a cada test
4. Ajusta descripciones de tests
5. **Importante:** Suma total de puntos debe ser 100

**Distribución de puntos recomendada:**
- 1 función simple → 15-20 puntos
- 1 función media → 25-30 puntos
- 1 función compleja → 35-40 puntos
- Quiz (si aplica) → 5 pts (completitud) + 30 pts (correctitud)

### Paso 5: Crear practica.ipynb
1. Crea notebook básico con:
   - Introducción al concepto
   - Ejemplo simple comentado
   - Experimentación interactiva
   - Recordatorio para implementar `ejercicio.py`

**Plantilla (copia y adapta de clase-01):**
```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Práctica: [Tema]"]
  },
  ...
 ]
}
```

### Paso 6: Verificar
```bash
python -m framework evaluar "cursos/curso-1-fundamentos-software-ia/m1-rol-y-sistemas/clase-XX-tema"
```

Debe mostrar 0/100 con todos los tests fallando por NotImplementedError.

## 📋 Checklist de clases pendientes

### Nivel Fácil (clases 02-04)
- [ ] clase-02-determinista-probabilistico (`ej02_determinista_vs_probabilistico.py`)
- [ ] clase-03-componentes (`ej03_componentes.py`)
- [ ] clase-04-costos-tokens (`ej04_costos_tokens.py`)

### Nivel Medio (clases 05-09)
- [ ] clase-05-retrieval-contexto (`ej05_retrieval_contexto.py`)
- [ ] clase-06-tools (`ej06_tools.py`)
- [ ] clase-07-memoria (`ej07_memoria.py`)
- [ ] clase-08-evaluacion (`ej08_evaluacion.py`)
- [ ] clase-09-observabilidad (`ej09_observabilidad.py`)

### Nivel Difícil (clases 10-13)
- [ ] clase-10-selector-modelos (`ej10_selector_modelos.py`)
- [ ] clase-11-build-vs-buy (`ej11_build_vs_buy.py`)
- [ ] clase-12-cuando-no-llm (`ej12_cuando_no_llm.py`)
- [ ] clase-13-confiabilidad-seguridad (`ej13_confiabilidad_seguridad.py`)

## 🎯 Proyecto final del Curso 1
- [ ] Adaptar `proyecto_final/` de ai-engineer-modulo1
- [ ] Crear en `proyectos/curso-1-asistente-soporte/`
- [ ] Incluir:
  - README con rúbrica
  - DISENO.md para documentar decisiones
  - asistente.py con plantilla
  - tests de integración

## 📚 Documentación actualizada
- [ ] Actualizar `ROADMAP.md` con detalle del Módulo 1
- [ ] Actualizar `README.md` mencionando FakeLLM y skills nuevos
- [ ] Actualizar `CLAUDE.md` con contexto de ejercicios con LLMs

## 🔄 Próximos pasos sugeridos

1. **Completar el resto de clases** siguiendo el proceso documentado
2. **Integrar el proyecto final** del Curso 1
3. **Actualizar documentación** principal
4. **Validar que todo funciona**:
   ```bash
   pytest cursos/curso-1-fundamentos-software-ia/m1-rol-y-sistemas/
   make progreso
   ```
5. **Merge a main** cuando esté todo completo

## 🎉 Beneficios de la integración

- ✅ **Un solo proyecto unificado** para todo el programa
- ✅ **Sistema de evaluación consistente** (pytest con puntos)
- ✅ **FakeLLM disponible** para todos los cursos futuros
- ✅ **Skills interactivos** mejoran la experiencia de aprendizaje
- ✅ **Visualización mejorada** del progreso
- ✅ **13 clases prácticas** del Módulo 1 (fundamentos de AI Engineering)

## 📖 Referencias

- Proyecto original: `../ai-engineer-modulo1/`
- Clase ejemplo: `cursos/curso-1-fundamentos-software-ia/m1-rol-y-sistemas/clase-01-roles/`
- Tests del framework: `tests/`
- Configuración pytest: `conftest.py` y `pytest.ini`
