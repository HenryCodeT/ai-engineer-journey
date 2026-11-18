---
name: siguiente
description: Indica qué estudiar a continuación según progreso.json y las dependencias del ROADMAP.md. Úsalo al iniciar una sesión de estudio.
---
# Siguiente paso

1. **Lee el progreso actual**:
   - Ejecuta `python -m framework progreso` para ver el estado
   - Lee `progreso.json` para ver últimas clases trabajadas y notas

2. **Analiza el contexto**:
   - Clases con nota < 70 (necesitan retrabajo)
   - Última clase aprobada
   - Patrón de errores (puntos débiles en tipos de ejercicio)

3. **Consulta las dependencias** en `ROADMAP.md`:
   - ¿Qué clase sigue según el orden del programa?
   - ¿Hay prerrequisitos que faltan? (🔗 en el ROADMAP)
   - ¿Corresponde hacer un proyecto modular ahora?

4. **Recomienda UN siguiente paso concreto**:
   - Formato: "Tu siguiente paso: [Clase XX - Tema]"
   - Ruta completa de la clase
   - Objetivo de la sesión (~30-45 min)
   - Prerequisitos que debe repasar si hay gaps

5. **Si está todo aprobado**:
   - Si terminó nivelación → Curso 1
   - Si terminó un curso → siguiente curso o capa de Mentor IA
   - Si terminó todo → proyecto final o extensiones

**Nunca menciones fechas o plazos**. El avance es por dominio, no por calendario.

**Ejemplo de salida:**
```
📍 Tu siguiente paso: Clase 06 - Detección de Outliers

Ruta: cursos/nivelacion/n1-python-data-science/m2-outliers-encoding/clase-06-outliers

Has aprobado las primeras 5 clases del Módulo 1. Ahora entramos al Módulo 2
donde verás cómo identificar y manejar valores atípicos con IQR y Z-score.

Objetivo de la sesión:
- Leer notas.md sobre outliers
- Ejecutar practica.ipynb
- Implementar las 3 funciones de detección en ejercicio.py

Nota: Repasa percentiles (clase 01) si necesitas recordar.
```
