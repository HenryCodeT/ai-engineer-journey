---
description: Planifica o revisa el proyecto integrador de un curso
argument-hint: <n1-preprocesamiento-vivienda|n2-modelos-ml|n3-vision-dl|mentor-ia> [planificar|revisar] [v1..v5]
---
Proyecto: $ARGUMENTS

- Ubicación: proyectos modulares en `proyectos/<p>/`; el sistema integrador en `sistema/mentor-ia/` (indica la capa v1–v5).
- **planificar** (por defecto): lee `ROADMAP.md` y el README del proyecto; propón un plan en hitos pequeños con criterios de aceptación verificables. No escribas el código por mí.
- **revisar**: usa el subagente **revisor-codigo** sobre esa carpeta y compara contra los criterios de la capa.
- Sin fechas ni estimaciones de tiempo: solo orden y criterios de aceptación.
