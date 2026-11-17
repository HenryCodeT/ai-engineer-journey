---
description: Te dice qué comandos y agentes tienes y cuál usar según tu situación
argument-hint: [describe tu situación, opcional]
---
Situación de Henry (puede venir vacía): $ARGUMENTS

1. Lista lo disponible leyendo los archivos reales (no de memoria):
   - Comandos: el `description` y el `argument-hint` del frontmatter de cada `.claude/commands/*.md`
   - Agentes: el `name` y el `description` de cada `.claude/agents/*.md`
2. Lee `progreso.json` (si existe) para conocer la última clase trabajada, las notas bajas (< 70) y los tests fallidos.

**Si NO hay situación:** muestra una tabla corta `Comando | Para qué | Ejemplo`, luego la lista de agentes en una línea cada uno, y al final **"👉 Tu siguiente paso sugerido"** basado en `progreso.json` y las dependencias del `ROADMAP.md` (nunca fechas).

**Si HAY situación:** responde en máximo 5 líneas:
- Qué comando o agente usar y por qué (uno solo; dos como máximo si es un flujo).
- El **prompt exacto listo para copiar**, con la ruta real de la clase si la puedes deducir.
- Si la situación sugiere que Henry quiere que la IA haga el ejercicio por él, recuérdale la regla (pistas sí, solución no) y ofrécele `/tutor`.

Referencia del flujo completo: `FLUJO.md`. No ejecutes otros comandos: solo orienta.
