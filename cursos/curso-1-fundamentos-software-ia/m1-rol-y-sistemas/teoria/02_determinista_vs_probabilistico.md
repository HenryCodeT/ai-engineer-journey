# 2. Software determinista vs. sistemas probabilísticos

> **Objetivo:** distinguir ambos paradigmas y saber cuándo cada uno es más adecuado.
> **Ejercicios:** `ej02_determinista_vs_probabilistico.py`, `ej08_evaluacion.py`

## La diferencia

| | Determinista | Probabilístico (LLM) |
|---|---|---|
| Misma entrada → | **misma salida, siempre** | una **distribución** de salidas posibles |
| Cómo se especifica | reglas explícitas (código) | ejemplos e instrucciones en lenguaje natural |
| Cómo se prueba | `assert f(x) == y` | **métricas** sobre un dataset (accuracy, % de respuestas válidas...) |
| Errores | bugs reproducibles | errores estadísticos, a veces irreproducibles |
| Fortaleza | exactitud, auditabilidad, costo ~0 | entender lenguaje ambiguo, generalizar, generar texto |
| Debilidad | no maneja ambigüedad ni lenguaje libre | alucina, varía, cuesta por token, es lento |

```python
# Determinista: lo pruebas con un assert y listo
assert calcular_total_con_igv(100, 2) == 236.0

# Probabilístico: lo pruebas con estadística
resultados = [clasificar(texto) for texto, _ in dataset]
accuracy = aciertos / total   # ¿≥ 0.9? -> aprobado
```

## La temperatura

`temperature` controla la aleatoriedad del muestreo.
- `0` → casi siempre elige el token más probable → respuestas casi idénticas (¡ojo: "casi", en APIs reales no hay garantía absoluta!).
- `1` → más variedad y creatividad, también más errores de formato.

Regla práctica: **tareas de extracción/clasificación → temperatura baja; tareas creativas → más alta.**

## El patrón más importante: "sándwich determinista"

Casi todo sistema de IA serio envuelve al componente probabilístico en código determinista:

```
entrada → [validación determinista] → [LLM] → [parseo/normalización/validación determinista] → salida
```

Ejemplos: normalizar "Positivo." → `"positivo"`; validar JSON contra un schema; verificar que una fecha extraída exista; reintentar si la salida no es válida.

## ¿Cuándo usar cada enfoque?

Usa **determinista** cuando:
- Las reglas son claras y finitas (impuestos, validar un RUC, ordenar, calcular).
- El error tolerado es **cero** (dinero, salud, legal, permisos).
- Necesitas auditar exactamente por qué se tomó una decisión.

Usa **probabilístico** cuando:
- La entrada es lenguaje natural, imágenes o datos muy variados.
- Escribir las reglas sería imposible ("¿este correo es una queja?").
- Una respuesta "buena la mayoría de veces" + supervisión es aceptable.

**Lo más común en la vida real: híbrido.** Reglas para lo que se puede, LLM para lo que no (ver ej12).

## Autoevaluación

1. ¿Por qué no basta con probar un prompt 3 veces a mano?
2. Diseña el sándwich determinista para "extraer la fecha de entrega de un correo".
3. ¿Qué temperatura usarías para clasificar tickets? ¿Y para generar nombres de productos?
