# 5. Build vs. Buy y cuándo NO usar un LLM

> **Objetivo:** aplicar criterios para construir o comprar, y reconocer cuándo un LLM es la herramienta equivocada.
> **Ejercicios:** `ej11_build_vs_buy.py`, `ej12_cuando_no_usar_llm.py`

## El espectro (no es binario)

```
BUY ◄─────────────────────────────────────────────────────────────► BUILD
SaaS listo      API de LLM +          API + RAG/evals/        Modelo open-source     Entrenar modelo
(chatbot        prompts               tools propios           self-hosted +          propio
 comercial)                           ("híbrido")             fine-tuning
```
La mayoría de empresas termina en el **medio**: compran el modelo (API) y construyen lo que las diferencia: sus datos, su retrieval, su evaluación, su integración.

## Criterios

| Criterio | Inclina a BUY | Inclina a BUILD |
|---|---|---|
| ¿Es el core del negocio / diferenciador? | No | Sí |
| Time to market | Urgente | Hay tiempo |
| Equipo | Sin experiencia en ML/infra | Equipo ML maduro |
| Volumen | Bajo / medio | Muy alto (costo variable domina) |
| Datos sensibles / regulación | Proveedor cumple la normativa | Ningún proveedor cumple |
| Personalización | Estándar | Muy específica |
| Lock-in | Aceptable | Inaceptable |

## TCO: el costo que nadie cuenta

"El modelo open-source es gratis" → **falso**. TCO = costo inicial (desarrollo, integración) + fijo mensual (GPUs, ingenieros, licencias, mantenimiento) + variable (por llamada). Calcula el **punto de equilibrio**: ¿desde qué mes construir sale más barato que comprar? Si es en el mes 40 y tu producto puede cambiar en 12, compra.

## Cuándo NO usar un LLM

No uses un LLM si:
1. **Hay reglas claras y finitas** → código. (Calcular impuestos, validar un RUC, ordenar, filtrar.)
2. **El error tolerado es cero y no hay supervisión** → código o humano. (Aprobar créditos automáticamente, dosis médicas.)
3. **Es una búsqueda exacta** → base de datos. (Estado del pedido #1002: ¡nunca dejes que un LLM lo "adivine"!)
4. **Es aritmética** → calculadora. Los LLMs se equivocan en cuentas.
5. **La latencia o el costo no lo permiten** → un modelo clásico (regex, clasificador pequeño) puede ser 1000× más barato.
6. **Necesitas explicabilidad exacta** (auditoría, regulación) → reglas o modelos interpretables.
7. **Un formulario o un botón resuelve mejor** la UX que un chat.

Sí usa un LLM cuando hay **lenguaje natural variado, ambigüedad, generación de texto, resumen, extracción de documentos heterogéneos**, y una respuesta "buena la mayoría de veces" + validación/supervisión es aceptable.

## Patrón: router híbrido (ej12)

```
consulta ─► ¿aritmética? ─► calculadora
         ─► ¿pedido #N?  ─► base de datos
         ─► ¿extraer email? ─► regex
         ─► si nada aplica ─► LLM (último recurso)
```
Beneficios: más barato, más rápido, exacto donde importa, y el LLM se concentra en lo que solo él puede hacer.

## Autoevaluación

1. Una fintech quiere un asistente para que sus analistas consulten normativa interna. ¿Build, buy o híbrido? Justifica con 3 criterios.
2. Da 3 ejemplos de tu trabajo/vida donde alguien usaría un LLM pero sería mejor código.
3. ¿Qué costos escondidos tiene self-hostear un modelo?
