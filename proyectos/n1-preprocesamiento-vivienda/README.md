# Proyecto modular N1 · Pipeline de preprocesamiento: viviendas en Lima
Proyecto integrador del N1 (Python para Ciencia de Datos). Se construye en tres entregas:

| Entrega | Al terminar | Qué agrega |
|---|---|---|
| **a** | clase 12 | Limpieza: duplicados, texto sucio, faltantes (imputación justificada), outliers, encoding |
| **b** | clase 17 | Selección de variables: baja varianza, correlación, importancia, PCA opcional |
| **c** | clase 24 | Versión de clasificación desbalanceada con `clientes_banco.csv` |

## Entregables
- `src/preprocesamiento.py`: funciones puras `limpiar(df)`, `imputar(df)`, `tratar_outliers(df)`, `codificar(df)`…
- `tests/`: tus propios tests (sí, ¡ahora los escribes tú!)
- `reporte_eda.ipynb`: hallazgos y **justificación de cada decisión**
- `README.md`: problema, decisiones, antes/después

## Criterios de aceptación
- [ ] `python -m pytest proyectos/n1-preprocesamiento-vivienda` pasa
- [ ] El dataset final no tiene NaN, duplicados ni texto inconsistente
- [ ] Cada decisión (por qué mediana y no media, por qué IQR…) está explicada
- [ ] Revisado con `/proyecto n1-preprocesamiento-vivienda revisar` y el feedback atendido

Planifícalo con `/proyecto n1-preprocesamiento-vivienda planificar` al terminar la clase 12.
