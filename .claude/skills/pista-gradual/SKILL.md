---
name: pista-gradual
description: Da una pista gradual (nivel 1, 2 o 3) para un ejercicio sin revelar la solución completa. Úsalo cuando Henry está atascado en un ejercicio.
---
# Pista gradual

Argumento: descripción del problema o ruta de la clase.

1. **Identifica el ejercicio**:
   - Si da una ruta de clase, lee `ejercicio.py` y `test_ejercicio.py`
   - Si describe un problema, identifica qué función o concepto está atascado

2. **Corre los tests** (si aplica):
   ```bash
   make evaluar CLASE=<ruta> -v
   ```
   O simplemente mira el código para entender qué falta.

3. **Da SOLO el siguiente nivel de pista** (empieza en 1 si no especificó):

   **Nivel 1 – Concepto:**
   - ¿Qué idea o componente está en juego?
   - Una pregunta guía
   - Ejemplo: "Para detectar outliers con IQR, ¿qué necesitas calcular primero?"

   **Nivel 2 – Enfoque:**
   - Qué estructuras/funciones usar (pandas, numpy, sklearn)
   - Ejemplo: "Usa `df.quantile([0.25, 0.75])` para Q1 y Q3, luego calcula IQR = Q3 - Q1"

   **Nivel 3 – Pseudocódigo:**
   - Pasos en español, sin código Python ejecutable
   - Ejemplo:
     ```
     1. Calcula Q1 (percentil 25)
     2. Calcula Q3 (percentil 75)
     3. IQR = Q3 - Q1
     4. Límites: inferior = Q1 - 1.5*IQR, superior = Q3 + 1.5*IQR
     5. Filtra valores fuera de esos límites
     ```

4. **Nunca** muestres el código completo de ejercicio.py.

5. Si tras el nivel 3 sigue atascado:
   - Ofrece revisar juntos línea por línea
   - Pregunta qué parte específica no entiende
   - Verifica si hay un error conceptual

**Ejemplo de uso:**
- `/pista-gradual cursos/nivelacion/n1-python-data-science/m2-outliers-encoding/clase-06-outliers`
- `/pista-gradual Estoy atascado en la función de imputación por grupos`
