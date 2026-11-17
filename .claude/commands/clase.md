---
description: Genera la carpeta de una clase (notas, notebook, ejercicio, tests) desde un transcript pegado o desde un archivo
argument-hint: <ruta-transcript | transcript pegado> [título oficial] [notas tuyas]
---
Entrada de Henry: $ARGUMENTS

## 1. Asegura que el transcript quede guardado en el repo
- **Si la entrada es una ruta** a un archivo existente en `transcripts/`, úsala.
- **Si la entrada es el texto del transcript pegado** (o el transcript viene en el mensaje):
  1. Identifica el curso y el número de clase a partir del título o del texto. Usa las carpetas `transcripts/n1-python`, `n2-ml`, `n3-dl`, `curso-1` … `curso-5` y compara el título con las tablas de clases en `cursos/**/README.md`.
  2. Si **no puedes deducir con seguridad** el curso o el número de clase, **pregunta antes de guardar** (una sola pregunta corta).
  3. Si el archivo destino ya existe, pregunta si sobrescribir.
  4. Guarda el texto tal cual en `transcripts/<curso>/clase-XX.txt`, con el título oficial en la primera línea (`# <título>`).
- Lo que Henry agregue fuera del transcript ("el profe enfatizó…", "no entendí…") es **contexto prioritario** para las notas y el ejercicio.

## 2. Genera la clase
Usa el subagente **generador-ejercicios** en *Modo 1* con la ruta del transcript, el título y las notas de Henry.
Si Henry indicó que es una clase de dudas o resultados ("solo notas"), genera solo `notas.md`.

## 3. Al terminar, muéstrame
1. Dónde se guardó el transcript y la carpeta de la clase creada.
2. Un resumen de 5 líneas de la clase.
3. Qué funciones tengo que implementar y el comando `make evaluar CLASE=...` exacto.
