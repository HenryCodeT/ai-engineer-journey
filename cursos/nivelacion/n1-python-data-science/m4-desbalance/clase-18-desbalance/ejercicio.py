"""
Clase 18 · Desbalance de datos: el problema (frontera de decisión y por qué accuracy engaña)
==========================================================================================

Implementa cada función reemplazando `raise NotImplementedError`.
Evalúa con:

    python -m framework evaluar cursos/nivelacion/n1-python-data-science/m4-desbalance/clase-18-desbalance

(o solo la nota: python -m pytest cursos/nivelacion/n1-python-data-science/m4-desbalance/clase-18-desbalance)

Reglas:
- No modifiques las entradas: trabaja sobre copias si hace falta.
- Solo necesitas pandas, numpy y scikit-learn.
- Datasets: los tests usan puntos SINTÉTICOS de 3 clases (como los 4 escenarios del
  profesor) y data/raw/clientes_banco.csv (target `abandono`, ~5 % de positivos).
- Esta clase es el PROBLEMA. Aquí NO se balancea nada (oversampling, undersampling,
  SMOTE… llegan en la clase 20). Solo se MIDE el desbalance y su efecto.

Hilo del ejercicio (sigue la clase):
  1. Medir cuánto desbalance hay: ratio mayoritaria / minoritaria.
  2. Generar un escenario de puntos con conteos exactos por clase (13 / 15 / 972…).
  3. El "modelo tonto": predecir siempre la clase mayoritaria (y ver su accuracy alto).
  4. Recall por clase: ¿cuántos de cada clase real acertó el modelo?
  5. Precision, recall y F1 de la clase que nos importa (la minoritaria).
  6. Balanced accuracy: el promedio de los recalls (no se deja engañar).
  7. Juntar todo: entrenar un SVC en un escenario y comparar accuracy vs. las otras métricas.

Ya hiciste en la clase 13: accuracy_manual, matriz_confusion_manual, proporciones_por_clase
y dividir_estratificado. No hace falta repetirlas aquí (puedes inspirarte en ellas).

Orden sugerido: 1 -> 7. La 6 puede usar la 4; la 7 usa la 2, la 4 y la 6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Centros FIJOS de las 3 nubes de puntos (clase 0, clase 1, clase 2).
# Están cerca a propósito: así las nubes se superponen y hay una zona de confusión
# en la frontera de decisión, como en el dibujo de triángulos y cuadrados del profesor.
CENTROS = [(-1.0, -0.5), (1.0, -0.5), (0.0, 1.0)]
DISPERSION = 0.8


# ---------------------------------------------------------------- 1
def ratio_desbalance(y) -> float:
    """Calcula el ratio de desbalance: (n.º de muestras de la clase MAYORITARIA) /
    (n.º de muestras de la clase MINORITARIA).

    - 1.0 significa perfectamente balanceado.
    - 18.6 significa "por cada caso de la clase rara hay 18.6 de la clase común".

    Args:
        y: etiquetas (lista, array de numpy o Series). Al menos una muestra.

    Devuelve:
        float >= 1.

    Ejemplo:
        >>> ratio_desbalance([0, 0, 0, 0, 0, 0, 1, 1])
        3.0
        >>> ratio_desbalance(["a", "b", "b", "c", "c", "c"])   # 3 / 1
        3.0

    💡 Pistas:
        - pd.Series(y).value_counts() cuenta cuántas muestras hay de cada clase.
        - .max() y .min() de esos conteos.
    """
    # 1. convierte y a Series
    # 2. cuenta las muestras de cada clase
    # 3. divide el conteo mayor entre el conteo menor
    # 4. devuelve el resultado como float
    raise NotImplementedError


# ---------------------------------------------------------------- 2
def generar_escenario(conteos: dict, random_state: int = 0) -> tuple:
    """Genera puntos 2D con EXACTAMENTE los conteos pedidos por clase, como los escenarios
    A, B, C y D del profesor (p. ej. {0: 13, 1: 15, 2: 972}).

    Usa sklearn.datasets.make_blobs, con estos argumentos (son obligatorios para que
    el resultado coincida con el de los tests):

        make_blobs(n_samples=[lista de conteos en el orden de las clases ordenadas],
                   centers=CENTROS[:k],          # k = número de clases
                   cluster_std=DISPERSION,
                   random_state=random_state)

    Cuando n_samples es una LISTA, make_blobs genera exactamente esa cantidad de puntos
    por nube y devuelve etiquetas 0, 1, 2… según la posición en la lista. Si las claves
    de `conteos` no son 0, 1, 2…, cambia cada etiqueta i por la i-ésima clase ordenada.

    Args:
        conteos: dict {clase: n.º de muestras}. Máximo 3 clases (hay 3 centros).
        random_state: semilla (mismo valor -> mismos puntos).

    Devuelve:
        tupla (X, y):
          - X: DataFrame con 2 columnas "x1" y "x2" (una fila por punto).
          - y: Series llamada "clase" con la etiqueta de cada punto.
          Ambos con índice 0..n-1.

    Errores:
        ValueError si `conteos` tiene más de 3 clases.

    Ejemplo:
        >>> X, y = generar_escenario({0: 13, 1: 15, 2: 972})
        >>> X.shape
        (1000, 2)
        >>> y.value_counts().sort_index().tolist()
        [13, 15, 972]

    💡 Pistas:
        - clases = sorted(conteos)  -> [0, 1, 2]
        - n_samples = [conteos[c] for c in clases]
        - make_blobs devuelve (X_array, y_array): conviértelos con pd.DataFrame(...)
          y pd.Series(..., name="clase").
        - Para cambiar etiquetas: np.array(clases)[y_array] (indexación con un array).
    """
    # 1. ordena las clases y valida que sean como máximo 3
    # 2. arma la lista de conteos en ese orden
    # 3. llama a make_blobs con los argumentos indicados
    # 4. traduce las etiquetas 0..k-1 a las clases originales
    # 5. devuelve (DataFrame con x1, x2 ; Series "clase")
    raise NotImplementedError


# ---------------------------------------------------------------- 3
def prediccion_mayoritaria(y_train, n: int) -> np.ndarray:
    """El "modelo tonto" (baseline): ignora las variables y SIEMPRE predice la clase más
    frecuente de y_train.

    Sirve para ver por qué accuracy engaña: en clientes_banco.csv este modelo tiene
    accuracy ~0.95 ¡sin haber aprendido nada! Cualquier modelo real debe superarlo en
    métricas que miren a la clase minoritaria.

    Si hay empate entre clases, usa la de etiqueta MENOR.

    Args:
        y_train: etiquetas de entrenamiento (lista, array o Series).
        n: cuántas predicciones devolver (normalmente len(y_test)).

    Devuelve:
        np.ndarray de longitud n, con la clase mayoritaria repetida.

    Ejemplo:
        >>> prediccion_mayoritaria([0, 0, 0, 1], 3)
        array([0, 0, 0])

    💡 Pistas:
        - pd.Series(y_train).value_counts() ordena de mayor a menor (en empate, no
          garantiza el orden): usa .mode(), que devuelve las modas ORDENADAS, y toma la 1.ª.
        - np.full(n, valor) crea un array de n copias de valor.
    """
    # 1. encuentra la clase más frecuente (en empate, la menor)
    # 2. crea un array de n posiciones con esa clase
    raise NotImplementedError


# ---------------------------------------------------------------- 4
def recall_por_clase(y_true, y_pred) -> pd.Series:
    """Para CADA clase real, qué proporción de sus muestras acertó el modelo.

        recall(c) = (muestras de clase c predichas como c) / (muestras REALES de clase c)

    Es la pregunta "de los que eran c, ¿a cuántos encontré?". En el escenario A del
    profesor, el recall de la clase 2 es ~1.0 y el de las clases 0 y 1 es 0.0.

    Se compara POSICIÓN por posición (como en accuracy_manual de la clase 13).

    Args:
        y_true: etiquetas reales (lista, array o Series).
        y_pred: etiquetas predichas (misma longitud).

    Devuelve:
        pd.Series con índice = clases que aparecen en y_true (orden ascendente) y
        valores float entre 0 y 1.

    Ejemplo:
        >>> recall_por_clase([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
        0    0.500000
        1    0.666667
        dtype: float64

    💡 Pistas:
        - Convierte a arrays con np.asarray para olvidar los índices.
        - Para cada clase c de np.unique(y_true): máscara = (real == c);
          recall = (pred[máscara] == c).mean().
    """
    # 1. convierte y_true / y_pred a arrays
    # 2. recorre las clases reales (ordenadas)
    # 3. para cada una, calcula aciertos / total de esa clase real
    # 4. devuelve una Series con índice = clases
    raise NotImplementedError


# ---------------------------------------------------------------- 5
def precision_recall_f1(y_true, y_pred, clase_positiva=1) -> dict:
    """Calcula precision, recall y F1 de UNA clase (la "positiva", normalmente la
    minoritaria: el cliente que abandona, el moroso…), SIN scikit-learn.

    Con la clase positiva definida:
        TP = real positivo  y predicho positivo    (acierto)
        FP = real NEGATIVO  y predicho positivo    (falsa alarma)
        FN = real positivo  y predicho NEGATIVO    (se nos escapó)

        precision = TP / (TP + FP)   "de los que marqué como positivos, ¿cuántos lo eran?"
        recall    = TP / (TP + FN)   "de los positivos reales, ¿cuántos encontré?"
        F1        = 2 * precision * recall / (precision + recall)   (media armónica)

    Si un denominador es 0, esa métrica vale 0.0 (igual que sklearn con zero_division=0).

    Args:
        y_true: etiquetas reales.
        y_pred: etiquetas predichas (misma longitud).
        clase_positiva: qué etiqueta se considera "positiva".

    Devuelve:
        dict {"precision": float, "recall": float, "f1": float}

    Ejemplo:
        >>> precision_recall_f1([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0])
        {'precision': 0.6666..., 'recall': 0.6666..., 'f1': 0.6666...}
        (TP=2, FP=1, FN=1)

    💡 Pistas:
        - real_pos = (np.asarray(y_true) == clase_positiva); igual con pred_pos.
        - TP = (real_pos & pred_pos).sum(); FP = (~real_pos & pred_pos).sum(); FN = ...
        - Antes de dividir, revisa si el denominador es 0.
    """
    # 1. crea las máscaras booleanas real_pos y pred_pos
    # 2. cuenta TP, FP y FN
    # 3. precision (0.0 si TP + FP == 0)
    # 4. recall (0.0 si TP + FN == 0)
    # 5. f1 (0.0 si precision + recall == 0)
    # 6. devuelve el dict con floats
    raise NotImplementedError


# ---------------------------------------------------------------- 6
def balanced_accuracy_manual(y_true, y_pred) -> float:
    """Balanced accuracy: el PROMEDIO del recall de cada clase real.

    A diferencia del accuracy, cada clase pesa lo mismo, tenga 13 o 972 muestras.
    Por eso el modelo tonto saca accuracy 0.97 en el escenario A pero balanced
    accuracy 0.33 (= (0 + 0 + 1) / 3).

    Debe coincidir con sklearn.metrics.balanced_accuracy_score(y_true, y_pred).

    Args:
        y_true: etiquetas reales.
        y_pred: etiquetas predichas.

    Devuelve:
        float entre 0 y 1.

    Ejemplo:
        >>> balanced_accuracy_manual([0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0])
        0.5
        (recall clase 0 = 1.0, recall clase 1 = 0.0 -> promedio 0.5;
         el accuracy sería 4/6 = 0.67)

    💡 Pistas:
        - Reutiliza recall_por_clase y calcula su .mean().
    """
    # 1. calcula el recall de cada clase
    # 2. devuelve el promedio como float
    raise NotImplementedError


# ---------------------------------------------------------------- 7
def evaluar_escenario(conteos: dict, random_state: int = 0) -> dict:
    """Reproduce el experimento de la clase: genera un escenario, entrena un clasificador
    de vectores de soporte (SVC) y compara métricas.

    Pasos (todos obligatorios, para que el resultado coincida con los tests):
      1. X, y = generar_escenario(conteos, random_state)
      2. Divide con train_test_split(X, y, test_size=0.3, stratify=y,
                                     random_state=random_state)
         (stratify para que la clase rara aparezca en train Y en test).
      3. Entrena SVC(kernel="linear") SOLO con train.
      4. Predice sobre test.
      5. Calcula sobre test:
         - "accuracy": aciertos / total (puedes usar sklearn.metrics.accuracy_score)
         - "balanced_accuracy": con tu balanced_accuracy_manual
         - "recall_minoritaria": recall de la clase con MENOS muestras en `conteos`
           (en empate, la de etiqueta menor). Usa tu recall_por_clase.

    Args:
        conteos: dict {clase: n.º de muestras}, como en generar_escenario.
        random_state: semilla para generar los datos y para dividir.

    Devuelve:
        dict {"accuracy": float, "balanced_accuracy": float, "recall_minoritaria": float}

    Ejemplo:
        >>> evaluar_escenario({0: 13, 1: 15, 2: 972})        # escenario A
        {'accuracy': 0.973..., 'balanced_accuracy': 0.333..., 'recall_minoritaria': 0.0}
        >>> evaluar_escenario({0: 336, 1: 334, 2: 330})      # escenario D
        {'accuracy': 0.83..., 'balanced_accuracy': 0.83..., 'recall_minoritaria': 0.71...}

    💡 Pistas:
        - from sklearn.model_selection import train_test_split
        - from sklearn.svm import SVC
        - Clase minoritaria: min(conteos, key=lambda c: (conteos[c], c))
    """
    # 1. genera el escenario
    # 2. divide estratificado (70 % train / 30 % test)
    # 3. entrena SVC(kernel="linear") con train
    # 4. predice test
    # 5. calcula accuracy, balanced accuracy y recall de la minoritaria
    # 6. devuelve el dict con floats
    raise NotImplementedError
