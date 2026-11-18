"""
fake_llm.py — Un LLM SIMULADO para aprender sin API keys ni costos.

¿Por qué existe?
    Para practicar AI Engineering no necesitas gastar dinero. Este "modelo"
    imita los comportamientos que importan al diseñar sistemas reales:

    * Es PROBABILÍSTICO: con temperature > 0 puede dar respuestas distintas.
    * Cobra por TOKENS de entrada y salida (precios ficticios por 1M tokens).
    * Tiene LATENCIA (simulada, no te hace esperar).
    * A veces FALLA (tasa_fallo) -> útil para practicar reintentos/fallback.
    * ALUCINA si no le das contexto (inventa números).
    * Devuelve JSON MAL FORMADO a veces cuando la temperature es alta.
    * Tiene VENTANA DE CONTEXTO limitada.

Cuando quieras usar un LLM real (OpenAI, Anthropic, Ollama...), solo necesitas
una clase con el mismo método `generar(prompt, temperature, max_tokens)` que
devuelva un objeto `Respuesta`. Tu código de los ejercicios no cambia.
Eso es una lección de AI Engineering en sí misma: depende de interfaces, no
de proveedores.
"""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass

# Precios FICTICIOS en USD por 1 millón de tokens. Calidad de 0 a 1.
CATALOGO: dict[str, dict] = {
    "nano": dict(precio_in=0.10, precio_out=0.40, latencia_ms=300, calidad=0.62, local=False, ventana=16_000),
    "mini": dict(precio_in=0.40, precio_out=1.60, latencia_ms=600, calidad=0.78, local=False, ventana=128_000),
    "grande": dict(precio_in=3.00, precio_out=15.00, latencia_ms=1800, calidad=0.92, local=False, ventana=200_000),
    "local-8b": dict(precio_in=0.0, precio_out=0.0, latencia_ms=900, calidad=0.70, local=True, ventana=8_000),
}

MS_POR_TOKEN_SALIDA = 2  # cada token generado suma ~2 ms


def contar_tokens(texto: str) -> int:
    """Aproximación clásica: ~4 caracteres por token."""
    if not texto:
        return 0
    return max(1, math.ceil(len(texto) / 4))


class LLMError(Exception):
    """Error del proveedor (timeout, rate limit, contexto excedido...)."""


@dataclass
class Respuesta:
    texto: str
    modelo: str
    tokens_entrada: int
    tokens_salida: int
    latencia_ms: float
    costo_usd: float


_POSITIVAS = {"excelente", "bueno", "buena", "encanta", "genial", "feliz", "rapido", "rápido", "recomiendo", "perfecto"}
_NEGATIVAS = {"malo", "mala", "terrible", "odio", "pesimo", "pésimo", "lento", "roto", "horrible", "nunca"}


class FakeLLM:
    def __init__(self, modelo: str = "mini", seed: int | None = None, tasa_fallo: float = 0.0):
        if modelo not in CATALOGO:
            raise ValueError(f"Modelo desconocido: {modelo}. Opciones: {list(CATALOGO)}")
        self.modelo = modelo
        self.info = CATALOGO[modelo]
        self.rng = random.Random(seed)
        self.tasa_fallo = tasa_fallo
        self.llamadas = 0

    # ------------------------------------------------------------------ API
    def generar(self, prompt: str, temperature: float = 0.7, max_tokens: int = 300) -> Respuesta:
        self.llamadas += 1
        if self.rng.random() < self.tasa_fallo:
            raise LLMError(f"[{self.modelo}] timeout simulado del proveedor")
        t_in = contar_tokens(prompt)
        if t_in > self.info["ventana"]:
            raise LLMError(f"[{self.modelo}] el prompt ({t_in} tokens) excede la ventana de {self.info['ventana']}")

        texto = self._responder(prompt, temperature)
        texto = texto[: max_tokens * 4]
        t_out = contar_tokens(texto)
        latencia = self.info["latencia_ms"] * (1 + 0.3 * self.rng.random()) + t_out * MS_POR_TOKEN_SALIDA
        costo = (t_in * self.info["precio_in"] + t_out * self.info["precio_out"]) / 1_000_000
        return Respuesta(texto, self.modelo, t_in, t_out, round(latencia, 1), costo)

    # -------------------------------------------------------- comportamiento
    def _elegir(self, opciones: list[str], temperature: float) -> str:
        if temperature <= 0:
            return opciones[0]
        return self.rng.choice(opciones)

    def _responder(self, prompt: str, temperature: float) -> str:
        if "HERRAMIENTAS:" in prompt:
            return self._tool_call(prompt, temperature)
        if "CONTEXTO:" in prompt:
            return self._responder_con_contexto(prompt)
        if "Clasifica el sentimiento" in prompt:
            return self._sentimiento(prompt, temperature)
        return self._generico(prompt, temperature)

    def _sentimiento(self, prompt: str, temperature: float) -> str:
        texto = prompt.split(":", 1)[-1].lower()
        palabras = set(re.findall(r"\w+", texto))
        pos, neg = len(palabras & _POSITIVAS), len(palabras & _NEGATIVAS)
        etiqueta = "positivo" if pos > neg else "negativo" if neg > pos else "neutral"
        # Ruido: con temperatura alta a veces se "equivoca"
        if temperature > 0 and self.rng.random() < temperature * 0.2:
            etiqueta = self.rng.choice(["positivo", "negativo", "neutral"])
        formatos = [etiqueta, etiqueta.capitalize() + ".", f"El sentimiento es {etiqueta}",
                    f"Sentimiento: {etiqueta.upper()}", f"Diría que es {etiqueta} 🙂"]
        return self._elegir(formatos, temperature)

    def _responder_con_contexto(self, prompt: str) -> str:
        contexto = prompt.split("CONTEXTO:", 1)[1].split("PREGUNTA:", 1)[0]
        pregunta = prompt.split("PREGUNTA:", 1)[1] if "PREGUNTA:" in prompt else ""
        claves = {w for w in re.findall(r"\w+", pregunta.lower()) if len(w) > 3}
        oraciones = [o.strip(" -\n") for o in re.split(r"[.\n]", contexto) if o.strip(" -\n")]
        mejor, puntaje = None, 0
        for o in oraciones:
            p = len(claves & set(re.findall(r"\w+", o.lower())))
            if p > puntaje:
                mejor, puntaje = o, p
        if not mejor:
            return "No encuentro esa información en el contexto proporcionado."
        return f"Según el contexto: {mejor}."

    def _generico(self, prompt: str, temperature: float) -> str:
        p = prompt.lower()
        if any(w in p for w in ("cuánto", "cuanto", "cuántos", "cuantos", "precio", "plazo")):
            # ALUCINACIÓN: inventa un número con total seguridad
            n = self.rng.randint(2, 90) if temperature > 0 else 30
            return self._elegir([f"La respuesta es {n}.", f"Son exactamente {n}, sin duda.",
                                 f"Según mis datos, {n}."], temperature)
        tema = prompt.strip().split("\n")[-1][:80]
        return self._elegir([f"Aquí tienes una respuesta sobre: {tema}",
                             f"Claro. Respecto a '{tema}', te explico brevemente.",
                             f"Buena pregunta. Sobre {tema}: depende del contexto."], temperature)

    def _tool_call(self, prompt: str, temperature: float) -> str:
        mensaje = prompt.split("USUARIO:", 1)[-1].strip()
        m_low = mensaje.lower()
        expr = re.search(r"[\d\s\.\+\-\*/\(\)]{3,}", mensaje)
        if "clima" in m_low:
            ciudad = re.search(r"\ben\s+([A-Za-zÁÉÍÓÚáéíóúñÑ]+)", mensaje)
            llamada = {"tool": "clima", "args": {"ciudad": ciudad.group(1) if ciudad else "Lima"}}
        elif expr and re.search(r"\d", expr.group()) and re.search(r"[\+\-\*/]", expr.group()):
            llamada = {"tool": "calculadora", "args": {"expresion": expr.group().strip()}}
        elif "borra" in m_low or "elimina" in m_low:
            llamada = {"tool": "borrar_base_de_datos", "args": {}}  # tool NO permitida
        else:
            llamada = {"tool": None, "respuesta": "Hola, ¿en qué te ayudo?"}
        limpio = json.dumps(llamada, ensure_ascii=False)
        if temperature >= 0.8 and self.rng.random() < 0.5:
            return self.rng.choice([
                f"Claro, aquí está la llamada: {limpio}\n¡Espero que ayude!",
                f"```json\n{limpio}\n```",
                limpio[:-2],  # JSON roto
            ])
        return limpio


if __name__ == "__main__":
    llm = FakeLLM("mini", seed=1)
    for t in (0.0, 1.0):
        print(f"temperature={t}:", [llm.generar("Clasifica el sentimiento: me encanta, excelente", t).texto for _ in range(4)])
    r = llm.generar("¿Cuántos días tengo para devolver un producto?")
    print("Sin contexto (alucina):", r.texto, f"| costo ${r.costo_usd:.6f} | {r.latencia_ms} ms")
