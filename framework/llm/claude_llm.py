"""
claude_llm.py — Adaptador a Claude REAL con la MISMA interfaz que FakeLLM.

    pip install anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."      (Windows: set ANTHROPIC_API_KEY=...)
    export LLM=claude                           -> los demos usan Claude

Lección de AI Engineering: tu código depende de la interfaz
`generar(prompt, temperature, max_tokens) -> Respuesta`, no del proveedor.
Cambiar de FakeLLM a Claude (o a otro proveedor) es cambiar UNA línea.

PRECIOS: verifica los precios vigentes en https://docs.claude.com (sección
pricing) y ajústalos con las variables de entorno PRECIO_IN / PRECIO_OUT
(USD por 1M tokens). Los valores por defecto son solo de referencia.
"""
from __future__ import annotations

import os
import time

from .fake_llm import FakeLLM, LLMError, Respuesta

MODELO_BARATO = os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")


class ClaudeLLM:
    def __init__(self, modelo: str = MODELO_BARATO):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("Instala el SDK: pip install anthropic") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Falta la variable de entorno ANTHROPIC_API_KEY")
        self._anthropic = anthropic
        self.cliente = anthropic.Anthropic()
        self.modelo = modelo
        self.precio_in = float(os.environ.get("PRECIO_IN", "1.0"))
        self.precio_out = float(os.environ.get("PRECIO_OUT", "5.0"))
        self.llamadas = 0

    def generar(self, prompt: str, temperature: float = 0.7, max_tokens: int = 300) -> Respuesta:
        self.llamadas += 1
        inicio = time.perf_counter()
        try:
            r = self.cliente.messages.create(
                model=self.modelo,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        except self._anthropic.APIError as e:
            # Traducimos el error del proveedor a NUESTRO tipo de error:
            # así reintentos/fallback (ej13) funcionan igual con cualquier proveedor.
            raise LLMError(str(e)) from e
        latencia = (time.perf_counter() - inicio) * 1000
        texto = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        t_in, t_out = r.usage.input_tokens, r.usage.output_tokens
        costo = (t_in * self.precio_in + t_out * self.precio_out) / 1_000_000
        return Respuesta(texto, self.modelo, t_in, t_out, round(latencia, 1), costo)


def crear_llm(modelo_fake: str = "mini", seed: int | None = None):
    """Devuelve ClaudeLLM si LLM=claude, si no un FakeLLM (gratis y reproducible)."""
    if os.environ.get("LLM", "").lower() == "claude":
        return ClaudeLLM()
    return FakeLLM(modelo_fake, seed=seed)


if __name__ == "__main__":
    llm = ClaudeLLM()
    r = llm.generar("En una frase: ¿qué hace un AI Engineer?", temperature=0)
    print(r.texto, f"\n{r.tokens_entrada}+{r.tokens_salida} tokens | {r.latencia_ms} ms | ${r.costo_usd:.6f}")
