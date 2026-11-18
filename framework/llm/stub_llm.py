"""LLM de prueba que devuelve respuestas predefinidas (o lanza errores). Útil para tests."""
from .fake_llm import LLMError, Respuesta, contar_tokens


class StubLLM:
    def __init__(self, salidas, modelo="stub"):
        self.salidas = list(salidas)
        self.modelo = modelo
        self.prompts = []

    def generar(self, prompt, temperature=0.7, max_tokens=300):
        self.prompts.append(prompt)
        s = self.salidas.pop(0) if len(self.salidas) > 1 else self.salidas[0]
        if isinstance(s, Exception):
            raise s
        return Respuesta(s, self.modelo, contar_tokens(prompt), contar_tokens(s), 100.0, 0.0001)


def error():
    return LLMError("fallo simulado")
