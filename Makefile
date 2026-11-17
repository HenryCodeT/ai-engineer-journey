PY ?= python
CLASE ?=

.PHONY: help datos evaluar evaluar-v progreso test-framework notebook

help:
	@echo "make datos                 -> genera los datasets de práctica"
	@echo "make evaluar CLASE=<ruta>  -> nota automática de una clase"
	@echo "make evaluar-v CLASE=<ruta>-> igual, con detalle de errores"
	@echo "make progreso              -> tabla de avance"
	@echo "make notebook              -> abre JupyterLab"

datos:
	$(PY) data/generar_datos.py

evaluar:
	@test -n "$(CLASE)" || (echo "Uso: make evaluar CLASE=cursos/..../clase-XX-tema" && exit 1)
	$(PY) -m framework evaluar $(CLASE)

evaluar-v:
	@test -n "$(CLASE)" || (echo "Uso: make evaluar-v CLASE=..." && exit 1)
	$(PY) -m framework evaluar $(CLASE) -v

progreso:
	$(PY) -m framework progreso

test-framework:
	$(PY) -m pytest tests -q

notebook:
	$(PY) -m jupyter lab
