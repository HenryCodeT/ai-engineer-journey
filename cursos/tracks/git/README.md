# 🌿 Track Git y terminal
Es requisito del programa, y en el Curso 1 (M2) trabajas con branches, pull requests y code review. **Practícalo en este mismo repo.**
Cada reto se marca ✅ cuando el comando de verificación da lo esperado. El agente `evaluador` puede revisarlo si le pides "revisa mi track git".

## Nivel 1 · Terminal
- [ ] Moverte por carpetas sin el explorador: `pwd`, `ls -la`, `cd`, `mkdir -p`, `mv`, `cp`
- [ ] Buscar sin abrir archivos: `grep -rn "NotImplementedError" cursos/` (¿cuántas funciones te faltan?)
- [ ] Ver el final de un archivo y contar líneas: `tail -n 20`, `wc -l`
- [ ] Usar variables de entorno: `export CLASE=...` y luego `make evaluar CLASE=$CLASE`

## Nivel 2 · Git local
- [ ] `git init`, primer commit, `.gitignore` entendido (¿por qué no se versiona `data/raw/`?)
- [ ] Un commit por clase aprobada, con mensajes claros: `clase 01: EDA (nota 85)`
  ✔️ Verificación: `git log --oneline | head`
- [ ] Ver qué cambiaste antes de commitear: `git status`, `git diff`, `git diff --staged`
- [ ] Deshacer sin miedo: `git restore <archivo>`, `git restore --staged`, `git commit --amend`

## Nivel 3 · Branches y PRs (como en el trabajo)
- [ ] Crear una rama por clase o feature: `git switch -c clase-02-faltantes`
- [ ] Subir a GitHub (repo privado) y abrir un **Pull Request** hacia `main`
  ✔️ Verificación: `gh pr list --state all`
- [ ] Pedir code review del PR al agente `revisor-codigo` y atender sus comentarios con nuevos commits
- [ ] Resolver un conflicto de merge a propósito: edita la misma línea en dos ramas y mézclalas
- [ ] `git stash` para guardar trabajo a medias

## Nivel 4 · Automatización (prepara el Curso 4)
- [ ] Un workflow de GitHub Actions que corra `make test-framework` en cada push
- [ ] Badge de CI en el README
