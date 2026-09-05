# Contribuir

1. Crea una rama desde `master`.
2. Instala con `uv sync --group dev`.
3. Modifica las fuentes de notebooks en `curso/modules/` o
   `tutorial/build_tutorial.py`, no solo el `.ipynb` generado.
4. Ejecuta `uv run ruff check .`, `uv run pytest`, los dos regeneradores y los
   dos validadores.
5. No incluyas datos clínicos reales, secretos ni artefactos de modelos.

Cada cambio metodológico debe añadir una prueba o un caso centinela, describir
el uso previsto y señalar al menos una limitación clínica.
