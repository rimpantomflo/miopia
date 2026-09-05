# Datos del curso

Todos los archivos versionados en este directorio son ficticios.

- `nefrologia_sintetica.jsonl`: 30 notas para reglas y fenotipo longitudinal.
- `conceptos_renales_sinteticos.json`: terminología docente.
- `doccano_nefrologia_sintetica.jsonl`: lote de anotación asistida.
- `renal_classification_synthetic.jsonl`: 320 notas, 80 pacientes y cuatro
  clases para el baseline ML de la Hero Track.

El último archivo se regenera de forma determinista con:

```bash
uv run python scripts/generate_advanced_synthetic_data.py
```

Los corpora públicos o hospitalarios deben vivir fuera del repositorio. Las
rutas `data/restricted/`, `data/private/` y `data/raw/` están ignoradas como
segunda barrera, no como control de acceso suficiente.
