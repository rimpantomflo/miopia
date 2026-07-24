# Entorno para transformers y modelos avanzados

Los módulos 00–03 no necesitan descargar modelos grandes. El módulo 04 incluye
celdas protegidas por una variable de activación para que el curso siga siendo
ejecutable en un portátil sin GPU ni acceso a internet.

## Antes de instalar

Comprueba para cada modelo:

- licencia y condiciones de uso;
- idiomas, corpus de preentrenamiento y tarea de ajuste;
- arquitectura necesaria y compatibilidad con la versión de `transformers`;
- memoria, tiempo y huella de carbono;
- autorización institucional si habrá texto clínico real;
- posibilidad de ejecución totalmente local.

Un modelo biomédico preentrenado no es automáticamente un NER, clasificador ni
extractor listo para producción. La cabeza de tarea y la validación local siguen
siendo necesarias.

## Entorno separado

Al llegar al módulo 04, crea una rama o copia del entorno y registra las
versiones resueltas:

```powershell
uv add --optional models transformers datasets evaluate seqeval accelerate torch
uv sync --extra models
```

Para integración Oracle opcional:

```powershell
uv add --optional oracle oracledb
uv sync --extra oracle
```

Estos comandos modifican `pyproject.toml` y `uv.lock`; ejecútalos cuando vayas a
realizar la práctica pesada, no por adelantado. Conserva el lockfile del
experimento.

## Ejecución segura

1. Prueba primero con las frases ficticias del repositorio.
2. Activa una sola celda pesada y controla memoria.
3. Guarda identificador exacto, revisión del modelo, tokenizer y parámetros.
4. Comprueba offsets tras tokenización y truncamiento.
5. Evalúa contra el mismo test bloqueado que el baseline.
6. No subas pesos, cachés ni datos clínicos al repositorio.

En datos reales, una API externa solo es aceptable si existe aprobación,
contrato, minimización, control de retención y evaluación de seguridad. La
opción predeterminada del curso es ejecución local en infraestructura
autorizada.
