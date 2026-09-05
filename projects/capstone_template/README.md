# Plantilla de capstone hospitalario

Copia esta carpeta y reemplaza `PROJECT_NAME`. No incluyas datos reales ni
secretos en la copia versionada.

## Estructura recomendada

```text
PROJECT_NAME/
  README.md                 uso previsto, alcance y responsables
  configs/                  configuración sin secretos
  docs/
    data_card.md
    annotation_protocol.md
    model_card.md
    validation_protocol.md
    risk_register.md
    deployment_runbook.md
  src/                      pipeline instalable
  tests/                    unitarios, contrato, regresión y seguridad
  scripts/                  entrenamiento y lotes idempotentes
  synthetic_data/           casos ficticios versionables
  artifacts/                ignorado; modelos, métricas y manifiestos
```

## Pull request 1 — Contrato y referencia

- [ ] pregunta, unidad, población y estimando;
- [ ] usos excluidos y revisión humana;
- [ ] guía de anotación y adjudicación;
- [ ] dataset sintético con casos límite;
- [ ] partición por paciente/tiempo;
- [ ] tests de offsets y fuga.

## Pull request 2 — Baselines

- [ ] reglas/diccionario;
- [ ] TF-IDF + modelo lineal;
- [ ] métricas en development;
- [ ] errores categorizados;
- [ ] manifiesto de cada ejecución.

## Pull request 3 — Candidato avanzado

- [ ] justificación de complejidad;
- [ ] mismo presupuesto y protocolo;
- [ ] tres semillas;
- [ ] test temporal bloqueado;
- [ ] comparación emparejada e intervalos;
- [ ] model card.

## Pull request 4 — Shadow mode

- [ ] contrato de entrada/salida;
- [ ] idempotencia por contenido y versión;
- [ ] logs por lista positiva;
- [ ] monitorización y cola de revisión;
- [ ] plan de incidente y rollback probado;
- [ ] aprobaciones locales documentadas fuera del repo público.

## Definition of Done

`validate_capstone_evidence()` debe devolver una lista vacía, CI debe pasar y el
equipo clínico debe poder reconstruir una predicción desde su evidencia. Aun así,
“terminado” solo significa listo para la fase que haya sido autorizada.

Copia `submission.example.json`, rellénalo y ejecuta:

```bash
uv run miopia-course-check check-capstone projects/MI_PROYECTO/submission.json
```

La plantilla vacía falla deliberadamente: el validador no certifica un capstone
sin entregables ni salvaguardas declaradas.
