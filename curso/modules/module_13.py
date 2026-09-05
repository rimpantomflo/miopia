from curso.notebook_factory import code, common_setup, md

TITLE = "13 · Corpora públicos y benchmark externo"


def build() -> list[dict]:
    return [
        md(
            """
            # 13 · Corpora públicos y benchmark externo

            Los datos sintéticos prueban código; no prueban validez clínica. En
            este módulo se diseña una evaluación sobre corpus públicos con acceso,
            licencia, esquema y procedencia respetados. El repositorio nunca los
            descarga ni redistribuye automáticamente.
            """
        ),
        md(
            """
            ## Objetivos

            - elegir corpus por tarea y no por fama;
            - documentar acceso y licencia;
            - adaptar BRAT/TSV sin perder offsets;
            - armonizar etiquetas sin borrar diferencias;
            - bloquear un benchmark externo y evitar contaminación;
            - distinguir portabilidad lingüística, institucional y temporal.
            """
        ),
        common_setup(),
        code(
            """
            from dataclasses import asdict

            import pandas as pd

            from clinical_nlp_course import PUBLIC_CORPORA, load_brat

            catalog = pd.DataFrame(
                [asdict(descriptor) for descriptor in PUBLIC_CORPORA.values()]
            )
            display(catalog[["name", "tasks", "access", "url"]])
            assert set(catalog["name"]) == {
                "CARMEN-I", "SympTEMIST", "MedProcNER", "CodiEsp"
            }
            """
        ),
        md(
            """
            ## 1. Selección por pregunta

            | Pregunta | Recurso inicial | Qué no demuestra |
            |---|---|---|
            | NER clínico ES/CA | CARMEN-I | rendimiento en tu hospital |
            | síntomas + SNOMED CT | SympTEMIST | normalización de procedimientos |
            | procedimientos | MedProcNER | fenotipo longitudinal |
            | codificación CIE-10 | CodiEsp | decisión clínica prospectiva |

            CARMEN-I es de acceso controlado: credenciales, formación y acuerdo de
            uso no son un obstáculo técnico que deba eludirse.
            """
        ),
        code(
            """
            benchmark_plan = pd.DataFrame([
                {
                    "stage": "internal_development",
                    "data": "hospital A, periodo temprano",
                    "allowed_use": "entrenar y elegir",
                },
                {
                    "stage": "internal_temporal_test",
                    "data": "hospital A, periodo posterior",
                    "allowed_use": "una evaluación bloqueada",
                },
                {
                    "stage": "public_external",
                    "data": "corpus público compatible",
                    "allowed_use": "portabilidad; no reajustar",
                },
                {
                    "stage": "external_site",
                    "data": "hospital B con autorización",
                    "allowed_use": "validación externa real",
                },
            ])
            benchmark_plan
            """
        ),
        md(
            """
            ## 2. Adaptador BRAT validado

            `load_brat(path)` lee cada par `.txt/.ann`, rechaza offsets que no
            recuperan exactamente la evidencia y falla de forma explícita ante
            spans discontinuos. No convierte silenciosamente un formato que no
            entiende.
            """
        ),
        code(
            """
            PUBLIC_DATA_ROOT = None  # Path("data/restricted/carmen-i/...")

            if PUBLIC_DATA_ROOT is not None:
                public_records = load_brat(PUBLIC_DATA_ROOT)
                print(len(public_records), "documentos autorizados")
            else:
                print("Ruta pública desactivada: consulta docs/public_corpora.md")
            """
        ),
        md(
            """
            ## 3. Tabla de armonización

            Nunca reasignes etiquetas solo porque sus nombres se parecen. Cada
            mapeo requiere definición, ejemplos y revisión clínica.
            """
        ),
        code(
            """
            harmonization = pd.DataFrame([
                {
                    "source": "SympTEMIST",
                    "source_label": "SYMPTOM",
                    "target_label": "FINDING",
                    "decision": "manual_review_required",
                    "reason": "el alcance ontológico puede diferir",
                },
                {
                    "source": "MedProcNER",
                    "source_label": "PROCEDIMIENTO",
                    "target_label": "PROCEDURE",
                    "decision": "candidate_mapping",
                    "reason": "validar inclusiones y exclusiones",
                },
            ])
            harmonization
            """
        ),
        md(
            """
            ## 4. Protocolo bloqueado

            Registra antes de abrir el test:

            - versión y checksum del corpus;
            - criterios de inclusión;
            - unidad de partición;
            - label map y postprocesado;
            - métrica primaria y umbral;
            - subgrupos y análisis de errores;
            - modelos/revisiones exactos;
            - política ante documentos incompatibles;
            - número de comparaciones.
            """
        ),
        code(
            """
            protocol = {
                "protocol_version": "external-benchmark-1.0",
                "primary_metric": "exact_span_micro_f1",
                "secondary_metrics": ["overlap_f1_iou_0.5", "recall_by_label"],
                "unit_of_resampling": "document_or_patient_as_available",
                "test_reuse": "forbidden_for_model_selection",
                "subgroups": ["language", "document_type", "entity_label"],
                "minimum_error_review": 100,
            }
            assert protocol["test_reuse"].startswith("forbidden")
            protocol
            """
        ),
        md(
            """
            ## Práctica obligatoria

            Elige una tarea pública. Entrega ficha de datos, prueba de offsets,
            label map versionado, baseline, intervalo de confianza y 50 errores
            categorizados. Separa adaptación de formato de cualquier adaptación
            del modelo.

            **Criterio de salida:** puedes demostrar qué datos usaste, bajo qué
            condiciones y por qué el benchmark responde —o no— a tu uso previsto.
            """
        ),
    ]
