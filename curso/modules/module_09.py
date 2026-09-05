from curso.notebook_factory import code, common_setup, md

TITLE = "09 · Producción, MLOps y monitorización clínica"


def build() -> list[dict]:
    return [
        md(
            """
            # 09 · Producción, MLOps y monitorización clínica

            Un modelo validado offline no es todavía un servicio clínico. Este
            módulo diseña datos, versiones, ejecución, logs, revisión, deriva,
            rollback e incidentes.
            """
        ),
        md(
            """
            ## Objetivos

            - crear configuración y manifiesto reproducibles;
            - validar contratos de entrada;
            - procesar lotes idempotentes;
            - seudonimizar correctamente;
            - diseñar logs sin texto clínico;
            - monitorizar distribución y rendimiento;
            - crear cola de revisión;
            - desplegar en fases;
            - responder a incidentes y hacer rollback.
            """
        ),
        common_setup(),
        code(
            """
            import hashlib
            import json
            from dataclasses import asdict, dataclass
            from datetime import datetime, timezone

            import numpy as np
            import pandas as pd

            from clinical_nlp_course import (
                operation_key,
                population_stability_index,
            )
            from miopia_nlp import pseudonymize_id

            renal_df = pd.read_json(
                PROJECT_ROOT / "data" / "nefrologia_sintetica.jsonl",
                lines=True,
            )
            """
        ),
        md(
            """
            ## 1. Arquitectura

            ```text
            Oracle/vistas aprobadas
                ↓ extracción incremental
            zona segura + seudonimización
                ↓
            validación de contrato
                ↓
            pipeline versionado
                ↓
            evidencia + predicciones
                ↓
            cola de revisión / tabla paciente
                ↓
            monitorización + auditoría
            ```

            Texto e identificadores no deben salir de la zona autorizada.
            """
        ),
        md(
            """
            ## 2. Configuración inmutable

            Incluye versiones, umbrales, fechas y reglas. Un hash permite
            demostrar qué configuración generó una salida.
            """
        ),
        code(
            """
            @dataclass(frozen=True)
            class RunConfig:
                pipeline_version: str
                dictionary_version: str
                guide_version: str
                date_from: str
                date_to: str
                threshold: float
                text_column: str = "text"

                def sha256(self):
                    canonical = json.dumps(
                        asdict(self),
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

            config = RunConfig(
                pipeline_version="renal-baseline-1.0.0",
                dictionary_version="1.0.0",
                guide_version="renal-trs-1.0",
                date_from="2024-01-01",
                date_to="2025-01-01",
                threshold=0.50,
            )
            print(config.sha256())
            """
        ),
        md(
            """
            ## 3. Data contract

            Rechaza entrada inesperada antes del modelo:

            - columnas;
            - tipos;
            - nulos;
            - fechas;
            - IDs;
            - longitud;
            - codificación;
            - duplicados.
            """
        ),
        code(
            """
            def validate_input_contract(frame):
                issues = []
                required = {
                    "patient_id", "course_id", "date", "language", "text"
                }
                missing = required - set(frame.columns)
                if missing:
                    issues.append(f"columnas ausentes: {sorted(missing)}")
                    return issues
                if frame["course_id"].duplicated().any():
                    issues.append("course_id duplicado")
                if frame["patient_id"].isna().any():
                    issues.append("patient_id nulo")
                if frame["text"].isna().any():
                    issues.append("texto nulo")
                if frame["text"].astype(str).str.len().gt(200_000).any():
                    issues.append("texto supera longitud permitida")
                unknown_languages = set(frame["language"].dropna()) - {"es", "ca"}
                if unknown_languages:
                    issues.append(f"idiomas no previstos: {sorted(unknown_languages)}")
                return issues

            assert validate_input_contract(renal_df) == []
            """
        ),
        code(
            """
            broken = pd.concat([renal_df, renal_df.iloc[[0]]], ignore_index=True)
            validate_input_contract(broken)
            """
        ),
        md(
            """
            ## 4. Seudonimización

            HMAC con secreto aleatorio gestionado fuera del código evita que un
            atacante pruebe identificadores con un hash simple. La tabla de
            correspondencia y la clave se separan.

            Seudonimizado sigue siendo dato personal.
            """
        ),
        code(
            """
            DEMO_SECRET = b"solo-demostracion-clave-no-productiva"
            demo_pseudonyms = renal_df["patient_id"].head().map(
                lambda identifier: pseudonymize_id(identifier, DEMO_SECRET)
            )
            demo_pseudonyms.tolist()
            """
        ),
        md(
            """
            En producción la clave no se imprime, no se versiona y no se pasa como
            argumento visible de línea de comandos.
            """
        ),
        md(
            """
            ## 5. Lotes e idempotencia

            La misma entrada + configuración + versión debe producir la misma
            clave. Un cambio del texto con el mismo ID debe producir otra; de lo
            contrario podríamos omitir una nota corregida.
            """
        ),
        code(
            """
            def batches(frame, size):
                for start in range(0, len(frame), size):
                    yield frame.iloc[start:start + size]

            config_hash = config.sha256()
            seen_keys = set()
            processed_count = 0
            for batch in batches(renal_df, size=7):
                for row in batch.itertuples():
                    key = operation_key(
                        row.course_id,
                        row.text,
                        config_hash,
                        pipeline_version=config.pipeline_version,
                    )
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    processed_count += 1
            assert processed_count == len(renal_df)
            print(processed_count)
            """
        ),
        md(
            """
            Guarda checkpoint por ventana/ID, estado del lote y error técnico.
            Nunca registres el texto en el mensaje de excepción.
            """
        ),
        md(
            """
            ## 6. Manifiesto y logs seguros

            El manifiesto identifica ejecución. El log contiene conteos y hashes,
            no PHI.
            """
        ),
        code(
            """
            run_manifest = {
                "run_id": "RUN-" + config_hash[:12],
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_sha256": config_hash,
                "pipeline_version": config.pipeline_version,
                "dictionary_version": config.dictionary_version,
                "input_rows": len(renal_df),
                "status": "validated_synthetic",
            }
            safe_log_event = {
                "run_id": run_manifest["run_id"],
                "event": "batch_complete",
                "batch_number": 1,
                "rows": 7,
                "errors": 0,
            }
            print(json.dumps(safe_log_event, indent=2))
            """
        ),
        md(
            """
            ### Ejercicio 1

            Revisa estos campos y elimina los inseguros:

            `patient_name`, `course_text`, `exception_with_text`,
            `course_id_original`, `run_id`, `duration_ms`, `model_version`.
            """
        ),
        md(
            """
            ## 7. Pruebas en capas

            - unitarias: reglas y funciones;
            - integración: Oracle → pipeline → salida;
            - contrato: esquemas;
            - regresión: casos centinela;
            - rendimiento: tiempo/memoria;
            - seguridad: permisos, secretos, inyección;
            - aceptación clínica: flujo y revisión;
            - recuperación: reintento/rollback.
            """
        ),
        code(
            """
            test_matrix = pd.DataFrame([
                {"layer": "unit", "example": "negación no se descarta", "frequency": "cada cambio"},
                {"layer": "integration", "example": "CLOB y Unicode", "frequency": "cada release"},
                {"layer": "contract", "example": "columna text ausente", "frequency": "cada lote"},
                {"layer": "performance", "example": "10k cursos", "frequency": "cada modelo"},
                {"layer": "clinical", "example": "cola revisada", "frequency": "piloto/periódica"},
            ])
            test_matrix
            """
        ),
        md(
            """
            ## 8. Monitorización de entrada

            Monitoriza volumen, nulos, idioma, longitud, secciones, conceptos y
            tipos documentales. No hace falta almacenar texto para detectar muchos
            cambios.
            """
        ),
        code(
            """
            reference_lengths = renal_df["text"].str.len()
            simulated_future_lengths = reference_lengths * 1.7

            input_monitor = {
                "reference_mean_length": float(reference_lengths.mean()),
                "future_mean_length": float(simulated_future_lengths.mean()),
                "relative_change": float(
                    simulated_future_lengths.mean() / reference_lengths.mean() - 1
                ),
            }
            input_monitor
            """
        ),
        md(
            """
            ## 9. PSI didáctico

            Population Stability Index compara distribuciones por bins. Es una
            señal, no una prueba de degradación clínica.
            """
        ),
        code(
            """
            bins = [0, 50, 100, 150, 250, 500]
            psi_length = population_stability_index(
                reference_lengths,
                simulated_future_lengths,
                bins=bins,
            )
            print("PSI longitud:", psi_length)
            """
        ),
        md(
            """
            No adoptes umbrales PSI universales sin justificar. Una plantilla más
            larga puede no afectar al objetivo; un cambio pequeño de abreviatura
            puede romper una regla crítica.
            """
        ),
        md(
            """
            ## 10. Monitorización de rendimiento

            Necesita nuevas etiquetas humanas. Diseña muestreo periódico:

            - aleatorio;
            - positivos;
            - negativos;
            - posibles/abstenciones;
            - desacuerdos;
            - subgrupos;
            - cambios de plantilla.
            """
        ),
        code(
            """
            review_queue = renal_df.assign(
                priority=np.where(
                    renal_df["gold_document_trs_evidence"],
                    "predicted_positive_demo",
                    "random_negative_demo",
                )
            )[["course_id", "priority"]].head(10)
            review_queue
            """
        ),
        md(
            """
            En producción la cola no usa gold. Combinarías incertidumbre,
            desacuerdo de modelos, novedad y una muestra aleatoria.
            """
        ),
        md(
            """
            ## 11. Despliegue por fases

            1. pruebas offline;
            2. retrospectivo bloqueado;
            3. modo silencioso;
            4. revisión paralela;
            5. piloto limitado;
            6. ampliación gradual;
            7. evaluación clínica formal cuando corresponda.

            Define criterio de parada antes del piloto.
            """
        ),
        code(
            """
            rollout_gates = pd.DataFrame([
                {"phase": "offline", "gate": "métricas + errores aceptados"},
                {"phase": "shadow", "gate": "sin impacto, flujo estable"},
                {"phase": "assisted", "gate": "revisión humana y carga aceptable"},
                {"phase": "expanded", "gate": "validación temporal + monitorización"},
            ])
            rollout_gates
            """
        ),
        md(
            """
            ## 12. Incidentes y rollback

            Prepara:

            - desactivar nueva versión;
            - volver a último artefacto validado;
            - preservar evidencias;
            - identificar resultados afectados;
            - notificar responsables;
            - corregir y revalidar;
            - documentar causa raíz.
            """
        ),
        code(
            """
            incident_record = {
                "incident_id": "INC-DEMO",
                "detected_by": "monitoring",
                "symptom": "caída de menciones tras cambio de plantilla",
                "affected_versions": ["renal-baseline-1.1.0"],
                "containment": "rollback a 1.0.0",
                "patient_impact_assessed": False,
                "root_cause": None,
                "revalidation_required": True,
            }
            incident_record
            """
        ),
        md(
            """
            ## 13. Registro de modelos

            Un artefacto promovible enlaza:

            - código/commit;
            - configuración;
            - datos y guía;
            - diccionario;
            - checkpoint/tokenizador;
            - métricas;
            - model card;
            - aprobaciones;
            - estado de despliegue.
            """
        ),
        md(
            """
            ## 14. Factores humanos

            Evalúa:

            - quién revisa;
            - cómo se muestran evidencia e incertidumbre;
            - cómo se corrige;
            - tiempo;
            - alert fatigue;
            - sobreconfianza;
            - discrepancias;
            - responsabilidad y escalado.

            «Human in the loop» no es una mitigación si nadie tiene tiempo,
            autoridad o información para detectar el error.
            """
        ),
        md(
            """
            ## Reto integrador

            Diseña un release `1.1.0`:

            1. cambio de diccionario;
            2. pruebas;
            3. nueva evaluación;
            4. model card;
            5. migración;
            6. shadow mode;
            7. umbrales de monitorización;
            8. rollback;
            9. responsables.
            """
        ),
        md(
            """
            ## Criterio para avanzar

            Debes poder:

            - reconstruir una ejecución por hash;
            - validar entrada;
            - diseñar idempotencia;
            - explicar seudonimización;
            - crear logs sin PHI;
            - distinguir deriva de degradación;
            - diseñar muestreo de monitorización;
            - ejecutar rollback conceptual;
            - describir factores humanos.
            """
        ),
    ]
