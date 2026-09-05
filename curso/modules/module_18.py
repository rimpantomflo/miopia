from curso.notebook_factory import code, common_setup, md

TITLE = "18 · Producción hospitalaria y capstone Hero"


def build() -> list[dict]:
    return [
        md(
            """
            # 18 · Producción hospitalaria y capstone Hero

            El objetivo no es “poner un modelo en un servidor”. Es operar un
            sistema reversible, auditable y supervisado dentro de la gobernanza del
            hospital. Este módulo conecta contrato, idempotencia, artefactos, API,
            monitorización, incidentes y aprobación clínica.
            """
        ),
        md(
            """
            ## Objetivos

            - rechazar entradas incompatibles antes del modelo;
            - garantizar idempotencia sensible al contenido y la versión;
            - crear manifiestos y logs por lista positiva;
            - ejecutar API/Docker con mínimos privilegios;
            - definir gates, parada y rollback;
            - entregar un capstone con evidencia técnica, clínica y operativa.
            """
        ),
        common_setup(),
        code(
            """
            import json

            import pandas as pd

            from clinical_nlp_course import (
                InputContract,
                build_run_manifest,
                canonical_sha256,
                operation_key,
                population_stability_index,
                safe_batch_event,
            )

            source_path = PROJECT_ROOT / "data" / "renal_classification_synthetic.jsonl"
            incoming = pd.read_json(source_path, lines=True).head(20)
            incoming = incoming.rename(columns={"document_id": "course_id"})
            incoming["date"] = pd.date_range("2025-01-01", periods=len(incoming), freq="D")
            """
        ),
        md(
            """
            ## 1. Contrato de entrada

            Se comprueban columnas, nulos, IDs, duplicados, idioma, longitud y
            fechas interpretables dentro de un intervalo. El lote no llega al
            modelo si falla.
            """
        ),
        code(
            """
            contract = InputContract(maximum_text_length=20_000)
            contract_issues = contract.validate(incoming)
            assert contract_issues == []
            print("Contrato:", contract_issues or "OK")

            duplicated = pd.concat([incoming, incoming.iloc[[0]]], ignore_index=True)
            assert "course_id duplicado" in contract.validate(duplicated)
            """
        ),
        md(
            """
            ## 2. Idempotencia correcta

            La clave antigua usaba solo `course_id + config_hash`: si cambiaba el
            texto con el mismo ID, se omitía erróneamente. La nueva incluye hash de
            contenido y versión del pipeline, sin revelar el texto.
            """
        ),
        code(
            """
            config = {
                "pipeline_version": "renal-classifier-2.0.0",
                "dictionary_version": "renal-terminology-1.1.0",
                "threshold": 0.62,
            }
            config_hash = canonical_sha256(config)
            first = incoming.iloc[0]
            key = operation_key(
                first.course_id,
                first.text,
                config_hash,
                pipeline_version=config["pipeline_version"],
            )
            changed_key = operation_key(
                first.course_id,
                first.text + " Cambio ficticio.",
                config_hash,
                pipeline_version=config["pipeline_version"],
            )
            assert key != changed_key
            print(key)
            """
        ),
        md(
            """
            ## 3. Manifiesto y evento seguro

            El manifiesto guarda checksums y configuración. El evento de ejecución
            usa una lista cerrada de campos y nunca acepta texto, nombre o ID
            clínico original.
            """
        ),
        code(
            """
            manifest = build_run_manifest(
                name="hospital-shadow-run",
                configuration=config,
                data_files={"synthetic_input": str(source_path)},
            )
            event = safe_batch_event(
                run_id=manifest["run_id"],
                batch_number=1,
                rows=len(incoming),
                errors=0,
                duration_ms=42.1876,
                model_version=config["pipeline_version"],
            )
            print(json.dumps(event, indent=2))
            assert "text" not in event and "patient_id" not in event
            """
        ),
        md(
            """
            ## 4. Monitorización sin PHI

            Longitud, volumen, idioma, tasa de abstención y distribución de salida
            pueden monitorizarse sin almacenar el texto. PSI abre las colas para no
            perder observaciones fuera de los bins, pero es una alarma, no evidencia
            automática de daño o degradación.
            """
        ),
        code(
            """
            reference_lengths = incoming["text"].str.len().tolist()
            future_lengths = [value * 1.8 for value in reference_lengths]
            psi = population_stability_index(reference_lengths, future_lengths)
            print("PSI longitud:", psi)
            """
        ),
        md(
            """
            ## 5. API y contenedor

            Prueba local:

            ```bash
            uv sync --extra service
            uv run uvicorn miopia_nlp.service:create_app --factory \
              --host 127.0.0.1 --port 8000 --no-access-log
            curl -s http://127.0.0.1:8000/health
            ```

            O bien `docker compose up --build`. El contenedor usa usuario no root,
            filesystem de solo lectura, capacidades eliminadas y puerto ligado a
            loopback. Eso reduce riesgo; no sustituye autenticación, TLS, red
            segmentada, secretos, auditoría ni hardening institucional.
            """
        ),
        code(
            """
            endpoint_contract = {
                "path": "/v1/phenotype",
                "request": {"text": "str, 1..200000 chars"},
                "response": "status + evidence + offsets + rule_id",
                "access_log": "disabled_to_avoid_request_bodies",
                "cache": "no-store",
                "health": "/health",
            }
            endpoint_contract
            """
        ),
        md(
            """
            ## 6. Gates de despliegue

            | Fase | Gate mínimo | Stop inmediato |
            |---|---|---|
            | offline | protocolo y test bloqueado | fuga o métrica crítica |
            | shadow | estabilidad y cero impacto | texto en logs/incidente |
            | asistida | revisión humana y carga viable | error clínico grave |
            | piloto | aprobación y monitorización | umbral de seguridad |
            | escala | validación temporal/externa | deriva no explicada |

            Define propietario, tiempo de respuesta, última versión estable y
            procedimiento para identificar resultados afectados antes del piloto.
            """
        ),
        code(
            """
            risk_register = pd.DataFrame([
                {"risk": "falso negativo", "control": "muestreo de negativos + sensibilidad", "owner": "clinical_lead"},
                {"risk": "PHI en logs", "control": "lista positiva + secret scan", "owner": "security"},
                {"risk": "cambio de plantilla", "control": "drift + casos centinela", "owner": "data_owner"},
                {"risk": "automation bias", "control": "UI con evidencia + estudio humano", "owner": "product_clinical"},
                {"risk": "modelo nuevo peor", "control": "canary + rollback", "owner": "ml_engineer"},
            ])
            risk_register
            """
        ),
        md(
            """
            ## Capstone Hero: definición de terminado

            Elige una tarea hospitalaria aprobada y entrega:

            1. uso previsto, exclusiones, población y responsables;
            2. protocolo de datos/anotación y 100+ casos sintéticos difíciles;
            3. corpus real autorizado con ficha y control de acceso;
            4. baseline de reglas y baseline ML clásico;
            5. candidato avanzado justificado;
            6. test temporal bloqueado y validación externa cuando sea posible;
            7. intervalos, calibración, subgrupos, utilidad y errores;
            8. model card, manifiestos y matriz de riesgos;
            9. API/lote idempotente, tests, CI, monitorización y rollback;
            10. demo en shadow mode con revisión clínica documentada.

            No está terminado si solo existe un notebook. Tampoco si el modelo es
            excelente pero no se puede auditar, detener o corregir.

            **Criterio final:** puedes defender el sistema ante clínica,
            metodología, ingeniería, seguridad y protección de datos, y puedes
            explicar con precisión qué falta antes de afectar atención real.
            """
        ),
    ]
