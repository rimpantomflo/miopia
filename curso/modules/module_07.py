from curso.notebook_factory import code, common_setup, md


TITLE = "07 · Validación avanzada de NLP clínico"


def build() -> list[dict]:
    return [
        md(
            """
            # 07 · Validación avanzada de NLP clínico

            Evaluar no es obtener un F1. Diseñaremos estimandos, unidades,
            intervalos, calibración, subgrupos, validación temporal y evidencia
            para pasar de rendimiento técnico a utilidad clínica.
            """
        ),
        md(
            """
            ## Objetivos

            - definir resultado primario antes de evaluar;
            - conectar prevalencia con VPP/VPN;
            - calcular intervalos agrupados por paciente;
            - evaluar calibración;
            - comparar match exacto/solapado;
            - detectar fuga;
            - diseñar validación temporal/externa;
            - evaluar subgrupos y deriva;
            - validar LLM por dimensiones;
            - elegir guías de reporte.
            """
        ),
        common_setup(),
        code(
            """
            import math

            import numpy as np
            import pandas as pd

            from clinical_nlp_course import (
                brier_score,
                cluster_bootstrap_binary_metric,
                exact_span_metrics,
                expected_calibration_error,
                overlap_span_metrics,
                patient_hash_split,
            )
            from miopia_nlp import binary_metrics

            renal_df = pd.read_json(
                PROJECT_ROOT / "data" / "nefrologia_sintetica.jsonl",
                lines=True,
            )
            """
        ),
        md(
            """
            ## 1. Define el estimando

            Ejemplo:

            > Sensibilidad a nivel de paciente para detectar TRS alguna vez,
            > durante 2024, en adultos del centro X, frente a revisión adjudicada,
            > usando umbral congelado.

            Cambiar unidad, periodo o población cambia la pregunta.
            """
        ),
        code(
            """
            estimand = {
                "population": "población definida antes de extracción",
                "unit": "patient",
                "target": "ever kidney replacement therapy",
                "reference": "adjudicated clinical review",
                "metric": "sensitivity",
                "period": "predefined",
                "threshold": "frozen on development",
            }
            pd.Series(estimand)
            """
        ),
        md(
            """
            ## 2. Matriz y métricas

            Usa siempre conteos junto a porcentajes. Una sensibilidad de 100 %
            puede significar 2/2.
            """
        ),
        code(
            """
            y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
            y_pred = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
            pd.Series(binary_metrics(y_true, y_pred))
            """
        ),
        md(
            """
            ## 3. Prevalencia y valores predictivos

            Sensibilidad/especificidad no determinan por sí solas VPP/VPN. Bajo
            prevalencia baja, incluso un sistema específico puede generar muchos
            falsos positivos.
            """
        ),
        code(
            """
            def predictive_values(sensitivity, specificity, prevalence):
                ppv = (
                    sensitivity * prevalence
                    / (
                        sensitivity * prevalence
                        + (1 - specificity) * (1 - prevalence)
                    )
                )
                npv = (
                    specificity * (1 - prevalence)
                    / (
                        (1 - sensitivity) * prevalence
                        + specificity * (1 - prevalence)
                    )
                )
                return ppv, npv

            prevalence_rows = []
            for prevalence in [0.01, 0.05, 0.20, 0.50]:
                ppv, npv = predictive_values(0.90, 0.95, prevalence)
                prevalence_rows.append({"prevalence": prevalence, "ppv": ppv, "npv": npv})
            pd.DataFrame(prevalence_rows)
            """
        ),
        md(
            """
            ### Ejercicio 1

            Explica por qué un corpus 50/50 enriquecido no proporciona el VPP de
            producción. Decide qué métricas son transportables y qué información
            adicional necesitas.
            """
        ),
        md(
            """
            ## 4. Intervalos agrupados

            Cursos del mismo paciente están correlacionados. Remuestreamos
            pacientes completos.
            """
        ),
        code(
            """
            bootstrap_rows = [
                {"patient": "A", "truth": 1, "pred": 1},
                {"patient": "A", "truth": 1, "pred": 0},
                {"patient": "B", "truth": 1, "pred": 1},
                {"patient": "C", "truth": 0, "pred": 0},
                {"patient": "D", "truth": 0, "pred": 1},
                {"patient": "E", "truth": 0, "pred": 0},
            ]
            cluster_bootstrap_binary_metric(
                bootstrap_rows,
                group_key="patient",
                truth_key="truth",
                prediction_key="pred",
                metric="sensitivity",
                n_resamples=1000,
                seed=7,
            )
            """
        ),
        md(
            """
            Un bootstrap no corrige una muestra sesgada ni crea positivos donde
            no existen. La unidad de remuestreo debe reflejar independencia.
            """
        ),
        md(
            """
            ## 5. Calibración

            Una probabilidad de 0,8 está calibrada si, entre casos similares,
            aproximadamente 80 % son positivos. Ranking y calibración son
            propiedades diferentes.
            """
        ),
        code(
            """
            calibration_truth = [1, 1, 1, 0, 1, 0, 0, 0]
            calibrated = [0.85, 0.75, 0.70, 0.55, 0.60, 0.30, 0.20, 0.10]
            overconfident = [0.99, 0.99, 0.20, 0.95, 0.25, 0.80, 0.70, 0.01]
            print("Brier calibrado:", brier_score(calibration_truth, calibrated))
            print("Brier sobreconfiado:", brier_score(calibration_truth, overconfident))
            print("ECE calibrado:", expected_calibration_error(calibration_truth, calibrated))
            print("ECE sobreconfiado:", expected_calibration_error(calibration_truth, overconfident))
            """
        ),
        md(
            """
            ECE depende de bins y Brier mezcla discriminación/calibración. Usa
            gráficos y tamaño suficiente; no declares calibración por un único
            número pequeño.
            """
        ),
        md(
            """
            ## 6. NER exacto y solapado

            Reporta ambos y una taxonomía de límites.
            """
        ),
        code(
            """
            span_gold = {
                "D1": [(0, 25, "DISEASE")],
                "D2": [(10, 22, "PROCEDURE")],
            }
            span_pred = {
                "D1": [(0, 17, "DISEASE")],
                "D2": [(10, 22, "PROCEDURE")],
                "D3": [(0, 3, "DISEASE")],
            }
            print("Exacto:", exact_span_metrics(span_gold, span_pred))
            print("Solapado:", overlap_span_metrics(span_gold, span_pred))
            """
        ),
        md(
            """
            Un F1 solapado alto puede ocultar límites inútiles para normalización
            o relaciones. El uso previsto decide la tolerancia.
            """
        ),
        md(
            """
            ## 7. Fuga y particiones

            Prueba automática: un paciente solo pertenece a una partición.
            Busca también duplicados exactos y casi duplicados.
            """
        ),
        code(
            """
            renal_df["split"] = renal_df["patient_id"].map(patient_hash_split)
            leakage = renal_df.groupby("patient_id")["split"].nunique()
            assert leakage.max() == 1
            print("Sin fuga directa por patient_id.")

            normalized_text = (
                renal_df["text"].str.casefold().str.replace(r"\\s+", " ", regex=True).str.strip()
            )
            print("Duplicados exactos:", int(normalized_text.duplicated().sum()))
            """
        ),
        md(
            """
            ## 8. Validación temporal y externa

            - desarrollo en periodo anterior;
            - test temporal posterior sin actualizar;
            - test externo en otro centro;
            - reentrenamiento solo después de cerrar análisis.

            Documenta cambios de EHR, plantillas y práctica clínica.
            """
        ),
        code(
            """
            renal_df["date"] = pd.to_datetime(renal_df["date"])
            temporal_split = np.where(
                renal_df["date"] < pd.Timestamp("2024-01-01"),
                "historical_development",
                "future_test",
            )
            pd.Series(temporal_split).value_counts()
            """
        ),
        md(
            """
            El corpus sintético está desequilibrado temporalmente y no sirve para
            conclusiones. El ejemplo muestra el diseño.
            """
        ),
        md(
            """
            ## 9. Subgrupos

            Define antes:

            - idioma;
            - centro;
            - servicio;
            - tipo documental;
            - longitud;
            - periodo;
            - grupos clínicos pertinentes.

            Reporta denominadores e intervalos. No sobreinterpretes grupos
            minúsculos ni uses subgrupos para buscar resultados favorables.
            """
        ),
        code(
            """
            subgroup_summary = (
                renal_df.groupby("language")
                .agg(
                    documents=("course_id", "count"),
                    patients=("patient_id", "nunique"),
                    positives=("gold_document_trs_evidence", "sum"),
                )
            )
            subgroup_summary
            """
        ),
        md(
            """
            ## 10. Comparación de modelos

            Usa predicciones pareadas sobre los mismos casos. Informa discordancias,
            no solo diferencia de F1.
            """
        ),
        code(
            """
            paired = pd.DataFrame([
                {"id": "1", "truth": 1, "rules": 1, "model": 1},
                {"id": "2", "truth": 1, "rules": 0, "model": 1},
                {"id": "3", "truth": 0, "rules": 0, "model": 1},
                {"id": "4", "truth": 0, "rules": 1, "model": 0},
            ])
            paired["rules_correct"] = paired["truth"].eq(paired["rules"])
            paired["model_correct"] = paired["truth"].eq(paired["model"])
            paired
            """
        ),
        md(
            """
            ### Ejercicio 2

            Analiza los dos casos discordantes. ¿El modelo añade valor sobre reglas?
            ¿Qué sistema híbrido propondrías? No respondas solo con el promedio.
            """
        ),
        md(
            """
            ## 11. Validación generativa

            Matriz mínima:

            | Dimensión | Ejemplo |
            |---|---|
            | Estructura | JSON válido |
            | Exactitud | concepto/aserción |
            | Fidelidad | cita sustenta afirmación |
            | Completitud | eventos no omitidos |
            | Abstención | evidencia insuficiente |
            | Robustez | prompt, idioma, adversarial |
            | Operación | coste, latencia, fallos |

            Un LLM-juez puede complementar, no reemplazar, la referencia humana.
            """
        ),
        md(
            """
            ## 12. De retrospectivo a clínico

            Fases:

            1. validación técnica;
            2. retrospectiva bloqueada;
            3. temporal/externa;
            4. modo silencioso;
            5. piloto asistido;
            6. estudio comparativo si la acción lo requiere;
            7. monitorización y revalidación.

            Estudia carga, automatización, correcciones, tiempos y seguridad.
            """
        ),
        md(
            """
            ## 13. Guías

            - [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378):
              modelos de predicción.
            - [PROBAST+AI](https://www.bmj.com/content/388/bmj-2024-082505):
              riesgo de sesgo/aplicabilidad.
            - [STARD-AI](https://www.nature.com/articles/s41591-025-03953-8):
              exactitud diagnóstica.
            - [DECIDE-AI](https://www.nature.com/articles/s41591-022-01772-9):
              evaluación temprana en vivo.

            Una guía de reporte no sustituye diseño metodológico.
            """
        ),
        md(
            """
            ## Reto integrador

            Escribe un plan estadístico para detectar TRS actual:

            - estimando;
            - muestra;
            - unidad;
            - referencia;
            - métrica primaria e IC;
            - umbral;
            - subgrupos;
            - indeterminados;
            - análisis de error;
            - validación temporal;
            - criterio para piloto.
            """
        ),
        md(
            """
            ## Criterio para avanzar

            Debes poder explicar:

            - estimando;
            - prevalencia y VPP;
            - bootstrap agrupado;
            - discriminación y calibración;
            - exacto frente a solapado;
            - fuga;
            - validación temporal/externa;
            - rendimiento técnico frente a utilidad clínica.
            """
        ),
    ]
