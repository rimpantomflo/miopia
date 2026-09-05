from curso.notebook_factory import code, common_setup, md

TITLE = "17 · Validación, utilidad y explicabilidad avanzada"


def build() -> list[dict]:
    return [
        md(
            """
            # 17 · Validación, utilidad y explicabilidad avanzada

            Una sola F1 no autoriza un piloto. Seleccionaremos umbral en
            development, lo congelaremos, mediremos utilidad en test, compararemos
            modelos de forma emparejada y examinaremos subgrupos y explicaciones.
            """
        ),
        md(
            """
            ## Objetivos

            - distinguir discriminación, calibración y utilidad;
            - elegir umbral sin tocar el test;
            - crear decision curves;
            - informar subgrupos con tamaño insuficiente explícito;
            - comparar sistemas por bootstrap de pacientes;
            - explicar modelos lineales y probar perturbaciones;
            - diseñar validación temporal y externa.
            """
        ),
        common_setup(),
        code(
            """
            import pandas as pd

            from clinical_nlp_course import (
                build_tfidf_classifier,
                choose_threshold,
                decision_curve,
                discrimination_report,
                expected_calibration_error,
                leave_one_segment_out,
                linear_feature_contributions,
                paired_cluster_bootstrap_difference,
                subgroup_report,
                threshold_metrics,
            )
            """
        ),
        md(
            """
            ## 1. Development: selección del umbral

            El requisito clínico ficticio exige sensibilidad ≥0,80. Entre los
            umbrales que cumplen, elegimos la mayor especificidad. Esta regla debe
            estar escrita antes de mirar el test.
            """
        ),
        code(
            """
            dev_truth = [1, 1, 1, 1, 0, 0, 0, 0]
            dev_scores = [0.95, 0.82, 0.63, 0.42, 0.55, 0.31, 0.15, 0.05]
            print(discrimination_report(dev_truth, dev_scores))
            print("ECE:", expected_calibration_error(dev_truth, dev_scores, n_bins=4))
            selected = choose_threshold(
                dev_truth, dev_scores, minimum_sensitivity=0.80
            )
            frozen_threshold = selected["threshold"]
            print("Umbral congelado:", frozen_threshold, selected)
            """
        ),
        md(
            """
            ## 2. Test bloqueado

            Aplicamos el valor sin optimizar de nuevo. Reporta conteos e intervalos,
            no solo porcentajes.
            """
        ),
        code(
            """
            test_truth = [1, 1, 1, 0, 0, 0, 1, 0]
            test_scores = [0.91, 0.77, 0.35, 0.62, 0.28, 0.09, 0.69, 0.41]
            locked_test = threshold_metrics(
                test_truth, test_scores, threshold=frozen_threshold
            )
            print(locked_test)
            """
        ),
        code(
            """
            curve = pd.DataFrame(
                decision_curve(
                    test_truth,
                    test_scores,
                    thresholds=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
                )
            )
            display(curve)
            print("Dibuja model/treat_all/treat_none y define el rango clínico relevante.")
            """
        ),
        md(
            """
            La net benefit requiere que el umbral represente una relación explícita
            entre falsos positivos y falsos negativos. No convierte una predicción
            retrospectiva en beneficio prospectivo.
            """
        ),
        md(
            """
            ## 3. Subgrupos y portabilidad

            No interpretes un 100 % sobre dos pacientes como equidad. Declara el
            tamaño, intervalos y `insufficient_sample`.
            """
        ),
        code(
            """
            subgroup_rows = [
                {"language": language, "truth": truth, "score": score}
                for language, truth, score in zip(
                    ["es", "es", "es", "es", "ca", "ca", "ca", "other"],
                    test_truth,
                    test_scores,
                )
            ]
            display(subgroup_report(
                subgroup_rows,
                group_key="language",
                truth_key="truth",
                score_key="score",
                threshold=frozen_threshold,
                minimum_size=3,
            ))
            """
        ),
        md(
            """
            ## 4. Comparación emparejada por paciente

            Si A y B evalúan los mismos pacientes, la incertidumbre debe conservar
            ese emparejamiento. El bootstrap remuestrea pacientes completos.
            """
        ),
        code(
            """
            paired_rows = [
                {"patient": "P1", "truth": 1, "baseline": 0.40, "candidate": 0.90},
                {"patient": "P1", "truth": 1, "baseline": 0.70, "candidate": 0.80},
                {"patient": "P2", "truth": 1, "baseline": 0.80, "candidate": 0.82},
                {"patient": "P3", "truth": 0, "baseline": 0.20, "candidate": 0.10},
                {"patient": "P4", "truth": 0, "baseline": 0.30, "candidate": 0.25},
            ]
            comparison = paired_cluster_bootstrap_difference(
                paired_rows,
                group_key="patient",
                truth_key="truth",
                score_a_key="baseline",
                score_b_key="candidate",
                threshold=0.5,
                metric="sensitivity",
                n_resamples=500,
            )
            print("Diferencia candidato - baseline:", comparison)
            """
        ),
        md(
            """
            ## 5. Explicabilidad que se puede falsar

            Los pesos lineales describen el modelo. La perturbación comprueba si su
            score cambia al ocultar un segmento. Ninguna de las dos demuestra
            causalidad ni razonamiento clínico.
            """
        ),
        code(
            """
            explanation_texts = [
                "continúa hemodiálisis por FAV",
                "sesión de HD completada",
                "sin necesidad de hemodiálisis",
                "manejo conservador sin TRS",
                "hemodiálisis crónica estable",
                "no precisa diálisis",
            ]
            explanation_labels = ["YES", "YES", "NO", "NO", "YES", "NO"]
            explainable_model = build_tfidf_classifier().fit(
                explanation_texts, explanation_labels
            )
            explanation = linear_feature_contributions(
                explainable_model, "continúa hemodiálisis estable", target_label="YES"
            )
            print(explanation)

            target = "hemodiálisis"
            example = "continúa hemodiálisis estable"
            start = example.index(target)
            perturbations = leave_one_segment_out(
                example,
                [(start, start + len(target), "treatment")],
                score=lambda text: float(
                    explainable_model.predict_proba([text])[0][
                        list(explainable_model.classes_).index("YES")
                    ]
                ),
            )
            print(perturbations)
            """
        ),
        md(
            """
            ## Protocolo senior de validación

            1. análisis y umbral en development;
            2. test interno temporal bloqueado;
            3. validación externa con protocolo sin reajuste;
            4. revisión de errores clínicamente estratificada;
            5. calibración y utilidad en población de destino;
            6. subgrupos preespecificados;
            7. modo silencioso y estudio de factores humanos;
            8. revalidación tras deriva o cambio material.

            Sigue TRIPOD+AI/PROBAST+AI para modelos predictivos, STARD-AI para
            precisión diagnóstica y DECIDE-AI para evaluación clínica temprana
            cuando sean aplicables al uso previsto.

            **Criterio de salida:** puedes separar una mejora estadística de una
            mejora clínicamente útil y sabes qué evidencia falta para un piloto.
            """
        ),
    ]
