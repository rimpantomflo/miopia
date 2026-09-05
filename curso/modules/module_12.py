from curso.notebook_factory import code, common_setup, md

TITLE = "12 · Baselines fuertes de ML clásico"


def build() -> list[dict]:
    return [
        md(
            """
            # 12 · Baselines fuertes de ML clásico

            Un senior no empieza por el modelo más caro. Construye una referencia
            difícil de superar, separa por paciente y conserva probabilidades,
            errores y señales aprendidas. Aquí entrenaremos de extremo a extremo
            sobre 320 notas enteramente ficticias en español y catalán.
            """
        ),
        md(
            """
            ## Objetivos

            - entrenar TF-IDF de palabras y caracteres + regresión logística;
            - impedir fuga por paciente;
            - comparar development y test bloqueado;
            - inspeccionar probabilidades, errores y rasgos;
            - crear un manifiesto reproducible;
            - saber cuándo un transformer todavía no está justificado.
            """
        ),
        common_setup(),
        code(
            """
            import json

            import pandas as pd
            from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

            from clinical_nlp_course import (
                build_run_manifest,
                build_tfidf_classifier,
                evaluate_classifier,
                predict_with_probabilities,
                top_linear_features,
            )

            data_path = PROJECT_ROOT / "data" / "renal_classification_synthetic.jsonl"
            rows = pd.read_json(data_path, lines=True)
            print(rows.shape)
            display(pd.crosstab(rows["split"], rows["label"]))
            assert rows["synthetic"].all()
            assert rows.groupby("patient_id")["split"].nunique().max() == 1
            """
        ),
        md(
            """
            ## 1. Define la tarea antes del modelo

            Unidad: nota. Salida: modalidad renal dominante explícita en la nota.
            Clases: `HEMODIALYSIS`, `PERITONEAL_DIALYSIS`, `TRANSPLANT` y
            `NO_REPLACEMENT`. No estamos infiriendo una decisión terapéutica ni el
            estado longitudinal completo del paciente.

            El corpus es un laboratorio: sirve para comprobar el flujo, no para
            estimar rendimiento hospitalario.
            """
        ),
        code(
            """
            train = rows.query("split == 'train'").copy()
            development = rows.query("split == 'development'").copy()
            test = rows.query("split == 'test'").copy()

            assert set(train.patient_id).isdisjoint(development.patient_id)
            assert set(train.patient_id).isdisjoint(test.patient_id)
            assert set(development.patient_id).isdisjoint(test.patient_id)
            print({name: len(frame) for name, frame in {
                "train": train, "development": development, "test": test
            }.items()})
            """
        ),
        md(
            """
            ## 2. Entrena el baseline

            Las palabras capturan frases clínicas; los n-gramas de caracteres
            ayudan con abreviaturas, tildes y pequeñas variaciones. `class_weight`
            evita ignorar una clase minoritaria, pero no sustituye un muestreo
            representativo.
            """
        ),
        code(
            """
            baseline = build_tfidf_classifier(seed=17)
            baseline.fit(train["text"], train["label"])

            development_report = evaluate_classifier(
                baseline, development["text"].tolist(), development["label"].tolist()
            )
            print("Macro F1 development:", development_report["macro avg"]["f1-score"])
            assert development_report["accuracy"] > 0.70
            """
        ),
        code(
            """
            dev_predictions = predict_with_probabilities(baseline, development["text"])
            dev_errors = development.assign(prediction=dev_predictions.labels)
            dev_errors = dev_errors.loc[
                dev_errors["label"] != dev_errors["prediction"],
                ["document_id", "language", "label", "prediction", "text"],
            ]
            print("Errores development:", len(dev_errors))
            display(dev_errors.head(10))
            """
        ),
        md(
            """
            ## 3. Abre el test una sola vez

            Decide la representación y cualquier hiperparámetro usando train/dev.
            Solo después calcula el test. Si vuelves a cambiar el modelo por un
            error del test, ese conjunto deja de ser test.
            """
        ),
        code(
            """
            test_report = evaluate_classifier(
                baseline, test["text"].tolist(), test["label"].tolist()
            )
            test_predictions = baseline.predict(test["text"])
            test_confusion = confusion_matrix(
                test["label"], test_predictions, labels=baseline.classes_
            )
            print("Macro F1 test:", test_report["macro avg"]["f1-score"])
            print(pd.DataFrame(
                test_confusion, index=baseline.classes_, columns=baseline.classes_
            ))
            assert test_report["accuracy"] > 0.70
            """
        ),
        md(
            """
            `ConfusionMatrixDisplay` está importado para que dibujes la matriz en
            tu copia de trabajo. En un informe conserva también los conteos: una
            matriz normalizada puede ocultar muestras pequeñas.
            """
        ),
        code(
            """
            signals = top_linear_features(baseline, n=8)
            for label, features in signals.items():
                print("\\n", label)
                for feature, weight in features[:5]:
                    print(f"  {feature}: {weight:.3f}")
            """
        ),
        md(
            """
            ## 4. Interpretación crítica

            Los pesos revelan asociaciones del corpus, no causas. Busca fugas:
            nombres de plantilla, servicios, fechas, códigos o palabras que sean
            proxies de la etiqueta. El modelo sintético será casi perfecto porque
            las plantillas son separables; esa facilidad es otra limitación.
            """
        ),
        code(
            """
            manifest = build_run_manifest(
                name="renal-tfidf-v1",
                configuration={
                    "model": "word_char_tfidf_logistic_regression",
                    "seed": 17,
                    "split": "patient_hash_65_20_15",
                    "primary_metric": "macro_f1",
                },
                data_files={"synthetic_corpus": str(data_path)},
            )
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            """
        ),
        md(
            """
            ## Práctica obligatoria

            1. Añade 40 contraejemplos difíciles y regenera el corpus.
            2. Compara solo palabras, solo caracteres y ambos.
            3. Ejecuta cinco semillas de partición por paciente.
            4. Informa media, dispersión y los diez errores clínicamente peores.
            5. Define por escrito qué mejora mínima justificaría un transformer.

            **Criterio de salida:** puedes reconstruir el experimento, explicar por
            qué no hay fuga y defender el baseline ante un clínico y un ingeniero.
            """
        ),
    ]
