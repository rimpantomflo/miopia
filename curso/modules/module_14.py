from curso.notebook_factory import code, common_setup, md

TITLE = "14 · Fine-tuning real de Transformers clínicos"


def build() -> list[dict]:
    return [
        md(
            """
            # 14 · Fine-tuning real de Transformers clínicos

            Este laboratorio reemplaza el pseudocódigo por una tubería completa:
            alineación, dataset, `Trainer`, `seqeval`, mejor checkpoint y test
            bloqueado. La ruta CPU verifica toda la preparación; entrenar pesos se
            activa de forma explícita porque requiere red, tiempo y normalmente GPU.
            """
        ),
        md(
            """
            ## Objetivos

            - convertir spans a BIO y BIO a offsets;
            - alinear subtokens con `-100`;
            - comprobar fuga por paciente;
            - lanzar un entrenamiento reproducible;
            - guardar tokenizador, label map y métricas;
            - comparar varias semillas y modelos con el mismo presupuesto.
            """
        ),
        common_setup(),
        code(
            """
            from clinical_nlp_course import (
                align_word_labels,
                assert_no_patient_leakage,
                bio_to_char_spans,
                char_spans_to_bio,
            )

            text = "ERC G5 inicia hemodiálisis hoy"
            offsets = [(0, 3), (4, 6), (7, 13), (14, 27), (28, 31)]
            start = text.index("hemodiálisis")
            labels = char_spans_to_bio(
                offsets, [(start, start + len("hemodiálisis"), "TREATMENT")]
            )
            print(list(zip(text.split(), labels)))
            assert bio_to_char_spans(labels, offsets)[0]["start"] == start
            """
        ),
        code(
            """
            # [CLS], ERC, hemo, ##diálisis, hoy, [SEP]
            word_ids = [None, 0, 1, 1, 2, None]
            word_label_ids = [0, 1, 0]
            aligned = align_word_labels(word_ids, word_label_ids)
            assert aligned == [-100, 0, 1, -100, 0, -100]
            aligned
            """
        ),
        md(
            """
            ## 1. Contrato JSONL

            Cada línea del script `scripts/train_token_classifier.py` requiere:

            ```json
            {"document_id":"D1","patient_id":"P1","split":"train",
             "tokens":["Inicia","hemodiálisis"],
             "ner_tags":["O","B-TREATMENT"]}
            ```

            El script falla si tokens/etiquetas difieren o un paciente aparece en
            más de una partición. No fabrica automáticamente un test conveniente.
            """
        ),
        code(
            """
            sample_rows = [
                {"patient_id": "P1", "split": "train"},
                {"patient_id": "P1", "split": "train"},
                {"patient_id": "P2", "split": "development"},
                {"patient_id": "P3", "split": "test"},
            ]
            assert_no_patient_leakage(sample_rows)

            leaked = [*sample_rows, {"patient_id": "P1", "split": "test"}]
            try:
                assert_no_patient_leakage(leaked)
            except ValueError as error:
                print("Control correcto:", error)
            """
        ),
        md(
            """
            ## 2. Ejecuta el entrenamiento

            Instala una sola vez:

            ```bash
            uv sync --extra transformers --group dev
            ```

            Lanza dentro de infraestructura autorizada:

            ```bash
            uv run python scripts/train_token_classifier.py \
              --data data/restricted/ner_train.jsonl \
              --model PlanTL-GOB-ES/roberta-base-biomedical-clinical-es \
              --output artifacts/roberta_ner_seed17 --seed 17
            ```

            El script contiene tokenización rápida, alineación, collator dinámico,
            `AutoModelForTokenClassification`, evaluación por época, carga del
            mejor F1, test final y guardado del modelo/tokenizador.
            """
        ),
        code(
            """
            transformer_experiment = {
                "base_model": "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
                "model_revision": "PIN_ME_BEFORE_TRAINING",
                "seeds": [13, 17, 23],
                "epochs": 3,
                "learning_rate": 2e-5,
                "batch_size": 8,
                "max_length": 256,
                "selection_metric": "development_f1",
                "final_metric": "exact_span_micro_f1_on_locked_test",
            }
            transformer_experiment
            """
        ),
        md(
            """
            ## 3. Modelos ya ajustados

            Para `BSC-NLP4BIA/bsc-bio-ehr-es-distemist`, revisa la tarjeta y
            reproduce su pretokenización si la exige; no asumas que pasar un string
            a `pipeline()` replica el protocolo original. Guarda la revisión del
            modelo y armoniza etiquetas antes de puntuar.
            """
        ),
        code(
            """
            RUN_PRETRAINED_INFERENCE = False

            if RUN_PRETRAINED_INFERENCE:
                from transformers import pipeline

                ner = pipeline(
                    task="token-classification",
                    model="BSC-NLP4BIA/bsc-bio-ehr-es-distemist",
                    aggregation_strategy="simple",
                )
                print(ner("Curso ficticio: ERC G5 por nefropatía diabética."))
            else:
                print("Inferencia pesada desactivada; la preparación CPU sí fue validada.")
            """
        ),
        md(
            """
            ## 4. Comparación senior

            Compara diccionario, spaCy, modelo BSC directo y encoder ajustado con:

            - mismas particiones y postprocesado;
            - exact span F1 primario y solapado secundario;
            - tres semillas y dispersión;
            - latencia, memoria y energía;
            - errores por etiqueta, longitud, idioma y sección;
            - calibración o abstención cuando se usan scores;
            - prueba temporal y externa.

            **Criterio de salida:** puedes lanzar el script sobre un corpus
            autorizado, reconstruir spans globales y explicar por qué cualquier
            mejora es real y no fuga, azar o cambio de evaluación.
            """
        ),
    ]
