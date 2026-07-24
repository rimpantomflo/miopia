from curso.notebook_factory import code, common_setup, md


TITLE = "04 · Transformers y modelos biomédicos del BSC"


def build() -> list[dict]:
    return [
        md(
            """
            # 04 · Transformers y modelos biomédicos del BSC

            Este módulo enseña a seleccionar y preparar modelos. La ruta básica no
            descarga pesos: todos los conceptos y comprobaciones son ejecutables.
            Las celdas pesadas están desactivadas explícitamente.
            """
        ),
        md(
            """
            ## Objetivos

            - distinguir preentrenamiento, ajuste e inferencia;
            - comprender subtokenización, atención y contexto;
            - alinear etiquetas de palabras con subtokens;
            - leer críticamente una model card;
            - conocer recursos BSC/PlanTL;
            - diseñar ajuste de NER y clasificación;
            - manejar cursos largos;
            - comparar modelos bajo el mismo protocolo;
            - estimar requisitos y riesgos.
            """
        ),
        common_setup(),
        code(
            """
            import json
            from dataclasses import dataclass, asdict

            import pandas as pd

            from clinical_nlp_course import exact_span_metrics
            """
        ),
        md(
            """
            ## 1. Qué hace un transformer encoder

            De forma simplificada:

            ```text
            texto → subtokens → embeddings + posición
                  → capas de autoatención → representaciones contextuales
                  → cabeza de tarea
            ```

            El mismo encoder puede recibir una cabeza de:

            - token classification para NER;
            - sequence classification;
            - multilabel;
            - extracción de relaciones;
            - embeddings/recuperación.

            El encoder no sabe nuestras etiquetas hasta ajustarlo.
            """
        ),
        md(
            """
            ## 2. BERT-like no significa LLM generativo

            RoBERTa/MrBERT con objetivo de lenguaje enmascarado son encoders.
            Producen representaciones y completan máscaras, pero no son
            asistentes conversacionales generativos.

            `AutoModelForMaskedLM` y `AutoModelForTokenClassification` usan el
            mismo punto de partida con cabezas distintas.
            """
        ),
        md(
            """
            ## 3. Subtokens

            Un token clínico puede dividirse:

            ```text
            "glomeruloesclerosis" → ["glomerulo", "esclerosis"]
            ```

            En NER debes decidir cómo propagar una etiqueta de palabra. Lo
            habitual es etiquetar el primer subtoken y usar `-100` para ignorar
            los restantes en la pérdida, o propagar BIO de forma coherente.
            """
        ),
        code(
            """
            word_labels = [0, 1, 0]  # O, B-DISEASE, O para tres palabras
            # Ejemplo ficticio de word_ids devueltos por un fast tokenizer:
            word_ids = [None, 0, 1, 1, 2, None]

            def align_first_subtoken(word_labels, word_ids):
                aligned = []
                previous = None
                for word_id in word_ids:
                    if word_id is None:
                        aligned.append(-100)
                    elif word_id != previous:
                        aligned.append(word_labels[word_id])
                    else:
                        aligned.append(-100)
                    previous = word_id
                return aligned

            aligned = align_first_subtoken(word_labels, word_ids)
            print(aligned)
            assert aligned == [-100, 0, 1, -100, 0, -100]
            """
        ),
        md(
            """
            ### Ejercicio 1

            Implementa una segunda política que propague `I-DISEASE` a subtokens
            interiores. Compara ventajas y asegúrate de que evaluación reconstruye
            offsets de caracteres, no solo IDs de subtokens.
            """
        ),
        md(
            """
            ## 4. Catálogo BSC/PlanTL orientativo

            Revisa siempre la tarjeta vigente.
            """
        ),
        code(
            """
            bsc_catalog = pd.DataFrame([
                {
                    "model": "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
                    "kind": "encoder MLM",
                    "languages": "es",
                    "context": "revisar tarjeta/config",
                    "direct_task": "fill-mask",
                    "clinical_use": "base para ajuste local",
                },
                {
                    "model": "PlanTL-GOB-ES/longformer-base-4096-biomedical-clinical-es",
                    "kind": "encoder MLM largo",
                    "languages": "es",
                    "context": "4096",
                    "direct_task": "fill-mask",
                    "clinical_use": "comparar documentos largos",
                },
                {
                    "model": "BSC-LT/MrBERT-biomed",
                    "kind": "encoder MLM 2026",
                    "languages": "principalmente en/es",
                    "context": "8192 declarado",
                    "direct_task": "fill-mask",
                    "clinical_use": "ajuste/recuperación a evaluar",
                },
                {
                    "model": "BSC-NLP4BIA/bsc-bio-ehr-es-distemist",
                    "kind": "token classification",
                    "languages": "es",
                    "context": "revisar tarjeta",
                    "direct_task": "NER enfermedad",
                    "clinical_use": "baseline/candidatos",
                },
                {
                    "model": "Clinical NMT-NER collection",
                    "kind": "token classification",
                    "languages": "es/ca",
                    "context": "según modelo",
                    "direct_task": "enfermedad/fármaco/profesión",
                    "clinical_use": "preanotación evaluada",
                },
                {
                    "model": "DT4H multilingual NER",
                    "kind": "NER STL/MTL",
                    "languages": "7, incluido es",
                    "context": "según modelo",
                    "direct_task": "enfermedad/síntoma/procedimiento",
                    "clinical_use": "baseline multilingüe",
                },
            ])
            bsc_catalog
            """
        ),
        md(
            """
            Fuentes:

            - [RoBERTa biomédico-clínico](https://huggingface.co/PlanTL-GOB-ES/roberta-base-biomedical-clinical-es)
            - [MrBERT-biomed](https://huggingface.co/BSC-LT/MrBERT-biomed)
            - [Clinical NMT-NER](https://huggingface.co/collections/BSC-NLP4BIA/clinical-nmt-ner)
            - [DT4H](https://huggingface.co/BSC-NLP4BIA/DT4H_XLM-R_mtl_multilingual_multilabel)

            Algunas variantes DT4H multitarea requieren arquitectura y script
            personalizados; la tarjeta indica que no se cargan directamente con
            `AutoModelForTokenClassification`.
            """
        ),
        md(
            """
            ## 5. Auditoría de model card

            Para cada candidato registra:

            - propietario, versión y licencia;
            - objetivo de preentrenamiento;
            - idiomas y distribución;
            - corpus, limpieza y posible solapamiento;
            - etiquetas y tarea;
            - longitud máxima;
            - arquitectura de carga;
            - métricas y conjunto;
            - sesgos/limitaciones;
            - uso previsto;
            - hardware.
            """
        ),
        code(
            """
            model_audit_template = {
                "model_id": "",
                "revision": "",
                "license": "",
                "base_or_finetuned": "",
                "pretraining_objective": "",
                "training_languages": {},
                "task_labels": [],
                "max_length": None,
                "evaluation_datasets": [],
                "load_class": "",
                "limitations": [],
                "local_validation_required": True,
            }
            model_audit_template
            """
        ),
        md(
            """
            ### Ejercicio 2 · Comparar dos modelos

            Completa la plantilla para RoBERTa biomédico-clínico y
            MrBERT-biomed. Decide cuál probarías para:

            1. NER español de ERC;
            2. recuperación bilingüe;
            3. curso de 3000 tokens;
            4. CPU con poca memoria.

            Puede no haber un ganador único. Formula el experimento que decidiría.
            """
        ),
        md(
            """
            ## 6. Cargar un NER ya ajustado — opcional

            Solo dentro de infraestructura autorizada. La descarga requiere
            `transformers`, PyTorch, red y espacio.
            """
        ),
        code(
            """
            RUN_BSC_NER = False

            if RUN_BSC_NER:
                from transformers import pipeline

                ner_pipe = pipeline(
                    "token-classification",
                    model="BSC-NLP4BIA/bsc-bio-ehr-es-distemist",
                    aggregation_strategy="simple",
                )
                synthetic = "ERC G4 por nefropatía diabética. Inicia hemodiálisis."
                print(ner_pipe(synthetic))
            """
        ),
        md(
            """
            Cuando lo actives:

            - guarda revisión/commit del modelo;
            - conserva offsets;
            - convierte etiquetas a tu esquema;
            - evalúa en test local;
            - inspecciona negados/familiares;
            - no uses el score como probabilidad clínica calibrada.
            """
        ),
        md(
            """
            ## 7. Ajuste de token classification — diseño

            Ruta:

            ```text
            offsets adjudicados
              → palabras/labels
              → fast tokenizer + word_ids
              → alineación -100
              → DataCollatorForTokenClassification
              → AutoModelForTokenClassification
              → Trainer
              → offsets reconstruidos
              → exact/overlap F1
            ```
            """
        ),
        code(
            """
            RUN_TRANSFORMER_TRAINING = False

            if RUN_TRANSFORMER_TRAINING:
                from datasets import Dataset
                from transformers import (
                    AutoModelForTokenClassification,
                    AutoTokenizer,
                    DataCollatorForTokenClassification,
                    Trainer,
                    TrainingArguments,
                )

                base_model = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
                tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
                model = AutoModelForTokenClassification.from_pretrained(
                    base_model,
                    num_labels=3,
                    id2label={0: "O", 1: "B-DISEASE", 2: "I-DISEASE"},
                    label2id={"O": 0, "B-DISEASE": 1, "I-DISEASE": 2},
                )
                # Completar: Dataset, tokenización/alineación, argumentos,
                # compute_metrics, entrenamiento y evaluación bloqueada.
            """
        ),
        md(
            """
            El esqueleto incompleto es deliberado: copiar un `Trainer` sin
            entender datos y alineación produce métricas engañosas.

            Documentación primaria:
            [Hugging Face token classification](https://huggingface.co/docs/transformers/tasks/token_classification).
            """
        ),
        md(
            """
            ## 8. Contexto largo

            Truncar puede eliminar la evidencia. Alternativas:

            - procesar secciones;
            - ventanas con solapamiento;
            - modelo de contexto largo;
            - recuperación previa;
            - agregación jerárquica.

            Compara, no supongas que más contexto mejora.
            """
        ),
        code(
            """
            def sliding_word_windows(words, *, size=8, overlap=2):
                if overlap >= size:
                    raise ValueError("overlap debe ser menor que size")
                step = size - overlap
                return [
                    {
                        "start_word": start,
                        "end_word": min(start + size, len(words)),
                        "text": " ".join(words[start:start + size]),
                    }
                    for start in range(0, len(words), step)
                ]

            long_text = (
                "Antecedentes ERC G5 sin TRS previa Evolución clínica estable "
                "Plan iniciar hemodiálisis mediante catéter tunelizado mañana"
            )
            sliding_word_windows(long_text.split(), size=7, overlap=2)
            """
        ),
        md(
            """
            ### Ejercicio 3

            Mueve la evidencia hasta un límite de ventana. ¿Se duplica? ¿Cómo
            fusionas spans? ¿Cómo recuperas offsets globales? Diseña tests para:

            - entidad partida;
            - duplicada;
            - contradicción entre secciones;
            - fecha lejana.
            """
        ),
        md(
            """
            ## 9. Comparación justa

            Mantén constantes:

            - train/dev/test;
            - guía y etiquetas;
            - preprocesado;
            - presupuesto de búsqueda;
            - semillas repetidas;
            - métrica y postprocesado;
            - unidad de hardware/latencia;
            - criterios de selección.

            Comparar tu mejor transformer contra un baseline sin ajustar no
            responde qué método es mejor.
            """
        ),
        code(
            """
            experiment_registry = pd.DataFrame(columns=[
                "run_id", "model_revision", "data_version", "seed",
                "learning_rate", "epochs", "dev_f1", "test_f1",
                "latency_ms", "notes",
            ])
            experiment_registry
            """
        ),
        md(
            """
            ## 10. Recursos y reproducibilidad

            Registra:

            - CPU/GPU y memoria;
            - precisión numérica;
            - batch y acumulación;
            - duración;
            - emisiones/coste cuando proceda;
            - checkpoints;
            - versiones de CUDA/PyTorch/transformers;
            - semillas;
            - fallos y reinicios.

            Un checkpoint sin tokenizador, label map y config no es suficiente.
            """
        ),
        md(
            """
            ## Reto integrador

            Diseña —sin ejecutar aún— un estudio que compare:

            1. diccionario;
            2. NER BSC ya ajustado;
            3. spaCy NER;
            4. RoBERTa clínico ajustado;
            5. MrBERT-biomed ajustado.

            Especifica datos, métricas, tres semillas, hardware, subgrupos,
            criterio de selección y análisis de errores.
            """
        ),
        md(
            """
            ## Criterio para avanzar

            Debes poder explicar:

            - MLM frente a token classification;
            - palabra frente a subtoken;
            - por qué `-100`;
            - modelo base frente a modelo ajustado;
            - problemas del contexto largo;
            - por qué una model card no sustituye validación local;
            - qué hace justa una comparación.
            """
        ),
    ]
