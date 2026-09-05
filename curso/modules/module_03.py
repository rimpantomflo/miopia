from curso.notebook_factory import code, common_setup, md

TITLE = "03 · Entrenar un NER con spaCy"


def build() -> list[dict]:
    return [
        md(
            """
            # 03 · Entrenar un NER con spaCy

            Entrenaremos un NER diminuto con frases ficticias. El objetivo no es
            obtener rendimiento clínico: es comprender datos, etiquetas,
            inicialización, optimización, evaluación, error y serialización.
            """
        ),
        md(
            """
            ## Objetivos

            - convertir offsets a ejemplos de spaCy;
            - comprender BILUO y entidades no solapadas;
            - crear `EntityRecognizer`;
            - inicializar y actualizar el modelo;
            - observar pérdida sin confundirla con validación;
            - evaluar spans exactos;
            - analizar errores;
            - serializar/reproducir;
            - saber cuándo usar `SpanCategorizer`.
            """
        ),
        common_setup(),
        code(
            """
            import random

            import pandas as pd
            import spacy
            from spacy.training import Example, offsets_to_biluo_tags
            from spacy.util import minibatch, fix_random_seed

            from clinical_nlp_course import exact_span_metrics, overlap_span_metrics

            fix_random_seed(13)
            """
        ),
        md(
            """
            ## 1. Formato de entrenamiento

            Cada ejemplo tiene texto y triples `(start, end, label)`. Creamos los
            offsets buscando frases porque son datos sintéticos; en corpus reales
            proceden de anotación.
            """
        ),
        code(
            """
            def one_entity(text, phrase, label):
                start = text.index(phrase)
                return (text, {"entities": [(start, start + len(phrase), label)]})

            train_data = [
                one_entity("Paciente con enfermedad renal crónica G4.", "enfermedad renal crónica", "DISEASE"),
                one_entity("Diagnóstico de nefropatía IgA confirmada.", "nefropatía IgA", "DISEASE"),
                one_entity("Presenta fracaso renal agudo.", "fracaso renal agudo", "DISEASE"),
                one_entity("Antecedente de poliquistosis renal.", "poliquistosis renal", "DISEASE"),
                one_entity("En hemodiálisis desde enero.", "hemodiálisis", "PROCEDURE"),
                one_entity("Recibe diálisis peritoneal automatizada.", "diálisis peritoneal", "PROCEDURE"),
                one_entity("Trasplante renal funcionante.", "Trasplante renal", "PROCEDURE"),
                one_entity("Se crea fístula arteriovenosa.", "fístula arteriovenosa", "PROCEDURE"),
                one_entity("Tratamiento con tacrolimus.", "tacrolimus", "DRUG"),
                one_entity("Se ajusta furosemida.", "furosemida", "DRUG"),
                one_entity("Suspender ibuprofeno.", "ibuprofeno", "DRUG"),
                one_entity("Mantiene prednisona 5 mg.", "prednisona", "DRUG"),
                ("Función renal estable.", {"entities": []}),
                ("Control analítico en tres meses.", {"entities": []}),
                ("Niega síntomas urinarios.", {"entities": []}),
            ]

            dev_data = [
                one_entity("ERC avanzada en seguimiento.", "ERC", "DISEASE"),
                one_entity("Inicia HD trisemanal.", "HD", "PROCEDURE"),
                one_entity("Continúa con ciclosporina.", "ciclosporina", "DRUG"),
                one_entity("Nefropatía diabética conocida.", "Nefropatía diabética", "DISEASE"),
                ("Sin cambios terapéuticos.", {"entities": []}),
            ]
            print(len(train_data), len(dev_data))
            """
        ),
        md(
            """
            ### Advertencia

            Quince frases no bastan para un NER clínico. Se usan para observar el
            mecanismo y sobreajustar deliberadamente. Un modelo puede memorizar
            términos sin aprender variabilidad.
            """
        ),
        md(
            """
            ## 2. Alineación BILUO

            Revisa todos los ejemplos antes de entrenar. Una etiqueta `-` indica
            offsets que no coinciden con límites de token.
            """
        ),
        code(
            """
            alignment_nlp = spacy.blank("es")
            for text, annotations in train_data[:5]:
                doc = alignment_nlp.make_doc(text)
                tags = offsets_to_biluo_tags(doc, annotations["entities"])
                print(list(zip([token.text for token in doc], tags)))
            """
        ),
        code(
            """
            for text, annotations in train_data + dev_data:
                doc = alignment_nlp.make_doc(text)
                tags = offsets_to_biluo_tags(doc, annotations["entities"])
                assert "-" not in tags, (text, tags)
            print("Todos los offsets se alinean.")
            """
        ),
        md(
            """
            ### Ejercicio 1

            Desplaza un offset un carácter y observa `-`. Explica por qué
            `alignment_mode="expand"` puede ser útil para inspección, pero no debe
            ocultar automáticamente anotaciones defectuosas.
            """
        ),
        md(
            """
            ## 3. Crear `Example`

            `Example` contiene un `Doc` predicho y otro de referencia alineado.
            """
        ),
        code(
            """
            preview_text, preview_annotations = train_data[0]
            preview_example = Example.from_dict(
                alignment_nlp.make_doc(preview_text),
                preview_annotations,
            )
            print("Predicho:", preview_example.predicted.ents)
            print("Referencia:", [(ent.text, ent.label_) for ent in preview_example.reference.ents])
            """
        ),
        md(
            """
            ## 4. Crear e inicializar el NER

            Añadimos etiquetas antes de `initialize`. La inicialización necesita
            ejemplos para inferir dimensiones.
            """
        ),
        code(
            """
            ner_nlp = spacy.blank("es")
            ner = ner_nlp.add_pipe("ner")
            labels = sorted({
                label
                for _, annotations in train_data
                for _, _, label in annotations["entities"]
            })
            for label in labels:
                ner.add_label(label)

            def make_examples(data):
                return [
                    Example.from_dict(ner_nlp.make_doc(text), annotations)
                    for text, annotations in data
                ]

            optimizer = ner_nlp.initialize(get_examples=lambda: make_examples(train_data))
            print("Etiquetas:", ner.labels)
            """
        ),
        md(
            """
            ## 5. Entrenamiento

            Una época recorre entrenamiento. `drop` regulariza. La pérdida guía
            optimización, pero no dice por sí sola si el modelo generaliza.
            """
        ),
        code(
            """
            history = []
            rng = random.Random(13)
            for epoch in range(25):
                shuffled = list(train_data)
                rng.shuffle(shuffled)
                losses = {}
                for batch in minibatch(shuffled, size=4):
                    ner_nlp.update(
                        make_examples(batch),
                        sgd=optimizer,
                        drop=0.20,
                        losses=losses,
                    )
                history.append({"epoch": epoch + 1, "ner_loss": losses.get("ner", 0.0)})

            pd.DataFrame(history).tail(10)
            """
        ),
        md(
            """
            Si la pérdida baja a casi cero, probablemente el modelo ha memorizado
            este corpus. Eso era esperable; ahora medimos desarrollo.
            """
        ),
        md(
            """
            ## 6. Evaluación de spans

            Convertimos gold y predicciones a offsets. El test real se mantendría
            bloqueado; aquí usamos desarrollo didáctico.
            """
        ),
        code(
            """
            def gold_mapping(data):
                return {
                    f"D{index}": [
                        (start, end, label)
                        for start, end, label in annotations["entities"]
                    ]
                    for index, (_, annotations) in enumerate(data)
                }

            def prediction_mapping(data, model):
                output = {}
                for index, (text, _) in enumerate(data):
                    doc = model(text)
                    output[f"D{index}"] = [
                        (ent.start_char, ent.end_char, ent.label_)
                        for ent in doc.ents
                    ]
                return output

            dev_gold = gold_mapping(dev_data)
            dev_pred = prediction_mapping(dev_data, ner_nlp)
            print("Exacto:", exact_span_metrics(dev_gold, dev_pred))
            print("Solapado:", overlap_span_metrics(dev_gold, dev_pred))
            """
        ),
        code(
            """
            error_rows = []
            for index, (text, annotations) in enumerate(dev_data):
                doc = ner_nlp(text)
                error_rows.append({
                    "text": text,
                    "gold": [
                        (text[start:end], label)
                        for start, end, label in annotations["entities"]
                    ],
                    "pred": [(ent.text, ent.label_) for ent in doc.ents],
                })
            pd.DataFrame(error_rows)
            """
        ),
        md(
            """
            ### Ejercicio 2

            Clasifica cada error:

            - término no visto;
            - abreviatura;
            - límite;
            - etiqueta;
            - falso positivo;
            - corpus insuficiente.

            No añadas el ejemplo de desarrollo a entrenamiento y vuelvas a
            publicar la misma métrica como si siguiera siendo desarrollo.
            """
        ),
        md(
            """
            ## 7. Inferencia y contexto

            El NER detecta menciones. No hemos entrenado negación, temporalidad ni
            experienciador.
            """
        ),
        code(
            """
            challenge = [
                "Madre con enfermedad renal crónica.",
                "No presenta fracaso renal agudo.",
                "Candidato a trasplante renal.",
            ]
            for text in challenge:
                doc = ner_nlp(text)
                print(text, "→", [(ent.text, ent.label_) for ent in doc.ents])
            """
        ),
        md(
            """
            Un span correcto en esas frases no implica un fenotipo positivo.
            Añade atributos con reglas/modelos separados y evalúalos por separado.
            """
        ),
        md(
            """
            ## 8. Spans solapados

            `EntityRecognizer` optimiza entidades no solapadas. Si necesitas
            `enfermedad renal` dentro de `enfermedad renal crónica`, o varias
            etiquetas sobre el mismo fragmento, estudia `SpanCategorizer` y
            `Doc.spans`.
            """
        ),
        md(
            """
            ## 9. Serialización

            Un modelo reproducible incluye pipeline, vocabulario y configuración.
            """
        ),
        code(
            """
            model_bytes = ner_nlp.to_bytes()
            restored_nlp = spacy.blank("es")
            restored_nlp.add_pipe("ner")
            restored_nlp.from_bytes(model_bytes)
            sample = "Tratamiento con tacrolimus."
            original = [(ent.text, ent.label_) for ent in ner_nlp(sample).ents]
            restored = [(ent.text, ent.label_) for ent in restored_nlp(sample).ents]
            print(original, restored)
            assert original == restored
            """
        ),
        md(
            """
            ## 10. Configuración de proyectos reales

            Para experimentos serios usa:

            - archivos `.spacy`;
            - `config.cfg`;
            - `spacy debug data`;
            - `spacy train`;
            - semillas;
            - directorio de salida versionado;
            - métrica en desarrollo;
            - test separado.

            La API Python de este notebook muestra el mecanismo; la CLI/config
            mejora reproducibilidad.
            """
        ),
        md(
            """
            ### Ejercicio 3 · Diseñar una ampliación

            Añade `LAB_TEST` con creatinina, eGFR y albuminuria:

            1. escribe guía de límites;
            2. añade 10 ejemplos y negativos;
            3. revisa alineación;
            4. reentrena con nueva semilla;
            5. evalúa por etiqueta;
            6. compara errores.

            No mezcles el valor numérico en el span si después quieres extraerlo
            como entidad relacionada.
            """
        ),
        md(
            """
            ## Model card mínima

            Completa [plantilla_model_card.md](../docs/plantilla_model_card.md):
            uso previsto, datos, etiquetas, entrenamiento, evaluación,
            limitaciones y usos excluidos.
            """
        ),
        md(
            """
            ## Criterio para avanzar

            Debes poder:

            - crear offsets y `Example`;
            - explicar BILUO;
            - inicializar y entrenar;
            - distinguir pérdida y F1;
            - producir una tabla de errores;
            - explicar por qué este modelo no es clínicamente válido;
            - elegir entre `EntityRecognizer` y `SpanCategorizer`.
            """
        ),
    ]
