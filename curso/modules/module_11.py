from curso.notebook_factory import code, common_setup, md


TITLE = "11 · Anotación clínica asistida con Doccano"


def build() -> list[dict]:
    return [
        md(
            """
            # 11 · Anotación clínica asistida con Doccano

            Usaremos una sola plataforma. Doccano ofrece una interfaz sencilla
            para clínicos y permite importar sugerencias creadas por reglas,
            spaCy, modelos BSC o un LLM local autorizado.

            La herramienta no crea la verdad de referencia: organiza el trabajo
            humano y conserva sus decisiones.
            """
        ),
        md(
            """
            ## Objetivos

            - convertir una guía clínica en un proyecto Doccano pequeño;
            - preparar JSONL con offsets comprobados;
            - diseñar doble anotación y aprobación;
            - añadir sugerencias sin confundirlas con gold;
            - medir aceptación, omisiones y ahorro de tiempo;
            - exportar a spaCy;
            - entender el contrato de una API de autoetiquetado;
            - diseñar un piloto seguro para usuarios no técnicos.
            """
        ),
        common_setup(),
        code(
            """
            import json

            import pandas as pd
            import spacy
            from spacy.tokens import DocBin

            from clinical_nlp_course import (
                annotation_edit_stats,
                dictionary_suggestions,
                exact_span_metrics,
                make_doccano_record,
                validate_doccano_labels,
            )

            concepts = json.loads(
                (PROJECT_ROOT / "data" / "conceptos_renales_sinteticos.json")
                .read_text(encoding="utf-8")
            )
            courses = pd.read_json(
                PROJECT_ROOT / "data" / "nefrologia_sintetica.jsonl",
                lines=True,
            )
            print(len(courses), "cursos ficticios")
            """
        ),
        md(
            """
            ## 1. Decisión de alcance

            Para identificar pacientes con miopía, comienza con dos proyectos
            dentro de Doccano:

            1. **Sequence labeling:** evidencia literal como `MYOPIA`,
               `REFRACTION` y `REFRACTIVE_SURGERY`.
            2. **Text classification:** `PRESENT`, `ABSENT`, `UNCERTAIN` o
               `INSUFFICIENT`.

            El primero enseña qué dice el texto; el segundo responde la pregunta
            documental. No intentes codificar todas las combinaciones de
            negación, sujeto y temporalidad en decenas de etiquetas.
            """
        ),
        code(
            """
            project_spec = {
                "platform": "doccano",
                "task": "SequenceLabeling",
                "collaborative_annotation": True,
                "allow_overlapping": False,
                "labels": [concept["concept_id"] for concept in concepts],
                "guideline_version": "renal-annotation-v1",
                "double_annotation_fraction": 1.0,
                "approver_required": True,
            }
            assert project_spec["collaborative_annotation"]
            assert project_spec["approver_required"]
            project_spec
            """
        ),
        md(
            """
            ### Ejercicio 1 · Especificación mínima

            Escribe la especificación equivalente para el primer proyecto de
            miopía. Limita el NER a tres etiquetas y la clasificación a cuatro
            estados. Para cada etiqueta añade una inclusión, una exclusión y un
            caso dudoso.
            """
        ),
        code(
            """
            myopia_project_spec = {
                "ner_labels": [],
                "document_labels": [],
                "guideline_version": "",
            }
            print("Completa:", myopia_project_spec)
            """
        ),
        md(
            """
            ## 2. Un contrato, no un archivo improvisado

            El registro conserva texto inmutable, identificador documental,
            spans `[start, end, label]`, indicador de preanotación y versión del
            generador. La procedencia separa una sugerencia del juicio humano.
            """
        ),
        code(
            """
            sample_text = (
                "Se explican opciones de hemodiálisis. "
                "Finalmente inicia hemodiálisis mediante CVC tunelizado."
            )
            suggestions = dictionary_suggestions(sample_text, concepts)
            for suggestion in suggestions:
                recovered = sample_text[suggestion["start"]:suggestion["end"]]
                print(suggestion["label"], repr(recovered))
                assert recovered == suggestion["evidence"]

            # La primera mención está dentro de una exclusión; la segunda se sugiere.
            hd = [item for item in suggestions if item["label"] == "HEMODIALYSIS"]
            assert len(hd) == 1
            """
        ),
        code(
            """
            sample_record = make_doccano_record(
                document_id="DEMO-001",
                text=sample_text,
                suggestions=suggestions,
                suggestion_version="renal-dictionary-v1",
            )
            print(json.dumps(sample_record, ensure_ascii=False, indent=2))
            assert validate_doccano_labels(
                sample_record["text"],
                sample_record["labels"],
            ) == []
            """
        ),
        md(
            """
            ### Error intencionado

            Cambiar el texto después de calcular offsets corrompe el corpus.
            Predice qué problemas detectará la siguiente celda.
            """
        ),
        code(
            """
            altered_text = "NOTA: " + sample_record["text"]
            offset_problems = validate_doccano_labels(
                altered_text,
                sample_record["labels"],
            )
            # Los rangos siguen siendo válidos: este control no sabe que cambió
            # el texto. Necesitamos hash y versionado.
            print("Problemas estructurales:", offset_problems)
            first_label = sample_record["labels"][0]
            print("Evidencia ahora:", altered_text[first_label[0]:first_label[1]])
            """
        ),
        md(
            """
            **Lección:** validar rangos es necesario, pero no suficiente. Congela
            el texto y guarda un hash del documento antes de preanotar.
            """
        ),
        md(
            """
            ## 3. Preparar un lote

            El script `scripts/preparar_doccano.py` ejecuta este proceso sobre
            todo el corpus ficticio. Aquí lo reproducimos en memoria para
            comprobarlo sin instalar Doccano.
            """
        ),
        code(
            """
            records = []
            for row in courses.head(10).to_dict("records"):
                predicted = dictionary_suggestions(row["text"], concepts)
                records.append(
                    make_doccano_record(
                        document_id=row["course_id"],
                        text=row["text"],
                        suggestions=predicted,
                        suggestion_version="renal-dictionary-v1",
                    )
                )

            jsonl_preview = "\\n".join(
                json.dumps(record, ensure_ascii=False) for record in records[:2]
            )
            print(jsonl_preview)
            assert all(
                validate_doccano_labels(record["text"], record["labels"]) == []
                for record in records
            )
            """
        ),
        md(
            """
            ## 4. Fuentes de sugerencias

            Añádelas y versiónalas por etapas: diccionario/reglas, NER spaCy,
            modelo BSC validado y, finalmente, LLM local para casos complejos.

            No mezcles varias fuentes antes de medir cada una. Toda sugerencia
            conserva `source`, versión y evidencia recuperable.
            """
        ),
        code(
            """
            suggestion_registry = pd.DataFrame([
                {
                    "version": "renal-dictionary-v1",
                    "source": "dictionary",
                    "status": "pilot",
                    "external_text_transfer": False,
                },
                {
                    "version": "renal-spacy-v1",
                    "source": "spacy_ner",
                    "status": "future",
                    "external_text_transfer": False,
                },
                {
                    "version": "renal-llm-local-v1",
                    "source": "llm",
                    "status": "future_after_validation",
                    "external_text_transfer": False,
                },
            ])
            suggestion_registry
            """
        ),
        md(
            """
            ## 5. Adaptar una salida de LLM

            Pide evidencia literal y estructura. Localiza después la evidencia
            de forma determinista. Si aparece cero o varias veces, abstente en
            lugar de inventar un offset.
            """
        ),
        code(
            """
            def llm_item_to_suggestion(text, item):
                evidence = item.get("evidence", "")
                starts = []
                cursor = 0
                while evidence:
                    position = text.find(evidence, cursor)
                    if position < 0:
                        break
                    starts.append(position)
                    cursor = position + 1
                if len(starts) != 1:
                    return None
                start = starts[0]
                return {
                    "start": start,
                    "end": start + len(evidence),
                    "label": item["concept"],
                    "evidence": evidence,
                    "source": "llm_local",
                }

            llm_text = "Continúa hemodiálisis mediante FAV."
            mock_llm_output = {
                "concept": "HEMODIALYSIS",
                "evidence": "hemodiálisis",
            }
            llm_suggestion = llm_item_to_suggestion(llm_text, mock_llm_output)
            assert llm_suggestion is not None
            assert (
                llm_text[llm_suggestion["start"]:llm_suggestion["end"]]
                == "hemodiálisis"
            )
            llm_suggestion
            """
        ),
        md(
            """
            ### Ejercicio 2 · Ambigüedad

            Con “hemodiálisis previa; reinicia hemodiálisis”, el adaptador debe
            abstenerse. Diseña un contrato que incluya contexto izquierdo y
            derecho para desambiguar sin aceptar offsets inventados.
            """
        ),
        code(
            """
            ambiguous_text = "hemodiálisis previa; reinicia hemodiálisis."
            ambiguous = llm_item_to_suggestion(
                ambiguous_text,
                {"concept": "HEMODIALYSIS", "evidence": "hemodiálisis"},
            )
            assert ambiguous is None
            """
        ),
        md(
            """
            ## 6. Conexión con Doccano

            Doccano puede llamar a una API REST propia y mapear la respuesta:

            ```json
            [{"label": "HEMODIALYSIS", "start_offset": 10, "end_offset": 22}]
            ```

            El endpoint vive junto al modelo en infraestructura autorizada.
            [Auto Labeling oficial](https://doccano.github.io/doccano/advanced/auto_labelling_config/)
            """
        ),
        code(
            """
            def to_autolabel_api_response(suggestions):
                return [
                    {
                        "label": item["label"],
                        "start_offset": item["start"],
                        "end_offset": item["end"],
                    }
                    for item in suggestions
                ]

            api_response = to_autolabel_api_response(suggestions)
            assert all(
                {"label", "start_offset", "end_offset"} == set(item)
                for item in api_response
            )
            api_response
            """
        ),
        md(
            """
            ## 7. Trabajo humano

            A y B anotan independientemente, ambos terminan y el aprobador
            revisa discrepancias. Se exportan las versiones originales y la
            aprobada. No muestres a A las decisiones de B.
            """
        ),
        code(
            """
            suggested_labels = [[0, 12, "HEMODIALYSIS"], [22, 25, "AV_FISTULA"]]
            annotator_a = [[0, 12, "HEMODIALYSIS"], [22, 25, "AV_FISTULA"]]
            annotator_b = [[0, 12, "HEMODIALYSIS"]]

            print("A:", annotation_edit_stats(suggested_labels, annotator_a))
            print("B:", annotation_edit_stats(suggested_labels, annotator_b))
            agreement = exact_span_metrics(
                {"D1": annotator_a},
                {"D1": annotator_b},
            )
            print("Acuerdo exacto A/B:", agreement)
            """
        ),
        md(
            """
            Una aceptación alta no demuestra que el modelo sea correcto: puede
            reflejar buenas sugerencias, prisa o automatización. Revisa omisiones
            y mantén una muestra manual.
            """
        ),
        md(
            """
            ## 8. Evaluar el beneficio

            Aleatoriza documentos entre brazo manual y asistido. Mide tiempo,
            exactitud adjudicada, entidades añadidas, eliminadas y esfuerzo. El
            objetivo es mantener calidad reduciendo carga.
            """
        ),
        code(
            """
            pilot = pd.DataFrame([
                {"arm": "manual", "minutes": 4.8, "errors": 1},
                {"arm": "manual", "minutes": 5.2, "errors": 0},
                {"arm": "manual", "minutes": 4.5, "errors": 1},
                {"arm": "assisted", "minutes": 3.0, "errors": 1},
                {"arm": "assisted", "minutes": 2.8, "errors": 0},
                {"arm": "assisted", "minutes": 3.4, "errors": 2},
            ])
            pilot.groupby("arm").agg(
                median_minutes=("minutes", "median"),
                total_errors=("errors", "sum"),
                n=("errors", "size"),
            )
            """
        ),
        md(
            """
            Seis ejemplos no permiten concluir. Redacta un protocolo A/B con
            unidad de aleatorización, resultado primario, familiarización y
            criterio para retirar las sugerencias.
            """
        ),
        md(
            """
            ## 9. Exportar a spaCy

            Tras aprobación, convierte offsets a `DocBin`. Rechaza spans que no
            respeten límites de tokenización.
            """
        ),
        code(
            """
            nlp = spacy.blank("es")
            doc_bin = DocBin()
            rejected = []

            for record in records:
                doc = nlp.make_doc(record["text"])
                spans = []
                for start, end, label in record["labels"]:
                    span = doc.char_span(start, end, label=label, alignment_mode="strict")
                    if span is None:
                        rejected.append(
                            (record["meta"]["document_id"], start, end, label)
                        )
                    else:
                        spans.append(span)
                doc.ents = spans
                doc_bin.add(doc)

            print("Bytes DocBin:", len(doc_bin.to_bytes()))
            print("Rechazados:", rejected)
            assert not rejected
            """
        ),
        md(
            """
            ## 10. Operación sencilla

            La persona técnica crea proyecto, etiquetas y usuarios; importa,
            versiona, exporta, respalda y ejecuta controles. Los clínicos solo
            inician sesión, corrigen sugerencias, añaden omisiones, marcan dudas
            y terminan el documento.

            Consulta `docs/doccano_operativo.md`.
            """
        ),
        md(
            """
            ## Práctica final

            1. Ejecuta `scripts/preparar_doccano.py`.
            2. Instala Doccano solo con datos ficticios.
            3. Importa el JSONL.
            4. Configura dos anotadores y un aprobador.
            5. Anota diez documentos en brazos manual y asistido.
            6. Exporta, valida offsets y calcula acuerdo.
            7. Escribe tres mejoras de la guía.
            """
        ),
        md(
            """
            ## Criterio para completar el módulo

            Debes explicar por qué elegimos Doccano, diferencia entre sugerencia
            y gold, procedencia, offsets de LLM, sesgo de automatización,
            conversión a spaCy y controles previos al texto hospitalario.
            """
        ),
    ]
