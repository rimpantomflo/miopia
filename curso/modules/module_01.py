from curso.notebook_factory import code, common_setup, md


TITLE = "01 · Corpus, anotación y diccionarios clínicos"


def build() -> list[dict]:
    return [
        md(
            """
            # 01 · Corpus, anotación y diccionarios clínicos

            El rendimiento máximo de un sistema está limitado por la calidad de
            su pregunta y su referencia. Este módulo enseña a construir los datos
            antes de entrenar.
            """
        ),
        md(
            """
            ## Objetivos

            - distinguir corpus de desarrollo, referencia y test;
            - diseñar muestreo sin buscar solo la palabra objetivo;
            - anotar spans y atributos con offsets;
            - medir acuerdo exacto y por solapamiento;
            - adjudicar;
            - crear y validar un diccionario versionado;
            - separar por paciente;
            - serializar anotaciones con `DocBin`;
            - documentar una ficha de datos.
            """
        ),
        common_setup(),
        code(
            """
            import json
            import warnings
            from pathlib import Path

            import pandas as pd
            import spacy
            from spacy.tokens import DocBin
            from spacy.training import offsets_to_biluo_tags

            from clinical_nlp_course import (
                exact_span_metrics,
                overlap_span_metrics,
                patient_hash_split,
                validate_concept_dictionary,
            )

            corpus_path = PROJECT_ROOT / "data" / "nefrologia_sintetica.jsonl"
            concepts_path = PROJECT_ROOT / "data" / "conceptos_renales_sinteticos.json"

            renal_df = pd.read_json(corpus_path, lines=True)
            concepts = json.loads(concepts_path.read_text(encoding="utf-8"))
            print(len(renal_df), "documentos ficticios")
            print(len(concepts), "conceptos")
            """
        ),
        md(
            """
            ## 1. ¿Qué corpus?

            - **Corpus exploratorio:** descubre lenguaje y problemas.
            - **Corpus de entrenamiento:** ajusta parámetros.
            - **Desarrollo:** elige reglas, umbrales y modelos.
            - **Test bloqueado:** estima rendimiento al final.
            - **Temporal/externo:** prueba transporte.

            Un mismo documento no cambia de función según convenga al resultado.
            """
        ),
        code(
            """
            renal_df[["patient_id", "course_id", "date", "language", "text"]].head()
            """
        ),
        md(
            """
            ## 2. Muestreo y sesgo de selección

            Si seleccionas únicamente `WHERE text LIKE '%diálisis%'`, no podrás
            estudiar:

            - pacientes positivos sin esa palabra;
            - verdaderos negativos representativos;
            - abreviaturas;
            - prevalencia operativa;
            - errores del método de búsqueda.

            Combina muestra aleatoria y enriquecimiento documentado. Conserva el
            mecanismo de selección.
            """
        ),
        code(
            """
            lexical_enrichment = renal_df["text"].str.contains(
                "diálisis|dialisis|HD|HDF|DPA",
                case=False,
                regex=True,
            )
            pd.crosstab(
                lexical_enrichment,
                renal_df["gold_document_trs_evidence"],
                rownames=["seleccion_lexica"],
                colnames=["gold"],
            )
            """
        ),
        md(
            """
            ### Ejercicio 1

            Explica qué pacientes o cursos perdería la selección léxica. Diseña
            una estrategia que combine:

            - aleatorio;
            - procedimientos/códigos;
            - laboratorio;
            - términos de alta sensibilidad;
            - tipos documentales e idiomas.

            Escribe también qué prevalencia podrías estimar con cada muestra.
            """
        ),
        md(
            """
            ## 3. Esquema multidimensional

            Para una mención:

            ```text
            label + start/end + aserción + experienciador
                  + temporalidad + certeza + sección
            ```

            Para un evento:

            ```text
            concepto + estado + fecha + relaciones + evidencias
            ```

            No uses una clase gigante para todas las combinaciones.
            """
        ),
        code(
            """
            text = "Madre en hemodiálisis. Paciente sin tratamiento renal sustitutivo."
            start = text.index("hemodiálisis")
            annotation = {
                "document_id": "DEMO",
                "start": start,
                "end": start + len("hemodiálisis"),
                "label": "HEMODIALYSIS",
                "assertion": "affirmed",
                "experiencer": "family",
                "temporality": "current",
            }
            print(annotation)
            print("Evidencia recuperada:", text[annotation["start"]:annotation["end"]])
            """
        ),
        md(
            """
            ## 4. Offsets: el contrato

            Los offsets son caracteres Python sobre el texto exacto. Cambiar
            Unicode, espacios o saltos después de anotar rompe el contrato.

            Verifica siempre:

            ```python
            text[start:end] == evidence
            ```
            """
        ),
        code(
            """
            unicode_text = "HD con CVC − flujo 300 ml/min."
            for index, character in enumerate(unicode_text):
                print(index, repr(character))
            """
        ),
        md(
            """
            ### Ejercicio 2

            Anota manualmente `CVC` y `300 ml/min` en `unicode_text`. Recupera
            ambos con slices. Después inserta un espacio al inicio y observa por
            qué los offsets antiguos ya no sirven.
            """
        ),
        code(
            """
            cvc_start = unicode_text.index("CVC")
            flow_start = unicode_text.index("300")
            exercise_spans = [
                (cvc_start, cvc_start + len("CVC"), "ACCESS"),
                (flow_start, flow_start + len("300 ml/min"), "FLOW"),
            ]
            assert [unicode_text[s:e] for s, e, _ in exercise_spans] == [
                "CVC",
                "300 ml/min",
            ]
            exercise_spans
            """
        ),
        md(
            """
            ## 5. Tokenización y alineación

            El NER de spaCy necesita que los offsets se alineen con tokens.
            BILUO representa:

            - `B`: inicio;
            - `I`: interior;
            - `L`: último;
            - `U`: entidad de un token;
            - `O`: fuera.

            Un `-` indica desalineación y debe investigarse.
            """
        ),
        code(
            """
            nlp = spacy.blank("es")
            alignment_text = "Paciente en hemodiálisis domiciliaria."
            doc = nlp(alignment_text)
            start = alignment_text.index("hemodiálisis")
            entity_offsets = [(start, start + len("hemodiálisis domiciliaria"), "PROCEDURE")]
            tags = offsets_to_biluo_tags(doc, entity_offsets)
            list(zip([token.text for token in doc], tags))
            """
        ),
        code(
            """
            bad_offsets = [(start + 1, start + 7, "PROCEDURE")]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                bad_tags = offsets_to_biluo_tags(doc, bad_offsets)
            print(bad_tags)
            assert "-" in bad_tags
            """
        ),
        md(
            """
            ## 6. Acuerdo: exacto y solapado

            El acuerdo exacto exige mismos límites y etiqueta. El acuerdo por
            solapamiento distingue errores de frontera de conceptos totalmente
            ausentes. Informa ambos; no elijas el más favorable después.
            """
        ),
        code(
            """
            annotator_a = {
                "D1": [(12, 36, "DISEASE")],
                "D2": [(0, 13, "PROCEDURE")],
            }
            annotator_b = {
                "D1": [(12, 33, "DISEASE")],  # límite distinto
                "D2": [(0, 13, "PROCEDURE")],
            }

            print("Exacto:", exact_span_metrics(annotator_a, annotator_b))
            print("Solapado:", overlap_span_metrics(annotator_a, annotator_b))
            """
        ),
        md(
            """
            ### Ejercicio 3

            Añade:

            - una entidad con etiqueta distinta;
            - una omisión;
            - un falso positivo.

            Predice TP/FP/FN antes de ejecutar. Explica qué desacuerdo requiere
            cambiar la guía y cuál podría ser un error accidental.
            """
        ),
        md(
            """
            ## 7. Adjudicación

            La adjudicación no es «la persona senior elige». Debe registrar:

            - versiones enfrentadas;
            - decisión;
            - regla aplicada;
            - cambio de guía, si existe;
            - necesidad de reanotar;
            - adjudicador y fecha.

            Los desacuerdos son datos sobre la ambigüedad de la tarea.
            """
        ),
        code(
            """
            adjudication = pd.DataFrame([
                {
                    "document_id": "D1",
                    "issue": "incluir estadio en el span",
                    "annotator_a": "ERC G4",
                    "annotator_b": "ERC",
                    "decision": "span enfermedad=ERC; estadio como atributo separado",
                    "guide_change": "v1.1, regla 3.2",
                }
            ])
            adjudication
            """
        ),
        md(
            """
            ## 8. Diccionario clínico versionado

            Un diccionario debe indicar concepto, término preferido, variantes,
            tipo semántico, idioma, exclusiones, procedencia y versión.

            Abreviaturas como `DP`, `HD` o `TR` requieren contexto local. Una
            variante compartida entre conceptos es un conflicto, no una decisión
            silenciosa.
            """
        ),
        code(
            """
            dictionary_issues = validate_concept_dictionary(concepts)
            print("Problemas:", dictionary_issues)
            assert dictionary_issues == []

            pd.DataFrame(concepts)[
                ["concept_id", "preferred_term", "semantic_type", "version"]
            ]
            """
        ),
        code(
            """
            # Provocamos una ambigüedad para comprobar el validador.
            ambiguous = [dict(concept) for concept in concepts]
            ambiguous[0] = {**ambiguous[0], "variants": [*ambiguous[0]["variants"], "HD"]}
            [issue for issue in validate_concept_dictionary(ambiguous) if "ambigua" in issue]
            """
        ),
        md(
            """
            ### Ejercicio 4

            Añade un concepto `PROTEINURIA` con variantes y exclusiones. Decide si
            `ACR` es el concepto, una prueba, una medida o una abreviatura. No hay
            una respuesta universal: documenta el uso previsto.
            """
        ),
        md(
            """
            ## 9. Partición por paciente

            Una función hash crea una asignación estable. No garantiza equilibrio
            perfecto; para estudios reales se revisan clase y subgrupos sin mover
            pacientes después de observar resultados del modelo.
            """
        ),
        code(
            """
            renal_df["split"] = renal_df["patient_id"].map(patient_hash_split)
            assert renal_df.groupby("patient_id")["split"].nunique().max() == 1
            renal_df.groupby(["split", "gold_document_trs_evidence"]).size().unstack(fill_value=0)
            """
        ),
        md(
            """
            ## 10. `DocBin`

            `DocBin` serializa tokens y anotaciones de spaCy eficientemente. La
            demostración usa bytes en memoria; en un proyecto se guardan
            `train.spacy` y `dev.spacy` junto a metadatos y versión.
            """
        ),
        code(
            """
            doc_bin = DocBin(attrs=["ENT_IOB", "ENT_TYPE"])
            demo_doc = nlp("Paciente en hemodiálisis.")
            mention_start = demo_doc.text.index("hemodiálisis")
            span = demo_doc.char_span(
                mention_start,
                mention_start + len("hemodiálisis"),
                label="PROCEDURE",
                alignment_mode="strict",
            )
            assert span is not None
            demo_doc.ents = [span]
            doc_bin.add(demo_doc)

            payload = doc_bin.to_bytes()
            recovered = list(DocBin().from_bytes(payload).get_docs(nlp.vocab))
            print(len(payload), "bytes")
            print([(ent.text, ent.label_) for ent in recovered[0].ents])
            """
        ),
        md(
            """
            ## 11. Ficha de datos

            Completa [plantilla_ficha_datos.md](../docs/plantilla_ficha_datos.md):

            - población, periodo y fuentes;
            - muestreo y prevalencia;
            - anotación y acuerdo;
            - particiones;
            - privacidad;
            - limitaciones y subgrupos.

            Un archivo con textos sin esta información no es un corpus
            reproducible.
            """
        ),
        code(
            """
            dataset_summary = {
                "patients": int(renal_df["patient_id"].nunique()),
                "documents": int(len(renal_df)),
                "languages": renal_df["language"].value_counts().to_dict(),
                "positive_documents": int(renal_df["gold_document_trs_evidence"].sum()),
                "date_min": str(renal_df["date"].min()),
                "date_max": str(renal_df["date"].max()),
            }
            dataset_summary
            """
        ),
        md(
            """
            ## Comprobación final

            Sin mirar, explica:

            1. por qué buscar la palabra objetivo sesga el corpus;
            2. qué significa un offset desalineado;
            3. diferencia entre acuerdo exacto y solapado;
            4. por qué una variante ambigua debe detectarse;
            5. por qué paciente, no documento, controla la partición;
            6. qué información aporta una ficha de datos.

            Continúa cuando puedas construir una anotación válida y detectar sus
            problemas automáticamente.
            """
        ),
    ]
