from curso.notebook_factory import code, common_setup, md


TITLE = "10 · Evaluación de competencias y capstone"


def build() -> list[dict]:
    return [
        md(
            """
            # 10 · Evaluación de competencias y capstone

            Este notebook comprueba si puedes diseñar y defender un proyecto, no
            solo ejecutar funciones. Resuelve antes de abrir las soluciones.
            """
        ),
        md(
            """
            ## Estructura

            - Parte A: conceptos.
            - Parte B: código.
            - Parte C: diseño clínico.
            - Parte D: capstone.

            Puntuación orientativa: 100. Para considerar el curso adquirido,
            busca ≥80 y ningún fallo crítico de privacidad, fuga o interpretación.
            """
        ),
        common_setup(),
        code(
            """
            import json

            import pandas as pd
            import spacy

            from clinical_nlp_course import (
                TfidfRetriever,
                exact_span_metrics,
                overlap_span_metrics,
                patient_hash_split,
                validate_concept_dictionary,
                validate_llm_extraction,
            )
            """
        ),
        md(
            """
            ## Parte A · Conceptos (20 puntos)

            Responde en tu cuaderno:

            1. NER frente a fenotipo.
            2. Diccionario frente a terminología.
            3. Encoder frente a LLM generativo.
            4. Match exacto frente a solapado.
            5. Sensibilidad frente a VPP.
            6. Discriminación frente a calibración.
            7. Development frente a test.
            8. Recuperación frente a generación.
            9. Seudonimización frente a anonimización.
            10. Validación retrospectiva frente a utilidad clínica.

            Dos puntos por definición y ejemplo clínico correcto.
            """
        ),
        md(
            """
            <details>
            <summary>Pistas de corrección</summary>

            Cada respuesta debe incluir unidad, salida y una limitación. Por
            ejemplo, NER detecta un span; el fenotipo agrega evidencias y contexto
            para responder sobre paciente/episodio. Un span correcto no implica
            fenotipo correcto.
            </details>
            """
        ),
        md(
            """
            ## Parte B1 · Offsets (5 puntos)

            Implementa `locate_all(text, term)` que devuelva todas las
            coincidencias con `start`, `end` y evidencia, ignorando mayúsculas
            pero conservando offsets originales.
            """
        ),
        code(
            """
            def my_locate_all(text, term):
                # TODO: sustituye por tu implementación.
                return []

            offset_text = "HD previa. Actualmente continúa HD."
            expected_count = 2
            print("Tu resultado:", my_locate_all(offset_text, "HD"))
            """
        ),
        code(
            """
            # Solución de referencia: ejecútala después.
            import re

            def reference_locate_all(text, term):
                return [
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "evidence": match.group(),
                    }
                    for match in re.finditer(re.escape(term), text, flags=re.IGNORECASE)
                ]

            reference_offsets = reference_locate_all(offset_text, "HD")
            assert len(reference_offsets) == expected_count
            assert all(
                offset_text[row["start"]:row["end"]] == row["evidence"]
                for row in reference_offsets
            )
            reference_offsets
            """
        ),
        md(
            """
            ## Parte B2 · Diccionario (5 puntos)

            Crea dos conceptos válidos. Después introduce deliberadamente una
            variante ambigua y comprueba que el validador la detecta.
            """
        ),
        code(
            """
            my_concepts = []
            print("Problemas actuales:", validate_concept_dictionary(my_concepts))
            """
        ),
        code(
            """
            reference_concepts = [
                {
                    "concept_id": "HEMODIALYSIS",
                    "preferred_term": "hemodiálisis",
                    "variants": ["HD"],
                    "semantic_type": "procedure",
                    "version": "1.0",
                },
                {
                    "concept_id": "PERITONEAL_DIALYSIS",
                    "preferred_term": "diálisis peritoneal",
                    "variants": ["DPA"],
                    "semantic_type": "procedure",
                    "version": "1.0",
                },
            ]
            assert validate_concept_dictionary(reference_concepts) == []
            """
        ),
        md(
            """
            ## Parte B3 · NER (10 puntos)

            Calcula mentalmente y luego con código exact/overlap. Explica cada
            TP, FP y FN.
            """
        ),
        code(
            """
            gold = {
                "A": [(0, 25, "DISEASE")],
                "B": [(10, 22, "PROCEDURE")],
            }
            pred = {
                "A": [(0, 17, "DISEASE")],
                "B": [(10, 22, "DRUG")],
                "C": [(0, 3, "DISEASE")],
            }
            print("Exacto:", exact_span_metrics(gold, pred))
            print("Solapado:", overlap_span_metrics(gold, pred))
            """
        ),
        md(
            """
            ## Parte B4 · Partición (5 puntos)

            Demuestra que todos los cursos de cada paciente quedan juntos.
            """
        ),
        code(
            """
            split_demo = pd.DataFrame([
                {"patient_id": "P1", "course_id": "C1"},
                {"patient_id": "P1", "course_id": "C2"},
                {"patient_id": "P2", "course_id": "C3"},
                {"patient_id": "P3", "course_id": "C4"},
            ])
            split_demo["split"] = split_demo["patient_id"].map(patient_hash_split)
            assert split_demo.groupby("patient_id")["split"].nunique().max() == 1
            split_demo
            """
        ),
        md(
            """
            ## Parte B5 · LLM estructurado (10 puntos)

            Encuentra todos los defectos de la salida.
            """
        ),
        code(
            """
            llm_source = "No precisa hemodiálisis."
            candidate_output = {
                "concept": "HEMODIALYSIS",
                "assertion": "affirmed",
                "evidence": "hemodialisis",
                "start": 11,
                "end": 24,
            }
            print(validate_llm_extraction(candidate_output, llm_source))
            """
        ),
        md(
            """
            Además del JSON/offset, la aserción es clínicamente incorrecta. Un
            validador estructural no sustituye la referencia semántica.
            """
        ),
        md(
            """
            ## Parte B6 · RAG (5 puntos)

            Añade un documento irrelevante con palabras parecidas y observa el
            ranking. Propón una evaluación Recall@k.
            """
        ),
        code(
            """
            docs = [
                {"id": "HD", "text": "Hemodiálisis mediante fístula o catéter."},
                {"id": "DP", "text": "Diálisis peritoneal mediante Tenckhoff."},
                {"id": "TX", "text": "Seguimiento del injerto renal tras trasplante."},
            ]
            retriever = TfidfRetriever().fit(docs)
            retriever.rank("acceso de hemodiálisis", k=3)
            """
        ),
        md(
            """
            ## Parte C · Diseño clínico (20 puntos)

            Diseña un proyecto de detección de biopsia compatible con nefropatía
            IgA. Debe incluir:

            - uso previsto y usos excluidos;
            - población;
            - unidades;
            - esquema de anotación;
            - muestreo;
            - baseline;
            - modelo candidato;
            - particiones;
            - métrica primaria;
            - subgrupos;
            - privacidad;
            - paso a piloto.

            Pierdes todos los puntos si el test se utiliza para mejorar reglas o
            si se propone enviar texto real a un servicio no autorizado.
            """
        ),
        md(
            """
            ## Parte D · Selección de tecnología (10 puntos)

            Elige y justifica:

            1. extraer creatinina y unidades;
            2. detectar enfermedades generales ES/CA;
            3. clasificar etiología renal con 2000 ejemplos;
            4. buscar cursos semánticamente similares;
            5. generar cronología con citas.

            Para cada uno especifica baseline, modelo avanzado y evaluación.
            """
        ),
        md(
            """
            <details>
            <summary>Pistas</summary>

            1. regex/parser + validación semántica.
            2. NER BSC como baseline, validado localmente.
            3. clasificador supervisado/encoder ajustado.
            4. embeddings + retrieval benchmark.
            5. extracción/RAG + generativo con citas y evaluación humana.
            </details>
            """
        ),
        md(
            """
            ## Capstone final (15 puntos)

            Construye un proyecto nuevo sin copiar el fenotipo de TRS:

            - acceso vascular;
            - trasplante;
            - biopsia;
            - etiología ERC;
            - medicación/dosis;
            - otro aprobado.

            Entregables:

            1. canvas;
            2. protocolo;
            3. 50+ documentos ficticios diversos;
            4. diccionario;
            5. doble anotación simulada y adjudicación;
            6. baseline;
            7. modelo;
            8. evaluación bloqueada;
            9. errores;
            10. ficha de datos/model card;
            11. producción/monitorización.
            """
        ),
        md(
            """
            ## Rúbrica

            | Nivel | Evidencia |
            |---|---|
            | Inicial | ejecuta ejemplos |
            | Competente | modifica y explica errores |
            | Autónomo | diseña tarea/corpus/modelo/evaluación |
            | Experto | anticipa sesgo, transferencia y operación |

            Un «experto» sabe cuándo un problema no está listo para modelarse.
            """
        ),
        code(
            """
            competency_matrix = pd.DataFrame([
                {"competency": "corpus_annotation", "evidence": "", "status": "pending"},
                {"competency": "spacy_rules", "evidence": "", "status": "pending"},
                {"competency": "ner_training", "evidence": "", "status": "pending"},
                {"competency": "transformers_bsc", "evidence": "", "status": "pending"},
                {"competency": "classification_relations", "evidence": "", "status": "pending"},
                {"competency": "llm_rag", "evidence": "", "status": "pending"},
                {"competency": "validation", "evidence": "", "status": "pending"},
                {"competency": "production", "evidence": "", "status": "pending"},
            ])
            competency_matrix
            """
        ),
        md(
            """
            ## Final

            El curso termina cuando puedes defender un proyecto nuevo ante:

            - un nefrólogo;
            - un metodólogo;
            - un ingeniero;
            - protección de datos;
            - el equipo que revisará errores.

            Si cada uno entiende qué hace, dónde falla y qué evidencia lo respalda,
            has adquirido la competencia que buscamos.
            """
        ),
    ]
