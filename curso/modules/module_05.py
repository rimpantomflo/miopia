from curso.notebook_factory import code, common_setup, md


TITLE = "05 · Clasificación, relaciones y normalización"


def build() -> list[dict]:
    return [
        md(
            """
            # 05 · Clasificación, relaciones y normalización

            NER solo localiza menciones. Ahora aprenderás a clasificar documentos,
            relacionar entidades y enlazarlas con conceptos canónicos.
            """
        ),
        md(
            """
            ## Objetivos

            - entrenar un clasificador pequeño con spaCy;
            - distinguir binario, multiclase y multietiqueta;
            - seleccionar umbral en desarrollo;
            - representar relaciones;
            - crear candidatos de normalización;
            - medir top-1/top-k;
            - entender propagación de errores entre etapas.
            """
        ),
        common_setup(),
        code(
            """
            import difflib
            import json
            import random

            import pandas as pd
            import spacy
            from spacy.training import Example
            from spacy.util import fix_random_seed, minibatch

            fix_random_seed(21)
            """
        ),
        md(
            """
            ## 1. Tipos de clasificación

            - **Binaria:** TRS afirmada sí/no.
            - **Multiclase:** una modalidad actual.
            - **Multietiqueta:** ERC + anemia + trasplante pueden coexistir.

            No uses softmax multiclase cuando varias etiquetas pueden ser
            verdaderas.
            """
        ),
        code(
            """
            classification_examples = pd.DataFrame([
                {"text": "Inicia hemodiálisis.", "HD": 1, "DP": 0, "TX": 0},
                {"text": "DPA nocturna.", "HD": 0, "DP": 1, "TX": 0},
                {"text": "Trasplantado renal, HD previa.", "HD": 1, "DP": 0, "TX": 1},
            ])
            classification_examples
            """
        ),
        md(
            """
            ## 2. `textcat` binario con spaCy

            Entrenaremos `TRS_POSITIVE` frente a `TRS_NEGATIVE`. Como el corpus es
            mínimo, el objetivo vuelve a ser comprender el mecanismo.
            """
        ),
        code(
            """
            positive_texts = [
                "Inicia hemodiálisis trisemanal.",
                "Continúa HD mediante FAV.",
                "En diálisis peritoneal automatizada.",
                "Trasplantado renal con injerto funcionante.",
                "HDF online por FAV.",
                "Hemodiálisis domiciliaria.",
            ]
            negative_texts = [
                "No precisa tratamiento renal sustitutivo.",
                "Candidato a trasplante renal.",
                "Se explican opciones de diálisis.",
                "Madre en hemodiálisis.",
                "ERC G4 estable sin indicación de TRS.",
                "Función renal normal.",
            ]
            textcat_train = [
                (text, {"cats": {"TRS_POSITIVE": 1.0, "TRS_NEGATIVE": 0.0}})
                for text in positive_texts
            ] + [
                (text, {"cats": {"TRS_POSITIVE": 0.0, "TRS_NEGATIVE": 1.0}})
                for text in negative_texts
            ]
            """
        ),
        code(
            """
            textcat_nlp = spacy.blank("es")
            textcat = textcat_nlp.add_pipe("textcat")
            textcat.add_label("TRS_POSITIVE")
            textcat.add_label("TRS_NEGATIVE")

            def textcat_examples(data):
                return [
                    Example.from_dict(textcat_nlp.make_doc(text), annotations)
                    for text, annotations in data
                ]

            optimizer = textcat_nlp.initialize(
                get_examples=lambda: textcat_examples(textcat_train)
            )
            rng = random.Random(21)
            losses_history = []
            for epoch in range(20):
                data = list(textcat_train)
                rng.shuffle(data)
                losses = {}
                for batch in minibatch(data, size=4):
                    textcat_nlp.update(
                        textcat_examples(batch),
                        sgd=optimizer,
                        drop=0.2,
                        losses=losses,
                    )
                losses_history.append(losses.get("textcat", 0.0))
            losses_history[-5:]
            """
        ),
        code(
            """
            textcat_challenges = [
                "Actualmente no está en hemodiálisis.",
                "Iniciará diálisis peritoneal el próximo mes.",
                "Trasplante renal previo; injerto perdido.",
                "Continúa tratamiento conservador.",
            ]
            for text in textcat_challenges:
                doc = textcat_nlp(text)
                print(text, doc.cats)
            """
        ),
        md(
            """
            Los fallos mostrarán por qué clasificación directa necesita más datos
            y puede ocultar evidencia. Un sistema híbrido usa menciones y contexto
            como señales auditables.
            """
        ),
        md(
            """
            ## 3. Umbral

            `argmax` fuerza una clase. En un sistema binario puedes elegir umbral
            en desarrollo según coste clínico. El umbral no se elige en test.
            """
        ),
        code(
            """
            development_scores = pd.DataFrame([
                {"truth": 1, "score": 0.91},
                {"truth": 1, "score": 0.68},
                {"truth": 1, "score": 0.42},
                {"truth": 0, "score": 0.61},
                {"truth": 0, "score": 0.35},
                {"truth": 0, "score": 0.08},
            ])

            def confusion_at_threshold(frame, threshold):
                pred = frame["score"].ge(threshold)
                truth = frame["truth"].astype(bool)
                return {
                    "threshold": threshold,
                    "tp": int((truth & pred).sum()),
                    "fp": int((~truth & pred).sum()),
                    "fn": int((truth & ~pred).sum()),
                    "tn": int((~truth & ~pred).sum()),
                }

            pd.DataFrame(
                confusion_at_threshold(development_scores, threshold)
                for threshold in [0.3, 0.5, 0.7]
            )
            """
        ),
        md(
            """
            ### Ejercicio 1

            Decide un coste relativo FP/FN para una cola de cribado. Elige umbral
            y justifica. Después cambia el uso previsto a «enviar una carta al
            paciente» y comprueba por qué el umbral y revisión deben cambiar.
            """
        ),
        md(
            """
            ## 4. Relaciones

            Una relación conecta entidades:

            ```text
            DRUG(tacrolimus) --HAS_DOSE--> DOSE(2 mg/12 h)
            DISEASE(ERC) --HAS_STAGE--> STAGE(G4)
            ACCESS(FAV) --HAS_STATUS--> STATUS(trombosada)
            ```

            Guarda IDs de spans, dirección, tipo, evidencia y confianza.
            """
        ),
        code(
            """
            relation_text = "ERC G4. Mantiene tacrolimus 2 mg cada 12 horas."
            entities = [
                {"id": "e1", "text": "ERC", "start": 0, "end": 3, "label": "DISEASE"},
                {"id": "e2", "text": "G4", "start": 4, "end": 6, "label": "STAGE"},
                {"id": "e3", "text": "tacrolimus", "start": 16, "end": 26, "label": "DRUG"},
                {"id": "e4", "text": "2 mg cada 12 horas", "start": 27, "end": 45, "label": "DOSE"},
            ]
            relations = [
                {"head": "e1", "tail": "e2", "label": "HAS_STAGE"},
                {"head": "e3", "tail": "e4", "label": "HAS_DOSE"},
            ]
            pd.DataFrame(relations)
            """
        ),
        md(
            """
            ### Baseline de proximidad

            Emparejar la dosis más cercana funciona en frases simples y falla con
            varios fármacos, listas, dosis previas o negación.
            """
        ),
        code(
            """
            def nearest_relation(entities, source_label, target_label):
                sources = [entity for entity in entities if entity["label"] == source_label]
                targets = [entity for entity in entities if entity["label"] == target_label]
                output = []
                for source in sources:
                    if not targets:
                        continue
                    target = min(
                        targets,
                        key=lambda item: abs(item["start"] - source["end"]),
                    )
                    output.append((source["id"], target["id"]))
                return output

            nearest_relation(entities, "DRUG", "DOSE")
            """
        ),
        md(
            """
            ### Ejercicio 2

            Haz fallar proximidad con:

            `Tacrolimus 2 mg y prednisona 5 mg cada 24 h.`

            Define gold de relaciones. Decide si necesitas dependencias,
            clasificación de pares o reglas por estructura.
            """
        ),
        md(
            """
            ## 5. Normalización

            NER produce `HD`, `hemodiàlisi`, `hemodiálisis`. Normalización enlaza
            todas al concepto `HEMODIALYSIS`.

            Pipeline:

            ```text
            mención → candidatos → reranking → concepto/abstención
            ```
            """
        ),
        code(
            """
            concepts = json.loads(
                (PROJECT_ROOT / "data" / "conceptos_renales_sinteticos.json")
                .read_text(encoding="utf-8")
            )
            candidate_terms = []
            for concept in concepts:
                for term in [concept["preferred_term"], *concept["variants"]]:
                    candidate_terms.append({
                        "concept_id": concept["concept_id"],
                        "term": term,
                    })

            def normalization_candidates(mention, k=5):
                scored = [
                    {
                        **candidate,
                        "score": difflib.SequenceMatcher(
                            None,
                            mention.casefold(),
                            candidate["term"].casefold(),
                        ).ratio(),
                    }
                    for candidate in candidate_terms
                ]
                return sorted(
                    scored,
                    key=lambda row: (-row["score"], row["concept_id"], row["term"]),
                )[:k]

            normalization_candidates("hemodialisi", k=3)
            """
        ),
        md(
            """
            Similaridad de caracteres no comprende significado. Es un baseline
            útil para erratas; un sistema avanzado usa bi-encoder para recuperar,
            cross-encoder para reordenar y reglas/abstención.
            """
        ),
        code(
            """
            normalization_gold = [
                ("HD", "HEMODIALYSIS"),
                ("DPA", "PERITONEAL_DIALYSIS"),
                ("FAVI", "AV_FISTULA"),
                ("trasplantada renal", "KIDNEY_TRANSPLANT"),
            ]

            rows = []
            for mention, gold in normalization_gold:
                candidates = normalization_candidates(mention, k=3)
                ids = [candidate["concept_id"] for candidate in candidates]
                rows.append({
                    "mention": mention,
                    "gold": gold,
                    "top1": ids[0],
                    "top1_correct": ids[0] == gold,
                    "top3_correct": gold in ids,
                })
            normalization_results = pd.DataFrame(rows)
            normalization_results
            """
        ),
        code(
            """
            normalization_results[["top1_correct", "top3_correct"]].mean()
            """
        ),
        md(
            """
            ## 6. Abstención y concepto desconocido

            Si ningún candidato supera el umbral o hay empate clínicamente
            ambiguo, devuelve `UNRESOLVED`, no un código inventado.
            """
        ),
        code(
            """
            def normalize_with_abstention(mention, threshold=0.80):
                best = normalization_candidates(mention, k=1)[0]
                if best["score"] < threshold:
                    return {"concept_id": "UNRESOLVED", "score": best["score"]}
                return {"concept_id": best["concept_id"], "score": best["score"]}

            for mention in ["hemodialisi", "DP", "síndrome renal misterioso"]:
                print(mention, normalize_with_abstention(mention))
            """
        ),
        md(
            """
            ## 7. Propagación de errores

            Si NER omite la mención, normalización no puede recuperarla. Si detecta
            límites erróneos, candidatos empeoran. Evalúa:

            1. normalización sobre spans gold;
            2. pipeline completo sobre spans predichos.

            La diferencia mide propagación.
            """
        ),
        md(
            """
            ## Reto integrador

            Diseña un sistema para `ERC + estadio + etiología`:

            - NER de enfermedad, estadio y etiología;
            - atributos de contexto;
            - relaciones;
            - normalización;
            - clase documental;
            - agregación por paciente.

            Define métricas independientes y end-to-end.
            """
        ),
        md(
            """
            ## Criterio para avanzar

            Debes explicar:

            - multiclase frente a multietiqueta;
            - umbral y uso previsto;
            - relación frente a coocurrencia;
            - NER frente a normalización;
            - top-1 frente a top-k;
            - abstención;
            - propagación de errores.
            """
        ),
    ]
