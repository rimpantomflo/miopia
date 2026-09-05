from curso.notebook_factory import code, common_setup, md

TITLE = "15 · Contexto, relaciones y normalización clínica"


def build() -> list[dict]:
    return [
        md(
            """
            # 15 · Contexto, relaciones y normalización clínica

            Detectar una palabra no basta. Un sistema útil debe responder quién,
            cuándo y con qué certeza; enlazar entidades relacionadas; y mapear la
            mención a una terminología con posibilidad de abstenerse.
            """
        ),
        md(
            """
            ## Objetivos

            - aplicar ConText ES/CA sin propagación entre cláusulas;
            - conservar los disparadores que justifican cada atributo;
            - generar candidatos de relación y negativos explícitos;
            - entrenar un clasificador de relaciones;
            - recuperar y reordenar conceptos terminológicos;
            - calibrar umbral y margen de abstención.
            """
        ),
        common_setup(),
        code(
            """
            from clinical_nlp_course import (
                ConceptNormalizer,
                Entity,
                RelationClassifier,
                annotate_context,
                relation_candidates,
            )
            """
        ),
        md(
            """
            ## 1. Contexto con alcance

            `No se descarta` tiene prioridad sobre el `no` aislado. Los
            disparadores se detienen en puntuación, saltos de línea y conectores
            adversativos para evitar que una negación contamine toda la nota.
            """
        ),
        code(
            """
            context_text = (
                "No se descarta nefropatía IgA. Sin hemodiálisis; "
                "actualmente diálisis peritoneal estable. "
                "Madre con trasplante renal."
            )
            mention_terms = ["nefropatía IgA", "hemodiálisis", "diálisis peritoneal", "trasplante renal"]
            mentions = []
            cursor = 0
            for term in mention_terms:
                start = context_text.index(term, cursor)
                mentions.append({"start": start, "end": start + len(term), "label": "CONCEPT"})
                cursor = start + len(term)

            contextualized = annotate_context(context_text, mentions)
            for row in contextualized:
                print(row["evidence"], row["assertion"], row["experiencer"], row["context_triggers"])

            assert contextualized[0]["assertion"] == "possible"
            assert contextualized[1]["assertion"] == "negated"
            assert contextualized[2]["assertion"] == "affirmed"
            assert contextualized[3]["experiencer"] == "family"
            """
        ),
        md(
            """
            ### Suite centinela

            Añade, como mínimo: `sin A ni B`, `niega A pero refiere B`, historia
            familiar, plan futuro, educación sanitaria, encabezados, listas y una
            frase catalana. Mide atributos por separado; un F1 NER alto no valida
            negación o sujeto.
            """
        ),
        md(
            """
            ## 2. Relaciones: primero candidatos

            Entrenar solo con relaciones positivas impide aprender cuándo no hay
            vínculo. Generamos pares compatibles dentro de una ventana y etiquetamos
            `NO_RELATION` de forma explícita.
            """
        ),
        code(
            """
            def make_relation_candidate(text, first, second):
                first_start = text.index(first)
                second_start = text.index(second)
                entities = [
                    Entity("E1", first_start, first_start + len(first), "TREATMENT", first),
                    Entity("E2", second_start, second_start + len(second), "ACCESS", second),
                ]
                return relation_candidates(
                    text,
                    entities,
                    allowed_type_pairs={("TREATMENT", "ACCESS")},
                )[0]

            relation_examples = [
                ("Hemodiálisis mediante FAV.", "Hemodiálisis", "FAV", "HAS_ACCESS"),
                ("Diálisis mediante catéter.", "Diálisis", "catéter", "HAS_ACCESS"),
                ("Tratamiento con fístula funcionante.", "Tratamiento", "fístula", "HAS_ACCESS"),
                ("Hemodiálisis por catéter tunelizado.", "Hemodiálisis", "catéter", "HAS_ACCESS"),
                ("Hemodiálisis. La FAV fue retirada.", "Hemodiálisis", "FAV", "NO_RELATION"),
                ("Diálisis; sin embargo el catéter no se utiliza.", "Diálisis", "catéter", "NO_RELATION"),
                ("Tratamiento. No consta vínculo con fístula.", "Tratamiento", "fístula", "NO_RELATION"),
                ("Hemodiálisis. Se descarta uso del catéter.", "Hemodiálisis", "catéter", "NO_RELATION"),
            ]
            relation_rows = [
                make_relation_candidate(text, first, second)
                for text, first, second, label in relation_examples
            ]
            relation_labels = [label for text, first, second, label in relation_examples]
            relation_model = RelationClassifier(seed=17).fit(relation_rows, relation_labels)
            relation_model.predict_records(relation_rows)[:2]
            """
        ),
        md(
            """
            El ejemplo enseña arquitectura, no rendimiento: ocho frases no bastan.
            En un proyecto real mide primero recall del generador de candidatos; una
            relación que nunca se propone es imposible de recuperar por el modelo.
            """
        ),
        md(
            """
            ## 3. Normalización con abstención

            La primera etapa recupera por n-gramas de caracteres. La segunda puede
            aprender `match/no-match` con pares locales. Aceptamos solo si score y
            margen sobre el segundo candidato superan umbrales elegidos en dev.
            """
        ),
        code(
            """
            concepts = [
                {"concept_id": "HD", "preferred_term": "hemodiálisis", "variants": ["HD"], "semantic_type": "procedure"},
                {"concept_id": "DP", "preferred_term": "diálisis peritoneal", "variants": ["DPA"], "semantic_type": "procedure"},
                {"concept_id": "TX", "preferred_term": "trasplante renal", "variants": ["injerto renal"], "semantic_type": "procedure"},
                {"concept_id": "FAV", "preferred_term": "fístula arteriovenosa", "variants": ["FAV"], "semantic_type": "device"},
            ]
            normalizer = ConceptNormalizer(concepts).fit_reranker([
                ("hemodialisis", "HD"),
                ("DPA", "DP"),
                ("injerto renal", "TX"),
                ("fistula AV", "FAV"),
            ])
            for mention in ["hemodialisis", "DPA", "injerto", "diálisis"]:
                print(normalizer.normalize(mention, threshold=0.45, min_margin=0.03))
            """
        ),
        md(
            """
            ## 4. Salida compuesta y trazable

            La salida final conserva: span, contexto, candidatos terminológicos,
            score/margen, relación, versión y evidencia. No colapses toda esa
            información a un booleano antes de revisar errores.
            """
        ),
        code(
            """
            trace = {
                "document_id": "SYN-DEMO",
                "mention": contextualized[2],
                "normalization": normalizer.normalize("diálisis peritoneal", threshold=0.45),
                "pipeline_versions": {
                    "context": "context-es-ca-v1",
                    "relations": "relation-baseline-v1",
                    "terminology": "synthetic-v1",
                },
            }
            assert trace["mention"]["evidence"] == "diálisis peritoneal"
            trace
            """
        ),
        md(
            """
            ## Práctica obligatoria

            Crea 100 casos centinela de contexto, 200 pares de relación y 100
            menciones normalizadas con candidatos difíciles. Evalúa por atributo,
            por tipo de relación y `accuracy@1/recall@k/coverage` para normalización.

            **Criterio de salida:** sabes diferenciar error de detección, contexto,
            candidato, ranking y umbral; cada predicción conserva evidencia auditable.
            """
        ),
    ]
