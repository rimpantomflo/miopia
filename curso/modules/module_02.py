from curso.notebook_factory import code, common_setup, md


TITLE = "02 · spaCy avanzado y pipelines clínicos"


def build() -> list[dict]:
    return [
        md(
            """
            # 02 · spaCy avanzado y pipelines clínicos

            Pasamos de llamar a `nlp(texto)` a diseñar una canalización propia:
            tokenización, extensiones, componentes, matchers, spans,
            procesamiento por lotes y serialización.
            """
        ),
        md(
            """
            ## Objetivos

            - comprender `Language`, `Vocab`, tokenizer y componentes;
            - inspeccionar y modificar tokenización de forma controlada;
            - usar extensiones de `Doc`, `Span` y `Token`;
            - construir un componente clínico;
            - dominar `Matcher`, `PhraseMatcher` y `SpanRuler`;
            - resolver spans solapados;
            - pasar metadatos con `nlp.pipe`;
            - serializar y comprobar reproducibilidad.
            """
        ),
        common_setup(),
        code(
            """
            import re
            import time

            import pandas as pd
            import spacy
            from spacy.language import Language
            from spacy.matcher import Matcher, PhraseMatcher
            from spacy.symbols import ORTH
            from spacy.tokens import Doc, Span, Token
            from spacy.util import filter_spans

            nlp = spacy.blank("es")
            nlp.add_pipe("sentencizer")
            print("Idioma:", nlp.lang)
            print("Componentes:", nlp.pipe_names)
            print("Vocabulario compartido:", id(nlp.vocab))
            """
        ),
        md(
            """
            ## 1. Arquitectura

            El tokenizer no aparece en `pipe_names`: transforma string → `Doc`.
            Cada componente posterior recibe y devuelve el mismo `Doc`.

            ```text
            string → tokenizer → Doc → sentencizer → reglas → modelo → Doc
            ```

            `Vocab` almacena strings, lexemas y vectores compartidos. Un `Doc`
            creado con otro vocabulario no debe mezclarse indiscriminadamente.
            """
        ),
        code(
            """
            doc = nlp("ERC-G4; HD por FAV.")
            [(token.i, token.text, token.is_alpha, token.idx) for token in doc]
            """
        ),
        md(
            """
            ## 2. Tokenización

            No cambies tokenización para «mejorar» un único ejemplo sin evaluar
            el corpus completo. Los offsets anotados y modelos entrenados dependen
            de ella.

            Los casos especiales permiten conservar una forma concreta.
            """
        ),
        code(
            """
            tokenizer_demo = spacy.blank("es")
            before = [token.text for token in tokenizer_demo("ERC/G4")]
            tokenizer_demo.tokenizer.add_special_case("ERC/G4", [{ORTH: "ERC/G4"}])
            after = [token.text for token in tokenizer_demo("ERC/G4")]
            print("Antes:", before)
            print("Después:", after)
            assert before == ["ERC", "/", "G4"] and after == ["ERC/G4"]
            """
        ),
        md(
            """
            ### Ejercicio 1

            Inspecciona `ml/min/1,73m²`, `ACR>300 mg/g`, `CVC-T`, `anti-PLA2R`
            y `p-ANCA`. Decide cuáles necesitan una excepción y cuáles deben
            conservarse como varios tokens para extraer estructura.

            La respuesta depende de las tareas posteriores; justifícala.
            """
        ),
        md(
            """
            ## 3. Extensiones

            Las extensiones añaden atributos sin modificar clases internas.
            Regístralas una vez y usa nombres específicos para evitar colisiones.
            """
        ),
        code(
            """
            if not Doc.has_extension("course_id"):
                Doc.set_extension("course_id", default=None)
            if not Doc.has_extension("sections"):
                Doc.set_extension("sections", default=[])
            if not Span.has_extension("assertion"):
                Span.set_extension("assertion", default="unknown")
            if not Token.has_extension("normalized_unit"):
                Token.set_extension("normalized_unit", default=None)

            doc = nlp("Paciente en HD.")
            doc._.course_id = "SYN-001"
            print(doc._.course_id, doc._.sections)
            """
        ),
        md(
            """
            ## 4. Componente personalizado

            El ejemplo detecta encabezados muy simples. Un componente real debe
            validar plantillas, spans, saltos y abreviaturas locales.
            """
        ),
        code(
            """
            @Language.component("course_section_detector")
            def course_section_detector(doc):
                pattern = re.compile(
                    r"(?im)^(antecedentes|evolución|plan|tratamiento)\\s*:"
                )
                sections = []
                matches = list(pattern.finditer(doc.text))
                for index, match in enumerate(matches):
                    end = matches[index + 1].start() if index + 1 < len(matches) else len(doc.text)
                    sections.append({
                        "name": match.group(1).casefold(),
                        "start": match.end(),
                        "end": end,
                    })
                doc._.sections = sections
                return doc

            if "course_section_detector" not in nlp.pipe_names:
                nlp.add_pipe("course_section_detector", last=True)

            section_text = (
                "ANTECEDENTES: ERC G4.\\n"
                "EVOLUCIÓN: inicia hemodiálisis.\\n"
                "PLAN: crear FAV."
            )
            section_doc = nlp(section_text)
            section_doc._.sections
            """
        ),
        code(
            """
            for section in section_doc._.sections:
                print(section["name"], "→", repr(section_doc.text[section["start"]:section["end"]].strip()))
            """
        ),
        md(
            """
            ### Ejercicio 2

            Haz fallar el detector con:

            - encabezado sin dos puntos;
            - espacios iniciales;
            - encabezado en catalán;
            - palabra «plan» dentro de una frase.

            Escribe primero los resultados esperados. La sección es una entidad
            estructural y necesita su propio test.
            """
        ),
        md(
            """
            ## 5. `Matcher`

            Los patrones operan sobre atributos de tokens. Usa `validate=True`
            para detectar claves inválidas.
            """
        ),
        code(
            """
            matcher = Matcher(nlp.vocab, validate=True)
            matcher.add(
                "CKD_STAGE",
                [[
                    {"LOWER": {"IN": ["erc", "ckd"]}},
                    {"TEXT": {"REGEX": "^[Gg]?[1-5][AaBb]?$"}},
                ]],
            )
            matcher.add(
                "DIALYSIS_ACCESS",
                [[
                    {"LOWER": {"IN": ["por", "mediante"]}},
                    {"LOWER": {"IN": ["fav", "cvc", "tenckhoff"]}},
                ]],
            )

            matcher_doc = nlp("ERC G4. Continúa HD mediante FAV.")
            [
                (nlp.vocab.strings[match_id], matcher_doc[start:end].text)
                for match_id, start, end in matcher(matcher_doc)
            ]
            """
        ),
        md(
            """
            ## 6. `PhraseMatcher`

            Es eficiente para listas de frases. `attr="LOWER"` ignora mayúsculas,
            no acentos ni errores. El diccionario sigue necesitando control.
            """
        ),
        code(
            """
            phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER", validate=True)
            renal_terms = [
                "enfermedad renal crónica",
                "fracaso renal agudo",
                "diálisis peritoneal",
                "trasplante renal",
            ]
            phrase_matcher.add("RENAL_CONCEPT", [nlp.make_doc(term) for term in renal_terms])

            phrase_doc = nlp("Antecedente de trasplante renal y enfermedad renal crónica.")
            [phrase_doc[start:end].text for _, start, end in phrase_matcher(phrase_doc)]
            """
        ),
        md(
            """
            ## 7. `SpanRuler` y solapamientos

            `Doc.ents` no admite entidades solapadas. `Doc.spans[key]` sí. Eso
            permite representar `enfermedad renal` y `enfermedad renal crónica`
            mientras decides una política.
            """
        ),
        code(
            """
            nlp_spans = spacy.blank("es")
            ruler = nlp_spans.add_pipe(
                "span_ruler",
                config={"spans_key": "renal", "phrase_matcher_attr": "LOWER"},
            )
            ruler.add_patterns([
                {"label": "DISEASE", "id": "RENAL_GENERIC", "pattern": "enfermedad renal"},
                {"label": "DISEASE", "id": "CKD", "pattern": "enfermedad renal crónica"},
                {"label": "PROCEDURE", "id": "HD", "pattern": "hemodiálisis"},
            ])
            overlap_doc = nlp_spans("Enfermedad renal crónica en hemodiálisis.")
            [(span.text, span.id_) for span in overlap_doc.spans["renal"]]
            """
        ),
        code(
            """
            longest = filter_spans(overlap_doc.spans["renal"])
            [(span.text, span.id_) for span in longest]
            """
        ),
        md(
            """
            `filter_spans` prefiere spans largos y elimina solapamientos. Esa es
            una política técnica, no necesariamente la semántica correcta. Para
            conceptos anidados pueden necesitarse varios grupos de spans o
            `SpanCategorizer`.
            """
        ),
        md(
            """
            ## 8. Contexto y atributos de span

            Las reglas de negación deben modificar la mención, no borrar el span.
            Así puedes medir detección y aserción por separado.
            """
        ),
        code(
            """
            context_doc = nlp_spans("No precisa hemodiálisis.")
            hd_span = context_doc.spans["renal"][0]
            prefix = context_doc.text[max(0, hd_span.start_char - 30):hd_span.start_char].casefold()
            hd_span._.assertion = "negated" if re.search(r"\\b(no|sin|niega)\\b", prefix) else "affirmed"
            print(hd_span.text, hd_span._.assertion)
            """
        ),
        md(
            """
            ## 9. `nlp.pipe` y metadatos

            Usa lotes y `as_tuples=True`. El contexto externo no se convierte en
            texto ni se pierde.
            """
        ),
        code(
            """
            stream = [
                ("ERC G4 estable.", {"course_id": "A"}),
                ("Inicia HD por FAV.", {"course_id": "B"}),
                ("Sin indicación de TRS.", {"course_id": "C"}),
            ]
            processed = []
            for batch_doc, metadata in nlp.pipe(stream, as_tuples=True, batch_size=2):
                batch_doc._.course_id = metadata["course_id"]
                processed.append((batch_doc._.course_id, len(batch_doc)))
            processed
            """
        ),
        code(
            """
            many_texts = ["Paciente en HD mediante FAV."] * 1000
            start = time.perf_counter()
            list(nlp.pipe(many_texts, batch_size=128))
            elapsed = time.perf_counter() - start
            print(f"1000 textos ficticios: {elapsed:.4f} s")
            """
        ),
        md(
            """
            La medición es local y no representa un transformer. Registra versión,
            hardware, longitud, componentes, batch y número de procesos.
            """
        ),
        md(
            """
            ## 10. Desactivar componentes

            Si solo necesitas tokenización o reglas, evita ejecutar modelos
            innecesarios. `select_pipes` restaura el pipeline al salir.
            """
        ),
        code(
            """
            print("Antes:", nlp.pipe_names)
            with nlp.select_pipes(disable=["course_section_detector"]):
                temporary_doc = nlp("PLAN: seguimiento.")
                print("Dentro:", nlp.pipe_names, temporary_doc._.sections)
            print("Después:", nlp.pipe_names)
            """
        ),
        md(
            """
            ## 11. Serialización

            Serializa tokenizer, vocabulario y componentes. Comprueba que el
            pipeline recuperado produce la misma salida relevante.
            """
        ),
        code(
            """
            serialized = nlp_spans.to_bytes()
            # from_bytes restaura estado dentro de una arquitectura ya creada.
            recovered_nlp = spacy.blank("es")
            recovered_nlp.add_pipe(
                "span_ruler",
                config={"spans_key": "renal", "phrase_matcher_attr": "LOWER"},
            )
            recovered_nlp.from_bytes(serialized)
            recovered_doc = recovered_nlp("Enfermedad renal crónica.")
            print(recovered_nlp.pipe_names)
            print([(span.text, span.id_) for span in recovered_doc.spans["renal"]])
            """
        ),
        md(
            """
            ## Reto integrador

            Construye un pipeline con:

            1. sentencizer;
            2. secciones;
            3. `SpanRuler` desde el diccionario renal;
            4. atributo de aserción;
            5. `course_id`;
            6. procesamiento por lotes;
            7. serialización.

            Añade ejemplos para negación, familiar, plan futuro y abreviatura
            ambigua. No consultes el NER entrenable todavía.
            """
        ),
        md(
            """
            ## Criterio para avanzar

            Debes poder explicar:

            - tokenizer frente a componente;
            - `Doc.ents` frente a `Doc.spans`;
            - `Matcher`, `PhraseMatcher` y `SpanRuler`;
            - por qué una extensión no es una columna de pandas;
            - cuándo desactivar componentes;
            - qué debe reproducirse tras serializar.
            """
        ),
    ]
