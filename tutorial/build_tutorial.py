"""Genera el notebook docente sin depender de nbformat.

Ejecutar desde la raíz:
    .venv/Scripts/python.exe tutorial/build_tutorial.py
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tutorial" / "tutorial_miopia_nlp.ipynb"


def _source(text: str) -> list[str]:
    text = dedent(text).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source(text),
    }


def code(text: str = "") -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text) if text.strip() else [],
    }


cells = [
    markdown(
        """
        # Fenotipado clínico de miopía con NLP

        ## De cero a un pipeline auditable · guía completa · versión 1.0

        **Misión:** identificar, para revisión por Oftalmología, pacientes con
        evidencia de miopía en cursos clínicos longitudinales.

        Este cuaderno construye un sistema realista sin ocultar sus límites:
        empieza con búsqueda literal, añade menciones, contexto y refracción,
        mide errores, agrega por paciente y termina con el diseño para Oracle,
        modelos entrenables y producción.

        > Todo el corpus es ficticio. No copies texto clínico real al notebook,
        > a Git, a una incidencia ni a un servicio externo no autorizado.
        """
    ),
    markdown(
        """
        ## Contrato de aprendizaje

        No basta con ejecutar las celdas. En cada etapa sigue este ciclo:

        1. formula exactamente qué quieres predecir;
        2. inspecciona ejemplos y offsets;
        3. implementa el método más sencillo;
        4. mide por qué falla;
        5. cambia una sola cosa;
        6. congela casos difíciles como pruebas.

        Al acabar podrás explicar la diferencia entre **mención**, **curso** y
        **paciente**, construir una referencia anotada, evaluar sin fuga entre
        pacientes y decidir cuándo unas reglas son mejores que un modelo.

        Las soluciones reutilizables viven en `src/miopia_nlp/pipeline.py` y su
        contrato está protegido por pruebas en `tests/test_pipeline.py`.
        """
    ),
    markdown(
        """
        ## Mapa del proyecto

        | Etapa | Competencia | Entregable |
        |---|---|---|
        | 1 | Corpus de prueba y fundamentos de spaCy | Entender `nlp`, `Doc`, `Token` y `Span` |
        | 2 | Reglas con `Matcher` y `SpanRuler` | Detectar menciones de miopía |
        | 3 | Expresiones regulares | Extraer ojo, dioptrías y equivalente esférico |
        | 4 | Contexto clínico | Negación, incertidumbre, familiar y temporalidad |
        | 5 | Modelos clínicos preentrenados | Comparar reglas con modelos del BSC |
        | 6 | Validación | Sensibilidad, VPP, F1 y análisis de errores |
        | 7 | Oracle | Extraer y procesar textos por lotes |
        | 8 | Agregación longitudinal | Crear el fenotipo final por paciente |

        Seguiremos estas ocho etapas en este orden. La definición clínica es el
        requisito previo y la producción será un anexo posterior al mapa.
        """
    ),
    markdown(
        """
        ## Antes de la etapa 1 · Definir antes de programar

        Nuestra salida primaria será `ever_myopia`: existe evidencia de que el
        **paciente** ha tenido miopía alguna vez.

        También conservaremos:

        - `document_status`: `confirmed`, `possible` o `not_supported`;
        - `current_status`: evidencia actual confirmada, negada, posible o
          desconocida;
        - `high_myopia_numeric`: evidencia refractiva compatible con miopía alta;
        - todas las menciones, mediciones, reglas y posiciones de caracteres.

        **Ausencia de mención no equivale a ausencia de miopía.** Por eso usamos
        `not_supported`/`unknown`, no «paciente sano».

        El [protocolo de anotación](../docs/protocolo_anotacion.md) es parte del
        sistema: el código no puede arreglar una definición clínica ambigua.
        """
    ),
    markdown(
        """
        ### Umbrales clínicos y alcance

        El International Myopia Institute propone, para estudios clínicos y
        epidemiológicos, miopía con equivalente esférico ≤ −0,50 D y miopía alta
        con ≤ −6,00 D, cuando la acomodación está relajada.

        En cursos rutinarios con frecuencia no consta cómo se obtuvo la
        refracción. El pipeline lo denomina **evidencia numérica**, no diagnóstico.
        Miopía alta y miopía patológica tampoco son sinónimos.

        Fuente primaria:
        [IMI — Defining and Classifying Myopia](https://myopiainstitute.org/imi-whitepaper/defining-and-classifying-myopia/2023-02-12_imi-defining-and-classifying-myopia-report_english/).
        """
    ),
    markdown(
        """
        ## Etapa 1 · Corpus de prueba y fundamentos de spaCy

        ### Entorno reproducible

        Desde la raíz del proyecto:

        ```powershell
        uv sync
        uv run python -m unittest discover -s tests -v
        ```

        El notebook no instala paquetes silenciosamente: `uv.lock` define el
        entorno. Ábrelo en VS Code/Jupyter y selecciona
        `.venv\\Scripts\\python.exe` como kernel.
        """
    ),
    code(
        """
        import hashlib
        import inspect
        import platform
        import sys
        from pathlib import Path

        import pandas as pd
        import spacy

        PROJECT_ROOT = Path.cwd()
        if not (PROJECT_ROOT / "src").exists():
            PROJECT_ROOT = PROJECT_ROOT.parent
        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        from miopia_nlp import (
            MYOPIA_HIGH_THRESHOLD_D,
            MYOPIA_THRESHOLD_D,
            aggregate_patient,
            binary_metrics,
            extract_mentions,
            parse_refractions,
            phenotype_course,
            process_courses,
            pseudonymize_id,
        )

        print("Python:", platform.python_version())
        print("pandas:", pd.__version__)
        print("spaCy:", spacy.__version__)
        print("Proyecto:", PROJECT_ROOT)
        """
    ),
    markdown(
        """
        ### `spaCy` no es `nlp`

        `spaCy` es la biblioteca. `nlp` es una canalización concreta. Al llamar
        `nlp(texto)`:

        `texto → Tokenizer → Doc → componentes → Doc anotado`

        Empezamos con castellano vacío y un `sentencizer`. No hay lemas, POS ni
        NER estadístico: únicamente tokenización y frases basadas en puntuación.
        """
    ),
    code(
        """
        nlp = spacy.blank("es")
        nlp.add_pipe("sentencizer")

        texto_demo = "Antecedente de miopía magna bilateral desde la infancia."
        doc = nlp(texto_demo)

        print("Componentes:", nlp.pipe_names)
        print([(token.i, token.text, token.idx) for token in doc])

        span = doc[2:4]
        print("Span:", span.text, "| tokens:", span.start, span.end)
        print("Offsets:", span.start_char, span.end_char)
        assert doc.text[span.start_char:span.end_char] == span.text
        """
    ),
    markdown(
        """
        **Idea experta:** los offsets son un contrato. Normalizar o destruir el
        texto antes de anotar puede hacer imposible recuperar la evidencia. Usa
        texto normalizado para comparar, pero conserva siempre el original.

        El `sentencizer` es suficiente para aprender, no para declarar resuelta la
        segmentación clínica: abreviaturas, listas, saltos y plantillas requieren
        una evaluación propia.
        """
    ),
    markdown(
        """
        ## Corpus sintético longitudinal

        La referencia ya no mezcla todo en una etiqueta. Para esta primera
        evaluación binaria, `gold_ever_document` indica si el **curso** contiene
        evidencia afirmada del paciente o evidencia refractiva suficiente.

        Las menciones conservarán por separado aserción, experienciador,
        temporalidad y contexto.
        """
    ),
    code(
        """
        ejemplos = [
            ("P001", "C001", "2023-01-10", "es",
             "Antecedente de miopía magna bilateral desde la infancia.", True),
            ("P002", "C002", "2023-01-11", "es",
             "No presenta miopía ni otros defectos refractivos.", False),
            ("P003", "C003", "2023-02-02", "ca",
             "Possible miopia; es recomana valoració per Oftalmologia.", False),
            ("P004", "C004", "2023-02-03", "es",
             "Madre con miopía magna y desprendimiento de retina.", False),
            ("P005", "C005", "2023-03-05", "es",
             "Intervenido mediante LASIK por miopía en 2018.", True),
            ("P006", "C006", "2023-03-06", "ca",
             "Pacient gran miop. OD −8,00 D; OI −7,50 D.", True),
            ("P007", "C007", "2023-04-01", "es",
             "Fondo de ojo sin hallazgos patológicos.", False),
            ("P008", "C008", "2023-04-02", "es",
             "Se explica que la miopía aumenta el riesgo de patología retiniana.", False),
            ("P001", "C009", "2024-01-12", "es",
             "Antecedente de mipía, actualmente usa gafas.", True),
            ("P003", "C010", "2024-02-12", "es",
             "No se descarta miopía; pendiente de refracción.", False),
            ("P009", "C011", "2024-02-14", "es",
             "Sin antecedentes familiares de miopía. Paciente miope desde los 12 años.", True),
            ("P010", "C012", "2024-03-03", "es",
             "Agudeza visual OD 0.8, OI 1.0 sin corrección.", False),
            ("P011", "C013", "2024-03-04", "es",
             "Refracción: OD −8,00 D; OI −7,50 D.", True),
            ("P012", "C014", "2024-03-05", "es",
             "Hipermetropía leve. Niega miopía.", False),
            ("P005", "C015", "2025-01-08", "es",
             "Actualmente no presenta miopía. Intervenido por miopía con LASIK en 2018.", True),
            ("P013", "C016", "2025-01-09", "es",
             "Se observa miopización nocturna transitoria.", False),
            ("P014", "C017", "2025-02-02", "es",
             "AF: miopía. Refracción OD +1,00 D; OI +0,75 D.", False),
            ("P015", "C018", "2025-02-03", "ca",
             "Pacient miop. Porta ulleres des de la infància.", True),
            ("P016", "C019", "2025-02-04", "es",
             "¿Miopía? Pendiente de refracción ciclopléjica.", False),
            ("P017", "C020", "2025-02-05", "es",
             "Dx: H52.1 Miopía bilateral.", True),
            ("P018", "C021", "2025-02-06", "es",
             "Plan: control de miopía con atropina a baja dosis.", True),
            ("P019", "C022", "2025-02-07", "es",
             "Niega antecedentes familiares de miopía; paciente con miopía axial.", True),
        ]

        df = pd.DataFrame(
            ejemplos,
            columns=[
                "patient_id", "course_id", "fecha", "idioma",
                "texto", "gold_ever_document",
            ],
        )
        df["fecha"] = pd.to_datetime(df["fecha"])
        df
        """
    ),
    code(
        """
        print("Cursos:", len(df), "| pacientes:", df["patient_id"].nunique())
        display(df.groupby(["idioma", "gold_ever_document"]).size().rename("n").to_frame())
        """
    ),
    markdown(
        """
        ### Tu turno 1

        1. Procesa C006 y muestra un token por línea.
        2. Comprueba cómo se tokeniza `−8,00`.
        3. Crea un `Span` con `gran miop` usando los índices observados.
        4. Verifica que los offsets recuperan exactamente el texto.
        """
    ),
    code(
        """
        # Solución de referencia: ejecútala después de intentarlo.
        doc_c006 = nlp(df.loc[df["course_id"].eq("C006"), "texto"].iloc[0])
        for token in doc_c006:
            print(token.i, repr(token.text), token.idx)

        span_gran_miop = doc_c006[1:3]
        print("Span:", span_gran_miop.text)
        assert span_gran_miop.text == "gran miop"
        """
    ),
    markdown(
        """
        ## Etapa 2 · Reglas con `Matcher` y `SpanRuler`

        ### Baseline literal

        Un baseline debe ser sencillo, comprensible y medible. Buscaremos la
        subcadena `miop`.

        La errata original `miopai` habría sido detectada porque contiene
        `miop`. El corpus usa ahora `mipía`, que sí produce el falso negativo
        que pretendía ilustrar.
        """
    ),
    code(
        """
        df["pred_literal"] = (
            df["texto"].str.casefold().str.contains("miop", regex=False)
        )
        metrics_literal = binary_metrics(
            df["gold_ever_document"].tolist(),
            df["pred_literal"].tolist(),
        )
        pd.Series(metrics_literal, name="baseline_literal")
        """
    ),
    code(
        """
        errores_literal = df.loc[
            df["gold_ever_document"].ne(df["pred_literal"]),
            ["course_id", "texto", "gold_ever_document", "pred_literal"],
        ].copy()
        errores_literal["tipo_error"] = errores_literal.apply(
            lambda row: "FP" if row["pred_literal"] else "FN",
            axis=1,
        )
        errores_literal
        """
    ),
    markdown(
        """
        Busca la causa, no solo el número:

        - negación, familiar y educación producen falsos positivos;
        - `mipía` y la refracción sin palabra producen falsos negativos;
        - `miopización` demuestra que una raíz demasiado permisiva también falla.

        Este inventario de errores dicta la siguiente versión. No se añaden
        reglas «porque suenan razonables»: se añade un caso, una regla y una
        prueba.
        """
    ),
    markdown(
        """
        ### Reglas de tokens con `SpanRuler`

        `SpanRuler` guarda coincidencias en `Doc.spans` sin obligarnos a tratarlas
        como NER estadístico. `phrase_matcher_attr="LOWER"` hace las frases
        insensibles a mayúsculas. Los identificadores de patrón permiten saber
        qué regla disparó cada span.

        Documentación oficial:
        [spaCy SpanRuler](https://spacy.io/api/spanruler) y
        [Matcher](https://spacy.io/api/matcher/).
        """
    ),
    code(
        """
        nlp_rules = spacy.blank("es")
        nlp_rules.add_pipe("sentencizer")
        ruler = nlp_rules.add_pipe(
            "span_ruler",
            config={
                "spans_key": "myopia_mentions",
                "phrase_matcher_attr": "LOWER",
                "validate": True,
            },
        )
        ruler.add_patterns([
            {"label": "MYOPIA", "id": "TERM_MYOPIA", "pattern": "miopía"},
            {"label": "MYOPIA", "id": "TERM_MYOPIA_NO_ACCENT", "pattern": "miopia"},
            {"label": "MYOPIA", "id": "TERM_MIOPE", "pattern": "miope"},
            {"label": "MYOPIA", "id": "TERM_MIOP_CA", "pattern": "miop"},
            {"label": "MYOPIA", "id": "TYPO_MIPIA", "pattern": "mipía"},
            {"label": "MYOPIA", "id": "SEVERITY_MAGNA", "pattern": "miopía magna"},
            {"label": "MYOPIA", "id": "SEVERITY_HIGH", "pattern": "alta miopía"},
            {"label": "MYOPIA", "id": "SEVERITY_GRAN_CA", "pattern": "gran miop"},
        ])

        demo_rules = nlp_rules("Paciente con alta miopía. Madre miope.")
        [(span.text, span.label_, span.id_, span.start_char, span.end_char)
         for span in demo_rules.spans["myopia_mentions"]]
        """
    ),
    markdown(
        """
        `SpanRuler` puede devolver spans solapados (`miopía` y `alta miopía`).
        Eso es útil si representan atributos distintos; si solo quieres una
        mención, define y prueba una política de resolución, por ejemplo
        «conservar el span más largo».

        En el módulo reutilizable usamos una expresión regular conservadora para
        mantener offsets y admitir las variantes validadas. Inspecciona su
        implementación: la trazabilidad también exige poder leer el código.
        """
    ),
    code(
        """
        print(inspect.getsource(extract_mentions))
        """
    ),
    markdown(
        """
        ## Etapa 3 · Expresiones regulares

        ### Refracción estructurada

        Un curso puede no decir «miopía» y sí contener `OD −8,00 D`. Extraer
        números exige más contexto que una regex global:

        - exigimos ojo (`OD`, `OI`, `AO`);
        - exigimos signo para no confundir agudeza visual;
        - aceptamos coma/punto y varios signos menos Unicode;
        - calculamos `esfera + cilindro/2`;
        - marcamos si solo consta esfera;
        - conservamos texto, offsets y regla.
        """
    ),
    code(
        """
        refracciones_demo = [
            "OD −8,00 D; OI −7,50 D.",
            "OD esf -3.00 cil -1.00 x 90",
            "AV OD 0.8, OI 1.0",
            "OD +1,00 D; OI +0,75 D",
        ]

        filas_refraccion = [
            {"entrada": text, **measurement}
            for text in refracciones_demo
            for measurement in parse_refractions(text)
        ]
        pd.DataFrame(filas_refraccion)
        """
    ),
    code(
        """
        assert MYOPIA_THRESHOLD_D == -0.50
        assert MYOPIA_HIGH_THRESHOLD_D == -6.00
        assert parse_refractions("AV OD 0.8, OI 1.0") == []
        assert (
            parse_refractions("OD esf -3.00 cil -1.00 x 90")[0]
            ["spherical_equivalent_d"]
            == -3.5
        )
        print("Comprobaciones numéricas correctas.")
        """
    ),
    markdown(
        """
        ### Tu turno 2

        Añade primero como prueba y luego como regla formatos reales
        seudonimizados por el hospital:

        - `Rx: OD -2.25 (-0.50 a 90º)`;
        - `AO: -1,5 esf`;
        - refracción en columnas o saltos de línea.

        Preguntas expertas: ¿es refracción objetiva o subjetiva?, ¿con
        cicloplejia?, ¿el cilindro se expresa positivo o negativo?, ¿puedes
        calcular el equivalente si falta información? No ocultes la incertidumbre.
        """
    ),
    markdown(
        """
        ## Etapa 4 · Contexto clínico

        ### Contexto multidimensional

        Una mención no recibe una clase única. Se describen dimensiones
        independientes:

        | Dimensión | Ejemplo |
        |---|---|
        | Aserción | afirmada, negada, posible |
        | Experienciador | paciente, familiar |
        | Temporalidad | actual, histórica |
        | Contexto | clínico, educativo |

        La ventana de reglas es un baseline tipo ConText. Su alcance debe
        validarse localmente. medspaCy implementa ConText, pero su repositorio
        indica que no incluye reglas ConText españolas listas para usar; no
        debemos asumir que instalarlo resuelve el castellano o catalán.
        """
    ),
    code(
        """
        cursos_contexto = ["C002", "C003", "C004", "C005", "C008", "C010", "C019", "C022"]
        contexto_rows = []
        for row in df[df["course_id"].isin(cursos_contexto)].itertuples():
            for mention in extract_mentions(row.texto):
                contexto_rows.append({"course_id": row.course_id, **mention})

        pd.DataFrame(contexto_rows)[[
            "course_id", "text", "assertion", "experiencer",
            "temporality", "context", "rule_id", "start", "end",
        ]]
        """
    ),
    code(
        """
        texto_offsets = df.loc[df["course_id"].eq("C022"), "texto"].iloc[0]
        for mention in extract_mentions(texto_offsets):
            recovered = texto_offsets[mention["start"]:mention["end"]]
            print(mention["text"], "→", mention["assertion"],
                  mention["experiencer"], "| offset:", recovered)
            assert recovered == mention["text"]
        """
    ),
    markdown(
        """
        **Limitación consciente:** las reglas actuales usan puntuación y ventanas.
        Fallarán con plantillas, encabezados locales, negaciones largas y frases
        coordinadas. Es una ventaja docente: podemos ver el error y convertirlo
        en dato anotado y test. Una caja negra no elimina esos fallos; solo puede
        hacerlos menos visibles.
        """
    ),
    markdown(
        """
        ## Pipeline de curso completo

        Política estricta:

        - confirmado: mención clínica afirmada del paciente o refracción miópica;
        - posible: solo evidencia incierta del paciente;
        - no sustentado: negada, familiar, educativa o ausente;
        - histórico confirmado cuenta para `ever_myopia`, pero no se transforma
          automáticamente en miopía actual.
        """
    ),
    code(
        """
        processed = process_courses(df.to_dict(orient="records"))
        results = pd.DataFrame(processed)
        results["pred_rules"] = results["ever_myopia"].astype(bool)

        results[[
            "patient_id", "course_id", "texto", "document_status",
            "current_status", "ever_myopia", "high_myopia_numeric",
            "evidence_count",
        ]]
        """
    ),
    code(
        """
        comparison = pd.DataFrame({
            "literal": binary_metrics(
                df["gold_ever_document"].tolist(),
                df["pred_literal"].tolist(),
            ),
            "rules_plus_refraction": binary_metrics(
                results["gold_ever_document"].tolist(),
                results["pred_rules"].tolist(),
            ),
        })
        comparison
        """
    ),
    code(
        """
        errores_rules = results.loc[
            results["gold_ever_document"].ne(results["pred_rules"]),
            [
                "course_id", "texto", "gold_ever_document", "pred_rules",
                "mentions", "refractions",
            ],
        ]
        print("Errores en el corpus sintético:", len(errores_rules))
        errores_rules
        """
    ),
    markdown(
        """
        Que el corpus sintético llegue a resultados perfectos solo demuestra que
        el código reproduce los ejemplos que nosotros inventamos. **No estima el
        rendimiento clínico.** El primer corpus real bloqueado debe descubrir
        vocabulario, plantillas y distribuciones que aún ignoramos.
        """
    ),
    code(
        """
        # Pruebas de humo dentro del propio tutorial.
        assert phenotype_course("Madre con miopía.")["ever_myopia"] is False
        assert phenotype_course("No se descarta miopía.")["document_status"] == "possible"
        assert phenotype_course("Intervenido por miopía con LASIK en 2018.")["ever_myopia"]
        assert phenotype_course("Refracción OD −8,00 D.")["high_myopia_numeric"]
        assert phenotype_course("Se explica que la miopía aumenta el riesgo.")["ever_myopia"] is False
        print("Pipeline coherente con los casos centinela.")
        """
    ),
    markdown(
        """
        ## Etapa 6 · Validación

        ### Evaluar como un profesional

        ### Tres evaluaciones, no una

        1. **Mención:** offsets exactos o solapados y atributos de contexto.
        2. **Curso:** evidencia positiva/posible/no sustentada.
        3. **Paciente:** fenotipo longitudinal final.

        ### Diseño recomendado

        - guía congelada antes del test;
        - doble anotación y adjudicación;
        - partición por paciente, nunca por curso;
        - conjunto de desarrollo para reglas y test bloqueado;
        - intervalos de confianza y prevalencia;
        - desglose por idioma, servicio, plantilla, centro y tiempo;
        - análisis cualitativo de todos los FN y una muestra de FP/TN/TP.

        Sensibilidad y VPP responden a costes diferentes. La lista de cribado
        puede favorecer sensibilidad; contactar pacientes o modificar registros
        exige más VPP y revisión humana.
        """
    ),
    code(
        """
        def stable_patient_split(patient_id: str) -> str:
            # Demostración reproducible. En un estudio real se estratifica y
            # documenta la semilla, manteniendo todos los cursos del paciente juntos.
            bucket = int(hashlib.sha256(patient_id.encode()).hexdigest()[:8], 16) % 10
            return "test" if bucket < 2 else "development"

        df["split"] = df["patient_id"].map(stable_patient_split)
        assert df.groupby("patient_id")["split"].nunique().max() == 1
        display(df.groupby(["split", "gold_ever_document"]).size().rename("n").to_frame())
        """
    ),
    markdown(
        """
        En un conjunto pequeño, una partición hash puede quedar desequilibrada.
        Para el estudio real usa una partición agrupada y estratificada, revisada
        antes de mirar resultados. El principio no negociable es que un paciente
        no cruce particiones.

        Mantén una tabla de errores con:

        `curso | gold | pred | FP/FN | causa | corrección | regla afectada`

        Categorías útiles: léxico, límites, negación, experienciador, temporalidad,
        sección, número, idioma, OCR/errata, plantilla y error de anotación.
        """
    ),
    markdown(
        """
        ## Etapa 5 · Modelos clínicos preentrenados

        ### Cuándo pasar a modelos

        ### Escalera razonable

        1. reglas auditables;
        2. clasificador de documento con caracteres/TF-IDF;
        3. modelo neuronal para spans y atributos;
        4. híbrido: modelo propone, reglas resuelven contexto y umbrales.

        El modelo
        [`PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`](https://huggingface.co/PlanTL-GOB-ES/roberta-base-biomedical-clinical-es)
        fue preentrenado con texto biomédico y clínico español. Su tarjeta dice
        que está listo para *masked language modeling* y pensado para ajustarse
        a tareas posteriores. **No es un NER de miopía listo para usar.**

        Para nuestro objetivo necesitas ejemplos locales anotados y ajustar:

        - detección de spans `MYOPIA`;
        - clasificación de aserción/experienciador/temporalidad;
        - o clasificación de curso, siempre conservando evidencia.

        Un F1 publicado en fármacos, tumores o ictus no es el F1 de miopía en tus
        cursos.
        """
    ),
    markdown(
        """
        ### Esqueleto opcional de ajuste

        No se ejecuta en la ruta básica: requiere `transformers`, `datasets`,
        PyTorch, GPU aprobada y un corpus real autorizado.

        ```python
        BASE_MODEL = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
        # 1. Convertir offsets adjudicados a etiquetas BIO por subtoken.
        # 2. Partir por patient_id.
        # 3. AutoTokenizer.from_pretrained(BASE_MODEL)
        # 4. AutoModelForTokenClassification.from_pretrained(
        #        BASE_MODEL, num_labels=len(label2id))
        # 5. Ajustar solo con desarrollo; elegir umbral en validación.
        # 6. Evaluar una vez sobre test bloqueado.
        # 7. Guardar checkpoint, tokenizer, label map y model card.
        ```

        Antes de descargar un modelo, revisa licencia, tarjeta, procedencia,
        limitaciones, capacidad del entorno y aprobación institucional.
        """
    ),
    markdown(
        """
        ### Aprendizaje activo sin contaminar el test

        Prioriza para anotación:

        - desacuerdos entre reglas y modelo;
        - predicciones cerca del umbral;
        - frases con nuevos términos o plantillas;
        - subgrupos con peor sensibilidad;
        - una muestra aleatoria que evite sesgo de selección.

        Añade esos ejemplos al **desarrollo**, reentrena y deja el test intacto.
        Versiona corpus, guía y modelo conjuntamente.
        """
    ),
    markdown(
        """
        ## Etapa 7 · Oracle

        ### Privacidad y procesamiento por lotes

        Los cursos clínicos son datos de salud. El RGPD los trata como categoría
        especial; la seudonimización es una garantía útil, pero no anonimiza.
        Requiere base jurídica, minimización, control de acceso y las aprobaciones
        del hospital.

        Fuentes oficiales:
        [RGPD en EUR-Lex](https://eur-lex.europa.eu/legal-content/es/TXT/?uri=CELEX%3A32016R0679)
        y
        [AEPD: los datos seudonimizados siguen siendo personales](https://www.aepd.es/preguntas-frecuentes/0-conceptos-basicos/FAQ-0006-sobre-los-datos-seudonimizados).

        La lista operativa está en
        [produccion_oracle.md](../docs/produccion_oracle.md).
        """
    ),
    code(
        """
        # Demostración con una clave FICTICIA. En producción se lee de un gestor
        # de secretos y nunca se guarda junto al corpus.
        demo_secret = b"clave-docente-no-usar-en-produccion"
        demo_ids = ["HOSP-0001", "HOSP-0002", "HOSP-0001"]
        [pseudonymize_id(value, demo_secret) for value in demo_ids]
        """
    ),
    code(
        """
        # Esqueleto no ejecutado: valores parametrizados, columnas mínimas y
        # fetchmany. La vista debe ser fija y aprobada.
        RUN_ORACLE_EXAMPLE = False

        if RUN_ORACLE_EXAMPLE:
            import os
            import oracledb

            connection = oracledb.connect(
                user=os.environ["MYOPIA_ORACLE_USER"],
                password=os.environ["MYOPIA_ORACLE_PASSWORD"],
                dsn=os.environ["MYOPIA_ORACLE_DSN"],
            )
            sql = '''
                SELECT patient_id, course_id, course_date, language_code, course_text
                FROM approved_clinical_course_view
                WHERE course_date >= :date_from
                  AND course_date < :date_to
                  AND course_text IS NOT NULL
                ORDER BY course_date, course_id
            '''
            with connection.cursor() as cursor:
                cursor.execute(sql, date_from="2025-01-01", date_to="2025-02-01")
                while batch := cursor.fetchmany(500):
                    # Seudonimizar y procesar dentro de la zona segura.
                    pass
            connection.close()
        """
    ),
    code(
        """
        # Procesamiento spaCy por lotes con metadatos; evita nlp(texto) en un bucle
        # cuando el pipeline estadístico sea grande.
        text_and_context = [
            (row.texto, {"course_id": row.course_id})
            for row in df.itertuples()
        ]
        batched = []
        for doc_item, context in nlp_rules.pipe(
            text_and_context, as_tuples=True, batch_size=64
        ):
            batched.append({
                "course_id": context["course_id"],
                "n_tokens": len(doc_item),
                "n_spans": len(doc_item.spans.get("myopia_mentions", [])),
            })

        pd.DataFrame(batched).head()
        """
    ),
    markdown(
        """
        Nunca registres texto o identificadores en logs. Registra conteos,
        tiempos, versión del pipeline, hash de configuración y errores técnicos
        sin contenido clínico.

        `nlp.pipe` admite multiproceso, pero en Windows copiar modelos entre
        procesos puede ser costoso. Mide `batch_size` y memoria antes de aumentar
        `n_process`.
        """
    ),
    markdown(
        """
        ## Etapa 8 · Agregación longitudinal

        ### Fenotipo final por paciente

        `ever_myopia` es acumulativo: una evidencia histórica afirmada basta.
        El estado actual usa la evidencia actual informativa más reciente:
        confirmada, negada o posible. Una mención histórica no se convierte en
        estado actual.

        Esta política es explícita, discutible y versionable. Oftalmología debe
        decidir qué hacer con cirugía refractiva, diagnósticos antiguos,
        refracciones contradictorias y periodos sin documentación.
        """
    ),
    code(
        """
        patient_rows = []
        for patient_id, group in results.groupby("patient_id", sort=True):
            summary = aggregate_patient(
                group.sort_values("fecha").to_dict(orient="records")
            )
            patient_rows.append({"patient_id": patient_id, **summary})

        patient_phenotypes = pd.DataFrame(patient_rows)
        patient_phenotypes
        """
    ),
    code(
        """
        # P005 conserva historia positiva y la última negación actual.
        timeline_p005 = results.loc[
            results["patient_id"].eq("P005"),
            [
                "fecha", "course_id", "texto", "ever_myopia",
                "current_status", "document_status",
            ],
        ].sort_values("fecha")
        display(timeline_p005)
        display(patient_phenotypes.loc[patient_phenotypes["patient_id"].eq("P005")])
        """
    ),
    markdown(
        """
        ### Tu turno 3

        Diseña con Oftalmología tres fenotipos distintos:

        1. miopía alguna vez;
        2. miopía actual;
        3. miopía alta numérica bilateral.

        Escribe antes del código cómo resolverás:

        - un ojo miópico y el otro no;
        - esfera sin cilindro;
        - medida antigua frente a negación reciente;
        - LASIK/PRK;
        - «miopía magna» sin refracción;
        - discordancia entre código diagnóstico y texto.
        """
    ),
    markdown(
        """
        ## Anexo experto · Producción y deriva

        Un pipeline validado deja de estar validado cuando cambia su entrada.
        Monitoriza, sin almacenar texto innecesario:

        - volumen y porcentaje de cursos vacíos;
        - idioma y tipo documental;
        - prevalencia de menciones y fenotipo;
        - reglas desconocidas y formatos numéricos no interpretados;
        - proporción de posibles y casos enviados a revisión;
        - sensibilidad/VPP en muestras periódicas anotadas;
        - resultados por centro, servicio, edad y otros subgrupos permitidos;
        - cambios de plantilla, EHR, versión de reglas o modelo.

        Despliega por fases: modo silencioso, retrospectivo, piloto siempre
        revisado y monitorizado. El sistema propone evidencia; una persona decide
        cualquier acción asistencial.
        """
    ),
    markdown(
        """
        ## Proyecto maestro

        Para pasar de alumno a responsable del sistema:

        1. revisa el protocolo con dos oftalmólogos;
        2. crea 100 cursos sintéticos adicionales antes de tocar datos reales;
        3. anota un piloto real autorizado con doble revisión;
        4. mide el baseline sin cambiar reglas;
        5. clasifica todos los errores;
        6. mejora únicamente en desarrollo y añade tests;
        7. bloquea un test por paciente;
        8. compara reglas, modelo clásico y transformer ajustado;
        9. redacta una model card y una ficha de datos;
        10. ejecuta un piloto silencioso y estudia deriva.

        **Criterio de maestría:** puedes justificar cada etiqueta, reconstruir la
        evidencia, explicar el denominador de cada métrica y decir con precisión
        en qué población y periodo el rendimiento es válido.
        """
    ),
    markdown(
        """
        <details>
        <summary><strong>Pruebas y retos de ampliación</strong></summary>

        - Haz fallar una prueba con `AF:` y corrige el alcance de experienciador.
        - Añade `miop.` como abreviatura solo si aparece realmente y mide sus FP.
        - Compara match exacto frente a match con solapamiento de offsets.
        - Calcula intervalos bootstrap agrupando por paciente.
        - Diseña una cola de revisión que priorice desacuerdos sin excluir una
          muestra aleatoria.
        - Mide deriva temporal entre dos años sin usar el test para reentrenar.
        - Crea una tarjeta del modelo con uso previsto, exclusiones y umbrales.

        Ejecuta las pruebas completas:

        ```powershell
        uv run python -m unittest discover -s tests -v
        ```

        </details>
        """
    ),
    markdown(
        """
        ## Referencias primarias y documentación

        - [International Myopia Institute: definición y clasificación](https://myopiainstitute.org/imi-whitepaper/defining-and-classifying-myopia/2023-02-12_imi-defining-and-classifying-myopia-report_english/)
        - [spaCy: canalizaciones y `nlp.pipe`](https://spacy.io/usage/processing-pipelines)
        - [spaCy: `SpanRuler`](https://spacy.io/api/spanruler)
        - [spaCy: `Matcher`](https://spacy.io/api/matcher/)
        - [BSC/PlanTL: RoBERTa biomédico-clínico español](https://huggingface.co/PlanTL-GOB-ES/roberta-base-biomedical-clinical-es)
        - [BSC-LT: MrBERT-biomed](https://huggingface.co/BSC-LT/MrBERT-biomed)
        - [BSC-NLP4BIA: colección Clinical NMT-NER](https://huggingface.co/collections/BSC-NLP4BIA/clinical-nmt-ner)
        - [BSC-NLP4BIA: DT4H NER multilingüe](https://huggingface.co/BSC-NLP4BIA/DT4H_XLM-R_mtl_multilingual_multilabel)
        - [medspaCy: componentes y soporte lingüístico](https://github.com/medspacy/medspacy)
        - [RGPD, texto oficial](https://eur-lex.europa.eu/legal-content/es/TXT/?uri=CELEX%3A32016R0679)
        - [AEPD: seudonimización](https://www.aepd.es/preguntas-frecuentes/0-conceptos-basicos/FAQ-0006-sobre-los-datos-seudonimizados)

        Última revisión documental: julio de 2026.
        """
    ),
    markdown(
        """
        ## Checklist final

        Puedes avanzar a datos autorizados cuando puedas responder «sí»:

        - [ ] La pregunta clínica y unidad de análisis están congeladas.
        - [ ] Dos revisores entienden igual la guía.
        - [ ] Los offsets reconstruyen toda evidencia.
        - [ ] Negación, familia, incertidumbre y tiempo son dimensiones separadas.
        - [ ] El test está separado por paciente y permanece bloqueado.
        - [ ] Cada cambio de regla incluye una prueba.
        - [ ] El rendimiento se informa con denominadores y subgrupos.
        - [ ] La extracción Oracle minimiza datos y usa parámetros.
        - [ ] La seudonimización y las claves están separadas.
        - [ ] Hay revisión humana y plan de monitorización.

        Si una casilla falla, ese es el siguiente experimento; no una razón para
        ocultar incertidumbre.
        """
    ),
]

# En la fuente, validación se declara antes para poder mantener juntos sus
# ejemplos ejecutables. El notebook publicado respeta el mapa docente original:
# etapa 5 (modelos) precede a etapa 6 (validación).
def _find_markdown_heading(prefix: str) -> int:
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith(prefix):
            return index
    raise ValueError(f"No se encontró el encabezado: {prefix}")


def _insert_after_heading(prefix: str, *new_cells: dict) -> None:
    index = _find_markdown_heading(prefix)
    cells[index + 1:index + 1] = list(new_cells)


_insert_after_heading(
    "## Mapa del proyecto",
    markdown(
        """
        ### Un caso vertical, competencias horizontales

        La miopía es el caso de uso concreto que recorrerá las ocho etapas. El
        objetivo profesional es dominar un método reutilizable después en
        nefrología: redefinir el fenotipo, crear otra guía, cambiar entidades y
        números, volver a anotar y validar.

        En cada etapa encontrarás una caja **Transferencia a nefrología**. El
        itinerario ampliado y el catálogo de tareas están en
        [transferencia_nefrologia.md](../docs/transferencia_nefrologia.md).

        No transferiremos reglas clínicas de miopía a riñón; transferiremos
        competencias de NLP, diseño experimental y gobernanza.
        """
    ),
)

_insert_after_heading(
    "## Etapa 1 · Corpus de prueba y fundamentos de spaCy",
    markdown(
        """
        > **Transferencia a nefrología · corpus.** Cambiarán el vocabulario y la
        > referencia, pero `Doc`, `Token`, `Span`, frases y offsets serán los
        > mismos. Un futuro corpus piloto podría contrastar «ERC G4», «fracaso
        > renal agudo», «en HD», «portador de FAV» y «trasplantado renal».
        > Empieza siempre con ejemplos ficticios que incluyan abreviaturas,
        > plantillas, castellano/catalán y ausencia de mención.
        """
    ),
)

_insert_after_heading(
    "## Etapa 2 · Reglas con `Matcher` y `SpanRuler`",
    markdown(
        """
        > **Transferencia a nefrología · reglas.** `SpanRuler` puede construir
        > baselines de alta trazabilidad para ERC/CKD, FRA/AKI, HD/HDF/DP,
        > trasplante, FAV y CVC. Cada abreviatura debe validarse por sección y
        > servicio: `DP`, por ejemplo, puede ser ambigua. El valor del ejercicio
        > no es crear un diccionario infinito, sino aprender a medir cuándo deja
        > de bastar.
        """
    ),
)

_insert_after_heading(
    "## Etapa 3 · Expresiones regulares",
    markdown(
        """
        > **Transferencia a nefrología · números.** El equivalente esférico se
        > sustituirá por extractores de creatinina, eGFR, ACR/PCR, potasio,
        > bicarbonato o diuresis. Una extracción experta une `valor + unidad +
        > fecha + espécimen + método + límites`, normaliza unidades y evita
        > confundir un resultado con un objetivo, una dosis o un valor copiado
        > de otra fecha. Para AKI/ERC habrá que combinar texto y datos
        > estructurados; una regex aislada no establece el fenotipo.
        """
    ),
)

_insert_after_heading(
    "## Etapa 4 · Contexto clínico",
    markdown(
        """
        > **Transferencia a nefrología · contexto.** «Descartar FRA», «sin ERC»,
        > «madre en diálisis», «antecedente de trasplante» y «riesgo de
        > nefrotoxicidad» ilustran las mismas dimensiones. Después aparecerán
        > atributos propios: estadio, etiología, modalidad, acceso, injerto y
        > relación fármaco–dosis. Mantén esas dimensiones separadas antes de
        > agregarlas.
        """
    ),
)

_insert_after_heading(
    "## Etapa 5 · Modelos clínicos preentrenados",
    markdown(
        """
        ### Ecosistema BSC y usos futuros

        Distingue tres objetos:

        1. **encoder base:** conoce regularidades del lenguaje, pero no nuestras
           etiquetas;
        2. **modelo ajustado:** resuelve una tarea y ontología concretas;
        3. **sistema clínico:** añade contexto, reglas, umbrales, agregación,
           trazabilidad y validación local.

        | Recurso BSC/PlanTL | Qué es | Uso que se puede estudiar |
        |---|---|---|
        | `roberta-base-biomedical-clinical-es` | Encoder español, objetivo MLM | Ajustar NER o clasificación renal |
        | `longformer-base-4096-biomedical-clinical-es` | Encoder para más contexto | Comparar cursos largos frente a ventanas |
        | `BSC-LT/MrBERT-biomed` | Encoder biomédico 2026, contexto declarado 8192 | Ajuste y recuperación ES/EN |
        | Clinical NMT-NER | NER de enfermedad, fármaco y profesión ES/CA | Generar candidatos y acelerar anotación |
        | DT4H NER | Enfermedad, síntoma y procedimiento multilingüe | Baseline NER general, con validación local |
        | Modelos de anonimización ES/CA | Token classification de identificadores | Una capa del proceso de desidentificación |

        Potenciales tareas nefrológicas: NER de diagnósticos/fármacos,
        clasificación de episodios, relaciones fármaco–dosis o
        acceso–complicación, normalización terminológica, recuperación semántica
        de cohortes y cronologías asistidas.

        **Límite crucial:** un NER general puede proponer «enfermedad renal
        crónica», pero no garantiza aserción, estadio, etiología, estado actual
        ni fenotipo longitudinal. Eso pertenece a nuestro sistema.
        """
    ),
    code(
        """
        # Experimento OPCIONAL. Requiere transformers, descarga autorizada,
        # revisión de la tarjeta y ejecución dentro de la infraestructura segura.
        RUN_BSC_DISEASE_NER = False

        if RUN_BSC_DISEASE_NER:
            from transformers import pipeline

            disease_ner = pipeline(
                "token-classification",
                model="BSC-NLP4BIA/bsc-bio-ehr-es-distemist",
                aggregation_strategy="simple",
            )
            synthetic_texts = [
                "Paciente con miopía magna bilateral.",
                "Antecedente de enfermedad renal crónica G4 por nefropatía diabética.",
            ]
            for synthetic_text in synthetic_texts:
                print(synthetic_text)
                print(disease_ner(synthetic_text))

        # La comparación válida anotaría spans locales y mediría match exacto y
        # solapado frente a nuestras reglas bajo el mismo conjunto de test.
        """
    ),
)

_insert_after_heading(
    "## Etapa 6 · Validación",
    markdown(
        """
        > **Transferencia a nefrología · evaluación.** Cada tarea tiene un
        > denominador: F1 de spans para NER, macro/micro F1 para multietiqueta,
        > exactitud/top-k para normalización, Recall@k/nDCG para búsqueda y
        > métricas clínicas por episodio/paciente para fenotipos. Analiza por
        > estadio, modalidad, idioma, centro y periodo. Un buen NER no demuestra
        > automáticamente un buen detector de AKI o ERC.
        """
    ),
)

_insert_after_heading(
    "## Etapa 7 · Oracle",
    markdown(
        """
        > **Transferencia a nefrología · datos.** Los fenotipos renales suelen
        > exigir cursos + laboratorio + procedimientos + farmacia. Diseña
        > extracciones incrementales separadas, conserva tiempos y unidades y
        > une mediante seudónimos dentro de la zona segura. No conviertas Oracle
        > en una descarga masiva: minimiza columnas y población para cada
        > pregunta.
        """
    ),
)

_insert_after_heading(
    "## Etapa 8 · Agregación longitudinal",
    markdown(
        """
        > **Transferencia a nefrología · tiempo.** Aquí aparecen los proyectos
        > más potentes: ERC sostenida, episodios de AKI, inicio/cambio de
        > modalidad de diálisis, creación y fallo de acceso, trasplante, rechazo
        > o pérdida de injerto. Se necesita una máquina de estados clínica
        > explícita; «alguna vez», «actual» y «última evidencia» no siempre
        > bastarán.
        """
    ),
)


validation_start = _find_markdown_heading("## Etapa 6 · Validación")
models_start = _find_markdown_heading("## Etapa 5 · Modelos clínicos preentrenados")
oracle_start = _find_markdown_heading("## Etapa 7 · Oracle")
assert validation_start < models_start < oracle_start

cells = (
    cells[:validation_start]
    + cells[models_start:oracle_start]
    + cells[validation_start:models_start]
    + cells[oracle_start:]
)


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "miopia (.venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(f"Generado {OUTPUT} con {len(cells)} celdas")
