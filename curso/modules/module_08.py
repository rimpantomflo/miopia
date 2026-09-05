from curso.notebook_factory import code, common_setup, md

TITLE = "08 · Proyecto integrador de nefrología"


def build() -> list[dict]:
    return [
        md(
            """
            # 08 · Proyecto integrador de nefrología

            Construiremos un baseline para detectar evidencia de tratamiento
            renal sustitutivo (TRS), modalidad actual y acceso. Es un ejercicio
            metodológico con datos ficticios, no una herramienta asistencial.
            """
        ),
        md(
            """
            ## Objetivos

            - definir un fenotipo renal explícito;
            - cargar corpus y diccionario versionados;
            - construir baseline literal;
            - extraer conceptos con offsets;
            - clasificar contexto y estado;
            - evaluar documento;
            - agregar longitudinalmente;
            - analizar errores;
            - diseñar ampliación a ERC/AKI sin inferencias indebidas.
            """
        ),
        common_setup(),
        code(
            """
            import json
            import re
            import unicodedata
            from collections import defaultdict

            import pandas as pd

            from clinical_nlp_course import validate_concept_dictionary
            from miopia_nlp import binary_metrics

            renal_df = pd.read_json(
                PROJECT_ROOT / "data" / "nefrologia_sintetica.jsonl",
                lines=True,
            )
            renal_df["date"] = pd.to_datetime(renal_df["date"])
            concepts = json.loads(
                (PROJECT_ROOT / "data" / "conceptos_renales_sinteticos.json")
                .read_text(encoding="utf-8")
            )
            assert validate_concept_dictionary(concepts) == []
            print(len(renal_df), "cursos", renal_df["patient_id"].nunique(), "pacientes")
            """
        ),
        md(
            """
            ## 1. Contrato clínico

            Salidas:

            - evidencia documental de TRS alguna vez;
            - modalidad actual mencionada;
            - acceso actual mencionado;
            - evidencia y offsets;
            - estado longitudinal final por paciente.

            No inferiremos:

            - indicación;
            - adecuación;
            - diagnóstico de ERC/AKI;
            - modalidad si solo existe un acceso;
            - uso de acceso si solo consta su creación.
            """
        ),
        code(
            """
            modality_labels = {
                "HEMODIALYSIS",
                "HEMODIAFILTRATION",
                "PERITONEAL_DIALYSIS",
                "KIDNEY_TRANSPLANT",
            }
            access_labels = {
                "AV_FISTULA",
                "TUNNELED_DIALYSIS_CATHETER",
                "NONTUNNELED_DIALYSIS_CATHETER",
                "PERITONEAL_CATHETER",
            }
            non_trs_labels = {"CONSERVATIVE_KIDNEY_CARE"}
            """
        ),
        md(
            """
            ### Ejercicio 1 · Anotación ciega

            Antes de ejecutar reglas, anota manualmente N006, N007, N012, N018,
            N019, N024, N027 y N028. Para cada mención:

            `concepto | aserción | experienciador | estado | modalidad actual`

            Compara después con gold y predicciones.
            """
        ),
        md(
            """
            ## 2. Baseline literal

            Busca cualquier variante de modalidad sin interpretar contexto.
            """
        ),
        code(
            """
            concept_by_id = {concept["concept_id"]: concept for concept in concepts}
            modality_terms = []
            for concept_id in modality_labels:
                concept = concept_by_id[concept_id]
                modality_terms.extend([concept["preferred_term"], *concept["variants"]])
            modality_terms = sorted(set(modality_terms), key=len, reverse=True)

            literal_pattern = re.compile(
                "|".join(re.escape(term) for term in modality_terms),
                flags=re.IGNORECASE,
            )
            renal_df["pred_literal"] = renal_df["text"].map(
                lambda text: bool(literal_pattern.search(text))
            )
            pd.Series(
                binary_metrics(
                    renal_df["gold_document_trs_evidence"].tolist(),
                    renal_df["pred_literal"].tolist(),
                ),
                name="literal",
            )
            """
        ),
        code(
            """
            literal_errors = renal_df.loc[
                renal_df["gold_document_trs_evidence"].ne(renal_df["pred_literal"]),
                ["course_id", "text", "gold_document_trs_evidence", "pred_literal"],
            ].copy()
            literal_errors["error"] = literal_errors.apply(
                lambda row: "FP" if row["pred_literal"] else "FN",
                axis=1,
            )
            literal_errors
            """
        ),
        md(
            """
            Clasifica causas: familiar, educación, plan, negación, abreviatura,
            retirada o término ausente. Esa taxonomía dirige la siguiente versión.
            """
        ),
        md(
            """
            ## 3. Compilar diccionario a reglas

            Ordenamos términos largos antes de cortos y conservamos
            `concept_id`. Los límites alrededor de abreviaturas requieren cuidado.
            """
        ),
        code(
            """
            dictionary_patterns = []
            for concept in concepts:
                for term in [concept["preferred_term"], *concept["variants"]]:
                    escaped = re.escape(term)
                    pattern = re.compile(
                        rf"(?<!\\w){escaped}(?!\\w)",
                        flags=re.IGNORECASE,
                    )
                    dictionary_patterns.append({
                        "concept_id": concept["concept_id"],
                        "term": term,
                        "pattern": pattern,
                    })
            dictionary_patterns.sort(key=lambda row: len(row["term"]), reverse=True)
            """
        ),
        code(
            """
            def fold(text):
                decomposed = unicodedata.normalize("NFKD", text.casefold())
                return "".join(
                    character
                    for character in decomposed
                    if not unicodedata.combining(character)
                )

            def clause_bounds(text, start, end):
                separators = ".;\\n"
                left = max(text.rfind(mark, 0, start) for mark in separators) + 1
                right_candidates = [
                    pos
                    for mark in separators
                    if (pos := text.find(mark, end)) != -1
                ]
                right = min(right_candidates) if right_candidates else len(text)
                return left, right
            """
        ),
        md(
            """
            ## 4. Contexto y estado

            Estados:

            - `current`;
            - `historical_or_stopped`;
            - `possible_or_planned`;
            - `negated`;
            - `family`;
            - `educational`.

            La prioridad importa. Una plantilla educativa con la palabra «no»
            no debe convertirse automáticamente en negación del paciente.
            """
        ),
        code(
            """
            FAMILY_RE = re.compile(r"\\b(madre|padre|herman[oa]|familiar)\\b")
            EDUCATIONAL_RE = re.compile(
                r"\\b(se informa|se explican|plantilla informativa|opciones|puede realizarse)\\b"
            )
            POSSIBLE_RE = re.compile(
                r"\\b(posible|candidato|futuro|si progresa|si empeora|pendiente|prefiere)\\b"
            )
            NEGATED_RE = re.compile(
                r"\\b(no|sin|niega|aun no ha iniciado|aún no ha iniciado)\\b"
            )
            STOPPED_RE = re.compile(
                r"\\b(retirada|retirado|suspendida|previa|previo|perdida|pérdida|"
                r"trombosada|injerto perdido)\\b"
            )

            def classify_mention_context(clause):
                normalized = fold(clause)
                if EDUCATIONAL_RE.search(normalized):
                    return "educational"
                if FAMILY_RE.search(normalized):
                    return "family"
                if POSSIBLE_RE.search(normalized):
                    return "possible_or_planned"
                if NEGATED_RE.search(normalized):
                    return "negated"
                if STOPPED_RE.search(normalized):
                    return "historical_or_stopped"
                return "current"
            """
        ),
        code(
            """
            def extract_renal_mentions(text):
                candidates = []
                for rule in dictionary_patterns:
                    for match in rule["pattern"].finditer(text):
                        left, right = clause_bounds(text, match.start(), match.end())
                        candidates.append({
                            "concept_id": rule["concept_id"],
                            "text": match.group(),
                            "start": match.start(),
                            "end": match.end(),
                            "clause": text[left:right].strip(),
                            "status": classify_mention_context(text[left:right]),
                            "rule_term": rule["term"],
                        })
                # El término largo gana cuando hay coincidencias contenidas del mismo concepto.
                candidates.sort(key=lambda row: (row["start"], -(row["end"] - row["start"])))
                kept = []
                for candidate in candidates:
                    if any(
                        previous["concept_id"] == candidate["concept_id"]
                        and previous["start"] <= candidate["start"]
                        and previous["end"] >= candidate["end"]
                        for previous in kept
                    ):
                        continue
                    kept.append(candidate)
                return kept

            sample_mentions = extract_renal_mentions(
                "Madre en hemodiálisis. Paciente pendiente de diálisis peritoneal."
            )
            pd.DataFrame(sample_mentions)
            """
        ),
        code(
            """
            for row in renal_df[renal_df["course_id"].isin(
                ["N006", "N007", "N012", "N018", "N019", "N024", "N027", "N028"]
            )].itertuples():
                print("\\n", row.course_id, row.text)
                for mention in extract_renal_mentions(row.text):
                    recovered = row.text[mention["start"]:mention["end"]]
                    assert recovered == mention["text"]
                    print(" ", mention["concept_id"], repr(recovered), mention["status"])
            """
        ),
        md(
            """
            ### Ejercicio 2

            Encuentra errores del alcance por cláusula. Prueba coordinación,
            doble negación y «No descarta iniciar HD». Añade un caso centinela
            antes de modificar prioridades.
            """
        ),
        md(
            """
            ## 5. Fenotipo por documento

            Evidencia documental incluye modalidad afirmada actual o histórica.
            Planes, educación, familia y negación no cuentan. Modalidad actual
            solo usa `current`.
            """
        ),
        code(
            """
            def phenotype_renal_course(text):
                mentions = extract_renal_mentions(text)
                trs_evidence = [
                    mention for mention in mentions
                    if mention["concept_id"] in modality_labels
                    and mention["status"] in {"current", "historical_or_stopped"}
                ]
                current_modalities = [
                    mention["concept_id"] for mention in mentions
                    if mention["concept_id"] in modality_labels
                    and mention["status"] == "current"
                ]
                current_access = [
                    mention["concept_id"] for mention in mentions
                    if mention["concept_id"] in access_labels
                    and mention["status"] == "current"
                ]
                return {
                    "pred_document_trs_evidence": bool(trs_evidence),
                    "pred_current_modalities": sorted(set(current_modalities)),
                    "pred_current_access": sorted(set(current_access)),
                    "mentions": mentions,
                }

            course_predictions = renal_df["text"].map(phenotype_renal_course)
            pred_df = pd.concat(
                [renal_df, pd.DataFrame(course_predictions.tolist())],
                axis=1,
            )
            """
        ),
        code(
            """
            comparison = pd.DataFrame({
                "literal": binary_metrics(
                    pred_df["gold_document_trs_evidence"].tolist(),
                    pred_df["pred_literal"].tolist(),
                ),
                "context_rules": binary_metrics(
                    pred_df["gold_document_trs_evidence"].tolist(),
                    pred_df["pred_document_trs_evidence"].tolist(),
                ),
            })
            comparison
            """
        ),
        code(
            """
            context_errors = pred_df.loc[
                pred_df["gold_document_trs_evidence"].ne(
                    pred_df["pred_document_trs_evidence"]
                ),
                [
                    "course_id", "text", "gold_document_trs_evidence",
                    "pred_document_trs_evidence", "mentions",
                ],
            ]
            context_errors
            """
        ),
        md(
            """
            El objetivo no es forzar cero errores sintéticos. Inspecciona si falta
            vocabulario, si el contexto es incorrecto o si la definición gold
            necesita aclaración.
            """
        ),
        md(
            """
            ## 6. Modalidad y acceso

            Medimos exactitud de conjunto por documento. Para un sistema real
            también usaríamos P/R/F1 por etiqueta.
            """
        ),
        code(
            """
            def list_or_empty(value):
                return value if isinstance(value, list) else []

            pred_df["gold_modality_set"] = pred_df["gold_current_modality"].map(
                lambda value: set() if pd.isna(value) else {value}
            )
            pred_df["pred_modality_set"] = pred_df["pred_current_modalities"].map(set)
            pred_df["modality_exact"] = pred_df["gold_modality_set"].eq(
                pred_df["pred_modality_set"]
            )
            pred_df["access_exact"] = pred_df.apply(
                lambda row: set(list_or_empty(row["gold_current_access"]))
                == set(row["pred_current_access"]),
                axis=1,
            )
            pred_df[["modality_exact", "access_exact"]].mean()
            """
        ),
        code(
            """
            pred_df.loc[
                ~(pred_df["modality_exact"] & pred_df["access_exact"]),
                [
                    "course_id", "text", "gold_current_modality",
                    "pred_current_modalities", "gold_current_access",
                    "pred_current_access",
                ],
            ]
            """
        ),
        md(
            """
            ### Ejercicio 3

            Para cada error decide:

            - ¿es abreviatura ambigua?
            - ¿estado de acceso?
            - ¿histórico frente a actual?
            - ¿información heredada de otro curso?
            - ¿gold demasiado exigente para ese documento?

            No arregles un error documental usando información futura.
            """
        ),
        md(
            """
            ## 7. Agregación longitudinal

            Una máquina de estados simple mantiene:

            - modalidades vistas;
            - modalidad actual;
            - accesos activos;
            - última evidencia.

            Eventos de retirada/suspensión deben cerrar estados; los cursos sin
            mención no los borran.
            """
        ),
        code(
            """
            def aggregate_renal_patient(group):
                ever_modalities = set()
                current_modalities = set()
                current_access = set()
                latest_evidence = None
                conflicts = []

                for row in group.sort_values("date").itertuples():
                    mentions = row.mentions
                    row_current_modalities = {
                        mention["concept_id"] for mention in mentions
                        if mention["concept_id"] in modality_labels
                        and mention["status"] == "current"
                    }
                    row_stopped_modalities = {
                        mention["concept_id"] for mention in mentions
                        if mention["concept_id"] in modality_labels
                        and mention["status"] == "historical_or_stopped"
                    }
                    for mention in mentions:
                        concept = mention["concept_id"]
                        status = mention["status"]
                        if concept in modality_labels and status in {
                            "current", "historical_or_stopped"
                        }:
                            ever_modalities.add(concept)
                            latest_evidence = row.date
                        if concept in access_labels:
                            if status == "current":
                                current_access.add(concept)
                            elif status == "historical_or_stopped":
                                current_access.discard(concept)

                    current_modalities -= row_stopped_modalities
                    if row_current_modalities:
                        if len(row_current_modalities) > 1:
                            conflicts.append({
                                "course_id": row.course_id,
                                "type": "multiple_current_modalities",
                                "values": sorted(row_current_modalities),
                            })
                        # Una mención actual nueva sustituye el estado heredado;
                        # no dejamos modalidades previas coexistir por accidente.
                        current_modalities = set(row_current_modalities)

                    normalized = fold(row.text)
                    if "cvc retirado" in normalized:
                        current_access -= {
                            "TUNNELED_DIALYSIS_CATHETER",
                            "NONTUNNELED_DIALYSIS_CATHETER",
                        }
                    compatible_access = {
                        "HEMODIALYSIS": {
                            "AV_FISTULA",
                            "TUNNELED_DIALYSIS_CATHETER",
                            "NONTUNNELED_DIALYSIS_CATHETER",
                        },
                        "HEMODIAFILTRATION": {
                            "AV_FISTULA",
                            "TUNNELED_DIALYSIS_CATHETER",
                            "NONTUNNELED_DIALYSIS_CATHETER",
                        },
                        "PERITONEAL_DIALYSIS": {"PERITONEAL_CATHETER"},
                        "KIDNEY_TRANSPLANT": set(),
                    }
                    if len(current_modalities) == 1:
                        modality = next(iter(current_modalities))
                        current_access &= compatible_access[modality]
                return {
                    "ever_trs": bool(ever_modalities),
                    "ever_modalities": sorted(ever_modalities),
                    "current_modalities": sorted(current_modalities),
                    "current_access": sorted(current_access),
                    "latest_evidence": latest_evidence,
                    "state_conflicts": conflicts,
                }

            patient_rows = []
            for patient_id, group in pred_df.groupby("patient_id"):
                patient_rows.append({
                    "patient_id": patient_id,
                    **aggregate_renal_patient(group),
                })
            patient_state = pd.DataFrame(patient_rows)
            patient_state.head()
            """
        ),
        code(
            """
            display(
                pred_df.loc[
                    pred_df["patient_id"].isin(["R001", "R007"]),
                    ["patient_id", "date", "course_id", "text", "pred_current_modalities", "pred_current_access"],
                ].sort_values(["patient_id", "date"])
            )
            display(patient_state[patient_state["patient_id"].isin(["R001", "R007"])])
            """
        ),
        md(
            """
            La versión corregida sustituye una modalidad heredada cuando existe
            evidencia actual nueva, cierra estados suspendidos, elimina accesos
            incompatibles y conserva conflictos explícitos. Una máquina real aún
            debe definir fechas iguales, evidencia negativa, pérdida de injerto,
            recuperación renal y fuentes estructuradas. Cada transición necesita
            prueba.
            """
        ),
        md(
            """
            ## 8. Comparar con modelo/LLM

            Tras congelar este baseline:

            - NER BSC propone enfermedad/procedimiento;
            - un transformer local clasifica modalidad/contexto;
            - un LLM puede estructurar eventos complejos;
            - reglas conservan estado y valores;
            - todos usan el mismo test.

            El modelo nuevo debe demostrar ganancia neta y aportar errores
            aceptables.
            """
        ),
        md(
            """
            ## 9. Extensión a ERC y AKI

            **ERC:** menciones, estadio, etiología, cronicidad y datos
            estructurados longitudinales.

            **AKI:** no basta encontrar «FRA». Los criterios requieren creatinina
            basal/actual, tiempo y posiblemente diuresis. El NLP puede extraer
            contexto y eventos, pero el fenotipo clínico debe integrar series
            estructuradas y política explícita.
            """
        ),
        md(
            """
            ### Proyecto de ampliación

            Elige uno:

            1. acceso vascular y estado;
            2. trasplante/injerto;
            3. biopsia renal;
            4. etiología de ERC;
            5. fármaco–dosis–fecha.

            Entregables: canvas, protocolo, 30+ casos ficticios, diccionario,
            baseline, tests, evaluación, tabla de errores, model card y plan de
            validación real.
            """
        ),
        md(
            """
            ## Criterio para avanzar

            Debes poder:

            - reconstruir cada predicción desde offsets;
            - explicar errores del baseline;
            - separar evidencia alguna vez y estado actual;
            - distinguir documento y paciente;
            - escribir transiciones de la máquina de estados;
            - justificar qué no debe inferirse;
            - diseñar una comparación justa con modelos.
            """
        ),
    ]
