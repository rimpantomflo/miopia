from curso.notebook_factory import code, common_setup, md


TITLE = "00 · Mapa, método y lenguaje común"


def build() -> list[dict]:
    return [
        md(
            """
            # 00 · Mapa, método y lenguaje común

            Este notebook convierte el repositorio en un curso. No enseña todavía
            un algoritmo nuevo: establece cómo pensar, estudiar y comprobar que
            realmente has adquirido cada competencia.

            **Caso guiado:** miopía.  
            **Dominio de transferencia:** nefrología.  
            **Datos:** exclusivamente ficticios.
            """
        ),
        md(
            """
            ## Resultados de aprendizaje

            Al terminar podrás:

            - diferenciar extracción, NER, clasificación, normalización, fenotipo,
              recuperación y generación;
            - transformar una necesidad clínica en una tarea evaluable;
            - distinguir paciente, episodio, documento, frase, span y token;
            - describir el ciclo completo de un proyecto;
            - decidir qué debes aprender antes de activar un modelo pesado.
            """
        ),
        common_setup(),
        code(
            """
            import platform
            import pandas as pd
            import spacy

            print("Python:", platform.python_version())
            print("pandas:", pd.__version__)
            print("spaCy:", spacy.__version__)
            """
        ),
        md(
            """
            ## El mapa mental

            ```text
            necesidad clínica
                ↓
            uso previsto y población
                ↓
            corpus + guía + referencia
                ↓
            baseline → modelo → sistema híbrido
                ↓
            evaluación por mención/documento/paciente
                ↓
            piloto silencioso → monitorización → revalidación
            ```

            Un proyecto no empieza eligiendo un modelo. Empieza decidiendo qué
            salida sería útil y qué error sería peligroso.
            """
        ),
        md(
            """
            ## Tareas que suelen confundirse

            | Tarea | Entrada → salida | Ejemplo |
            |---|---|---|
            | NER | texto → spans | localizar «enfermedad renal crónica» |
            | Atributos | span → contexto | afirmada, negada, posible |
            | Clasificación | documento → clase | curso compatible con TRS |
            | Relación | dos entidades → vínculo | tacrolimus–dosis |
            | Normalización | mención → concepto | «IRC» → ERC |
            | Fenotipo | eventos → estado paciente | diálisis actual |
            | Recuperación | consulta → documentos | cohortes similares |
            | Generación | contexto → texto nuevo | cronología citada |

            NER no es diagnóstico. Un buen resumen tampoco garantiza que todos sus
            hechos estén sustentados.
            """
        ),
        code(
            """
            task_examples = pd.DataFrame([
                {
                    "pregunta": "¿Dónde se menciona hemodiálisis?",
                    "unidad": "span",
                    "tarea": "NER",
                },
                {
                    "pregunta": "¿Este curso confirma TRS en el paciente?",
                    "unidad": "documento",
                    "tarea": "clasificación + contexto",
                },
                {
                    "pregunta": "¿Cuál es la modalidad renal actual?",
                    "unidad": "paciente",
                    "tarea": "fenotipo longitudinal",
                },
                {
                    "pregunta": "¿Qué fragmentos responden a acceso vascular?",
                    "unidad": "fragmento",
                    "tarea": "recuperación",
                },
            ])
            task_examples
            """
        ),
        md(
            """
            ### Ejercicio 1 · Traducir preguntas

            Para cada pregunta escribe: unidad, tarea, salida y referencia.

            1. ¿Qué pacientes tienen una biopsia compatible con nefropatía IgA?
            2. ¿Qué dosis de tacrolimus se asocia a cada fecha?
            3. ¿Qué pacientes comenzaron hemodiálisis durante 2025?
            4. Resume la trayectoria del acceso vascular con citas.

            No continúes hasta haber escrito una respuesta. Después compárala con
            la solución.
            """
        ),
        md(
            """
            <details>
            <summary>Solución razonada</summary>

            1. NER de hallazgos + relación/normalización + fenotipo por biopsia.
            2. NER de fármaco/dosis/fecha + extracción de relaciones.
            3. extracción de modalidad y fecha + máquina de estados longitudinal.
            4. recuperación de eventos + orden temporal + generación sustentada.

            En todos los casos la referencia final debe anotarse al nivel de la
            salida clínica, no solo al nivel de palabras.
            </details>
            """
        ),
        md(
            """
            ## Canvas de pregunta clínica

            Completa siempre:

            - usuario y acción;
            - población y periodo;
            - entrada disponible al predecir;
            - salida y unidad;
            - referencia;
            - falso positivo y falso negativo;
            - abstención;
            - revisión humana;
            - usos excluidos.
            """
        ),
        code(
            """
            clinical_canvas = {
                "usuario": "nefrólogo revisor",
                "accion": "priorizar revisión retrospectiva",
                "poblacion": "adultos con cursos del periodo definido",
                "entrada": "texto disponible hasta la fecha índice",
                "salida": "evidencia de modalidad de TRS + offsets",
                "unidad": "curso y paciente",
                "referencia": "doble revisión clínica adjudicada",
                "coste_fp": "revisión innecesaria",
                "coste_fn": "paciente no recuperado",
                "abstencion": "evidencia contradictoria o insuficiente",
                "revision_humana": True,
                "uso_excluido": "activar tratamiento automáticamente",
            }
            pd.Series(clinical_canvas)
            """
        ),
        md(
            """
            ## Unidades y fuga

            - **Paciente:** unidad humana; agrupa todos sus datos.
            - **Episodio:** ventana de eventos clínicamente relacionados.
            - **Documento:** curso, informe o alta.
            - **Span:** evidencia `[start:end]`.
            - **Token:** unidad del tokenizador.
            - **Subtoken:** pieza usada por un transformer.

            Si cursos del mismo paciente aparecen en entrenamiento y test, el
            modelo puede reconocer estilo, antecedentes y plantillas. La métrica
            deja de representar pacientes nuevos.
            """
        ),
        code(
            """
            hierarchy = {
                "R001": {
                    "episodio_2024": {
                        "N001": ["span_1", "span_2"],
                        "N002": ["span_3"],
                    }
                }
            }
            hierarchy
            """
        ),
        md(
            """
            ## Método de estudio

            Para cada bloque:

            1. predice la salida antes de ejecutar;
            2. ejecuta;
            3. explica cualquier diferencia;
            4. modifica un ejemplo;
            5. provoca un error;
            6. formula una prueba;
            7. resume la limitación.

            Copiar una solución produce familiaridad, no competencia.
            """
        ),
        md(
            """
            ## Pretest

            Responde sin buscar:

            1. ¿Qué diferencia hay entre precisión y sensibilidad?
            2. ¿Por qué `miopía` detectada no implica paciente miope?
            3. ¿Qué es un offset?
            4. ¿Qué diferencia hay entre encoder y LLM generativo?
            5. ¿Por qué el test se bloquea?
            6. ¿Qué evalúas primero en un RAG?
            """
        ),
        code(
            """
            # Autoevaluación orientativa. Escribe tus respuestas antes.
            keywords = {
                1: {"fp", "fn", "precision", "sensibilidad"},
                2: {"negacion", "familiar", "contexto"},
                3: {"caracter", "inicio", "fin"},
                4: {"representacion", "generar"},
                5: {"sobreajuste", "estimacion"},
                6: {"recuperacion", "fuente"},
            }

            my_answers = {
                1: "",
                2: "",
                3: "",
                4: "",
                5: "",
                6: "",
            }
            print("Completa my_answers; la comprobación no sustituye revisión razonada.")
            """
        ),
        code(
            """
            def normalize_answer(text):
                return (
                    text.casefold()
                    .replace("ó", "o")
                    .replace("í", "i")
                    .replace("é", "e")
                    .replace("á", "a")
                )

            for number, required in keywords.items():
                answer = normalize_answer(my_answers[number])
                found = sorted(word for word in required if word in answer)
                print(number, "pistas presentes:", found)
            """
        ),
        md(
            """
            ## Seguridad desde la primera celda

            - solo datos ficticios en este repositorio;
            - identificadores ficticios no son seudónimos reales;
            - no pegar cursos en servicios externos;
            - no imprimir texto real en logs;
            - no guardar claves en notebooks;
            - permisos mínimos;
            - revisión humana definida.

            La privacidad no es un módulo final: condiciona corpus, herramientas,
            modelos y arquitectura desde el inicio.
            """
        ),
        md(
            """
            ## Diario de aprendizaje

            Al terminar cada notebook registra:

            ```text
            concepto que puedo explicar:
            código que puedo escribir sin copiar:
            error que sé diagnosticar:
            limitación clínica:
            duda pendiente:
            ejemplo renal de transferencia:
            ```

            El repositorio no evalúa tu comprensión interna. Este diario evita
            confundir ejecución correcta con aprendizaje.
            """
        ),
        md(
            """
            ## Criterio para avanzar

            Continúa a `01_corpus_y_diccionarios.ipynb` cuando puedas:

            - clasificar los ocho tipos de tarea de la tabla;
            - completar el canvas para un caso nuevo;
            - explicar la jerarquía paciente–episodio–documento–span–token;
            - describir el ciclo completo sin mirar;
            - justificar por qué el test no se usa para crear reglas.
            """
        ),
    ]
