# Currículo completo de NLP clínico

Este currículo está diseñado para un nefrólogo que quiere adquirir autonomía
técnica y metodológica en NLP de textos clínicos. La miopía es el primer caso
vertical; nefrología es el dominio de transferencia y proyecto final.

## Filosofía

El curso no pretende que memorices APIs. Al terminar debes poder:

1. traducir una pregunta clínica a una tarea de NLP;
2. construir una referencia reproducible;
3. implementar un baseline interpretable;
4. entrenar o adaptar un modelo cuando esté justificado;
5. evaluar extracción, clasificación y fenotipo;
6. explicar errores y límites a clínicos y técnicos;
7. desplegar de forma segura, asistida y monitorizada.

Cada notebook usa el mismo ciclo:

```text
teoría → predicción mental → ejemplo → ejercicio → comprobación
       → error intencionado → solución → criterio para avanzar
```

No es necesario completar varios notebooks en una sesión. Es preferible
dominar un módulo, modificar ejemplos y explicar sus errores antes de avanzar.

## Orden del curso

### Proyecto guiado

0. `tutorial/tutorial_miopia_nlp.ipynb`

Recorre el mapa original:

1. corpus y fundamentos de spaCy;
2. `Matcher` y `SpanRuler`;
3. expresiones regulares;
4. contexto clínico;
5. modelos BSC;
6. validación;
7. Oracle;
8. agregación longitudinal.

### Bloque A — Datos y spaCy

1. `curso/00_mapa_y_metodo.ipynb`
2. `curso/01_corpus_y_diccionarios.ipynb`
3. `curso/02_spacy_avanzado.ipynb`
4. `curso/03_ner_entrenable_spacy.ipynb`

### Bloque B — Modelos

5. `curso/04_transformers_y_bsc.ipynb`
6. `curso/05_clasificacion_relaciones_normalizacion.ipynb`
7. `curso/06_llm_extraccion_y_rag.ipynb`

### Bloque C — Evidencia y práctica clínica

8. `curso/07_validacion_avanzada.ipynb`
9. `curso/08_proyecto_nefrologia.ipynb`
10. `curso/09_produccion_y_monitorizacion.ipynb`
11. `curso/10_evaluacion_de_competencias.ipynb`
12. `curso/11_anotacion_asistida_doccano.ipynb`

## Resultados de aprendizaje

### Tras el Bloque A

Podrás:

- diseñar unidades de anotación;
- crear corpus y diccionarios versionados;
- medir acuerdo exacto y por solapamiento;
- manejar offsets, BILUO y `DocBin`;
- crear componentes personalizados de spaCy;
- entrenar y evaluar un NER pequeño de principio a fin.

### Tras el Bloque B

Podrás:

- distinguir encoder, modelo ajustado y LLM generativo;
- leer una model card críticamente;
- preparar etiquetas para subtokens;
- diseñar el ajuste de un modelo BSC;
- construir clasificación, relaciones y normalización;
- diseñar prompts estructurados;
- construir y evaluar un RAG básico;
- validar evidencia y abstención de un LLM.

### Tras el Bloque C

Podrás:

- calcular métricas con incertidumbre;
- evaluar calibración, prevalencia y subgrupos;
- evitar fuga por paciente y tiempo;
- construir un fenotipo renal longitudinal;
- diseñar ejecución Oracle por lotes;
- monitorizar deriva y fallos;
- redactar una ficha de datos y una model card;
- decidir si el sistema está listo para piloto silencioso.
- desplegar un proyecto Doccano ficticio con doble anotación y preanotación;
- medir si las sugerencias reducen carga sin degradar la referencia.

## Reglas para avanzar

No avances de módulo hasta poder:

- explicar sus conceptos sin leer;
- ejecutar todas sus comprobaciones;
- resolver al menos el 80 % de ejercicios;
- inventar un caso que rompa el método;
- añadir o describir la prueba que protegería la corrección;
- escribir una limitación que no se resuelve con más código.

## Niveles de dominio

### Nivel 1 — Baseline clínico

- reglas, regex y contexto;
- corpus sintético;
- métricas binarias;
- trazabilidad por offsets.

### Nivel 2 — Modelado supervisado

- anotación independiente;
- NER y clasificación entrenables;
- transformers;
- comparación reproducible.

### Nivel 3 — Sistema clínico

- integración de fuentes;
- fenotipo longitudinal;
- calibración;
- validación temporal/externa;
- revisión humana y monitorización.

### Nivel 4 — Experto

- identifica sesgos antes de entrenar;
- diseña referencias para tareas nuevas;
- selecciona el modelo mínimo suficiente;
- evalúa utilidad clínica y factores humanos;
- revalida ante deriva;
- comunica límites, incertidumbre y ámbito de uso.

## Uso de datos reales

Los notebooks utilizan exclusivamente textos ficticios. Antes de incorporar
datos hospitalarios:

- obtener base jurídica y aprobaciones;
- trabajar en infraestructura autorizada;
- definir minimización, acceso, retención y auditoría;
- separar la clave de seudonimización;
- impedir texto clínico en commits, notebooks, incidencias y logs;
- documentar qué modelos o servicios externos están autorizados.

## Fuentes metodológicas de referencia

- [spaCy: training pipelines](https://spacy.io/usage/training)
- [spaCy: DocBin](https://spacy.io/api/docbin)
- [Hugging Face: token classification](https://huggingface.co/docs/transformers/tasks/token_classification)
- [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378)
- [PROBAST+AI](https://www.bmj.com/content/388/bmj-2024-082505)
- [STARD-AI](https://www.nature.com/articles/s41591-025-03953-8)
- [DECIDE-AI](https://www.nature.com/articles/s41591-022-01772-9)
- [NIST Generative AI Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)
