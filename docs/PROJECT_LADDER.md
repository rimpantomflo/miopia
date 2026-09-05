# Escalera de proyectos para construir portfolio clínico

No empieces con una herramienta que afecte decisiones. Sube de nivel solo cuando
el anterior tiene referencia, tests, análisis de errores y documentación.

## Proyecto 1 — Extractor auditable sintético

**Pregunta:** extraer fármaco, dosis, vía y frecuencia de notas ficticias.

- baseline regex/diccionario;
- spans, contexto y normalización;
- 100 casos difíciles con offsets;
- exact/overlap F1 y tabla de errores;
- paquete, CLI, tests y model card.

Producto de portfolio: repositorio totalmente público porque no contiene datos
reales.

## Proyecto 2 — Benchmark público reproducible

**Pregunta:** NER + normalización de síntomas o procedimientos.

- SympTEMIST o MedProcNER bajo sus condiciones;
- baseline TF-IDF/diccionario;
- spaCy o Transformer ajustado;
- test bloqueado y tres semillas;
- `accuracy@1`, `recall@k`, cobertura y abstención;
- ficha de datos y experimento reproducible.

Producto: código y resultados permitidos por la licencia, nunca una
redistribución no autorizada del corpus.

## Proyecto 3 — Estado longitudinal renal retrospectivo

**Pregunta:** modalidad de TRS y acceso actual con evidencia temporal.

- protocolo con Nefrología;
- extracción de eventos + máquina de estados;
- conflictos explícitos y fuentes estructuradas;
- validación por paciente y tiempo;
- dashboard de errores sin PHI;
- ejecución por lotes idempotente.

Producto: demo ficticia pública y validación real únicamente en entorno seguro.

## Proyecto 4 — RAG de guías, sin historias clínicas

**Pregunta:** recuperar fragmentos de protocolos institucionales autorizados.

- ACL antes del ranking;
- BM25, embeddings, fusión y reranking;
- 100 preguntas con relevancia adjudicada;
- Recall@k/MRR/nDCG, citas y abstención;
- pruebas de inyección y documento malicioso;
- LLM local con salida estructurada.

Producto: asistente documental; no responde sobre pacientes.

## Proyecto 5 — Shadow mode hospitalario

**Pregunta:** una tarea aprobada de priorización o revisión, sin automatizar
decisiones.

- gates del protocolo hospitalario;
- validación temporal y externa;
- API o lote con autenticación institucional;
- monitorización y muestra humana continua;
- factores humanos, parada y rollback ensayado;
- evaluación prospectiva adecuada al uso.

Producto: dossier técnico-clínico. No publiques datos, capturas, métricas de
subgrupos pequeños ni arquitectura sensible sin autorización.

## Qué enseña seniority

En una entrevista o comité, muestra la cadena completa:

`pregunta → referencia → baseline → candidato → test → errores → utilidad → operación`

La madurez no se demuestra usando el modelo más grande, sino identificando qué
evidencia falta y evitando que un prototipo se convierta accidentalmente en una
decisión clínica.
