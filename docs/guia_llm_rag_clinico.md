# Guía de LLM y RAG para texto clínico

## 1. Decide si necesitas generación

Para detectar una lista cerrada de conceptos, reglas o un encoder suelen ser
más baratos, reproducibles y auditables. Un LLM puede ser útil cuando:

- la salida tiene estructura compleja;
- existe variabilidad lingüística amplia;
- se requieren varias tareas en una misma pasada;
- se necesita explicación o resumen, siempre verificable;
- pocos ejemplos deben definir inicialmente la tarea.

## 2. Contrato de salida

Evita texto libre si necesitas extracción. Define esquema:

```json
{
  "concept_id": "HEMODIALYSIS",
  "assertion": "affirmed",
  "evidence": "continúa en hemodiálisis",
  "start": 20,
  "end": 45,
  "abstain": false
}
```

Valida tipos, enums, offsets y correspondencia literal con la fuente. Rechaza o
reintenta salidas inválidas; no las repares silenciosamente.

El módulo `clinical_nlp_course.llm` implementa este contrato con Pydantic y
`OllamaBackend`. Por defecto solo acepta endpoints de loopback. Actívalo así:

```bash
ollama serve
uv run jupyter lab
```

No cambies `allow_remote` para datos clínicos sin autorización institucional.

## 3. Prompt

Un prompt reproducible contiene:

- rol limitado;
- objetivo;
- definición de etiquetas;
- inclusiones/exclusiones;
- política de evidencia;
- política de abstención;
- formato;
- ejemplos difíciles;
- separación inequívoca entre instrucciones y texto no confiable.

Guarda versión del prompt, modelo, parámetros y fecha.

## 4. RAG

Pipeline:

```text
documentos → ACL → fragmentación → índices → fusión/reranking → top-k
           → contexto → generación → citas → validación
```

Evalúa por separado:

1. ¿se recuperó el fragmento correcto?
2. ¿el modelo usó correctamente ese fragmento?

Si la fuente no está en top-k, la generación no puede compensarlo de forma
fiable.

`HybridRetriever` combina BM25 y TF-IDF mediante Reciprocal Rank Fusion; acepta
un encoder local opcional. `retrieval_metrics` calcula Recall@k, MRR y nDCG.

## 5. Fragmentación

Comparar:

- frases;
- párrafos;
- secciones clínicas;
- ventanas con solapamiento;
- eventos longitudinales.

Conservar `document_id`, offsets, fecha, sección y permisos. Un fragmento sin
procedencia no permite citar ni filtrar acceso.

## 6. Seguridad

- no enviar PHI a proveedores no autorizados;
- no incluir secretos en prompts;
- tratar el texto recuperado como entrada no confiable;
- mitigar prompt injection;
- controlar herramientas y permisos;
- limitar retención y logs;
- registrar modelo y versión;
- prever indisponibilidad y cambio del proveedor.

## 7. Evaluación

Crear un conjunto de casos:

- normales;
- negados;
- inciertos;
- familiares;
- contradictorios;
- sin respuesta;
- textos largos;
- abreviaturas;
- adversariales;
- cambios de idioma;
- prompt injection.

Medir:

- validez estructural;
- exactitud por campo;
- fidelidad de evidencia;
- omisiones;
- alucinaciones;
- abstención;
- estabilidad;
- latencia/coste;
- rendimiento por subgrupo.

## 8. Arquitectura recomendada

En tareas clínicas, empezar por:

```text
reglas/NER recuperan candidatos
        ↓
LLM estructura o relaciona
        ↓
validador determinista
        ↓
revisión humana
```

La generación no debe borrar la evidencia del extractor.

## 9. Referencias

- [RAG original, Lewis et al.](https://arxiv.org/abs/2005.11401)
- [HELM](https://crfm.stanford.edu/helm/)
- [MedHELM](https://crfm.stanford.edu/helm/medhelm/latest/)
- [NIST Generative AI Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)
