# Transferir el aprendizaje de miopía a nefrología

La miopía es el **caso vertical**: un problema concreto que permite recorrer
todo el ciclo de NLP clínico. El objetivo formativo es reutilizar después el
mismo método en nefrología cambiando el contrato clínico, el corpus y la
validación, no copiando sin más las palabras o los umbrales.

## El patrón que se transfiere

```text
pregunta clínica
    ↓
guía de anotación
    ↓
menciones + números + contexto
    ↓
predicción por documento
    ↓
agregación temporal por paciente
    ↓
validación local y análisis de errores
    ↓
despliegue asistido y monitorizado
```

En miopía extraemos términos, refracción y lateralidad. En nefrología podrían
ser diagnósticos, creatinina, filtrado glomerular, albuminuria, tratamiento
renal sustitutivo, acceso vascular, biopsia o estado del trasplante. El proceso
metodológico es el mismo; la definición clínica no.

## Transferencia etapa por etapa

| Etapa del proyecto | Miopía | Ejemplo nefrológico |
|---|---|---|
| 1. Corpus y spaCy | «miopía», OD/OI | «ERC», «fracaso renal», hemodiálisis, trasplante |
| 2. Reglas | Variantes castellano/catalán | ERC/CKD, FRA/AKI, HD/HDF/DP, TR, FAV/CVC |
| 3. Regex | Esfera, cilindro, dioptrías | Creatinina, eGFR, ACR/PCR, K, bicarbonato, diuresis |
| 4. Contexto | Negada, familiar, histórica | «descartar FRA», «sin ERC», «madre en diálisis», antecedente |
| 5. Modelos | Mención/fenotipo de miopía | Enfermedad, síntoma, procedimiento, fármaco, clasificación |
| 6. Validación | Curso y paciente | Episodio, estadio, modalidad, etiología, centro y periodo |
| 7. Oracle | Cursos oftalmológicos | Cursos + laboratorio + procedimientos + farmacia |
| 8. Longitudinal | Miopía alguna vez/actual | ERC sostenida, episodio de AKI, diálisis o trasplante |

## Proyectos nefrológicos posibles

Ordenados aproximadamente desde reglas transparentes hasta sistemas que
requieren más datos y modelado:

1. **Detectar tratamiento renal sustitutivo:** hemodiálisis, hemodiafiltración,
   diálisis peritoneal, trasplante y tratamiento conservador.
2. **Extraer acceso vascular:** FAV, prótesis, catéter tunelizado/no tunelizado,
   localización, complicación y estado.
3. **Extraer valores y unidades:** creatinina, urea, eGFR, potasio, bicarbonato,
   albuminuria/proteinuria y diuresis.
4. **Detectar enfermedad renal crónica mencionada:** aserción, estadio,
   etiología y temporalidad.
5. **Identificar episodios de lesión renal aguda:** combinar texto con
   creatinina y diuresis estructuradas. Los criterios clínicos deben codificarse
   explícitamente; un modelo de texto solo no basta.
6. **Fenotipar trasplante renal:** trasplantado, fecha, injerto funcionante,
   rechazo, pérdida del injerto y vuelta a diálisis.
7. **Extraer histología renal:** diagnóstico, patrones, inmunofluorescencia,
   cronicidad y adecuación de la muestra.
8. **Detectar toxicidad o ajuste renal de fármacos:** entidad farmacológica,
   dosis, pauta, función renal, acción recomendada y experienciador.
9. **Clasificar etiología renal:** diabética, vascular, glomerular, hereditaria,
   obstructiva, intersticial o indeterminada, usando una taxonomía acordada.
10. **Búsqueda semántica de cohortes:** recuperar cursos conceptualmente
    relacionados aunque no compartan una palabra exacta.
11. **Normalización terminológica:** enlazar menciones locales con SNOMED CT,
    ICD-10 u otra terminología aprobada.
12. **Resumen longitudinal asistido:** generar una cronología verificable con
    enlaces a cada evidencia. Requiere controles adicionales de fidelidad.

Cada proyecto necesita su propio protocolo, referencia y test bloqueado.

## Qué tipo de modelo resuelve qué tarea

| Tarea | Salida | Ejemplo renal | Arquitectura habitual |
|---|---|---|---|
| NER/token classification | spans | «nefropatía IgA», «tacrolimus» | encoder + cabeza por token |
| Clasificación binaria | etiqueta del curso | episodio compatible con AKI | encoder + cabeza de clase |
| Clasificación multietiqueta | varias etiquetas | ERC + HD + anemia | encoder + sigmoides |
| Atributos de mención | contexto | negada, histórica, familiar | reglas o clasificador de span |
| Extracción de relaciones | pares relacionados | fármaco–dosis, biopsia–hallazgo | clasificador de pares/grafo |
| Normalización | código | mención → concepto SNOMED | bi-encoder + reranker |
| Recuperación semántica | documentos ordenados | cursos similares | embeddings + índice vectorial |
| Pregunta-respuesta extractiva | fragmento fuente | «¿qué acceso lleva?» | modelo ajustado para QA |
| Resumen generativo | texto nuevo | cronología renal | modelo generativo + evidencias |

Un encoder preentrenado no realiza todas esas tareas de forma automática. Se le
añade una cabeza, ejemplos anotados y un procedimiento de ajuste/evaluación.

## Ecosistema BSC: mapa práctico

La oferta cambia con el tiempo; revisar siempre la tarjeta y la versión exacta.

### 1. Encoders base para ajustar

**`PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`**

- preentrenado con texto biomédico y clínico español;
- objetivo original de lenguaje enmascarado;
- buen punto de partida para NER o clasificación local;
- no detecta por sí solo miopía, ERC o AKI con las etiquetas que necesitamos.

**`PlanTL-GOB-ES/longformer-base-4096-biomedical-clinical-es`**

- candidato para documentos más largos;
- debe compararse localmente con segmentación por secciones o ventanas;
- mayor contexto no garantiza mejor fenotipado.

**`BSC-LT/MrBERT-biomed`**

- encoder biomédico multilingüe publicado en 2026;
- 308 millones de parámetros y contexto declarado de 8192 tokens;
- corpus predominantemente inglés y, en segundo lugar, español;
- objetivo de lenguaje enmascarado: también requiere ajuste para la tarea.

Un experimento correcto compara varios encoders bajo la misma partición,
anotación, presupuesto de ajuste y métrica, no por popularidad.

### 2. NER ya ajustado para generar candidatos

La colección Clinical NMT-NER de BSC-NLP4BIA incluye modelos de
**enfermedades, fármacos y profesiones** en castellano y catalán. Pueden servir
para:

- proponer spans que después revise una persona;
- acelerar la anotación inicial;
- medir cuánto se transfiere un NER general a nefrología;
- iniciar un ajuste posterior con datos locales.

No resuelven necesariamente aserción, temporalidad, experienciador, estadio,
relaciones ni fenotipo longitudinal.

Los modelos DT4H amplían el objetivo a **enfermedades, síntomas y
procedimientos** en varios idiomas, incluido castellano. Algunas variantes
multitarea usan una arquitectura personalizada y no se cargan directamente con
`AutoModelForTokenClassification`; hay que seguir su script oficial. Su propia
tarjeta informa resultados diferentes según entidad e idioma, por lo que se
deben validar antes de reutilizarlos.

### 3. Anonimización

PlanTL publica modelos de anonimización para castellano y catalán. Pueden
formar parte de una defensa por capas, pero no sustituyen:

- el entorno seguro;
- la minimización;
- la revisión de tipos de identificador locales;
- pruebas específicas de recall;
- la gobernanza y base jurídica.

Un falso negativo de anonimización puede exponer datos personales.

### 4. Recuperación y normalización

El ecosistema BSC-NLP4BIA incluye recursos de normalización con bi-encoders y
cross-encoders. El patrón avanzado es:

1. NER detecta la mención;
2. un bi-encoder recupera conceptos candidatos;
3. un cross-encoder los reordena;
4. reglas y revisión resuelven ambigüedad;
5. se mide exactitud del código, además del span.

Esto permitiría transformar «IRC estadio 4», «ERC G4» y variantes locales en un
concepto común, siempre que la terminología y licencia estén aprobadas.

## Cómo elegir

| Situación | Primer método |
|---|---|
| Pocas expresiones estables, alta trazabilidad | Reglas |
| Muchos valores con formato repetido | Regex + validación semántica |
| Quieres spans generales de enfermedad/fármaco | NER BSC como baseline |
| Tienes cientos/miles de ejemplos locales | Ajustar encoder biomédico |
| Cursos largos con evidencia dispersa | Secciones/ventanas y comparar Longformer/MrBERT |
| Quieres buscar casos similares | Modelo de embeddings ajustado a recuperación |
| Quieres códigos terminológicos | NER + normalización |
| Quieres resúmenes | Extracción primero; generación con citas después |

En clínica suele ganar un sistema híbrido: reglas para valores y excepciones,
modelo para variabilidad lingüística y agregador temporal explícito.

## Itinerario para adquirir experiencia

1. Dominar offsets, tokenización, reglas y regex.
2. Aprender diseño de anotación y acuerdo entre revisores.
3. Implementar baselines clásicos con caracteres/TF-IDF.
4. Entender tokenización subword, BIO, padding, máscaras y pérdida.
5. Ajustar un encoder para NER y otro para clasificación.
6. Evaluar por paciente, subgrupo, tiempo y centro.
7. Aprender calibración, umbrales y curvas precisión–recall.
8. Estudiar aprendizaje activo y deriva.
9. Construir recuperación/normalización.
10. Desplegar en silencio, con auditoría y revisión humana.

La prueba de experiencia no es ejecutar un modelo grande: es poder demostrar
qué predice, sobre qué población, con qué errores, durante cuánto tiempo y con
qué evidencia verificable.

## Referencias BSC

- [RoBERTa biomédico-clínico español](https://huggingface.co/PlanTL-GOB-ES/roberta-base-biomedical-clinical-es)
- [Catálogo PlanTL-GOB-ES](https://huggingface.co/PlanTL-GOB-ES/models)
- [MrBERT-biomed](https://huggingface.co/BSC-LT/MrBERT-biomed)
- [BSC-NLP4BIA](https://huggingface.co/BSC-NLP4BIA)
- [Colección Clinical NMT-NER](https://huggingface.co/collections/BSC-NLP4BIA/clinical-nmt-ner)
- [DT4H NER multilingüe multitarea](https://huggingface.co/BSC-NLP4BIA/DT4H_XLM-R_mtl_multilingual_multilabel)

Última revisión documental: julio de 2026.
