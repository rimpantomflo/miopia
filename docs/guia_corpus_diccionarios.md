# Guía para corpus, anotaciones y diccionarios clínicos

## 1. Empieza por el uso previsto

Escribe una frase:

> El sistema se utilizará para ___ en ___ y su salida será revisada por ___
> antes de ___.

Después define explícitamente qué no hará. «Detectar menciones» y «diagnosticar
pacientes» son tareas diferentes.

## 2. Define las unidades

- unidad de muestreo: paciente, episodio o documento;
- unidad de anotación: span, frase, documento o evento;
- unidad de evaluación;
- unidad de despliegue.

Si se evalúa por paciente, la referencia debe permitir esa agregación.

## 3. Diseña el marco de muestreo

Evita construir el corpus solo buscando la palabra objetivo: perderías casos
sin el término y distorsionarías prevalencia. Combina, según la pregunta:

- muestra aleatoria;
- enriquecimiento de positivos;
- términos de alta sensibilidad;
- códigos/procedimientos/laboratorio;
- tipos documentales;
- periodos, centros e idiomas;
- casos difíciles deliberados.

Guarda el mecanismo de selección para poder ponderar o interpretar métricas.

## 4. Esquema de anotación

Un span clínico suele necesitar:

```json
{
  "document_id": "D001",
  "start": 12,
  "end": 36,
  "label": "CHRONIC_KIDNEY_DISEASE",
  "assertion": "affirmed",
  "experiencer": "patient",
  "temporality": "current",
  "certainty": "certain",
  "annotator": "A",
  "guide_version": "1.2"
}
```

No mezcles dimensiones en etiquetas como
`ERC_NEGADA_FAMILIAR_HISTORICA`.

## 5. Límites de spans

Decide si se incluye:

- severidad: «ERC avanzada»;
- estadio: «ERC G4»;
- lateralidad/localización;
- artículos y preposiciones;
- coordinación;
- abreviatura y expansión;
- spans discontinuos;
- entidades anidadas.

Incluye ejemplos positivos y negativos para cada decisión.

## 6. Piloto

1. Selecciona un lote pequeño y diverso.
2. Dos clínicos anotan independientemente.
3. Mide acuerdo de etiqueta y límites.
4. Discute desacuerdos sin modificar retroactivamente reglas en secreto.
5. Actualiza la guía y registra cambios.
6. Reanota si el cambio altera decisiones previas.
7. Congela una versión antes del corpus final.

El tamaño del piloto es una decisión práctica; no sustituye un cálculo de
tamaño para la validación final.

## 7. Particiones

- todos los documentos de un paciente permanecen juntos;
- el test se bloquea antes de ajustar reglas;
- si interesa despliegue futuro, añade test temporal;
- para generalización, reserva otro centro o fuente;
- evita duplicados y plantillas casi idénticas entre particiones.

## 8. Diccionario clínico

Un concepto versionado:

```json
{
  "concept_id": "PERITONEAL_DIALYSIS",
  "preferred_term": "diálisis peritoneal",
  "variants": ["DP", "DPA", "DPCA"],
  "semantic_type": "procedure",
  "languages": ["es", "ca"],
  "exclusions": ["pendiente de iniciar DP"],
  "section_constraints": ["antecedentes", "tratamiento_actual"],
  "provenance": "consenso_nefrologia_2026",
  "version": "1.0.0"
}
```

Comprueba:

- identificadores únicos;
- variantes duplicadas o ambiguas;
- mayúsculas, acentos y Unicode;
- abreviaturas por sección;
- errores observados, no imaginados;
- exclusiones;
- procedencia y fecha;
- licencia de terminologías externas.

## 9. Diccionario, terminología y modelo

- diccionario: formas textuales y reglas locales;
- terminología: conceptos, códigos y relaciones;
- NER: detecta spans;
- normalizador: enlaza spans con conceptos;
- fenotipo: combina evidencias a nivel de paciente.

Ninguno sustituye automáticamente a los otros.

## 10. Control de calidad

Validaciones automáticas:

- offsets dentro del texto;
- `text[start:end]` coincide con evidencia;
- etiquetas y atributos permitidos;
- ausencia de spans imposibles;
- identificadores y versiones presentes;
- distribución por anotador;
- duplicados;
- pacientes en una única partición;
- texto sin identificadores directos en repositorios.

## 11. Entregables

- protocolo clínico;
- guía de anotación;
- corpus versionado;
- adjudicaciones;
- tabla de acuerdo;
- diccionario;
- script de validación;
- ficha de datos;
- separación train/dev/test;
- registro de cambios.

