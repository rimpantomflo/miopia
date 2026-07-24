# Cómo estudiar este curso

Este directorio contiene doce notebooks acumulativos. Cada uno combina teoría,
predicción mental, ejemplos, ejercicios, comprobaciones, solución razonada y un
criterio explícito para avanzar.

## Preparación

Desde la raíz del repositorio:

```powershell
uv sync
uv run python curso/validate_course.py
```

Abre Jupyter o VS Code y selecciona `.venv` como kernel. Duplica el notebook que
vayas a trabajar y añade, por ejemplo, el sufijo `_rafa`. No edites la única
copia de datos reales ni guardes texto hospitalario en el repositorio.

## Ritual de cada módulo

1. Lee los objetivos y escribe qué crees que significa cada uno.
2. Antes de ejecutar una celda, predice su resultado.
3. Ejecuta las celdas en orden.
4. Resuelve el ejercicio sin desplegar la solución.
5. Ejecuta la comprobación; si falla, explica por qué antes de corregir.
6. Compara con la solución y anota una alternativa.
7. Inventa un caso límite clínico que rompa el enfoque.
8. Responde el criterio de salida sin consultar el notebook.
9. Registra dudas, errores y decisiones en tu diario de aprendizaje.

Usa tres marcas en tus notas:

- `SÉ`: puedo explicarlo y aplicarlo sin ayuda.
- `DUDO`: lo entiendo, pero necesito practicarlo.
- `NO SÉ`: debo repetir el bloque y pedir o buscar una explicación.

## Itinerario recomendado

| Fase | Notebook | Producto que debes conservar |
|---|---|---|
| Orientación | `00_mapa_y_metodo` | canvas de una pregunta clínica |
| Datos | `01_corpus_y_diccionarios` | mini-guía y diccionario validados |
| spaCy | `02_spacy_avanzado` | pipeline con spans y contexto |
| NER | `03_ner_entrenable_spacy` | experimento y análisis de errores |
| Transformers/BSC | `04_transformers_y_bsc` | ficha comparativa de modelos |
| Tareas clínicas | `05_clasificacion_relaciones_normalizacion` | diseño multitarea |
| LLM/RAG | `06_llm_extraccion_y_rag` | contrato JSON y evaluación |
| Evidencia | `07_validacion_avanzada` | plan de análisis estadístico |
| Nefrología | `08_proyecto_nefrologia` | baseline longitudinal renal |
| Producción | `09_produccion_y_monitorizacion` | plan de piloto silencioso |
| Competencia | `10_evaluacion_de_competencias` | examen y capstone |
| Anotación asistida | `11_anotacion_asistida_doccano` | piloto colaborativo con sugerencias |

El tutorial de miopía se realiza antes de esta tabla. Sirve para ver el paisaje;
estos módulos enseñan a construir cada parte con profundidad.

## Ritmo realista

Dedica una o dos semanas a cada módulo, con dos o tres sesiones breves. Alterna:

- una sesión de lectura y predicciones;
- una sesión de código y ejercicios;
- una sesión de casos límite, explicación oral y notas.

Tras los módulos 03, 06 y 09, haz una semana de consolidación sin contenido
nuevo. Repite ejercicios con ejemplos diferentes y revisa tu registro de
errores.

## Qué significa dominarlo

Completar celdas no equivale a competencia. Estás listo para avanzar cuando
puedes:

- formular el estimando clínico y la unidad de análisis;
- identificar fuga, sesgo y errores de referencia;
- construir un baseline sencillo antes de un modelo complejo;
- escoger métricas y umbrales según el uso;
- inspeccionar errores con offsets y evidencia;
- explicar qué no sabe el sistema;
- proponer una validación temporal o externa;
- diseñar revisión humana, monitorización y retirada segura.

## Regeneración y validación

Los archivos `modules/module_00.py` a `module_11.py` son la fuente canónica. Para
regenerar las copias limpias:

```powershell
uv run python curso/build_course.py
uv run python curso/validate_course.py
```

La regeneración sobrescribe los doce notebooks oficiales, pero no una copia
personal con otro nombre.
