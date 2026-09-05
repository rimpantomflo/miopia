# NLP clínico de cero a Hero

Megacurso práctico y repositorio reproducible para aprender a construir,
evaluar y operar sistemas de NLP sobre texto clínico. La miopía es el caso
guiado de principio a fin; la nefrología es el dominio de transferencia y del
proyecto final hospitalario.

No es un producto sanitario ni realiza diagnóstico autónomo. Todo el material
usa texto ficticio y enseña a producir resultados trazables para revisión
humana.

## Empieza aquí

La puerta de entrada es el [currículo completo](docs/CURRICULUM.md). No estudies
todo de golpe. Sigue este orden:

1. Ejecuta `tutorial/tutorial_miopia_nlp.ipynb` para recorrer una vez el mapa
   original de ocho etapas.
2. Trabaja los fundamentos de `curso/00` a `curso/11`.
3. Completa la ruta avanzada Hero de `curso/12` a `curso/18`.
4. No avances hasta superar el criterio de salida de cada módulo.
5. Usa el módulo 10 como examen intermedio y el 18 como capstone de producción.

La [guía operativa del curso](curso/README.md) explica exactamente qué leer,
qué ejercicios hacer y cómo conservar tus respuestas.

## Instalación y comprobación

Con `uv` instalado:

```powershell
uv sync --group dev
uv run ruff check src tests scripts curso tutorial main.py
uv run pytest
uv run python tutorial/validate_tutorial.py
uv run python curso/validate_course.py
uv run miopia-course-check doctor
```

El paquete queda instalado en editable; ya no es necesario manipular `sys.path`
en scripts propios. En VS Code o Jupyter, selecciona el Python de `.venv`. Los
módulos centrales funcionan sin descargar modelos pesados. Para las rutas
opcionales consulta
[entorno de modelos avanzados](docs/entorno_modelos_avanzados.md).

| Ruta | Instalación | Uso |
|---|---|---|
| CPU completa | `uv sync --group dev` | notebooks, ML clásico y tests |
| Transformers | `uv sync --extra transformers --group dev` | fine-tuning NER |
| Embeddings | `uv sync --extra embeddings` | RAG denso/híbrido |
| API | `uv sync --extra service` | FastAPI/monitorización |
| Oracle | `uv sync --extra oracle` | integración autorizada |

## Mapa original del proyecto

El tutorial de miopía conserva las ocho etapas solicitadas:

1. Corpus de prueba y fundamentos de spaCy.
2. Reglas con `Matcher` y `SpanRuler`.
3. Expresiones regulares para refracción.
4. Contexto: negación, incertidumbre, sujeto y temporalidad.
5. Modelos clínicos preentrenados del BSC.
6. Validación y análisis de errores.
7. Oracle, lotes y seudonimización con HMAC.
8. Agregación longitudinal por paciente.

El curso amplía cada etapa y añade ML clásico, NER entrenable, transformers,
contexto ES/CA, relaciones, normalización con abstención, LLM local, RAG
híbrido, explicabilidad, calibración, utilidad clínica, validación externa,
MLOps y transferencia a un fenotipo renal. Doccano cubre anotación asistida;
la Hero Track culmina en una API y un capstone hospitalario.

## Estructura

```text
tutorial/
  tutorial_miopia_nlp.ipynb   proyecto guiado de miopía
  build_tutorial.py           fuente reproducible
  validate_tutorial.py        ejecución automática
curso/
  00_...ipynb a 11_...ipynb  fundamentos
  12_...ipynb a 18_...ipynb  Hero Track avanzada
  modules/                     fuentes reproducibles de los notebooks
  build_course.py              regenerador
  validate_course.py           ejecutor y comprobador
data/
  *.json, *.jsonl              corpus y terminología enteramente ficticios
src/miopia_nlp/
  pipeline.py                  baseline auditable de miopía
src/clinical_nlp_course/
  classical.py, context.py     ML clásico y contexto auditable
  transformers.py             spans, BIO, subtokens y fuga
  relations.py                candidatos y clasificación de relaciones
  normalization.py            linking en dos etapas con abstención
  retrieval.py, llm.py        RAG híbrido y Ollama local estructurado
  evaluation.py               utilidad, subgrupos y bootstrap emparejado
  monitoring.py               contratos, idempotencia, PSI y logs seguros
docs/
  CURRICULUM.md                objetivos, orden y niveles de dominio
  guia_*.md                    manuales de corpus, validación y LLM/RAG
  plantilla_*.md               ficha de datos y model card
scripts/
  preparar_doccano.py          JSONL con sugerencias y offsets validados
  train_token_classifier.py    fine-tuning Hugging Face completo
tests/
  test_*.py                    pruebas centinela y de utilidades
.github/workflows/
  ci.yml                       lint, tests y ejecución de 19 notebooks
```

Los notebooks de `curso/` se generan desde `curso/modules/`. Escribe tus
respuestas en una copia personal. Regenerar un notebook reemplaza su versión
generada:

```powershell
uv run python curso/build_course.py
uv run python curso/validate_course.py
```

## Probar el baseline de miopía

```powershell
uv run python main.py "Intervenido por miopía con LASIK en 2018."
uv run miopia-nlp analyze "No se descarta miopía."
```

La salida conserva estado documental, estado actual, menciones, refracciones,
offsets y reglas activadas.

## Antes de usar datos reales

- Acordar la definición de caso con los especialistas implicados.
- Obtener aprobaciones institucionales y de protección de datos.
- Trabajar exclusivamente en infraestructura autorizada.
- No enviar texto clínico a servicios externos no aprobados.
- Separar pacientes entre desarrollo, test temporal y validación externa.
- Validar por subgrupos y mantener revisión humana.
- Documentar versión de datos, diccionario, código, modelo y umbral.

La seudonimización no convierte los cursos clínicos en datos anónimos.

Consulta el [protocolo de proyecto hospitalario](docs/proyecto_hospitalario_seguro.md)
antes de mover un solo dato y la [guía de corpora públicos](docs/public_corpora.md)
antes de construir el primer benchmark externo. La
[escalera de proyectos](docs/PROJECT_LADDER.md) propone cinco entregables para
convertir el curso en portfolio, y `projects/capstone_template/` evita empezar
un proyecto real desde una carpeta vacía.
