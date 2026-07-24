# NLP clínico: de la miopía a la nefrología

Curso práctico y repositorio reproducible para aprender a construir sistemas de
NLP sobre texto clínico. La miopía es el caso guiado de principio a fin; la
nefrología es el dominio de transferencia y del proyecto final.

No es un producto sanitario ni realiza diagnóstico autónomo. Todo el material
usa texto ficticio y enseña a producir resultados trazables para revisión
humana.

## Empieza aquí

La puerta de entrada es el [currículo completo](docs/CURRICULUM.md). No estudies
todo de golpe. Sigue este orden:

1. Ejecuta `tutorial/tutorial_miopia_nlp.ipynb` para recorrer una vez el mapa
   original de ocho etapas.
2. Trabaja los notebooks de `curso/` en orden, del `00` al `11`.
3. No avances hasta superar el criterio de salida de cada módulo.
4. Usa `curso/10_evaluacion_de_competencias.ipynb` como examen y proyecto final.

La [guía operativa del curso](curso/README.md) explica exactamente qué leer,
qué ejercicios hacer y cómo conservar tus respuestas.

## Instalación y comprobación

Con `uv` instalado:

```powershell
uv sync
uv run python tutorial/validate_tutorial.py
uv run python curso/validate_course.py
uv run python -m unittest discover -s tests -v
```

En VS Code o Jupyter, selecciona como kernel el Python de `.venv`. Los módulos
centrales funcionan sin descargar modelos pesados. Las prácticas opcionales con
transformers se activan solo al llegar al módulo 04; consulta
[entorno de modelos avanzados](docs/entorno_modelos_avanzados.md).

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

El curso amplía cada etapa y añade NER entrenable, clasificación, relaciones,
normalización, LLM, RAG, calibración, validación externa, MLOps y transferencia
a un fenotipo renal. El módulo 11 añade anotación colaborativa asistida
exclusivamente con Doccano.

## Estructura

```text
tutorial/
  tutorial_miopia_nlp.ipynb   proyecto guiado de miopía
  mi_cuaderno_trabajo.ipynb   copia personal para experimentar
  build_tutorial.py           fuente reproducible
  validate_tutorial.py        ejecución automática
curso/
  00_...ipynb a 11_...ipynb  itinerario completo
  modules/                     fuentes reproducibles de los notebooks
  build_course.py              regenerador
  validate_course.py           ejecutor y comprobador
data/
  *.json, *.jsonl              corpus y terminología enteramente ficticios
src/miopia_nlp/
  pipeline.py                  baseline auditable de miopía
src/clinical_nlp_course/
  utils.py                     métricas y utilidades didácticas reutilizables
docs/
  CURRICULUM.md                objetivos, orden y niveles de dominio
  guia_*.md                    manuales de corpus, validación y LLM/RAG
  plantilla_*.md               ficha de datos y model card
scripts/
  preparar_doccano.py          JSONL con sugerencias y offsets validados
tests/
  test_*.py                    pruebas centinela y de utilidades
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
