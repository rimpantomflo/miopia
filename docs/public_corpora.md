# Corpora públicos para pasar de sintético a evidencia

Este repositorio no descarga ni redistribuye corpora clínicos. Cada usuario debe
leer y aceptar sus condiciones, obtener las aprobaciones necesarias y guardar
los datos en `data/restricted/`, que está excluido de Git.

## Recursos recomendados

| Corpus | Tareas | Acceso | Laboratorio |
|---|---|---|---|
| [CARMEN-I](https://physionet.org/content/carmen-i/1.0.1/) | información clínica y NER ES/CA | credencial PhysioNet, formación, DUA y aprobación del proyecto | transferencia institucional/lingüística |
| [SympTEMIST](https://temu.bsc.es/symptemist/) | síntomas/signos y normalización SNOMED CT | condiciones de la tarea | NER + entity linking |
| [MedProcNER](https://temu.bsc.es/medprocner/) | procedimientos, normalización e indexación | condiciones de la tarea | procedimientos y retrieval |
| [CodiEsp](https://temu.bsc.es/codiesp/) | codificación CIE-10 y evidencia | condiciones de distribución | clasificación multilabel + evidencia |

## Flujo seguro

1. Escribe la pregunta, población, unidad y métrica antes de elegir corpus.
2. Registra URL, release, fecha, condiciones, checksum y responsable.
3. Mantén los originales de solo lectura y crea una adaptación versionada.
4. Valida encoding, IDs, offsets, duplicados y spans discontinuos.
5. Documenta cada armonización de etiquetas con definición y revisión clínica.
6. Separa por paciente si es posible; si no, declara la limitación.
7. Bloquea test, modelo, postprocesado y análisis antes de evaluar.
8. No interpretes un benchmark público como validación de tu hospital.

El comando `miopia-course-check validate-brat RUTA` verifica pares BRAT
`.txt/.ann`. El adaptador falla ante evidencia que no coincide con offsets y
ante spans discontinuos que aún no sabe representar.

## Qué debe acompañar al resultado

- ficha de datos y diagrama de flujo de inclusiones;
- tabla de armonización;
- conteos por split, idioma, documento y etiqueta;
- baseline y modelo candidato bajo el mismo protocolo;
- intervalos de confianza y errores por subgrupo;
- lista de incompatibilidades con el uso previsto;
- declaración explícita de si hubo adaptación al corpus externo.
