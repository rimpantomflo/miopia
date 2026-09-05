# Protocolo de proyecto hospitalario seguro

Este documento es una lista operativa docente, no asesoramiento jurídico ni una
autorización para usar datos clínicos.

## Gate 0 — problema y gobernanza

- uso previsto, usuarios, salida, decisión apoyada y usos excluidos;
- responsable clínico, responsable técnico, propietario del dato y seguridad;
- base jurídica, aprobación institucional y evaluación de protección de datos;
- minimización, retención, accesos, auditoría y plan de destrucción;
- infraestructura y proveedores expresamente autorizados;
- criterio de parada y canal privado de incidentes.

No se extraen datos hasta que este gate esté firmado por los responsables
locales.

## Gate 1 — datos y referencia

- cohorte e índice temporal reproducibles;
- esquema y protocolo de anotación pilotados;
- doble anotación de una muestra y adjudicación;
- separación por paciente, tiempo y centro;
- calidad de offsets, nulos, duplicados, idioma y plantillas;
- ficha de datos con exclusiones y limitaciones;
- ausencia de PHI en Git, notebooks, logs y sistemas de incidencias.

## Gate 2 — modelado

- baseline de reglas y baseline ML clásico;
- modelo complejo justificado por una ganancia predefinida;
- búsqueda solo en train/development;
- versiones de datos, código, terminología, prompt/modelo y semillas;
- análisis de contexto, relaciones, normalización o retrieval por separado;
- casos centinela y pruebas adversariales.

## Gate 3 — evaluación bloqueada

- métrica primaria y umbral preespecificados;
- test temporal abierto una sola vez;
- intervalos por paciente y comparación emparejada;
- calibración, subgrupos, errores graves y cobertura/abstención;
- validación externa o justificación de por qué falta;
- evaluación de utilidad y carga de revisión;
- model card aprobada.

## Gate 4 — shadow mode

- contrato de entrada y salida;
- lotes idempotentes sensibles al contenido y la versión;
- gestión externa de secretos y seudonimización HMAC;
- lista positiva de logs sin texto ni identificadores originales;
- autenticación, autorización, TLS y segmentación según el hospital;
- monitorización de calidad de entrada, deriva, errores y latencia;
- muestra humana periódica para medir rendimiento real;
- artefacto anterior disponible y rollback ensayado.

En shadow mode la salida no cambia la atención.

## Gate 5 — piloto asistido

- aprobación explícita para el alcance limitado;
- interfaz que muestra evidencia, fecha, incertidumbre y procedencia;
- revisores con tiempo, formación, autoridad y vía de corrección;
- medición de automation bias, alert fatigue y discrepancias;
- umbrales de suspensión y responsable de guardia;
- evaluación prospectiva adecuada al uso previsto.

## Definición de terminado

Un proyecto no está terminado porque tenga una API o un F1 alto. Está listo para
su fase autorizada cuando puede reproducirse, auditarse, monitorizarse,
corregirse, detenerse y revertirse, y cuando las personas afectadas comprenden
sus límites.
