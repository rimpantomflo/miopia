# Paso de prototipo a Oracle

Este documento es una lista de control. Los nombres de tablas, permisos y
procedimientos concretos deben acordarse con el hospital.

## Zona segura

La conexión, la seudonimización y el procesamiento deben ejecutarse en
infraestructura autorizada. Un notebook personal no es el lugar de destino de
los textos reales.

Antes de extraer:

- aprobación clínica, institucional y de protección de datos;
- cuenta Oracle de solo lectura y mínimo privilegio;
- tabla o vista aprobada;
- ventana temporal definida;
- lista mínima de columnas;
- secreto HMAC en un gestor de secretos;
- ruta de salida cifrada y con acceso restringido;
- política de logs que excluya texto e identificadores.

## Consulta incremental

Usar parámetros para valores y una tabla/vista aprobada fija:

```sql
SELECT patient_id, course_id, course_date, language_code, course_text
FROM approved_clinical_course_view
WHERE course_date >= :date_from
  AND course_date < :date_to
  AND course_text IS NOT NULL
ORDER BY course_date, course_id
```

No usar `SELECT *`. Los nombres de tabla no se parametrizan del mismo modo que
los valores: deben proceder de configuración validada, no de entrada libre.

## Procesamiento por lotes

1. Recuperar lotes con `fetchmany`.
2. Convertir CLOB a texto dentro de la zona segura.
3. Crear el seudónimo antes de escribir resultados analíticos.
4. Procesar por lotes y conservar únicamente offsets, evidencia mínima y
   predicciones requeridas.
5. Guardar un checkpoint incremental.
6. Registrar conteos, tiempos y versión del pipeline, nunca texto clínico.

Para spaCy, `nlp.pipe` evita procesar documento por documento y permite pasar
metadatos con `as_tuples=True`. En Windows, muchos procesos pueden empeorar el
rendimiento por el método `spawn`; medir primero con un proceso y lotes mayores.

## Salida recomendada

Separar tablas:

- `run`: versión, fecha, parámetros y estado;
- `course_prediction`: seudónimo, curso, fecha y estado documental;
- `mention_evidence`: curso, offsets, atributos y regla;
- `refraction_evidence`: curso, ojo y valores numéricos;
- `patient_phenotype`: agregación longitudinal y última evidencia;
- `review_queue`: casos dudosos, sin texto completo salvo necesidad aprobada.

Las claves de reidentificación permanecen en un sistema separado.

## Despliegue por fases

1. **Silencioso:** ejecutar sin mostrar resultados a clínicos.
2. **Validación retrospectiva:** comparar con revisión manual.
3. **Piloto asistido:** lista revisada siempre por una persona.
4. **Monitorizado:** estudiar errores, deriva y subgrupos.
5. **Revalidación:** ante cambios de plantilla, centro, idioma, reglas o modelo.

El pipeline produce evidencia para revisión; no debe activar por sí solo
tratamientos, citas ni cambios en la historia.
