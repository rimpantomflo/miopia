# Guía de validación para NLP clínico

## Capas de evaluación

1. **Técnica:** ¿el código produce salidas válidas?
2. **Extracción:** ¿spans, atributos, valores y relaciones son correctos?
3. **Documento:** ¿clasifica correctamente el curso?
4. **Paciente/episodio:** ¿la agregación responde al fenotipo?
5. **Flujo clínico:** ¿ayuda sin crear daño o carga inaceptable?

Una métrica alta en una capa no garantiza la siguiente.

## Plan previo

Antes de evaluar, congelar:

- uso previsto;
- población y exclusiones;
- referencia;
- unidad;
- resultado primario;
- umbral;
- subgrupos;
- manejo de indeterminados;
- análisis de errores;
- versión del sistema.

## Métricas por tarea

### NER

- precisión, recall y F1 micro por entidad;
- match exacto;
- match por solapamiento;
- errores de límite frente a etiqueta;
- atributos por separado.

### Clasificación

- matriz de confusión;
- sensibilidad/especificidad;
- VPP/VPN;
- F1 y balanced accuracy;
- ROC-AUC y PR-AUC cuando sean apropiadas;
- calibración y Brier;
- rendimiento al umbral operativo.

### Normalización

- top-1;
- top-k;
- MRR;
- errores por concepto ausente, ambiguo o mal detectado.

### Recuperación/RAG

- Recall@k, Precision@k, MRR o nDCG;
- cobertura de la fuente correcta;
- fidelidad de cada afirmación;
- completitud;
- abstención;
- seguridad y robustez.

## Dependencia y particiones

Los cursos de un paciente no son independientes. Partir por documento:

- filtra estilo y hechos longitudinales;
- estrecha artificialmente intervalos;
- puede duplicar plantillas o fragmentos.

Usa particiones y bootstrap agrupados por paciente.

## Prevalencia

VPP y VPN dependen de prevalencia. Un corpus enriquecido puede medir
sensibilidad/especificidad, pero sus valores predictivos no representan
automáticamente producción. Informa prevalencia y mecanismo de muestreo.

## Incertidumbre

Incluye intervalos de confianza. Para datos longitudinales, remuestrea la unidad
independiente adecuada. Un intervalo amplio no se arregla reportando más
decimales.

## Subgrupos y deriva

Como mínimo considerar:

- idioma;
- centro;
- servicio;
- tipo documental;
- periodo;
- sexo/edad cuando sea pertinente y permitido;
- longitud y calidad del texto;
- prevalencia;
- dispositivo o versión del EHR.

Las diferencias requieren interpretación y suficiente tamaño; no convertir
subgrupos pequeños en conclusiones definitivas.

## Modelos generativos

Separar:

- validez del JSON;
- exactitud de campos;
- evidencia y offsets;
- afirmaciones no sustentadas;
- omisiones;
- consistencia entre ejecuciones;
- sensibilidad al prompt;
- abstención;
- juicio clínico;
- tiempo y coste.

Un LLM evaluador puede escalar revisión, pero no debe ser la única referencia.
Audita concordancia con humanos y sesgos del juez.

## Validación clínica

La validación retrospectiva no demuestra beneficio. El paso a uso real debe
estudiar:

- quién usa la salida;
- qué información ve;
- cómo se corrige;
- tiempo y carga;
- sobreconfianza y automatización;
- fallos y recuperación;
- efecto sobre decisiones y pacientes.

## Guías de reporte

- TRIPOD+AI: modelos de predicción.
- PROBAST+AI: riesgo de sesgo y aplicabilidad.
- STARD-AI: exactitud diagnóstica con IA.
- DECIDE-AI: evaluación clínica temprana en vivo.
- CONSORT-AI/SPIRIT-AI: ensayos e intervenciones con IA.

Elegir según diseño; una checklist de reporte no sustituye la buena metodología.

