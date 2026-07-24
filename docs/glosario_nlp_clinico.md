# Glosario de NLP clínico

## Datos y anotación

**Corpus:** colección delimitada de documentos utilizada para desarrollo,
evaluación o análisis.

**Documento:** unidad textual original, por ejemplo un curso, informe o alta.

**Episodio:** conjunto temporal de eventos clínicamente relacionados.

**Span:** intervalo contiguo del texto definido por offsets `[start:end]`.

**Offset:** posición de carácter que permite recuperar evidencia exacta.

**Gold standard / referencia:** anotación aceptada tras el procedimiento
definido, idealmente con revisión independiente y adjudicación.

**Guía de anotación:** contrato que define inclusión, exclusión, límites,
atributos y resolución de casos difíciles.

**Adjudicación:** resolución explícita de desacuerdos entre anotadores.

**Acuerdo interanotador:** medida de consistencia entre anotadores. Debe
adaptarse a la tarea; para spans interesa acuerdo de límites y etiqueta.

**Fuga de información:** presencia en desarrollo/entrenamiento de información
que no estaría disponible al predecir o que pertenece al test.

**Deriva:** cambio en entrada, prevalencia, documentación o relación
entrada–salida a lo largo del tiempo.

## Representación lingüística

**Token:** unidad producida por el tokenizador.

**Subtoken:** pieza de palabra utilizada por tokenizadores de transformers.

**Vocabulario:** mapeo interno de formas, atributos e identificadores.

**Lema:** forma canónica de una palabra.

**POS:** categoría gramatical (*part of speech*).

**Dependencia:** relación sintáctica entre tokens.

**BIO/BILUO:** esquemas para representar spans como etiquetas por token.

**NER:** reconocimiento de entidades nombradas; detecta spans y tipos.

**Span categorizado:** fragmento con una o varias etiquetas; puede permitir
solapamientos que `Doc.ents` no admite.

**Normalización de entidades:** enlace de una mención a un concepto canónico.

**Ontología/terminología:** sistema formal de conceptos y relaciones. No es
sinónimo de diccionario de variantes.

## Modelos

**Baseline:** método inicial sencillo y medible.

**Regla:** patrón explícito diseñado por una persona.

**Feature:** representación de entrada utilizada por un modelo.

**Encoder:** modelo que transforma texto en representaciones contextuales.

**Cabeza de tarea:** capa añadida a un encoder para NER, clasificación u otra
tarea.

**Fine-tuning / ajuste:** actualización de un modelo preentrenado usando datos
de una tarea concreta.

**Transfer learning:** reutilización de conocimiento aprendido en otro corpus o
tarea.

**Transformer:** arquitectura basada en atención.

**Atención:** mecanismo que combina representaciones de posiciones del texto;
no debe interpretarse automáticamente como explicación clínica.

**Embedding:** vector que representa tokens, fragmentos o documentos.

**LLM generativo:** modelo que predice secuencias y produce texto nuevo.

**Temperatura:** parámetro de muestreo; no equivale a confianza calibrada.

**Prompt:** entrada estructurada con instrucciones, contexto y ejemplos.

**Few-shot:** inclusión de unos pocos ejemplos en el prompt.

**RAG:** recuperación de fragmentos seguida de generación condicionada a ellos.

**Alucinación:** afirmación no sustentada por la entrada o fuente requerida.

**Abstención:** salida explícita que reconoce evidencia insuficiente.

## Evaluación

**TP/TN/FP/FN:** verdaderos positivos, verdaderos negativos, falsos positivos y
falsos negativos.

**Sensibilidad/recall:** `TP / (TP + FN)`.

**Especificidad:** `TN / (TN + FP)`.

**Precisión/VPP:** `TP / (TP + FP)`.

**VPN:** `TN / (TN + FN)`.

**F1:** media armónica de precisión y recall.

**Micro promedio:** agrega decisiones antes de calcular la métrica.

**Macro promedio:** promedia métricas de clases o grupos por igual.

**Match exacto:** etiqueta y límites coinciden por completo.

**Match por solapamiento:** acepta intersección según una regla predefinida.

**Calibración:** correspondencia entre probabilidad predicha y frecuencia
observada.

**Brier score:** error cuadrático medio de probabilidades binarias.

**Intervalo de confianza:** rango de incertidumbre del estimador bajo el método
utilizado.

**Bootstrap agrupado:** remuestreo de pacientes completos para respetar
dependencia entre documentos.

**Validación interna:** evaluación dentro de la población/fuente de desarrollo.

**Validación temporal:** evaluación en un periodo posterior.

**Validación externa:** evaluación en otra institución, entorno o población.

**Utilidad clínica:** efecto del sistema en decisiones, flujo, resultados,
seguridad y carga de trabajo; no se deduce únicamente de F1.

## Producción

**Data contract:** esquema y condiciones que debe cumplir la entrada.

**Model card:** documentación de uso, datos, rendimiento y limitaciones del
modelo.

**Dataset card/ficha de datos:** documentación de procedencia, población,
anotación, particiones y riesgos del corpus.

**Shadow mode / modo silencioso:** ejecución sin influir en decisiones clínicas.

**Human in the loop:** intervención humana definida; debe especificarse quién,
cuándo, qué ve y qué autoridad conserva.

**MLOps:** prácticas para versionar, desplegar, observar y mantener modelos.

**Trazabilidad:** capacidad de reconstruir datos, versión, configuración,
evidencia y decisión de una salida.

