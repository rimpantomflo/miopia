# Protocolo de anotación de miopía en cursos clínicos

Versión docente 1.0. Antes de usarlo con datos reales, debe revisarlo el equipo
de Oftalmología, Documentación Clínica, Protección de Datos y metodología.

## 1. Pregunta y unidad de análisis

La pregunta primaria es:

> ¿Existe en el curso clínico evidencia textual de que el paciente ha tenido
> miopía?

No es lo mismo que:

- detectar la palabra «miopía»;
- afirmar que la miopía está activa hoy;
- detectar miopía magna o patológica;
- inferir miopía a partir de una prueba aislada;
- establecer un diagnóstico clínico.

Se anotan tres unidades distintas:

1. **Mención:** fragmento mínimo que nombra miopía.
2. **Documento/curso:** resumen de todas las menciones y mediciones del curso.
3. **Paciente:** agregación longitudinal de todos sus cursos.

Las métricas deben comunicarse por separado para cada unidad.

## 2. Esquema mínimo por mención

Cada mención conserva:

| Campo | Valores | Regla |
|---|---|---|
| `start`, `end` | enteros | Offsets de caracteres; `texto[start:end]` recupera la evidencia |
| `concepto` | `MYOPIA` | No mezclar con miopía patológica o complicaciones |
| `asercion` | `afirmada`, `negada`, `posible` | «No se descarta» es posible, no negada |
| `experienciador` | `paciente`, `familiar`, `otro` | «Madre con miopía» no etiqueta al paciente |
| `temporalidad` | `actual`, `historica`, `incierta` | La cirugía previa prueba historia, no necesariamente estado actual |
| `contexto` | `clinico`, `educativo`, `plantilla` | Una explicación general no es evidencia del paciente |
| `regla_o_modelo` | identificador/versionado | Hace auditable la extracción |

No se recomienda una sola etiqueta como `confirmada_historica`: aserción y
temporalidad son dimensiones independientes.

## 3. Evidencia numérica

Para cada ojo se anotan, cuando estén presentes:

- ojo: OD, OI o AO;
- esfera;
- cilindro;
- eje;
- equivalente esférico calculado;
- técnica y estado de acomodación, si el documento los especifica;
- offsets y texto original.

El equivalente esférico se calcula como:

`esfera + cilindro / 2`

Como referencia epidemiológica, el International Myopia Institute propone
miopía con equivalente esférico ≤ −0,50 D y miopía alta con ≤ −6,00 D, con la
acomodación relajada. En notas rutinarias a menudo no consta la técnica; por
eso el pipeline denomina el resultado **evidencia numérica** y no diagnóstico.

Miopía alta no equivale automáticamente a miopía patológica. Esta última
requiere criterios propios y queda fuera del objetivo inicial.

## 4. Reglas de inclusión

Cuenta para `ever_myopia = sí`:

- mención afirmada, clínica y referida al paciente;
- antecedente de cirugía refractiva explícitamente «por miopía»;
- refracción ocular interpretable que cruza el umbral acordado por el equipo.

No cuenta por sí solo:

- mención negada;
- sospecha o petición de descartar;
- antecedente familiar;
- texto educativo o plantilla sin evidencia individual;
- ausencia de mención;
- agudeza visual, longitud axial o uso de gafas sin regla clínica validada;
- códigos o abreviaturas cuyo significado local no haya sido confirmado.

## 5. Casos difíciles y adjudicación

Registrar la duda en vez de forzar una etiqueta:

- «No se descarta miopía»: `posible`.
- «Operado de LASIK por miopía»: `afirmada + histórica`.
- «No presenta miopía; LASIK por miopía en 2018»: historia positiva, estado
  actual negado.
- «Madre miope. Paciente pendiente de refracción»: familiar afirmada; paciente
  sin confirmación.
- «AF: miopía»: familiar, si `AF` significa antecedentes familiares en el
  centro.
- «Miopización»: no asumir miopía establecida sin acuerdo clínico.
- «OD −8 D»: evidencia numérica; documentar que falta cilindro si no aparece.

Los desacuerdos se resuelven por consenso o por un tercer anotador. Las reglas
nuevas deben añadirse primero a este documento y después al código.

## 6. Muestreo y calidad de la referencia

1. Crear una guía y un lote piloto de 50–100 cursos.
2. Anotar el piloto de forma independiente por dos revisores.
3. Revisar desacuerdos y actualizar la guía.
4. Congelar una versión de la guía.
5. Anotar doblemente una fracción representativa del corpus final.
6. Mantener un conjunto de prueba bloqueado que no se use para crear reglas.

El muestreo debe incluir deliberadamente:

- cursos sin la palabra objetivo;
- especialidades y tipos documentales distintos;
- castellano, catalán, abreviaturas y errores reales;
- casos con negación, familia, incertidumbre y cirugía;
- pacientes con muchos cursos y con uno solo;
- periodos temporales y centros diferentes, si aplica.

Nunca dividir cursos del mismo paciente entre entrenamiento y prueba. De lo
contrario hay fuga de información y las métricas serán optimistas.

## 7. Métricas y criterios de salida

Informar como mínimo:

- TP, TN, FP y FN;
- sensibilidad;
- especificidad;
- valor predictivo positivo (VPP);
- valor predictivo negativo (VPN);
- F1;
- intervalos de confianza;
- prevalencia en la muestra;
- resultados por idioma, tipo documental, centro y periodo.

Definir antes de evaluar qué coste clínico tiene un falso negativo y un falso
positivo. Para una lista de cribado suele priorizarse sensibilidad; para
contactar pacientes o modificar registros se exige un VPP mucho mayor y
revisión humana.

## 8. Privacidad y gobernanza

- Trabajar dentro del entorno autorizado del hospital.
- Obtener base jurídica, aprobaciones y evaluación de impacto cuando proceda.
- Extraer únicamente los campos necesarios.
- Seudonimizar con una clave secreta gestionada fuera del código.
- Separar la tabla de correspondencias del corpus analítico.
- No enviar texto clínico a servicios externos no aprobados.
- No incluir texto real en notebooks, commits, incidencias o logs.
- Mantener control de acceso, auditoría, retención y borrado.

La seudonimización no es anonimización: si la persona puede reidentificarse con
información adicional, los datos siguen siendo personales.

## 9. Versionado

Cada resultado debe registrar:

- versión de este protocolo;
- versión del corpus de referencia;
- hash/versión de reglas;
- modelo y checkpoint, si existe;
- fecha de ejecución;
- parámetros y umbrales;
- responsable de la validación.

