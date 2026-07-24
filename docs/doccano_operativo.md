# Doccano para anotación clínica asistida

Doccano es la única plataforma de anotación utilizada en este curso. La
selección reduce complejidad: los clínicos trabajan en una interfaz web y la
persona técnica mantiene importación, preanotación, exportación y controles de
calidad.

## Alcance inicial

El primer piloto usa exclusivamente textos ficticios y dos proyectos:

1. **NER de evidencia:** `MYOPIA`, `REFRACTION`, `REFRACTIVE_SURGERY`.
2. **Clasificación documental:** `PRESENT`, `ABSENT`, `UNCERTAIN`,
   `INSUFFICIENT`.

Separar evidencia y decisión evita crear una combinación inmanejable de
etiquetas de contexto. El texto real solo se incorporará tras aprobación y
despliegue institucional.

## Instalación para aprendizaje

La documentación oficial ofrece instalación mediante `pip`, Docker y Docker
Compose. Para el curso, utiliza Docker y un volumen persistente. No reutilices
las credenciales de ejemplo de la documentación.

1. Instala Docker Desktop o usa un servidor Linux autorizado.
2. Sigue la sección vigente de
   [instalación oficial](https://doccano.github.io/doccano/install_and_upgrade_doccano/).
3. Crea una contraseña de administrador robusta.
4. Expón el servicio solo en `localhost` durante las prácticas.
5. Verifica que el volumen persiste tras reiniciar el contenedor.

El servidor de aprendizaje no debe contener información clínica real.

## Preparar el ejercicio

Desde la raíz:

```powershell
uv run python scripts/preparar_doccano.py
```

Se genera `data/doccano_nefrologia_sintetica.jsonl`. Cada registro contiene
texto, sugerencias por offsets y metadatos de procedencia.

En Doccano:

1. crea un proyecto `Sequence Labeling`;
2. activa anotación colaborativa;
3. mantén desactivados los spans solapados en el primer piloto;
4. crea las etiquetas incluidas en el archivo;
5. importa el JSONL;
6. crea dos usuarios anotadores y un aprobador;
7. añade la guía de anotación;
8. completa diez documentos y exporta el resultado.

## Sugerencias previas

La primera versión usa el diccionario del repositorio. Su objetivo es comprobar
el flujo, no producir una referencia definitiva.

Después pueden conectarse, uno por vez:

- reglas y `SpanRuler`;
- NER spaCy;
- NER BSC evaluado localmente;
- LLM local autorizado con salida estructurada.

Doccano permite autoetiquetado mediante una API REST. Para NER espera una lista
con etiqueta, offset inicial y offset final. Consulta la
[configuración oficial de autoetiquetado](https://doccano.github.io/doccano/advanced/auto_labelling_config/).

El adaptador debe:

1. recibir texto;
2. ejecutar el modelo local;
3. convertir su salida al esquema de Doccano;
4. comprobar que `text[start:end]` es la evidencia;
5. registrar versión y origen de la sugerencia;
6. devolver una lista vacía si la salida no es válida.

No se corrigen offsets silenciosamente y no se envía texto a proveedores
externos no autorizados.

## Flujo de trabajo

```text
selección de documentos
  → preanotación versionada
  → anotación A y B independientes
  → revisión de discrepancias
  → aprobación/adjudicación
  → exportación inmutable
  → validación automática
  → corpus versionado
```

Las anotaciones originales no se sustituyen por la adjudicada. Se conservan
para estudiar acuerdo, ambigüedad y cambios de la guía.

## Medir si la ayuda funciona

Asigna aleatoriamente documentos comparables a:

- anotación manual;
- anotación con sugerencias.

Mide:

- minutos por documento;
- sugerencias aceptadas, eliminadas y corregidas;
- entidades añadidas que el modelo omitió;
- exactitud contra la adjudicación;
- desacuerdo entre clínicos;
- percepción de esfuerzo.

Reserva al menos un 20 % sin sugerencias. Si solo se revisan preanotaciones, no
podrás detectar con fiabilidad el sesgo de automatización.

## Paso a infraestructura hospitalaria

Antes de usar cursos reales:

- revisión de seguridad y protección de datos;
- PostgreSQL y copias de seguridad probadas;
- HTTPS y acceso restringido a red/VPN institucional;
- usuarios nominales y mínimo privilegio;
- política de actualización y reversión;
- logs sin texto clínico;
- retención y borrado definidos;
- modelo de preanotación ejecutado dentro del perímetro autorizado;
- prueba de restauración de datos.

Doccano facilita la interfaz, pero no sustituye estos controles.
