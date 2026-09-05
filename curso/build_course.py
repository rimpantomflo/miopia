"""Genera todos los notebooks avanzados del curso."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from curso.modules import (
    module_00,
    module_01,
    module_02,
    module_03,
    module_04,
    module_05,
    module_06,
    module_07,
    module_08,
    module_09,
    module_10,
    module_11,
    module_12,
    module_13,
    module_14,
    module_15,
    module_16,
    module_17,
    module_18,
)
from curso.notebook_factory import write_notebook

MODULES = [
    ("00_mapa_y_metodo.ipynb", module_00.TITLE, module_00.build()),
    ("01_corpus_y_diccionarios.ipynb", module_01.TITLE, module_01.build()),
    ("02_spacy_avanzado.ipynb", module_02.TITLE, module_02.build()),
    ("03_ner_entrenable_spacy.ipynb", module_03.TITLE, module_03.build()),
    ("04_transformers_y_bsc.ipynb", module_04.TITLE, module_04.build()),
    (
        "05_clasificacion_relaciones_normalizacion.ipynb",
        module_05.TITLE,
        module_05.build(),
    ),
    ("06_llm_extraccion_y_rag.ipynb", module_06.TITLE, module_06.build()),
    ("07_validacion_avanzada.ipynb", module_07.TITLE, module_07.build()),
    ("08_proyecto_nefrologia.ipynb", module_08.TITLE, module_08.build()),
    ("09_produccion_y_monitorizacion.ipynb", module_09.TITLE, module_09.build()),
    ("10_evaluacion_de_competencias.ipynb", module_10.TITLE, module_10.build()),
    ("11_anotacion_asistida_doccano.ipynb", module_11.TITLE, module_11.build()),
    ("12_ml_clasico_fuerte.ipynb", module_12.TITLE, module_12.build()),
    ("13_corpora_publicos_benchmark.ipynb", module_13.TITLE, module_13.build()),
    ("14_finetuning_transformers.ipynb", module_14.TITLE, module_14.build()),
    (
        "15_contexto_relaciones_normalizacion.ipynb",
        module_15.TITLE,
        module_15.build(),
    ),
    ("16_llm_local_rag.ipynb", module_16.TITLE, module_16.build()),
    ("17_validacion_explicabilidad.ipynb", module_17.TITLE, module_17.build()),
    ("18_produccion_capstone_hero.ipynb", module_18.TITLE, module_18.build()),
]


for filename, title, cells in MODULES:
    write_notebook(ROOT / "curso" / filename, cells, title=title)
    print(f"{filename}: {len(cells)} celdas")
