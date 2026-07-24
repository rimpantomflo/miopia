"""Ejecuta en orden la ruta básica de todos los notebooks del curso."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback
import types


ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT / "curso"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

notebooks = sorted(COURSE.glob("[0-9][0-9]_*.ipynb"))
if not notebooks:
    raise RuntimeError("No se encontraron notebooks; ejecuta curso/build_course.py")

total_code = 0
for notebook_path in notebooks:
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    module_name = f"__validate_{notebook_path.stem}__"
    validation_module = types.ModuleType(module_name)
    sys.modules[module_name] = validation_module
    namespace = validation_module.__dict__
    namespace["display"] = lambda *values, **_: None
    code_number = 0
    for cell_number, cell in enumerate(payload["cells"]):
        if cell["cell_type"] != "code":
            continue
        code_number += 1
        source = "".join(cell["source"])
        try:
            exec(
                compile(source, f"{notebook_path.name}:cell_{cell_number}", "exec"),
                namespace,
            )
        except Exception as error:
            traceback.print_exc()
            raise RuntimeError(
                f"Fallo en {notebook_path.name}, celda {cell_number} "
                f"(código {code_number})"
            ) from error
    total_code += code_number
    print(f"OK {notebook_path.name}: {code_number} celdas de código")

print(f"CURSO OK: {len(notebooks)} notebooks, {total_code} celdas de código")
