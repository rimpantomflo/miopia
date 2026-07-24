"""Ejecuta las celdas de código en orden para una validación rápida.

No reemplaza la ejecución con un kernel Jupyter, pero detecta errores de Python,
imports, aserciones y dependencias en la ruta principal.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorial" / "tutorial_miopia_nlp.ipynb"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
namespace = {
    "__name__": "__tutorial_validation__",
    "display": lambda *values, **_: None,
}

code_number = 0
for cell_number, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    code_number += 1
    source = "".join(cell["source"])
    try:
        exec(compile(source, f"{NOTEBOOK.name}:cell_{cell_number}", "exec"), namespace)
    except Exception as error:
        raise RuntimeError(
            f"Fallo en celda de notebook {cell_number} "
            f"(celda de código {code_number})"
        ) from error

print(f"OK: {code_number} celdas de código ejecutadas en orden")
