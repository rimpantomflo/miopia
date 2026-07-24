"""Fábrica mínima de notebooks para mantener el curso versionable."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Iterable


def _source(text: str) -> list[str]:
    text = dedent(text).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str = "") -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text) if text.strip() else [],
    }


def notebook(cells: Iterable[dict], *, title: str) -> dict:
    return {
        "cells": list(cells),
        "metadata": {
            "course": {"title": title, "synthetic_data_only": True},
            "kernelspec": {
                "display_name": "miopia (.venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, cells: Iterable[dict], *, title: str) -> None:
    path.write_text(
        json.dumps(notebook(cells, title=title), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def common_setup() -> dict:
    return code(
        """
        import sys
        from pathlib import Path

        PROJECT_ROOT = Path.cwd()
        if not (PROJECT_ROOT / "src").exists():
            PROJECT_ROOT = PROJECT_ROOT.parent
        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        print("Proyecto:", PROJECT_ROOT)
        """
    )
