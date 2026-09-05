"""CLI estable del baseline de miopía."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from .pipeline import phenotype_course


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="miopia")
def main() -> None:
    """Fenotipado docente y trazable de miopía."""


@main.command("analyze")
@click.argument("text", required=False)
@click.option(
    "--input-file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Archivo UTF-8 local. No uses datos reales fuera del entorno autorizado.",
)
@click.option("--pretty/--compact", default=True, help="Formato de salida JSON.")
def analyze(text: str | None, input_file: Path | None, pretty: bool) -> None:
    """Analiza TEXT o un archivo; nunca escribe el contenido en logs."""

    if (text is None) == (input_file is None):
        raise click.UsageError("indica TEXT o --input-file, pero no ambos")
    source = input_file.read_text(encoding="utf-8") if input_file else str(text)
    result = phenotype_course(source)
    click.echo(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2 if pretty else None,
            default=str,
        )
    )


@main.command("stdin")
@click.option("--pretty/--compact", default=False)
def analyze_stdin(pretty: bool) -> None:
    """Lee texto por stdin para integraciones que evitan argumentos visibles."""

    source = sys.stdin.read()
    if not source.strip():
        raise click.UsageError("stdin está vacío")
    click.echo(
        json.dumps(
            phenotype_course(source),
            ensure_ascii=False,
            indent=2 if pretty else None,
            default=str,
        )
    )
