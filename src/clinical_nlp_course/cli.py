"""Comandos de diagnóstico del entorno docente."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import click

from .assessment import validate_capstone_evidence
from .public_data import load_brat


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Comprueba el entorno y corpora antes de entrenar."""


@main.command("doctor")
def doctor() -> None:
    """Muestra qué rutas opcionales están disponibles."""

    packages = {
        "core": ("spacy", "sklearn", "pydantic"),
        "transformers": ("transformers", "datasets", "torch", "seqeval"),
        "embeddings": ("sentence_transformers",),
        "service": ("fastapi", "uvicorn"),
        "oracle": ("oracledb",),
    }
    status = {
        extra: {
            package: importlib.util.find_spec(package) is not None
            for package in dependencies
        }
        for extra, dependencies in packages.items()
    }
    click.echo(json.dumps(status, indent=2, sort_keys=True))


@main.command("validate-brat")
@click.argument(
    "directory", type=click.Path(path_type=Path, exists=True, file_okay=False)
)
def validate_brat(directory: Path) -> None:
    """Valida offsets de un corpus BRAT local sin modificarlo."""

    records = load_brat(directory)
    entities = sum(len(record["entities"]) for record in records)
    click.echo(f"OK: {len(records)} documentos, {entities} entidades")


@main.command("check-capstone")
@click.argument(
    "submission", type=click.Path(path_type=Path, exists=True, dir_okay=False)
)
def check_capstone(submission: Path) -> None:
    """Comprueba que el JSON de entrega no tenga evidencias críticas vacías."""

    payload = json.loads(submission.read_text(encoding="utf-8"))
    issues = validate_capstone_evidence(payload)
    if issues:
        for issue in issues:
            click.echo(f"ERROR: {issue}", err=True)
        raise click.exceptions.Exit(1)
    click.echo("CAPSTONE OK: entregables y salvaguardas declarados")
