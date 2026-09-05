"""Adaptadores explícitos para corpora públicos; nunca descargan datos."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class CorpusDescriptor:
    name: str
    url: str
    tasks: tuple[str, ...]
    access: str
    license_note: str


PUBLIC_CORPORA = {
    "carmen_i": CorpusDescriptor(
        name="CARMEN-I",
        url="https://physionet.org/content/carmen-i/1.0.1/",
        tasks=("NER", "de-identification", "clinical information extraction"),
        access="credentialed",
        license_note="PhysioNet credentialing, training, DUA and project approval apply.",
    ),
    "symptemist": CorpusDescriptor(
        name="SympTEMIST",
        url="https://temu.bsc.es/symptemist/",
        tasks=("symptom NER", "SNOMED CT normalization"),
        access="challenge terms",
        license_note="Accept the task data terms before use.",
    ),
    "medprocner": CorpusDescriptor(
        name="MedProcNER",
        url="https://temu.bsc.es/medprocner/",
        tasks=("procedure NER", "normalization", "document indexing"),
        access="challenge terms",
        license_note="Accept the task data terms before use.",
    ),
    "codiesp": CorpusDescriptor(
        name="CodiEsp",
        url="https://temu.bsc.es/codiesp/",
        tasks=("ICD-10 coding", "evidence extraction"),
        access="challenge terms",
        license_note="Review CodiEsp distribution and use conditions.",
    ),
}


def load_brat(directory: str | Path) -> list[dict[str, Any]]:
    """Carga pares ``.txt/.ann`` BRAT y valida que cada span recupere su texto."""

    root = Path(directory)
    records: list[dict[str, Any]] = []
    for text_path in sorted(root.glob("*.txt")):
        text = text_path.read_text(encoding="utf-8")
        annotations: list[dict[str, Any]] = []
        annotation_path = text_path.with_suffix(".ann")
        if annotation_path.exists():
            for line_number, line in enumerate(
                annotation_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line or not line.startswith("T"):
                    continue
                fields = line.split("\t")
                if len(fields) != 3:
                    raise ValueError(f"{annotation_path}:{line_number}: BRAT inválido")
                annotation_id, metadata, evidence = fields
                metadata_parts = metadata.split()
                if ";" in metadata:
                    raise ValueError(
                        f"{annotation_path}:{line_number}: span discontinuo no soportado"
                    )
                if len(metadata_parts) != 3:
                    raise ValueError(
                        f"{annotation_path}:{line_number}: metadatos inválidos"
                    )
                label, start_raw, end_raw = metadata_parts
                start, end = int(start_raw), int(end_raw)
                if text[start:end] != evidence:
                    raise ValueError(
                        f"{annotation_path}:{line_number}: evidencia no coincide con offsets"
                    )
                annotations.append(
                    {
                        "annotation_id": annotation_id,
                        "start": start,
                        "end": end,
                        "label": label,
                        "evidence": evidence,
                    }
                )
        records.append(
            {"document_id": text_path.stem, "text": text, "entities": annotations}
        )
    if not records:
        raise ValueError(f"no se encontraron .txt en {root}")
    return records


def load_entity_tsv(
    path: str | Path,
    *,
    document_column: str = "document_id",
    label_column: str = "label",
    start_column: str = "start",
    end_column: str = "end",
    text_column: str = "text",
    delimiter: str = "\t",
) -> list[dict[str, Any]]:
    """Carga el formato tabular común en tareas BSC sin asumir una release."""

    output: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        required = {
            document_column,
            label_column,
            start_column,
            end_column,
            text_column,
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"columnas ausentes: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                start, end = int(row[start_column]), int(row[end_column])
            except (TypeError, ValueError) as error:
                raise ValueError(f"línea {line_number}: offsets inválidos") from error
            output.append(
                {
                    "document_id": row[document_column],
                    "label": row[label_column],
                    "start": start,
                    "end": end,
                    "text": row[text_column],
                }
            )
    return output


def split_document_ids(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {str(row["document_id"]) for row in rows}
