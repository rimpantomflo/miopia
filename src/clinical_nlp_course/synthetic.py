"""Generadores deterministas de datos ficticios para pruebas y laboratorios."""

from __future__ import annotations

import random

from .utils import patient_hash_split

RENAL_CLASSIFICATION_TEMPLATES = {
    "HEMODIALYSIS": [
        "Inicia hemodiálisis crónica mediante fístula arteriovenosa.",
        "Continúa programa de HD tres sesiones por semana.",
        "Avui segueix hemodiàlisi a través de FAV funcionant.",
        "Paciente en hemodiálisis; se revisa el acceso vascular.",
    ],
    "PERITONEAL_DIALYSIS": [
        "Realiza diálisis peritoneal automatizada nocturna.",
        "Se mantiene DPA mediante catéter de Tenckhoff.",
        "Continua diàlisi peritoneal sense incidències.",
        "Entrenamiento completado para diálisis peritoneal domiciliaria.",
    ],
    "TRANSPLANT": [
        "Portador de trasplante renal funcionante desde 2021.",
        "Seguimiento del injerto renal con función estable.",
        "Trasplantament renal previ; manté tacròlimus.",
        "Control postrasplante sin datos de rechazo agudo.",
    ],
    "NO_REPLACEMENT": [
        "Enfermedad renal crónica en manejo conservador, sin diálisis.",
        "No precisa tratamiento renal sustitutivo actualmente.",
        "Es comenten opcions de diàlisi, però encara no les inicia.",
        "Se descarta inicio de hemodiálisis en esta visita.",
    ],
}

NOTE_PREFIXES = [
    "Consulta programada. ",
    "Evolución clínica: ",
    "Nota de nefrología. ",
    "Seguimiento ambulatorio. ",
]


def generate_renal_classification_rows(seed: int = 17) -> list[dict[str, object]]:
    """Crea 320 notas y garantiza que cada paciente pertenece a un split."""

    rng = random.Random(seed)
    labels = list(RENAL_CLASSIFICATION_TEMPLATES)
    rows: list[dict[str, object]] = []
    for patient_number in range(1, 81):
        patient_id = f"SYN-P{patient_number:03d}"
        split = patient_hash_split(
            patient_id,
            seed="renal-classification-v2",
            train_pct=65,
            development_pct=20,
        )
        for visit in range(4):
            label = labels[(patient_number + visit) % len(labels)]
            template = RENAL_CLASSIFICATION_TEMPLATES[label][
                (patient_number * 3 + visit) % 4
            ]
            text = rng.choice(NOTE_PREFIXES) + template
            rows.append(
                {
                    "document_id": f"SYN-D{patient_number:03d}-{visit + 1}",
                    "patient_id": patient_id,
                    "language": "ca"
                    if any(char in text for char in "àèìòùç")
                    else "es",
                    "text": text,
                    "label": label,
                    "split": split,
                    "synthetic": True,
                }
            )
    return rows
