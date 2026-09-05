from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx2")

from fastapi.testclient import TestClient

from miopia_nlp.service import create_app


def test_health_and_phenotype_contract() -> None:
    client = TestClient(create_app())
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.headers["cache-control"] == "no-store"

    response = client.post("/v1/phenotype", json={"text": "No se descarta miopía."})
    assert response.status_code == 200
    assert response.json()["document_status"] == "possible"
    assert response.headers["x-request-id"]
    assert "text" not in response.json()
    metrics = client.get("/metrics")
    assert "miopia_http_requests_total" in metrics.text


def test_api_rejects_unknown_fields_and_oversized_text() -> None:
    client = TestClient(create_app())
    unknown = client.post("/v1/phenotype", json={"text": "Miopía", "name": "PHI"})
    assert unknown.status_code == 422
    oversized = client.post("/v1/phenotype", json={"text": "x" * 200_001})
    assert oversized.status_code == 422
    too_large = client.post(
        "/v1/phenotype",
        content=b"x" * 1_000_001,
        headers={"content-type": "application/json"},
    )
    assert too_large.status_code == 413
