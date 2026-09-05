"""Factoría FastAPI opcional, sin estado ni logs de texto clínico."""

from __future__ import annotations

import re
import time
import uuid
from importlib.metadata import version
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .pipeline import phenotype_course


class PhenotypeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=200_000)


class PhenotypeResponse(BaseModel):
    document_status: str
    ever_myopia: bool
    current_status: str
    high_myopia_numeric: bool
    mentions: list[dict[str, Any]]
    refractions: list[dict[str, Any]]
    evidence_count: int


def create_app():
    """Crea la app solo cuando se instaló ``miopia[service]``."""

    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, Response
        from prometheus_client import (
            CollectorRegistry,
            Counter,
            Histogram,
            generate_latest,
        )
    except ImportError as error:
        raise RuntimeError("instala la API con: uv sync --extra service") from error

    app = FastAPI(
        title="Miopia NLP",
        version=version("miopia"),
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    registry = CollectorRegistry()
    request_count = Counter(
        "miopia_http_requests_total",
        "HTTP requests without payload or identifier labels.",
        ("method", "route", "status"),
        registry=registry,
    )
    request_latency = Histogram(
        "miopia_http_request_duration_seconds",
        "HTTP request duration without payload or identifier labels.",
        ("method", "route"),
        registry=registry,
    )

    @app.middleware("http")
    async def request_metadata(request: Request, call_next):
        supplied_request_id = request.headers.get("X-Request-ID", "")
        request_id = re.sub(r"[^A-Za-z0-9._-]", "", supplied_request_id)[:128]
        request_id = request_id or str(uuid.uuid4())
        started = time.perf_counter()
        content_length = request.headers.get("content-length")
        try:
            body_too_large = bool(content_length) and int(content_length) > 1_000_000
        except ValueError:
            body_too_large = True
        if body_too_large:
            return JSONResponse(
                status_code=413,
                content={"detail": "request body too large"},
                headers={"X-Request-ID": request_id, "Cache-Control": "no-store"},
            )
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        route = getattr(request.scope.get("route"), "path", "unmatched")
        request_count.labels(request.method, route, str(response.status_code)).inc()
        request_latency.labels(request.method, route).observe(elapsed)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Elapsed-MS"] = f"{elapsed * 1000:.3f}"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, error: Exception):
        del request, error
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": version("miopia")}

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(
            content=generate_latest(registry),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post("/v1/phenotype", response_model=PhenotypeResponse)
    def phenotype(payload: PhenotypeRequest) -> dict[str, Any]:
        return phenotype_course(payload.text)

    return app
