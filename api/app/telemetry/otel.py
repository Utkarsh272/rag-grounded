"""
OpenTelemetry tracing setup.

Exports to:
  - Console (local dev — default when OTEL_EXPORTER is unset or "console")
  - OTLP HTTP (prod — set OTEL_EXPORTER=otlp + OTEL_EXPORTER_OTLP_ENDPOINT)

Call configure_tracing() once at app startup, then use get_tracer() anywhere:

    from app.telemetry.otel import get_tracer
    tracer = get_tracer()

    with tracer.start_as_current_span("retrieval") as span:
        span.set_attribute("mode", "hybrid")
        span.set_attribute("chunk_count", len(chunks))
"""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

SERVICE_NAME = "rag-grounded"
SERVICE_VERSION = "1.0.0"

_tracer: trace.Tracer | None = None


def configure_tracing() -> None:
    """Call once at FastAPI startup."""
    global _tracer

    resource = Resource.create({
        "service.name": SERVICE_NAME,
        "service.version": SERVICE_VERSION,
    })

    provider = TracerProvider(resource=resource)
    exporter_type = os.getenv("OTEL_EXPORTER", "console").lower()

    if exporter_type == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            endpoint = os.getenv(
                "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces"
            )
            exporter = OTLPSpanExporter(endpoint=endpoint)
            print(f"[otel] OTLP exporter → {endpoint}")
        except ImportError:
            print("[otel] OTLP exporter not installed, falling back to console")
            exporter = ConsoleSpanExporter()
    else:
        exporter = ConsoleSpanExporter()
        print("[otel] Console span exporter enabled")

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(SERVICE_NAME, SERVICE_VERSION)


def get_tracer() -> trace.Tracer:
    """Return the configured tracer. configure_tracing() must have been called first."""
    if _tracer is None:
        configure_tracing()
    return _tracer  # type: ignore[return-value]
