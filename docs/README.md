# ai-kit documentation

## Overview
ai-kit is a monorepo that standardizes model discovery, routing, and inference across
Node.js, Go, and Python. Each SDK exposes a Hub for generation, a model registry with caching,
optional model routing, and SSE support for progressive UI rendering.

Key building blocks:
- Hub: orchestration layer for list, generate, and stream.
- Model registry: caches provider model lists, applies curated metadata and pricing.
- Model router: resolves a preferred or cheapest model based on constraints.
- Provider adapters: OpenAI, Anthropic, Google Gemini, and xAI.
- HTTP handlers (Node/Go) and ASGI adapter (Python): drop-in endpoints for REST + SSE.

## Architecture
![Hub architecture](diagrams/hub-architecture.png)

The Hub delegates list/generate to adapters, the registry caches and enriches models, and the
router selects a primary model plus fallbacks when you need to auto-resolve across providers.

## Streaming (SSE)
![SSE streaming flow](diagrams/sse-stream.png)

SSE streams expose incremental chunks so UIs can render text as it arrives. Node/Go ship HTTP
handlers and Python ships an ASGI adapter that format `event: chunk` and `event: done` responses.

## Evaluation notes
- Consistent core concepts across languages (Hub, registry, router, SSE chunks).
- Pricing is derived from curated model metadata; if a model is missing pricing metadata,
  cost estimation returns `None`/`undefined`.
- Python/Go support a static "catalog" provider when `models/catalog_models.json` is present;
  Node focuses on live provider lists plus curated metadata.
- Python exposes optional local pipelines (Transformers + Torch) for basic vision tasks.

## Related docs
- HTTP API: `docs/http-api.md`
- OpenAPI spec: `servers/openapi.yaml`
