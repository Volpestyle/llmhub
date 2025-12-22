# Migration notes

## Registry changes

- The registry no longer returns the curated list by default. It now pulls live provider model lists and uses the curated JSON as an overlay for display names, capabilities, context windows, and pricing.
- Default cache TTL is now 30 minutes (refresh with `refresh=true` to bypass the cache).
- The registry tracks short-lived learned unavailability when provider calls fail with model-not-found/invalid-model errors.
- Provider configs now accept `APIKeys`/`apiKeys` arrays for key pools; the hub will rotate keys automatically for generate/stream calls.

## New APIs

- Go/Node now expose `listModelRecords` for entitlement-aware model records with availability metadata.
- Go/Node now expose `generateWithContext` / `streamGenerateWithContext` to pass an entitlement context (API key + fingerprint) per request.
- Python package `llmhub` is available under `packages/python` with adapters, registry, and router.
