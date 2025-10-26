import { AdapterMap, ProviderAdapter } from "../core/provider.js";
import { LLMHubError, toHubError } from "../core/errors.js";
import {
  ErrorKind,
  ListModelsParams,
  ModelMetadata,
  Provider,
} from "../core/types.js";

interface CacheEntry {
  expiresAt: number;
  data: ModelMetadata[];
}

export interface ModelRegistryOptions {
  ttlMs: number;
}

export class ModelRegistry {
  private readonly cache = new Map<Provider, CacheEntry>();
  private readonly adapters: AdapterMap;
  private readonly ttlMs: number;

  constructor(adapters: AdapterMap, options?: Partial<ModelRegistryOptions>) {
    this.adapters = adapters;
    this.ttlMs = options?.ttlMs ?? 6 * 60 * 60 * 1000; // default 6h
  }

  async list(params?: ListModelsParams): Promise<ModelMetadata[]> {
    const providers = (params?.providers ??
      (Object.keys(this.adapters) as Provider[])).filter(
      (p): p is Provider => Boolean(p),
    );
    if (providers.length === 0) {
      throw new LLMHubError({
        kind: ErrorKind.Validation,
        message: "No providers configured",
      });
    }
    const results = await Promise.all(
      providers.map((provider) => this.forProvider(provider, params?.refresh)),
    );
    return results.flat().sort((a, b) =>
      a.displayName.localeCompare(b.displayName),
    );
  }

  private async forProvider(
    provider: Provider,
    refresh?: boolean,
  ): Promise<ModelMetadata[]> {
    const cached = this.cache.get(provider);
    const now = Date.now();
    if (!refresh && cached && cached.expiresAt > now) {
      return cached.data;
    }
    const adapter = this.adapters[provider];
    if (!adapter) {
      throw new LLMHubError({
        kind: ErrorKind.Validation,
        message: `Provider ${provider} is not configured`,
      });
    }
    try {
      const models = await adapter.listModels();
      this.cache.set(provider, {
        data: models,
        expiresAt: now + this.ttlMs,
      });
      return models;
    } catch (err) {
      throw toHubError(err);
    }
  }
}
