import { AdapterMap, ProviderAdapter } from "../core/provider.js";
import { AiKitError, toKitError } from "../core/errors.js";
import { fingerprintApiKey } from "../core/entitlements.js";
import {
  EntitlementContext,
  ErrorKind,
  ListModelsParams,
  ModelAvailability,
  ModelFeatures,
  ModelLimits,
  ModelMetadata,
  ModelModalities,
  ModelPricing,
  ModelRecord,
  Provider,
} from "../core/types.js";
import { applyCuratedMetadata } from "../core/pricing.js";

interface CacheEntry {
  expiresAt: number;
  fetchedAt: number;
  data: ModelMetadata[];
}

export interface ModelRegistryOptions {
  ttlMs: number;
  learnedTtlMs: number;
}

export type AdapterFactory = (
  provider: Provider,
  entitlement?: EntitlementContext,
) => ProviderAdapter | undefined;

export class ModelRegistry {
  private readonly cache = new Map<string, CacheEntry>();
  private readonly learned = new Map<string, { expiresAt: number; reason: string }>();
  private readonly adapters: AdapterMap;
  private readonly ttlMs: number;
  private readonly learnedTtlMs: number;
  private readonly adapterFactory?: AdapterFactory;

  constructor(
    adapters: AdapterMap,
    options?: Partial<ModelRegistryOptions>,
    adapterFactory?: AdapterFactory,
  ) {
    this.adapters = adapters;
    this.ttlMs = options?.ttlMs ?? 30 * 60 * 1000; // default 30m
    this.learnedTtlMs = options?.learnedTtlMs ?? 20 * 60 * 1000;
    this.adapterFactory = adapterFactory;
  }

  async list(params?: ListModelsParams): Promise<ModelMetadata[]> {
    const providers = this.resolveProviders(params).filter(
      (p): p is Provider => Boolean(p),
    );
    if (providers.length === 0) {
      throw new AiKitError({
        kind: ErrorKind.Validation,
        message: "No providers configured",
      });
    }
    const results = await Promise.all(
      providers.map((provider) => this.forProvider(provider, params)),
    );
    return results.flatMap((entry) => entry.data).sort((a, b) =>
      a.displayName.localeCompare(b.displayName),
    );
  }

  async listModelRecords(params?: ListModelsParams): Promise<ModelRecord[]> {
    const providers = this.resolveProviders(params).filter(
      (p): p is Provider => Boolean(p),
    );
    if (providers.length === 0) {
      throw new AiKitError({
        kind: ErrorKind.Validation,
        message: "No providers configured",
      });
    }
    const results = await Promise.all(
      providers.map(async (provider) => ({
        provider,
        entry: await this.forProvider(provider, params),
      })),
    );
    const entitlement = params?.entitlement;
    const records = results.flatMap(({ provider, entry }) =>
      entry.data.map((model) =>
        this.toModelRecord(model, provider, entry.fetchedAt, entitlement),
      ),
    );
    return records.sort((a, b) =>
      (a.displayName ?? "").localeCompare(b.displayName ?? ""),
    );
  }

  learnModelUnavailable(
    entitlement: EntitlementContext | undefined,
    provider: Provider,
    modelId: string,
    err: unknown,
  ) {
    const reason = learnReason(err);
    if (!reason) {
      return;
    }
    const key = this.learnedKey(provider, entitlement, modelId);
    this.learned.set(key, {
      expiresAt: Date.now() + this.learnedTtlMs,
      reason,
    });
  }

  private resolveProviders(params?: ListModelsParams): Provider[] {
    if (params?.providers?.length) {
      return params.providers;
    }
    if (params?.entitlement?.provider) {
      return [params.entitlement.provider];
    }
    return Object.keys(this.adapters) as Provider[];
  }

  private async forProvider(
    provider: Provider,
    params?: ListModelsParams,
  ): Promise<CacheEntry> {
    const refresh = Boolean(params?.refresh);
    const entitlement = params?.entitlement;
    const key = this.cacheKey(provider, entitlement);
    if (!refresh) {
      const cached = this.cache.get(key);
      if (cached && cached.expiresAt > Date.now()) {
        return cached;
      }
    }
    try {
      return await this.fetchFromAdapter(provider, entitlement, key);
    } catch (err) {
      const cached = this.cache.get(key);
      if (cached && cached.expiresAt > Date.now()) {
        return cached;
      }
      throw toKitError(err);
    }
  }

  private async fetchFromAdapter(
    provider: Provider,
    entitlement: EntitlementContext | undefined,
    key: string,
  ): Promise<CacheEntry> {
    const adapter = this.resolveAdapter(provider, entitlement);
    if (!adapter) {
      throw new AiKitError({
        kind: ErrorKind.Validation,
        message: `Provider ${provider} is not configured`,
      });
    }
    const models = await adapter.listModels();
    const curatedModels = models.map((model) => applyCuratedMetadata(model));
    const entry = {
      data: curatedModels,
      fetchedAt: Date.now(),
      expiresAt: Date.now() + this.ttlMs,
    };
    this.cache.set(key, entry);
    return entry;
  }

  private resolveAdapter(
    provider: Provider,
    entitlement?: EntitlementContext,
  ): ProviderAdapter | undefined {
    if (this.adapterFactory) {
      return this.adapterFactory(provider, entitlement);
    }
    return this.adapters[provider];
  }

  private cacheKey(provider: Provider, entitlement?: EntitlementContext): string {
    const fingerprint =
      entitlement?.apiKeyFingerprint?.trim() ||
      fingerprintApiKey(entitlement?.apiKey) ||
      "default";
    return [
      provider,
      fingerprint,
      entitlement?.accountId?.trim() ?? "",
      entitlement?.region?.trim() ?? "",
      entitlement?.environment?.trim() ?? "",
      entitlement?.tenantId?.trim() ?? "",
      entitlement?.userId?.trim() ?? "",
    ].join("|");
  }

  private learnedKey(
    provider: Provider,
    entitlement: EntitlementContext | undefined,
    modelId: string,
  ): string {
    return `${this.cacheKey(provider, entitlement)}|${modelId}`;
  }

  private learnedStatus(
    provider: Provider,
    entitlement: EntitlementContext | undefined,
    modelId: string,
  ): { reason: string } | undefined {
    const key = this.learnedKey(provider, entitlement, modelId);
    const entry = this.learned.get(key);
    if (!entry) {
      return undefined;
    }
    if (entry.expiresAt < Date.now()) {
      this.learned.delete(key);
      return undefined;
    }
    return { reason: entry.reason };
  }

  private toModelRecord(
    model: ModelMetadata,
    provider: Provider,
    fetchedAt: number,
    entitlement: EntitlementContext | undefined,
  ): ModelRecord {
    const modalities: ModelModalities = {
      text: model.capabilities.text,
      vision: model.capabilities.vision,
      audioIn: model.capabilities.audio_in,
      audioOut: model.capabilities.audio_out,
      imageOut: model.capabilities.image,
    };
    const features: ModelFeatures = {
      tools: model.capabilities.tool_use,
      jsonMode: model.capabilities.structured_output,
      jsonSchema: model.capabilities.structured_output,
      streaming: true,
    };
    const limits: ModelLimits | undefined =
      model.contextWindow && model.contextWindow > 0
        ? { contextTokens: model.contextWindow }
        : undefined;
    const pricing: ModelPricing | undefined = model.tokenPrices
      ? {
          currency: "USD",
          inputPer1M: model.tokenPrices.input,
          outputPer1M: model.tokenPrices.output,
          source: "config",
        }
      : undefined;
    const tags: string[] = [];
    if (model.inPreview) {
      tags.push("preview");
    }
    if (model.deprecated) {
      tags.push("deprecated");
    }
    const availability: ModelAvailability = {
      entitled: true,
      confidence: "listed",
      lastVerifiedAt: new Date(fetchedAt).toISOString(),
    };
    const learned = this.learnedStatus(provider, entitlement, model.id);
    if (learned) {
      availability.entitled = false;
      availability.confidence = "learned";
      availability.reason = learned.reason;
    }
    return {
      id: `${provider}:${model.id}`,
      provider,
      providerModelId: model.id,
      displayName: model.displayName,
      modalities,
      features,
      limits,
      tags,
      pricing,
      availability,
    };
  }
}

function learnReason(err: unknown): string | undefined {
  if (!err) {
    return undefined;
  }
  if (err instanceof AiKitError) {
    if (err.kind === ErrorKind.ProviderNotFound || err.kind === ErrorKind.Validation) {
      return err.message;
    }
    if (err.upstreamStatus && [400, 403, 404].includes(err.upstreamStatus)) {
      return err.message;
    }
  }
  return undefined;
}
