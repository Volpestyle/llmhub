import { ModelRegistry } from "../registry/modelRegistry.js";
import { AnthropicAdapter } from "../adapters/anthropic.js";
import { OpenAIAdapter } from "../adapters/openai.js";
import { XAIAdapter } from "../adapters/xai.js";
import { GoogleAdapter } from "../adapters/gemini.js";
import { AdapterMap, ProviderAdapter } from "./provider.js";
import { fingerprintApiKey } from "./entitlements.js";
import {
  EntitlementContext,
  GenerateInput,
  GenerateOutput,
  Hub,
  HubConfig,
  ListModelsParams,
  ModelRecord,
  Provider,
  StreamChunk,
} from "./types.js";
import { ErrorKind } from "./types.js";
import { LLMHubError, toHubError } from "./errors.js";

class KeyPool {
  private index = 0;

  constructor(private readonly keys: string[]) {}

  next(): string {
    if (!this.keys.length) {
      return "";
    }
    const key = this.keys[this.index % this.keys.length];
    this.index = (this.index + 1) % this.keys.length;
    return key;
  }
}

function normalizeKeys(primary?: string, extras?: string[]): string[] {
  const keys: string[] = [];
  const seen = new Set<string>();
  const addKey = (value?: string) => {
    const trimmed = value?.trim();
    if (!trimmed || seen.has(trimmed)) {
      return;
    }
    seen.add(trimmed);
    keys.push(trimmed);
  };
  addKey(primary);
  extras?.forEach(addKey);
  return keys;
}

class DefaultHub implements Hub {
  constructor(
    private readonly adapters: AdapterMap,
    private readonly registry: ModelRegistry,
    private readonly adapterFactory: (
      provider: Provider,
      entitlement?: EntitlementContext,
    ) => ProviderAdapter | undefined,
    private readonly keyPools: Map<Provider, KeyPool>,
  ) {}

  async listModels(params?: ListModelsParams) {
    return this.registry.list(params);
  }

  async listModelRecords(params?: ListModelsParams): Promise<ModelRecord[]> {
    return this.registry.listModelRecords(params);
  }

  async generate(input: GenerateInput): Promise<GenerateOutput> {
    const entitlement = this.entitlementForProvider(input.provider);
    if (entitlement) {
      return this.generateWithContext(entitlement, input);
    }
    const adapter = this.requireAdapter(input.provider);
    try {
      return await adapter.generate({ ...input, stream: false });
    } catch (err) {
      const hubErr = toHubError(err);
      this.registry.learnModelUnavailable(undefined, input.provider, input.model, hubErr);
      throw hubErr;
    }
  }

  async generateWithContext(
    entitlement: EntitlementContext | undefined,
    input: GenerateInput,
  ): Promise<GenerateOutput> {
    const adapter = this.requireAdapter(input.provider, entitlement);
    try {
      return await adapter.generate({ ...input, stream: false });
    } catch (err) {
      const hubErr = toHubError(err);
      this.registry.learnModelUnavailable(entitlement, input.provider, input.model, hubErr);
      throw hubErr;
    }
  }

  streamGenerate(input: GenerateInput): AsyncIterable<StreamChunk> {
    const entitlement = this.entitlementForProvider(input.provider);
    if (entitlement) {
      return this.streamGenerateWithContext(entitlement, input);
    }
    const adapter = this.requireAdapter(input.provider);
    const merged: GenerateInput = { ...input, stream: true };
    return adapter.streamGenerate(merged);
  }

  streamGenerateWithContext(
    entitlement: EntitlementContext | undefined,
    input: GenerateInput,
  ): AsyncIterable<StreamChunk> {
    const adapter = this.requireAdapter(input.provider, entitlement);
    const merged: GenerateInput = { ...input, stream: true };
    return adapter.streamGenerate(merged);
  }

  private requireAdapter(
    provider: Provider,
    entitlement?: EntitlementContext,
  ): ProviderAdapter {
    const adapter =
      this.adapterFactory?.(provider, entitlement) ?? this.adapters[provider];
    if (!adapter) {
      throw new LLMHubError({
        kind: ErrorKind.Validation,
        message: `Provider ${provider} is not configured`,
        provider,
      });
    }
    return adapter;
  }

  private entitlementForProvider(provider: Provider): EntitlementContext | undefined {
    const pool = this.keyPools.get(provider);
    if (!pool) {
      return undefined;
    }
    const apiKey = pool.next();
    if (!apiKey) {
      return undefined;
    }
    return {
      provider,
      apiKey,
      apiKeyFingerprint: fingerprintApiKey(apiKey),
    };
  }
}

export function createHub(config: HubConfig): Hub {
  if (!config.providers || Object.keys(config.providers).length === 0) {
    throw new LLMHubError({
      kind: ErrorKind.Validation,
      message: "At least one provider configuration is required",
    });
  }
  const adapters: AdapterMap = {};
  const keyPools = new Map<Provider, KeyPool>();
  if (config.providers[Provider.OpenAI]) {
    const providerConfig = config.providers[Provider.OpenAI]!;
    const keys = normalizeKeys(providerConfig.apiKey, providerConfig.apiKeys);
    if (!keys.length) {
      throw new LLMHubError({
        kind: ErrorKind.Validation,
        message: "OpenAI api key is required",
      });
    }
    adapters[Provider.OpenAI] = new OpenAIAdapter(
      { ...providerConfig, apiKey: keys[0] },
      config.httpClient,
    );
    keyPools.set(Provider.OpenAI, new KeyPool(keys));
  }
  if (config.providers[Provider.Anthropic]) {
    const providerConfig = config.providers[Provider.Anthropic]!;
    const keys = normalizeKeys(providerConfig.apiKey, providerConfig.apiKeys);
    if (!keys.length) {
      throw new LLMHubError({
        kind: ErrorKind.Validation,
        message: "Anthropic api key is required",
      });
    }
    adapters[Provider.Anthropic] = new AnthropicAdapter(
      { ...providerConfig, apiKey: keys[0] },
      config.httpClient,
    );
    keyPools.set(Provider.Anthropic, new KeyPool(keys));
  }
  if (config.providers[Provider.XAI]) {
    const providerConfig = config.providers[Provider.XAI]!;
    const keys = normalizeKeys(providerConfig.apiKey, providerConfig.apiKeys);
    if (!keys.length) {
      throw new LLMHubError({
        kind: ErrorKind.Validation,
        message: "xAI api key is required",
      });
    }
    adapters[Provider.XAI] = new XAIAdapter(
      { ...providerConfig, apiKey: keys[0] },
      config.httpClient,
    );
    keyPools.set(Provider.XAI, new KeyPool(keys));
  }
  if (config.providers[Provider.Google]) {
    const providerConfig = config.providers[Provider.Google]!;
    const keys = normalizeKeys(providerConfig.apiKey, providerConfig.apiKeys);
    if (!keys.length) {
      throw new LLMHubError({
        kind: ErrorKind.Validation,
        message: "Google api key is required",
      });
    }
    adapters[Provider.Google] = new GoogleAdapter(
      { ...providerConfig, apiKey: keys[0] },
      config.httpClient,
    );
    keyPools.set(Provider.Google, new KeyPool(keys));
  }

  const adapterFactory = (
    provider: Provider,
    entitlement?: EntitlementContext,
  ): ProviderAdapter | undefined => {
    if (!entitlement?.apiKey) {
      return adapters[provider];
    }
    const baseConfig = config.providers[provider];
    if (!baseConfig) {
      return undefined;
    }
    switch (provider) {
      case Provider.OpenAI:
        return new OpenAIAdapter(
          { ...baseConfig, apiKey: entitlement.apiKey },
          config.httpClient,
        );
      case Provider.Anthropic:
        return new AnthropicAdapter(
          { ...baseConfig, apiKey: entitlement.apiKey },
          config.httpClient,
        );
      case Provider.XAI:
        return new XAIAdapter(
          { ...baseConfig, apiKey: entitlement.apiKey },
          config.httpClient,
        );
      case Provider.Google:
        return new GoogleAdapter(
          { ...baseConfig, apiKey: entitlement.apiKey },
          config.httpClient,
        );
      default:
        return undefined;
    }
  };

  const registry = new ModelRegistry(
    adapters,
    {
      ttlMs: config.registry?.ttlMs,
    },
    adapterFactory,
  );
  return new DefaultHub(adapters, registry, adapterFactory, keyPools);
}
