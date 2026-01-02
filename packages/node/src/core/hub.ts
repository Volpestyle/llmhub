import { ModelRegistry } from "../registry/modelRegistry.js";
import { AnthropicAdapter } from "../adapters/anthropic.js";
import { OpenAIAdapter } from "../adapters/openai.js";
import { XAIAdapter } from "../adapters/xai.js";
import { GoogleAdapter } from "../adapters/gemini.js";
import { OllamaAdapter } from "../adapters/ollama.js";
import { AdapterMap, ProviderAdapter } from "./provider.js";
import { fingerprintApiKey } from "./entitlements.js";
import {
  EntitlementContext,
  GenerateInput,
  GenerateOutput,
  Kit,
  KitConfig,
  ImageGenerateInput,
  ImageGenerateOutput,
  ListModelsParams,
  MeshGenerateInput,
  MeshGenerateOutput,
  ModelRecord,
  OpenAIProviderConfig,
  AnthropicProviderConfig,
  XAIProviderConfig,
  GoogleProviderConfig,
  OllamaProviderConfig,
  Provider,
  StreamChunk,
  TranscribeInput,
  TranscribeOutput,
} from "./types.js";
import { ErrorKind } from "./types.js";
import { AiKitError, toKitError } from "./errors.js";
import { estimateCost } from "./pricing.js";

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

class DefaultKit implements Kit {
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
      const output = await adapter.generate({ ...input, stream: false });
      return attachCost(input, output);
    } catch (err) {
      const kitErr = toKitError(err);
      this.registry.learnModelUnavailable(undefined, input.provider, input.model, kitErr);
      throw kitErr;
    }
  }

  async generateWithContext(
    entitlement: EntitlementContext | undefined,
    input: GenerateInput,
  ): Promise<GenerateOutput> {
    const adapter = this.requireAdapter(input.provider, entitlement);
    try {
      const output = await adapter.generate({ ...input, stream: false });
      return attachCost(input, output);
    } catch (err) {
      const kitErr = toKitError(err);
      this.registry.learnModelUnavailable(entitlement, input.provider, input.model, kitErr);
      throw kitErr;
    }
  }

  async generateImage(
    input: ImageGenerateInput,
  ): Promise<ImageGenerateOutput> {
    const entitlement = this.entitlementForProvider(input.provider);
    if (entitlement) {
      return this.generateImageWithContext(entitlement, input);
    }
    const adapter = this.requireAdapter(input.provider);
    if (!adapter.generateImage) {
      throw new AiKitError({
        kind: ErrorKind.Unsupported,
        message: `Provider ${input.provider} does not support image generation`,
        provider: input.provider,
      });
    }
    try {
      return await adapter.generateImage(input);
    } catch (err) {
      const kitErr = toKitError(err);
      this.registry.learnModelUnavailable(undefined, input.provider, input.model, kitErr);
      throw kitErr;
    }
  }

  async generateMesh(
    input: MeshGenerateInput,
  ): Promise<MeshGenerateOutput> {
    const entitlement = this.entitlementForProvider(input.provider);
    if (entitlement) {
      return this.generateMeshWithContext(entitlement, input);
    }
    const adapter = this.requireAdapter(input.provider);
    if (!adapter.generateMesh) {
      throw new AiKitError({
        kind: ErrorKind.Unsupported,
        message: `Provider ${input.provider} does not support mesh generation`,
        provider: input.provider,
      });
    }
    try {
      return await adapter.generateMesh(input);
    } catch (err) {
      const kitErr = toKitError(err);
      this.registry.learnModelUnavailable(undefined, input.provider, input.model, kitErr);
      throw kitErr;
    }
  }

  async transcribe(input: TranscribeInput): Promise<TranscribeOutput> {
    const entitlement = this.entitlementForProvider(input.provider);
    if (entitlement) {
      return this.transcribeWithContext(entitlement, input);
    }
    const adapter = this.requireAdapter(input.provider);
    if (!adapter.transcribe) {
      throw new AiKitError({
        kind: ErrorKind.Unsupported,
        message: `Provider ${input.provider} does not support transcription`,
        provider: input.provider,
      });
    }
    try {
      return await adapter.transcribe(input);
    } catch (err) {
      const kitErr = toKitError(err);
      this.registry.learnModelUnavailable(undefined, input.provider, input.model, kitErr);
      throw kitErr;
    }
  }

  streamGenerate(input: GenerateInput): AsyncIterable<StreamChunk> {
    const entitlement = this.entitlementForProvider(input.provider);
    if (entitlement) {
      return this.streamGenerateWithContext(entitlement, input);
    }
    const adapter = this.requireAdapter(input.provider);
    const merged: GenerateInput = { ...input, stream: true };
    return attachCostToStream(adapter.streamGenerate(merged), input.provider, input.model);
  }

  streamGenerateWithContext(
    entitlement: EntitlementContext | undefined,
    input: GenerateInput,
  ): AsyncIterable<StreamChunk> {
    const adapter = this.requireAdapter(input.provider, entitlement);
    const merged: GenerateInput = { ...input, stream: true };
    return attachCostToStream(adapter.streamGenerate(merged), input.provider, input.model);
  }

  private async generateImageWithContext(
    entitlement: EntitlementContext | undefined,
    input: ImageGenerateInput,
  ): Promise<ImageGenerateOutput> {
    const adapter = this.requireAdapter(input.provider, entitlement);
    if (!adapter.generateImage) {
      throw new AiKitError({
        kind: ErrorKind.Unsupported,
        message: `Provider ${input.provider} does not support image generation`,
        provider: input.provider,
      });
    }
    try {
      return await adapter.generateImage(input);
    } catch (err) {
      const kitErr = toKitError(err);
      this.registry.learnModelUnavailable(entitlement, input.provider, input.model, kitErr);
      throw kitErr;
    }
  }

  private async generateMeshWithContext(
    entitlement: EntitlementContext | undefined,
    input: MeshGenerateInput,
  ): Promise<MeshGenerateOutput> {
    const adapter = this.requireAdapter(input.provider, entitlement);
    if (!adapter.generateMesh) {
      throw new AiKitError({
        kind: ErrorKind.Unsupported,
        message: `Provider ${input.provider} does not support mesh generation`,
        provider: input.provider,
      });
    }
    try {
      return await adapter.generateMesh(input);
    } catch (err) {
      const kitErr = toKitError(err);
      this.registry.learnModelUnavailable(entitlement, input.provider, input.model, kitErr);
      throw kitErr;
    }
  }

  private async transcribeWithContext(
    entitlement: EntitlementContext | undefined,
    input: TranscribeInput,
  ): Promise<TranscribeOutput> {
    const adapter = this.requireAdapter(input.provider, entitlement);
    if (!adapter.transcribe) {
      throw new AiKitError({
        kind: ErrorKind.Unsupported,
        message: `Provider ${input.provider} does not support transcription`,
        provider: input.provider,
      });
    }
    try {
      return await adapter.transcribe(input);
    } catch (err) {
      const kitErr = toKitError(err);
      this.registry.learnModelUnavailable(entitlement, input.provider, input.model, kitErr);
      throw kitErr;
    }
  }

  private requireAdapter(
    provider: Provider,
    entitlement?: EntitlementContext,
  ): ProviderAdapter {
    const adapter =
      this.adapterFactory?.(provider, entitlement) ?? this.adapters[provider];
    if (!adapter) {
      throw new AiKitError({
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

function attachCost(input: GenerateInput, output: GenerateOutput): GenerateOutput {
  const cost = estimateCost(input.provider, input.model, output.usage);
  if (!cost) {
    return output;
  }
  return { ...output, cost };
}

async function* attachCostToStream(
  stream: AsyncIterable<StreamChunk>,
  provider: Provider,
  model: string,
): AsyncIterable<StreamChunk> {
  for await (const chunk of stream) {
    if (chunk.type === "message_end") {
      const cost = estimateCost(provider, model, chunk.usage);
      if (cost) {
        yield { ...chunk, cost };
        continue;
      }
    }
    yield chunk;
  }
}

export function createKit(config: KitConfig): Kit {
  const providerCount = Object.keys(config.providers ?? {}).length;
  const adapterCount = Object.keys(config.adapters ?? {}).length;
  if (providerCount === 0 && adapterCount === 0 && !config.adapterFactory) {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "At least one provider configuration or adapter is required",
    });
  }
  const adapters: AdapterMap = { ...(config.adapters ?? {}) };
  const keyPools = new Map<Provider, KeyPool>();
  if (config.providers[Provider.OpenAI] && !adapters[Provider.OpenAI]) {
    const providerConfig = config.providers[Provider.OpenAI]! as OpenAIProviderConfig;
    const keys = normalizeKeys(providerConfig.apiKey, providerConfig.apiKeys);
    if (!keys.length) {
      throw new AiKitError({
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
  if (config.providers[Provider.Anthropic] && !adapters[Provider.Anthropic]) {
    const providerConfig = config.providers[Provider.Anthropic]! as AnthropicProviderConfig;
    const keys = normalizeKeys(providerConfig.apiKey, providerConfig.apiKeys);
    if (!keys.length) {
      throw new AiKitError({
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
  if (config.providers[Provider.XAI] && !adapters[Provider.XAI]) {
    const providerConfig = config.providers[Provider.XAI]! as XAIProviderConfig;
    const keys = normalizeKeys(providerConfig.apiKey, providerConfig.apiKeys);
    if (!keys.length) {
      throw new AiKitError({
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
  if (config.providers[Provider.Google] && !adapters[Provider.Google]) {
    const providerConfig = config.providers[Provider.Google]! as GoogleProviderConfig;
    const keys = normalizeKeys(providerConfig.apiKey, providerConfig.apiKeys);
    if (!keys.length) {
      throw new AiKitError({
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
  if (config.providers[Provider.Ollama] && !adapters[Provider.Ollama]) {
    const providerConfig = config.providers[Provider.Ollama]! as OllamaProviderConfig;
    adapters[Provider.Ollama] = new OllamaAdapter(
      providerConfig,
      config.httpClient,
    );
  }

  const baseAdapterFactory = (
    provider: Provider,
    entitlement?: EntitlementContext,
  ): ProviderAdapter => {
    if (!entitlement?.apiKey) {
      const adapter = adapters[provider];
      if (!adapter) {
        throw new AiKitError({
          kind: ErrorKind.Validation,
          message: `Provider ${provider} is not configured`,
          provider,
        });
      }
      return adapter;
    }
    const baseConfig = config.providers[provider];
    if (!baseConfig) {
      throw new AiKitError({
        kind: ErrorKind.Validation,
        message: `Provider ${provider} is not configured`,
        provider,
      });
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
      case Provider.Ollama:
        return new OllamaAdapter(
          { ...baseConfig, apiKey: entitlement.apiKey },
          config.httpClient,
        );
      default:
        throw new AiKitError({
          kind: ErrorKind.Validation,
          message: `Provider ${provider} is not configured`,
        });
    }
  };

  const adapterFactory = (
    provider: Provider,
    entitlement?: EntitlementContext,
  ): ProviderAdapter | undefined => {
    const adapter = config.adapterFactory?.(provider, entitlement);
    if (adapter) {
      return adapter;
    }
    return baseAdapterFactory(provider, entitlement);
  };

  const registry = new ModelRegistry(
    adapters,
    {
      ttlMs: config.registry?.ttlMs,
    },
    adapterFactory,
  );
  return new DefaultKit(adapters, registry, adapterFactory, keyPools);
}
