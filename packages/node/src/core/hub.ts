import { ModelRegistry } from "../registry/modelRegistry.js";
import { AnthropicAdapter } from "../adapters/anthropic.js";
import { OpenAIAdapter } from "../adapters/openai.js";
import { XAIAdapter } from "../adapters/xai.js";
import { GoogleAdapter } from "../adapters/gemini.js";
import { AdapterMap, ProviderAdapter } from "./provider.js";
import {
  GenerateInput,
  GenerateOutput,
  Hub,
  HubConfig,
  ListModelsParams,
  Provider,
  StreamChunk,
} from "./types.js";
import { ErrorKind } from "./types.js";
import { LLMHubError, toHubError } from "./errors.js";

class DefaultHub implements Hub {
  constructor(
    private readonly adapters: AdapterMap,
    private readonly registry: ModelRegistry,
  ) {}

  async listModels(params?: ListModelsParams) {
    return this.registry.list(params);
  }

  async generate(input: GenerateInput): Promise<GenerateOutput> {
    const adapter = this.requireAdapter(input.provider);
    try {
      return await adapter.generate({ ...input, stream: false });
    } catch (err) {
      throw toHubError(err);
    }
  }

  streamGenerate(input: GenerateInput): AsyncIterable<StreamChunk> {
    const adapter = this.requireAdapter(input.provider);
    const merged: GenerateInput = { ...input, stream: true };
    return adapter.streamGenerate(merged);
  }

  private requireAdapter(provider: Provider): ProviderAdapter {
    const adapter = this.adapters[provider];
    if (!adapter) {
      throw new LLMHubError({
        kind: ErrorKind.Validation,
        message: `Provider ${provider} is not configured`,
        provider,
      });
    }
    return adapter;
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
  if (config.providers[Provider.OpenAI]) {
    adapters[Provider.OpenAI] = new OpenAIAdapter(
      config.providers[Provider.OpenAI]!,
      config.httpClient,
    );
  }
  if (config.providers[Provider.Anthropic]) {
    adapters[Provider.Anthropic] = new AnthropicAdapter(
      config.providers[Provider.Anthropic]!,
      config.httpClient,
    );
  }
  if (config.providers[Provider.XAI]) {
    adapters[Provider.XAI] = new XAIAdapter(
      config.providers[Provider.XAI]!,
      config.httpClient,
    );
  }
  if (config.providers[Provider.Google]) {
    adapters[Provider.Google] = new GoogleAdapter(
      config.providers[Provider.Google]!,
      config.httpClient,
    );
  }

  const registry = new ModelRegistry(adapters, {
    ttlMs: config.registry?.ttlMs,
  });
  return new DefaultHub(adapters, registry);
}
