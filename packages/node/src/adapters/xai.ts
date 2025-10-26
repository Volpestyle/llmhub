import { ProviderAdapter } from "../core/provider.js";
import {
  GenerateInput,
  GenerateOutput,
  ModelMetadata,
  Provider,
  StreamChunk,
  XAIProviderConfig,
  FetchLike,
} from "../core/types.js";
import { OpenAIAdapter } from "./openai.js";
import { AnthropicAdapter } from "./anthropic.js";

const COMPATIBILITY_KEY = "xai:compatibility";

export class XAIAdapter implements ProviderAdapter {
  readonly provider = Provider.XAI;
  private readonly openAICompat: OpenAIAdapter;
  private readonly anthropicCompat: AnthropicAdapter;
  private readonly config: XAIProviderConfig;

  constructor(config: XAIProviderConfig, fetchImpl?: FetchLike) {
    this.config = config;
    const baseURL = config.baseURL ?? "https://api.x.ai";
    this.openAICompat = new OpenAIAdapter(
      {
        apiKey: config.apiKey,
        baseURL,
      },
      fetchImpl,
      {
        providerOverride: Provider.XAI,
        baseURLOverride: baseURL,
      },
    );
    this.anthropicCompat = new AnthropicAdapter(
      {
        apiKey: config.apiKey,
        baseURL,
        version: "2023-06-01",
      },
      fetchImpl,
      {
        providerOverride: Provider.XAI,
        baseURLOverride: baseURL,
      },
    );
  }

  async listModels(): Promise<ModelMetadata[]> {
    return this.openAICompat.listModels();
  }

  async generate(input: GenerateInput): Promise<GenerateOutput> {
    const adapter = this.selectAdapter(input);
    return adapter.generate({ ...input, provider: Provider.XAI });
  }

  streamGenerate(input: GenerateInput): AsyncIterable<StreamChunk> {
    const adapter = this.selectAdapter(input);
    return adapter.streamGenerate({ ...input, provider: Provider.XAI });
  }

  private selectAdapter(input: GenerateInput) {
    const metadataMode = input.metadata?.[COMPATIBILITY_KEY];
    const mode =
      (metadataMode as "openai" | "anthropic" | undefined) ??
      this.config.compatibilityMode ??
      "openai";
    return mode === "anthropic" ? this.anthropicCompat : this.openAICompat;
  }
}
