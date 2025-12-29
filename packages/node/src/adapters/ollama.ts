import { ProviderAdapter } from "../core/provider.js";
import {
  GenerateInput,
  GenerateOutput,
  ModelMetadata,
  OllamaProviderConfig,
  Provider,
  StreamChunk,
  FetchLike,
} from "../core/types.js";
import { OpenAIAdapter } from "./openai.js";

export class OllamaAdapter implements ProviderAdapter {
  readonly provider = Provider.Ollama;
  private readonly openAICompat: OpenAIAdapter;

  constructor(config: OllamaProviderConfig, fetchImpl?: FetchLike) {
    const baseURL = config.baseURL ?? "http://localhost:11434";
    this.openAICompat = new OpenAIAdapter(
      {
        apiKey: config.apiKey ?? "",
        baseURL,
        defaultUseResponses: config.defaultUseResponses ?? false,
      },
      fetchImpl,
      {
        providerOverride: Provider.Ollama,
        baseURLOverride: baseURL,
      },
    );
  }

  async listModels(): Promise<ModelMetadata[]> {
    return this.openAICompat.listModels();
  }

  async generate(input: GenerateInput): Promise<GenerateOutput> {
    return this.openAICompat.generate({ ...input, provider: Provider.Ollama });
  }

  streamGenerate(input: GenerateInput): AsyncIterable<StreamChunk> {
    return this.openAICompat.streamGenerate({
      ...input,
      provider: Provider.Ollama,
    });
  }
}
