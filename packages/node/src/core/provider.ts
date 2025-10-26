import {
  GenerateInput,
  GenerateOutput,
  ListModelsParams,
  ModelMetadata,
  Provider,
  StreamChunk,
} from "./types.js";

export interface ProviderAdapter {
  readonly provider: Provider;
  listModels(params?: ListModelsParams): Promise<ModelMetadata[]>;
  generate(input: GenerateInput): Promise<GenerateOutput>;
  streamGenerate(input: GenerateInput): AsyncIterable<StreamChunk>;
}

export type AdapterMap = Partial<Record<Provider, ProviderAdapter>>;
