import {
  GenerateInput,
  GenerateOutput,
  ImageGenerateInput,
  ImageGenerateOutput,
  ListModelsParams,
  MeshGenerateInput,
  MeshGenerateOutput,
  ModelMetadata,
  Provider,
  SpeechGenerateInput,
  SpeechGenerateOutput,
  TranscribeInput,
  TranscribeOutput,
  VideoGenerateInput,
  VideoGenerateOutput,
  StreamChunk,
} from "./types.js";

export interface ProviderAdapter {
  readonly provider: Provider;
  listModels(params?: ListModelsParams): Promise<ModelMetadata[]>;
  generate(input: GenerateInput): Promise<GenerateOutput>;
  streamGenerate(input: GenerateInput): AsyncIterable<StreamChunk>;
  generateImage?(input: ImageGenerateInput): Promise<ImageGenerateOutput>;
  generateMesh?(input: MeshGenerateInput): Promise<MeshGenerateOutput>;
  generateSpeech?(input: SpeechGenerateInput): Promise<SpeechGenerateOutput>;
  generateVideo?(input: VideoGenerateInput): Promise<VideoGenerateOutput>;
  transcribe?(input: TranscribeInput): Promise<TranscribeOutput>;
}

export type AdapterMap = Partial<Record<Provider, ProviderAdapter>>;
