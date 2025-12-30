import { createHash } from "crypto";
import { AiKitError } from "../core/errors.js";
import {
  ErrorKind,
  GenerateInput,
  GenerateOutput,
  ImageGenerateInput,
  ImageGenerateOutput,
  ListModelsParams,
  MeshGenerateInput,
  MeshGenerateOutput,
  ModelMetadata,
  Provider,
  StreamChunk,
  TranscribeInput,
  TranscribeOutput,
} from "../core/types.js";
import type { ProviderAdapter } from "../core/provider.js";

export type FixtureKeyInput =
  | { type: "generate"; input: GenerateInput }
  | { type: "stream"; input: GenerateInput }
  | { type: "image"; input: ImageGenerateInput }
  | { type: "mesh"; input: MeshGenerateInput }
  | { type: "transcribe"; input: TranscribeInput };

export type FixtureEntry = {
  generate?: GenerateOutput;
  stream?: StreamChunk[];
  image?: ImageGenerateOutput;
  mesh?: MeshGenerateOutput;
  transcribe?: TranscribeOutput;
};

export interface FixtureAdapterOptions {
  provider: Provider;
  fixtures: Record<string, FixtureEntry>;
  models?: ModelMetadata[];
  resolveKey?: (input: FixtureKeyInput) => string;
  defaultChunkSize?: number;
}

export interface FixtureCalls {
  generate: GenerateInput[];
  streamGenerate: GenerateInput[];
  generateImage: ImageGenerateInput[];
  generateMesh: MeshGenerateInput[];
  transcribe: TranscribeInput[];
}

export class FixtureAdapter implements ProviderAdapter {
  readonly provider: Provider;
  private readonly fixtures: Record<string, FixtureEntry>;
  private readonly models: ModelMetadata[];
  private readonly resolveKey: (input: FixtureKeyInput) => string;
  private readonly defaultChunkSize: number;
  readonly calls: FixtureCalls = {
    generate: [],
    streamGenerate: [],
    generateImage: [],
    generateMesh: [],
    transcribe: [],
  };

  constructor(options: FixtureAdapterOptions) {
    this.provider = options.provider;
    this.fixtures = options.fixtures;
    this.models = options.models ?? [];
    this.resolveKey = options.resolveKey ?? createFixtureKey;
    this.defaultChunkSize = options.defaultChunkSize ?? 24;
  }

  async listModels(params?: ListModelsParams): Promise<ModelMetadata[]> {
    if (params?.providers && !params.providers.includes(this.provider)) {
      return [];
    }
    return this.models;
  }

  async generate(input: GenerateInput): Promise<GenerateOutput> {
    this.calls.generate.push(input);
    const entry = this.requireFixture({ type: "generate", input });
    if (!entry.generate) {
      throw this.missingFixtureError("generate", input);
    }
    return entry.generate;
  }

  async *streamGenerate(input: GenerateInput): AsyncIterable<StreamChunk> {
    this.calls.streamGenerate.push(input);
    const entry = this.requireFixture({ type: "stream", input });
    if (entry.stream) {
      for (const chunk of entry.stream) {
        yield chunk;
      }
      return;
    }
    if (!entry.generate) {
      throw this.missingFixtureError("stream", input);
    }
    const chunks = buildStreamChunks(entry.generate, this.defaultChunkSize);
    for (const chunk of chunks) {
      yield chunk;
    }
  }

  async generateImage(input: ImageGenerateInput): Promise<ImageGenerateOutput> {
    this.calls.generateImage.push(input);
    const entry = this.requireFixture({ type: "image", input });
    if (!entry.image) {
      throw this.missingFixtureError("image", input);
    }
    return entry.image;
  }

  async generateMesh(input: MeshGenerateInput): Promise<MeshGenerateOutput> {
    this.calls.generateMesh.push(input);
    const entry = this.requireFixture({ type: "mesh", input });
    if (!entry.mesh) {
      throw this.missingFixtureError("mesh", input);
    }
    return entry.mesh;
  }

  async transcribe(input: TranscribeInput): Promise<TranscribeOutput> {
    this.calls.transcribe.push(input);
    const entry = this.requireFixture({ type: "transcribe", input });
    if (!entry.transcribe) {
      throw this.missingFixtureError("transcribe", input);
    }
    return entry.transcribe;
  }

  private requireFixture(input: FixtureKeyInput): FixtureEntry {
    const key = this.resolveKey(input);
    const entry = this.fixtures[key];
    if (!entry) {
      throw new AiKitError({
        kind: ErrorKind.Validation,
        message: `Fixture not found (key: ${key}).`,
        provider: this.provider,
      });
    }
    return entry;
  }

  private missingFixtureError(
    type: FixtureKeyInput["type"],
    input: GenerateInput | ImageGenerateInput | MeshGenerateInput | TranscribeInput,
  ): AiKitError {
    const key = this.resolveKey({ type, input } as FixtureKeyInput);
    return new AiKitError({
      kind: ErrorKind.Validation,
      message: `Fixture for ${type} is missing (key: ${key}).`,
      provider: this.provider,
    });
  }
}

export function createFixtureKey(input: FixtureKeyInput): string {
  const payload = stableStringify({
    type: input.type,
    provider: input.input.provider,
    model: input.input.model,
    input: scrubInput(input.input),
  });
  const hash = createHash("sha256").update(payload).digest("hex").slice(0, 12);
  return `${input.type}:${input.input.provider}:${input.input.model}:${hash}`;
}

export function buildStreamChunks(
  output: GenerateOutput,
  chunkSize: number,
): StreamChunk[] {
  const chunks: StreamChunk[] = [];
  if (output.text) {
    for (const part of chunkText(output.text, chunkSize)) {
      chunks.push({ type: "delta", textDelta: part });
    }
  }
  if (output.toolCalls) {
    for (const call of output.toolCalls) {
      chunks.push({ type: "tool_call", call });
    }
  }
  chunks.push({
    type: "message_end",
    usage: output.usage,
    finishReason: output.finishReason,
  });
  return chunks;
}

function chunkText(text: string, chunkSize: number): string[] {
  if (!text) {
    return [];
  }
  const size = chunkSize > 0 ? chunkSize : text.length;
  const chunks: string[] = [];
  for (let index = 0; index < text.length; index += size) {
    chunks.push(text.slice(index, index + size));
  }
  return chunks;
}

function scrubInput<T>(input: T): T {
  if (Array.isArray(input)) {
    return input.map((item) => scrubInput(item)) as T;
  }
  if (input && typeof input === "object") {
    const result: Record<string, unknown> = {};
    for (const key of Object.keys(input).sort()) {
      if (key === "signal") {
        continue;
      }
      const value = (input as Record<string, unknown>)[key];
      if (typeof value === "undefined" || typeof value === "function") {
        continue;
      }
      result[key] = scrubInput(value as object);
    }
    return result as T;
  }
  return input;
}

function stableStringify(value: unknown): string {
  return JSON.stringify(scrubInput(value));
}
