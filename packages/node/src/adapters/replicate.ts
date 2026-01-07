import Replicate from "replicate";
import { ProviderAdapter } from "../core/provider.js";
import {
  GenerateInput,
  GenerateOutput,
  ListModelsParams,
  ModelMetadata,
  Provider,
  ReplicateProviderConfig,
  StreamChunk,
  VideoGenerateInput,
  VideoGenerateOutput,
} from "../core/types.js";
import { AiKitError } from "../core/errors.js";
import { ErrorKind } from "../core/types.js";

const VIDEO_MODELS: ModelMetadata[] = [
  {
    id: "kwaivgi/kling-v2.5-turbo-pro",
    displayName: "Kling v2.5 Turbo Pro",
    provider: Provider.Replicate,
    family: "i2v",
    capabilities: {
      text: true,
      vision: true,
      image: false,
      video: true,
      tool_use: false,
      structured_output: false,
      reasoning: false,
    },
    videoPrices: {
      per_second_usd: 0.07,
    },
  },
  {
    id: "kwaivgi/kling-v2.6",
    displayName: "Kling v2.6",
    provider: Provider.Replicate,
    family: "i2v",
    capabilities: {
      text: true,
      vision: true,
      image: false,
      video: true,
      tool_use: false,
      structured_output: false,
      reasoning: false,
      audio_out: true,
    },
    videoPrices: {
      per_second_usd: 0.07,
      per_second_usd_with_audio: 0.14,
    },
  },
];

async function downloadAsBase64(url: string): Promise<string> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download video: ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  return Buffer.from(buffer).toString("base64");
}

export class ReplicateAdapter implements ProviderAdapter {
  readonly provider = Provider.Replicate;
  private readonly client: Replicate;

  constructor(config: ReplicateProviderConfig) {
    this.client = new Replicate({
      auth: config.apiKey,
    });
  }

  async listModels(_params?: ListModelsParams): Promise<ModelMetadata[]> {
    return VIDEO_MODELS;
  }

  async generate(_input: GenerateInput): Promise<GenerateOutput> {
    throw new AiKitError({
      kind: ErrorKind.Unsupported,
      message: "Replicate text generation is not supported",
      provider: this.provider,
    });
  }

  async *streamGenerate(_input: GenerateInput): AsyncIterable<StreamChunk> {
    throw new AiKitError({
      kind: ErrorKind.Unsupported,
      message: "Replicate text generation is not supported",
      provider: this.provider,
    });
  }

  async generateVideo(input: VideoGenerateInput): Promise<VideoGenerateOutput> {
    const modelInput: Record<string, unknown> = {
      prompt: input.prompt,
    };

    if (input.startImage) {
      modelInput.start_image = input.startImage;
    } else if (input.inputImages?.length) {
      const img = input.inputImages[0];
      if (img.url) {
        modelInput.start_image = img.url;
      } else if (img.base64) {
        const prefix = img.mediaType ? `data:${img.mediaType};base64,` : "data:image/png;base64,";
        modelInput.start_image = prefix + img.base64;
      }
    }

    if (input.duration !== undefined) {
      modelInput.duration = input.duration;
    }

    if (input.aspectRatio) {
      modelInput.aspect_ratio = input.aspectRatio;
    }

    if (input.negativePrompt) {
      modelInput.negative_prompt = input.negativePrompt;
    }

    if (input.generateAudio !== undefined) {
      modelInput.generate_audio = input.generateAudio;
    }

    if (input.parameters) {
      Object.assign(modelInput, input.parameters);
    }

    const output = await this.client.run(input.model as `${string}/${string}`, {
      input: modelInput,
    });

    let videoUrl: string;
    if (typeof output === "string") {
      videoUrl = output;
    } else if (Array.isArray(output) && output.length > 0) {
      videoUrl = String(output[0]);
    } else if (output && typeof output === "object" && "url" in output) {
      videoUrl = String((output as { url: string }).url);
    } else {
      throw new AiKitError({
        kind: ErrorKind.Unknown,
        message: "Unexpected output format from Replicate",
        provider: this.provider,
      });
    }

    const data = await downloadAsBase64(videoUrl);

    return {
      mime: "video/mp4",
      data,
      raw: output,
    };
  }
}
