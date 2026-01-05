import { ProviderAdapter } from "../core/provider.js";
import {
  GenerateInput,
  GenerateOutput,
  ModelMetadata,
  Provider,
  SpeechGenerateInput,
  SpeechGenerateOutput,
  StreamChunk,
  XAIProviderConfig,
  FetchLike,
} from "../core/types.js";
import { AiKitError } from "../core/errors.js";
import { ErrorKind } from "../core/types.js";
import { OpenAIAdapter } from "./openai.js";
import { AnthropicAdapter } from "./anthropic.js";
import WebSocket from "ws";

const COMPATIBILITY_KEY = "xai:compatibility";
const SPEECH_MODE_KEY = "xai:speech-mode";
const DEFAULT_VOICE = "ara";
const DEFAULT_SAMPLE_RATE = 24000;
const REALTIME_PROTOCOLS = ["realtime", "openai-beta.realtime-v1"];

/** xAI Voice Catalog - https://docs.x.ai/docs/guides/voice */
export const XAI_VOICES = {
  ara: { type: "female", tone: "warm, friendly", description: "Default voice, balanced and conversational" },
  rex: { type: "male", tone: "confident, clear", description: "Professional, ideal for business" },
  sal: { type: "neutral", tone: "smooth, balanced", description: "Versatile for various contexts" },
  eve: { type: "female", tone: "energetic, upbeat", description: "Engaging, great for interactive experiences" },
  leo: { type: "male", tone: "authoritative, strong", description: "Decisive, suitable for instructional content" },
} as const;

export type XAIVoice = keyof typeof XAI_VOICES;

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
    const adapter = this.selectAdapter(input.metadata);
    return adapter.generate({ ...input, provider: Provider.XAI });
  }

  async generateSpeech(
    input: SpeechGenerateInput,
  ): Promise<SpeechGenerateOutput> {
    const speechMode = resolveSpeechMode(this.config, input.metadata);
    if (speechMode === "realtime") {
      return this.generateSpeechRealtime(input);
    }
    if (!this.openAICompat.generateSpeech) {
      throw new AiKitError({
        kind: ErrorKind.Unsupported,
        message: "xAI speech generation is not supported in OpenAI compatibility mode",
        provider: Provider.XAI,
      });
    }
    return this.openAICompat.generateSpeech({ ...input, provider: Provider.XAI });
  }

  streamGenerate(input: GenerateInput): AsyncIterable<StreamChunk> {
    const adapter = this.selectAdapter(input.metadata);
    return adapter.streamGenerate({ ...input, provider: Provider.XAI });
  }

  private selectAdapter(metadata?: Record<string, string>) {
    const metadataMode = metadata?.[COMPATIBILITY_KEY];
    const mode =
      (metadataMode as "openai" | "anthropic" | undefined) ??
      this.config.compatibilityMode ??
      "openai";
    return mode === "anthropic" ? this.anthropicCompat : this.openAICompat;
  }

  private async generateSpeechRealtime(
    input: SpeechGenerateInput,
  ): Promise<SpeechGenerateOutput> {
    if (!this.config.apiKey) {
      throw new AiKitError({
        kind: ErrorKind.ProviderAuth,
        message: "xAI api key is required for realtime speech",
        provider: Provider.XAI,
      });
    }
    const { formatType, mime, sampleRate, sessionOverrides, responseOverrides } =
      resolveXaiSpeechOptions(input);
    const session: Record<string, unknown> = {
      voice: input.voice ?? DEFAULT_VOICE,
      turn_detection: { type: null },
      audio: {
        output: {
          format: {
            type: formatType,
            rate: sampleRate,
          },
        },
      },
    };
    if (sessionOverrides) {
      Object.assign(session, sessionOverrides);
    }
    const response: Record<string, unknown> = {
      modalities: ["audio"],
    };
    if (responseOverrides) {
      Object.assign(response, responseOverrides);
    }
    const url = resolveRealtimeURL(this.config.baseURL);
    const signal = input.signal;
    if (signal?.aborted) {
      throw new AiKitError({
        kind: ErrorKind.Timeout,
        message: "xAI realtime speech request was aborted",
        provider: Provider.XAI,
      });
    }
    return await new Promise<SpeechGenerateOutput>((resolve, reject) => {
      const ws = new WebSocket(url, REALTIME_PROTOCOLS, {
        headers: {
          Authorization: `Bearer ${this.config.apiKey}`,
        },
      });
      const audioChunks: Buffer[] = [];
      let sessionSent = false;
      let responseSent = false;
      let resolved = false;
      const cleanup = () => {
        ws.removeAllListeners();
        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
          ws.close();
        }
        if (signal && abortHandler) {
          signal.removeEventListener("abort", abortHandler);
        }
      };
      const fail = (err: unknown) => {
        if (resolved) {
          return;
        }
        resolved = true;
        cleanup();
        reject(err);
      };
      const finish = () => {
        if (resolved) {
          return;
        }
        resolved = true;
        cleanup();
        const data = Buffer.concat(audioChunks).toString("base64");
        resolve({ mime, data });
      };
      const abortHandler = signal
        ? () => {
            fail(
              new AiKitError({
                kind: ErrorKind.Timeout,
                message: "xAI realtime speech request was aborted",
                provider: Provider.XAI,
              }),
            );
          }
        : null;
      if (signal && abortHandler) {
        signal.addEventListener("abort", abortHandler, { once: true });
      }
      ws.on("message", (data) => {
        let payload: Record<string, unknown>;
        try {
          payload = JSON.parse(data.toString()) as Record<string, unknown>;
        } catch {
          return;
        }
        const type = typeof payload.type === "string" ? payload.type : "";
        if (type === "conversation.created" && !sessionSent) {
          sessionSent = true;
          ws.send(JSON.stringify({ type: "session.update", session }));
          return;
        }
        if (type === "session.updated" && !responseSent) {
          responseSent = true;
          ws.send(
            JSON.stringify({
              type: "conversation.item.create",
              item: {
                type: "message",
                role: "user",
                content: [{ type: "input_text", text: input.text }],
              },
            }),
          );
          ws.send(
            JSON.stringify({
              type: "response.create",
              response,
            }),
          );
          return;
        }
        if (type === "response.output_audio.delta") {
          const delta = payload.delta;
          if (typeof delta === "string" && delta.length) {
            audioChunks.push(Buffer.from(delta, "base64"));
          }
          return;
        }
        if (type === "response.output_audio.done" || type === "response.done") {
          finish();
          return;
        }
        if (type === "error" || type === "response.error") {
          const message =
            typeof payload.error === "object" && payload.error
              ? String((payload.error as Record<string, unknown>).message ?? "xAI realtime error")
              : "xAI realtime error";
          fail(
            new AiKitError({
              kind: ErrorKind.Unknown,
              message,
              provider: Provider.XAI,
            }),
          );
        }
      });
      ws.on("error", (err) => {
        fail(
          new AiKitError({
            kind: ErrorKind.Unknown,
            message: err instanceof Error ? err.message : "xAI realtime socket error",
            provider: Provider.XAI,
          }),
        );
      });
      ws.on("close", () => {
        if (!resolved) {
          fail(
            new AiKitError({
              kind: ErrorKind.Unknown,
              message: "xAI realtime socket closed before audio was completed",
              provider: Provider.XAI,
            }),
          );
        }
      });
    });
  }
}

function resolveSpeechMode(
  config: XAIProviderConfig,
  metadata?: Record<string, string>,
): "openai" | "realtime" {
  const metadataMode = metadata?.[SPEECH_MODE_KEY];
  const mode =
    (metadataMode as "openai" | "realtime" | undefined) ??
    config.speechMode ??
    "realtime";
  return mode.toLowerCase() === "openai" ? "openai" : "realtime";
}

function resolveRealtimeURL(baseURL?: string): string {
  const fallback = "https://api.x.ai";
  let parsed: URL;
  try {
    parsed = new URL(baseURL ?? fallback);
  } catch {
    parsed = new URL(fallback);
  }
  parsed.protocol = parsed.protocol === "http:" ? "ws:" : "wss:";
  parsed.pathname = "/v1/realtime";
  parsed.search = "";
  parsed.hash = "";
  return parsed.toString();
}

function resolveXaiSpeechOptions(input: SpeechGenerateInput): {
  formatType: string;
  mime: string;
  sampleRate: number;
  sessionOverrides?: Record<string, unknown>;
  responseOverrides?: Record<string, unknown>;
} {
  const responseFormat = input.responseFormat ?? input.format;
  const normalized = responseFormat ? responseFormat.toLowerCase() : "";
  let formatType = "audio/pcm";
  let mime = "audio/pcm";
  if (normalized && normalized !== "pcm") {
    if (normalized === "pcmu") {
      formatType = "audio/pcmu";
      mime = "audio/pcmu";
    } else if (normalized === "pcma") {
      formatType = "audio/pcma";
      mime = "audio/pcma";
    } else {
      throw new AiKitError({
        kind: ErrorKind.Unsupported,
        message: `xAI realtime speech only supports pcm/pcmu/pcma output (received ${responseFormat})`,
        provider: Provider.XAI,
      });
    }
  }
  const params =
    input.parameters && typeof input.parameters === "object"
      ? (input.parameters as Record<string, unknown>)
      : undefined;
  const sampleRateValue =
    typeof params?.sampleRate === "number" && Number.isFinite(params.sampleRate)
      ? params.sampleRate
      : DEFAULT_SAMPLE_RATE;
  const sampleRate = formatType === "audio/pcm" ? sampleRateValue : 8000;
  const sessionOverrides =
    params?.session && typeof params.session === "object" && !Array.isArray(params.session)
      ? (params.session as Record<string, unknown>)
      : undefined;
  const responseOverrides =
    params?.response && typeof params.response === "object" && !Array.isArray(params.response)
      ? (params.response as Record<string, unknown>)
      : undefined;
  return { formatType, mime, sampleRate, sessionOverrides, responseOverrides };
}
