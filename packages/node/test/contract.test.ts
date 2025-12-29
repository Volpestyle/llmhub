/// <reference types="vitest" />
import { describe, expect, it } from "vitest";
import {
  createKit,
  Provider,
  GenerateInput,
} from "../src/index.js";
import type { FetchLike } from "../src/core/types.js";

const STANDARD_TEXT = "Unified response";

const baseMessages: GenerateInput["messages"] = [
  {
    role: "system",
    content: [{ type: "text", text: "You are a helpful assistant." }],
  },
  {
    role: "user",
    content: [{ type: "text", text: "Say hello." }],
  },
];

const fetchImpl: FetchLike = async (input, init) => {
  const url = typeof input === "string" ? input : input.toString();
  if (url.includes("api.openai.com")) {
    return jsonResponse(buildChatCompletion(STANDARD_TEXT));
  }
  if (url.includes("api.anthropic.com")) {
    return jsonResponse(buildAnthropicResponse(STANDARD_TEXT));
  }
  if (url.includes("api.x.ai")) {
    return jsonResponse(buildChatCompletion(STANDARD_TEXT));
  }
  if (url.includes("generativelanguage.googleapis.com")) {
    if (url.includes(":generateContent")) {
      return jsonResponse(buildGeminiResponse(STANDARD_TEXT));
    }
  }
  throw new Error(`No mock response for ${url}`);
};

const kit = createKit({
  providers: {
    [Provider.OpenAI]: { apiKey: "test-openai-key" },
    [Provider.Anthropic]: { apiKey: "test-anthropic-key" },
    [Provider.XAI]: { apiKey: "test-xai-key" },
    [Provider.Google]: { apiKey: "test-google-key" },
  },
  httpClient: fetchImpl,
});

describe("contract generate normalization", () => {
  const input: Omit<GenerateInput, "provider"> = {
    model: "test-model",
    messages: baseMessages,
  };

  for (const provider of [
    Provider.OpenAI,
    Provider.Anthropic,
    Provider.Google,
    Provider.XAI,
  ]) {
    it(`normalizes output for ${provider}`, async () => {
      const output = await kit.generate({ ...input, provider });
      expect(output.text).toBe(STANDARD_TEXT);
      expect(output.finishReason).toBeDefined();
    });
  }
});

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

function buildChatCompletion(text: string) {
  return {
    id: "chatcmpl-1",
    choices: [
      {
        finish_reason: "stop",
        message: {
          content: [{ type: "text", text }],
          tool_calls: [],
        },
      },
    ],
    usage: {
      prompt_tokens: 5,
      completion_tokens: 10,
      total_tokens: 15,
    },
  };
}

function buildAnthropicResponse(text: string) {
  return {
    id: "msg_1",
    content: [{ type: "text", text }],
    stop_reason: "end_turn",
    usage: { input_tokens: 5, output_tokens: 10 },
  };
}

function buildGeminiResponse(text: string) {
  return {
    candidates: [
      {
        content: {
          parts: [{ text }],
        },
        finishReason: "STOP",
      },
    ],
    usageMetadata: {
      promptTokenCount: 1,
      candidatesTokenCount: 1,
      totalTokenCount: 2,
    },
  };
}
