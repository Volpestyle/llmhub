/// <reference types="vitest" />
import { describe, expect, it, vi } from "vitest";
import { httpHandlers } from "../src/http/handlers.js";
import { Provider, type Hub, type StreamChunk } from "../src/core/types.js";

function createMockResponse() {
  return {
    statusCode: 200,
    jsonPayload: undefined as unknown,
    headers: new Map<string, string>(),
    body: "",
    status(code: number) {
      this.statusCode = code;
      return this;
    },
    json(payload: unknown) {
      this.jsonPayload = payload;
    },
    setHeader(name: string, value: string) {
      this.headers.set(name.toLowerCase(), value);
    },
    write(chunk: string) {
      this.body += chunk;
    },
    end() {
      this.body += "[ended]";
    },
    flush: vi.fn(),
  };
}

describe("http handlers", () => {
  it("returns model list", async () => {
    const hub = createHubMock({
      listModels: vi.fn().mockResolvedValue([{ id: "x", provider: Provider.OpenAI }]),
    });
    const handlers = httpHandlers(hub);
    const res = createMockResponse();
    await handlers.models()(
      { query: { providers: "openai,xai" } },
      res,
    );
    expect(hub.listModels).toHaveBeenCalledWith({
      providers: [Provider.OpenAI, Provider.XAI],
    });
    expect(res.statusCode).toBe(200);
    expect(res.jsonPayload).toEqual([{ id: "x", provider: Provider.OpenAI }]);
  });

  it("invokes generate handler", async () => {
    const hub = createHubMock({
      generate: vi.fn().mockResolvedValue({ text: "ok" }),
    });
    const handlers = httpHandlers(hub);
    const res = createMockResponse();
    await handlers.generate()(
      { body: { provider: Provider.OpenAI, model: "gpt", messages: [] } },
      res,
    );
    expect(hub.generate).toHaveBeenCalled();
    expect(res.jsonPayload).toEqual({ text: "ok" });
  });

  it("streams SSE chunks", async () => {
    const chunks: StreamChunk[] = [
      { type: "delta", textDelta: "hel" },
      { type: "delta", textDelta: "lo" },
      { type: "message_end", finishReason: "stop" },
    ];
    const hub = createHubMock({
      streamGenerate: vi.fn().mockReturnValue(streamFrom(chunks)),
    });
    const handlers = httpHandlers(hub);
    const res = createMockResponse();
    await handlers.generateSSE()(
      { method: "POST", body: { provider: Provider.OpenAI, model: "gpt", messages: [] } },
      res,
    );
    expect(res.headers.get("content-type")).toBe("text/event-stream");
    expect(res.body).toContain("event: chunk");
    expect(res.body).toContain("event: done");
  });
});

function createHubMock(overrides: Partial<Hub>): Hub {
  return {
    listModels: vi.fn().mockResolvedValue([]),
    generate: vi.fn().mockResolvedValue({}),
    streamGenerate: vi.fn(),
    ...overrides,
  };
}

async function* streamFrom(chunks: StreamChunk[]) {
  for (const chunk of chunks) {
    yield chunk;
  }
}
