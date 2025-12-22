/// <reference types="vitest" />
import { describe, expect, it, vi } from "vitest";
import { ModelRegistry } from "../src/registry/modelRegistry.js";
import { Provider } from "../src/core/types.js";
import type {
  GenerateInput,
  GenerateOutput,
  ModelMetadata,
  StreamChunk,
} from "../src/core/types.js";
import type { ProviderAdapter } from "../src/core/provider.js";

describe("ModelRegistry", () => {
  it("calls provider to list models", async () => {
    const listModels = vi.fn<[], Promise<ModelMetadata[]>>().mockResolvedValue([
      {
        id: "custom-model",
        displayName: "Custom Model",
        provider: Provider.OpenAI,
        capabilities: {
          text: true,
          vision: false,
          tool_use: true,
          structured_output: false,
          reasoning: false,
        },
      },
    ]);
    const adapter = createAdapter(Provider.OpenAI, listModels);
    const registry = new ModelRegistry({ [Provider.OpenAI]: adapter });
    const models = await registry.list({
      providers: [Provider.OpenAI],
      refresh: true,
    });
    expect(listModels).toHaveBeenCalled();
    expect(models).toEqual([
      expect.objectContaining({ id: "custom-model", provider: Provider.OpenAI }),
    ]);
  });
});

function createAdapter(
  provider: Provider,
  listModels: ProviderAdapter["listModels"],
): ProviderAdapter {
  return {
    provider,
    listModels,
    async generate(_input: GenerateInput): Promise<GenerateOutput> {
      return {};
    },
    async *streamGenerate(
      _input: GenerateInput,
    ): AsyncIterable<StreamChunk> {
      yield { type: "message_end" } as StreamChunk;
    },
  };
}
