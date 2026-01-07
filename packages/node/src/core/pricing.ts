import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import type {
  CostBreakdown,
  ModelCapabilities,
  ModelMetadata,
  Provider,
  TokenPrices,
  Usage,
} from "./types.js";

interface CuratedModel {
  id: string;
  provider: Provider;
  displayName?: string;
  family?: string;
  capabilities?: ModelCapabilities;
  contextWindow?: number;
  tokenPrices?: TokenPrices;
  videoPrices?: Record<string, number>;
  deprecated?: boolean;
  inPreview?: boolean;
}

let cachedCuratedModels: CuratedModel[] | null = null;

function modelsRoot(): string {
  return fileURLToPath(new URL("../../../../models", import.meta.url));
}

function loadCuratedModels(): CuratedModel[] {
  if (cachedCuratedModels) {
    return cachedCuratedModels;
  }
  const collected: CuratedModel[] = [];
  try {
    const entries = fs.readdirSync(modelsRoot(), { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) {
        continue;
      }
      const provider = entry.name;
      const filePath = path.join(modelsRoot(), provider, "scraped_models.json");
      if (!fs.existsSync(filePath)) {
        continue;
      }
      const raw = fs.readFileSync(filePath, "utf-8");
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) {
        continue;
      }
      for (const item of parsed) {
        if (!item || typeof item !== "object") {
          continue;
        }
        if (!("provider" in item)) {
          (item as CuratedModel).provider = provider as Provider;
        }
        collected.push(item as CuratedModel);
      }
    }
  } catch {
    // ignore and return empty list
  }
  cachedCuratedModels = collected;
  return cachedCuratedModels;
}

function normalizeModelId(provider: Provider, modelId: string): string {
  const prefix = `${provider}/`;
  if (modelId.startsWith(prefix)) {
    return modelId.slice(prefix.length);
  }
  return modelId;
}

function findCuratedModel(
  provider: Provider,
  modelId: string,
): CuratedModel | undefined {
  const normalized = normalizeModelId(provider, modelId);
  let best: CuratedModel | undefined;
  for (const model of loadCuratedModels()) {
    if (model.provider !== provider) {
      continue;
    }
    if (model.id === normalized) {
      return model;
    }
    if (normalized.startsWith(model.id)) {
      if (!best || model.id.length > best.id.length) {
        best = model;
      }
    }
  }
  return best;
}

export function applyCuratedMetadata(model: ModelMetadata): ModelMetadata {
  const curated = findCuratedModel(model.provider, model.id);
  if (!curated) {
    return model;
  }
  return {
    ...model,
    displayName: curated.displayName ?? model.displayName,
    family: curated.family ?? model.family,
    capabilities: curated.capabilities ?? model.capabilities,
    contextWindow: curated.contextWindow ?? model.contextWindow,
    tokenPrices: curated.tokenPrices ?? model.tokenPrices,
    videoPrices: curated.videoPrices ?? model.videoPrices,
    deprecated: curated.deprecated ?? model.deprecated,
    inPreview: curated.inPreview ?? model.inPreview,
  };
}

export function lookupTokenPrices(
  provider: Provider,
  modelId: string,
): TokenPrices | undefined {
  return findCuratedModel(provider, modelId)?.tokenPrices;
}

export function estimateCost(
  provider: Provider,
  modelId: string,
  usage?: Usage,
): CostBreakdown | undefined {
  if (!usage) {
    return undefined;
  }
  const pricing = lookupTokenPrices(provider, modelId);
  if (!pricing) {
    return undefined;
  }
  const inputRate = pricing.input ?? 0;
  const outputRate = pricing.output ?? 0;
  if (!inputRate && !outputRate) {
    return undefined;
  }
  if (usage.inputTokens === undefined && usage.outputTokens === undefined) {
    return undefined;
  }
  const inputTokens = usage.inputTokens ?? 0;
  const outputTokens = usage.outputTokens ?? 0;
  const inputCost = (inputTokens * inputRate) / 1_000_000;
  const outputCost = (outputTokens * outputRate) / 1_000_000;
  return {
    input_cost_usd: roundUsd(inputCost),
    output_cost_usd: roundUsd(outputCost),
    total_cost_usd: roundUsd(inputCost + outputCost),
    pricing_per_million: pricing,
  };
}

function roundUsd(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}
