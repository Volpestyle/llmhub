import { ModelRecord, ModelResolutionRequest, ResolvedModel } from "./types.js";

export class ModelRouter {
  resolve(models: ModelRecord[], request: ModelResolutionRequest): ResolvedModel {
    const candidates = filterModels(models, request);
    if (!candidates.length) {
      throw new Error("router: no models match constraints");
    }
    return {
      primary: candidates[0],
      fallback: candidates.length > 1 ? candidates.slice(1) : undefined,
    };
  }
}

function filterModels(
  models: ModelRecord[],
  request: ModelResolutionRequest,
): ModelRecord[] {
  const constraints = request.constraints ?? {};
  const preferred = (request.preferredModels ?? [])
    .map((entry) => entry.trim())
    .filter(Boolean);
  const allowPreview = constraints.allowPreview ?? true;
  const filtered = models.filter((model) => {
    if (!model.availability.entitled) {
      return false;
    }
    if (constraints.requireTools && !model.features.tools) {
      return false;
    }
    if (
      constraints.requireJson &&
      !(model.features.jsonMode || model.features.jsonSchema)
    ) {
      return false;
    }
    if (constraints.requireVision && !model.modalities.vision) {
      return false;
    }
    if (constraints.requireVideo && !model.modalities.videoOut) {
      return false;
    }
    if (!allowPreview && model.tags?.includes("preview")) {
      return false;
    }
    if (constraints.maxCostUsd && !withinCost(model, constraints.maxCostUsd)) {
      return false;
    }
    if (preferred.length && !matchesPreferred(model, preferred)) {
      return false;
    }
    return true;
  });
  return filtered.sort((a, b) => {
    const aRank = preferredRank(a, preferred);
    const bRank = preferredRank(b, preferred);
    if (aRank !== bRank) {
      return aRank - bRank;
    }
    const aCost = priceScore(a);
    const bCost = priceScore(b);
    if (aCost !== bCost) {
      return aCost - bCost;
    }
    return (a.displayName ?? "").localeCompare(b.displayName ?? "");
  });
}

function matchesPreferred(model: ModelRecord, preferred: string[]): boolean {
  return preferred.some(
    (entry) => entry === model.id || entry === model.providerModelId,
  );
}

function preferredRank(model: ModelRecord, preferred: string[]): number {
  if (!preferred.length) {
    return preferred.length + 1;
  }
  const index = preferred.findIndex(
    (entry) => entry === model.id || entry === model.providerModelId,
  );
  return index === -1 ? preferred.length + 1 : index;
}

function withinCost(model: ModelRecord, maxCost: number): boolean {
  const pricing = model.pricing;
  if (!pricing) {
    return true;
  }
  if (pricing.inputPer1M && pricing.inputPer1M > maxCost) {
    return false;
  }
  if (pricing.outputPer1M && pricing.outputPer1M > maxCost) {
    return false;
  }
  return true;
}

function priceScore(model: ModelRecord): number {
  if (!model.pricing) {
    return 0;
  }
  const input = model.pricing.inputPer1M ?? 0;
  const output = model.pricing.outputPer1M ?? 0;
  if (!input) {
    return output;
  }
  if (!output) {
    return input;
  }
  return Math.min(input, output);
}
