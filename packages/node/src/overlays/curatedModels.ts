import curated from "../../../../packages/go/curated_models.json";
import { ModelMetadata, Provider } from "../core/types.js";

const curatedArray = curated as ModelMetadata[];

const byProvider = new Map<Provider, Map<string, ModelMetadata>>();
for (const model of curatedArray) {
  const provider = model.provider as Provider;
  if (!byProvider.has(provider)) {
    byProvider.set(provider, new Map());
  }
  byProvider.get(provider)!.set(model.id, {
    ...model,
    provider,
  });
}

export function lookupCuratedModel(
  provider: Provider,
  modelId: string,
): ModelMetadata | undefined {
  return byProvider.get(provider)?.get(modelId);
}

export function listCuratedModels(provider?: Provider): ModelMetadata[] {
  if (!provider) {
    return Array.from(byProvider.values()).flatMap((map) =>
      Array.from(map.values()),
    );
  }
  return Array.from(byProvider.get(provider)?.values() ?? []);
}
