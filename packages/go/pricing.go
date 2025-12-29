package aikit

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

type curatedModel struct {
	ID            string            `json:"id"`
	DisplayName   string            `json:"displayName"`
	Provider      Provider          `json:"provider"`
	Family        string            `json:"family"`
	Capabilities  ModelCapabilities `json:"capabilities"`
	ContextWindow int               `json:"contextWindow"`
	TokenPrices   *TokenPrices      `json:"tokenPrices"`
	Deprecated    bool              `json:"deprecated"`
	InPreview     bool              `json:"inPreview"`
}

var curatedOnce sync.Once
var curatedModels []curatedModel

func loadCuratedModels() []curatedModel {
	curatedOnce.Do(func() {
		path := curatedModelsPath()
		if path == "" {
			curatedModels = []curatedModel{}
			return
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			curatedModels = []curatedModel{}
			return
		}
		if err := json.Unmarshal(raw, &curatedModels); err != nil {
			curatedModels = []curatedModel{}
			return
		}
	})
	return curatedModels
}

func curatedModelsPath() string {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return ""
	}
	return filepath.Join(filepath.Dir(filename), "..", "..", "models", "curated_models.json")
}

func normalizeModelID(provider Provider, modelID string) string {
	prefix := string(provider) + "/"
	if strings.HasPrefix(modelID, prefix) {
		return strings.TrimPrefix(modelID, prefix)
	}
	return modelID
}

func findCuratedModel(provider Provider, modelID string) *curatedModel {
	normalized := normalizeModelID(provider, modelID)
	var best *curatedModel
	for i := range loadCuratedModels() {
		model := &curatedModels[i]
		if model.Provider != provider {
			continue
		}
		if model.ID == normalized {
			return model
		}
		if strings.HasPrefix(normalized, model.ID) {
			if best == nil || len(model.ID) > len(best.ID) {
				best = model
			}
		}
	}
	return best
}

func applyCuratedMetadata(model ModelMetadata) ModelMetadata {
	curated := findCuratedModel(model.Provider, model.ID)
	if curated == nil {
		return model
	}
	if curated.DisplayName != "" {
		model.DisplayName = curated.DisplayName
	}
	if curated.Family != "" {
		model.Family = curated.Family
	}
	model.Capabilities = curated.Capabilities
	if curated.ContextWindow > 0 {
		model.ContextWindow = curated.ContextWindow
	}
	if curated.TokenPrices != nil {
		model.TokenPrices = curated.TokenPrices
	}
	model.Deprecated = curated.Deprecated
	model.InPreview = curated.InPreview
	return model
}

func lookupTokenPrices(provider Provider, modelID string) *TokenPrices {
	curated := findCuratedModel(provider, modelID)
	if curated == nil {
		return nil
	}
	return curated.TokenPrices
}

func estimateCost(provider Provider, modelID string, usage *Usage) *CostBreakdown {
	if usage == nil {
		return nil
	}
	pricing := lookupTokenPrices(provider, modelID)
	if pricing == nil {
		return nil
	}
	if pricing.Input == 0 && pricing.Output == 0 {
		return nil
	}
	inputCost := float64(usage.InputTokens) * pricing.Input / 1_000_000
	outputCost := float64(usage.OutputTokens) * pricing.Output / 1_000_000
	return &CostBreakdown{
		InputCostUSD:    roundUsd(inputCost),
		OutputCostUSD:   roundUsd(outputCost),
		TotalCostUSD:    roundUsd(inputCost + outputCost),
		PricingPerMillion: pricing,
	}
}

func roundUsd(value float64) float64 {
	return math.Round(value*1_000_000) / 1_000_000
}
