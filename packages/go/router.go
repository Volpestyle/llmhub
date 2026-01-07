package aikit

import (
	"errors"
	"sort"
	"strings"
)

type ModelConstraints struct {
	RequireTools  bool    `json:"requireTools,omitempty"`
	RequireJSON   bool    `json:"requireJson,omitempty"`
	RequireVision bool    `json:"requireVision,omitempty"`
	RequireVideo  bool    `json:"requireVideo,omitempty"`
	MaxCostUSD    float64 `json:"maxCostUsd,omitempty"`
	LatencyClass  string  `json:"latencyClass,omitempty"`
	AllowPreview  *bool   `json:"allowPreview,omitempty"`
}

type ModelResolutionRequest struct {
	Constraints     ModelConstraints `json:"constraints,omitempty"`
	PreferredModels []string         `json:"preferredModels,omitempty"`
}

type ResolvedModel struct {
	Primary  ModelRecord   `json:"primary"`
	Fallback []ModelRecord `json:"fallback,omitempty"`
}

type ModelRouter struct{}

func (r *ModelRouter) Resolve(models []ModelRecord, req ModelResolutionRequest) (ResolvedModel, error) {
	candidates := filterModels(models, req)
	if len(candidates) == 0 {
		return ResolvedModel{}, errors.New("router: no models match constraints")
	}
	primary := candidates[0]
	var fallback []ModelRecord
	if len(candidates) > 1 {
		fallback = candidates[1:]
	}
	return ResolvedModel{Primary: primary, Fallback: fallback}, nil
}

func filterModels(models []ModelRecord, req ModelResolutionRequest) []ModelRecord {
	allowPreview := true
	if req.Constraints.AllowPreview != nil {
		allowPreview = *req.Constraints.AllowPreview
	}
	preferredOrder := normalizePreferred(req.PreferredModels)
	candidates := make([]ModelRecord, 0, len(models))
	for _, model := range models {
		if !model.Availability.Entitled {
			continue
		}
		if req.Constraints.RequireTools && !model.Features.Tools {
			continue
		}
		if req.Constraints.RequireJSON && !(model.Features.JSONMode || model.Features.JSONSchema) {
			continue
		}
		if req.Constraints.RequireVision && !model.Modalities.Vision {
			continue
		}
		if req.Constraints.RequireVideo && !model.Modalities.VideoOut {
			continue
		}
		if !allowPreview && hasTag(model.Tags, "preview") {
			continue
		}
		if req.Constraints.MaxCostUSD > 0 && !withinCost(model, req.Constraints.MaxCostUSD) {
			continue
		}
		if len(preferredOrder) > 0 && !matchesPreferred(model, preferredOrder) {
			continue
		}
		candidates = append(candidates, model)
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		iRank := preferredRank(candidates[i], preferredOrder)
		jRank := preferredRank(candidates[j], preferredOrder)
		if iRank != jRank {
			return iRank < jRank
		}
		iCost := priceScore(candidates[i])
		jCost := priceScore(candidates[j])
		if iCost != jCost {
			return iCost < jCost
		}
		return candidates[i].DisplayName < candidates[j].DisplayName
	})
	return candidates
}

func normalizePreferred(models []string) []string {
	out := make([]string, 0, len(models))
	for _, entry := range models {
		if trimmed := strings.TrimSpace(entry); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

func matchesPreferred(model ModelRecord, preferred []string) bool {
	if len(preferred) == 0 {
		return true
	}
	for _, entry := range preferred {
		if entry == model.ID || entry == model.ProviderModelID {
			return true
		}
	}
	return false
}

func preferredRank(model ModelRecord, preferred []string) int {
	if len(preferred) == 0 {
		return len(preferred) + 1
	}
	for idx, entry := range preferred {
		if entry == model.ID || entry == model.ProviderModelID {
			return idx
		}
	}
	return len(preferred) + 1
}

func withinCost(model ModelRecord, maxCost float64) bool {
	if model.Pricing == nil {
		return true
	}
	if model.Pricing.InputPer1M > 0 && model.Pricing.InputPer1M > maxCost {
		return false
	}
	if model.Pricing.OutputPer1M > 0 && model.Pricing.OutputPer1M > maxCost {
		return false
	}
	return true
}

func priceScore(model ModelRecord) float64 {
	if model.Pricing == nil {
		return 0
	}
	score := 0.0
	if model.Pricing.InputPer1M > 0 {
		score = model.Pricing.InputPer1M
	}
	if model.Pricing.OutputPer1M > 0 && (score == 0 || model.Pricing.OutputPer1M < score) {
		score = model.Pricing.OutputPer1M
	}
	return score
}

func hasTag(tags []string, target string) bool {
	for _, tag := range tags {
		if tag == target {
			return true
		}
	}
	return false
}
