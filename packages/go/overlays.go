package llmhub

import (
	_ "embed"
	"encoding/json"
	"sync"
)

//go:embed curated_models.json
var curatedModelsRaw []byte

var (
	curatedOnce sync.Once
	curatedData map[Provider]map[string]ModelMetadata
)

func loadCurated() {
	curatedOnce.Do(func() {
		var models []ModelMetadata
		if err := json.Unmarshal(curatedModelsRaw, &models); err != nil {
			curatedData = map[Provider]map[string]ModelMetadata{}
			return
		}
		curated := make(map[Provider]map[string]ModelMetadata)
		for _, model := range models {
			if _, ok := curated[model.Provider]; !ok {
				curated[model.Provider] = make(map[string]ModelMetadata)
			}
			curated[model.Provider][model.ID] = model
		}
		curatedData = curated
	})
}

func lookupCurated(provider Provider, id string) (ModelMetadata, bool) {
	loadCurated()
	if providerModels, ok := curatedData[provider]; ok {
		model, found := providerModels[id]
		return model, found
	}
	return ModelMetadata{}, false
}

func curatedList(provider Provider) []ModelMetadata {
	loadCurated()
	if providerModels, ok := curatedData[provider]; ok {
		models := make([]ModelMetadata, 0, len(providerModels))
		for _, model := range providerModels {
			models = append(models, model)
		}
		return models
	}
	return nil
}
