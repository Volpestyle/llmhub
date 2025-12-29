package aikit

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"sync"
)

type catalogAdapter struct {
	models []ModelMetadata
}

var catalogOnce sync.Once
var catalogModels []ModelMetadata

func catalogModelsPath() string {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return ""
	}
	return filepath.Join(filepath.Dir(filename), "..", "..", "models", "catalog_models.json")
}

func loadCatalogModels() []ModelMetadata {
	catalogOnce.Do(func() {
		path := catalogModelsPath()
		if path == "" {
			catalogModels = []ModelMetadata{}
			return
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			catalogModels = []ModelMetadata{}
			return
		}
		var models []ModelMetadata
		if err := json.Unmarshal(raw, &models); err != nil {
			catalogModels = []ModelMetadata{}
			return
		}
		catalogModels = models
	})
	return catalogModels
}

func HasCatalogModels() bool {
	return len(loadCatalogModels()) > 0
}

func newCatalogAdapter() *catalogAdapter {
	models := loadCatalogModels()
	if len(models) == 0 {
		return nil
	}
	return &catalogAdapter{models: models}
}

func (a *catalogAdapter) ListModels(_ context.Context) ([]ModelMetadata, error) {
	out := make([]ModelMetadata, len(a.models))
	copy(out, a.models)
	return out, nil
}

func (a *catalogAdapter) Generate(_ context.Context, _ GenerateInput) (GenerateOutput, error) {
	return GenerateOutput{}, &HubError{
		Kind:     ErrorUnsupported,
		Message:  "catalog provider does not support generate",
		Provider: ProviderCatalog,
	}
}

func (a *catalogAdapter) GenerateImage(_ context.Context, _ ImageGenerateInput) (ImageGenerateOutput, error) {
	return ImageGenerateOutput{}, &HubError{
		Kind:     ErrorUnsupported,
		Message:  "catalog provider does not support image generation",
		Provider: ProviderCatalog,
	}
}

func (a *catalogAdapter) GenerateMesh(_ context.Context, _ MeshGenerateInput) (MeshGenerateOutput, error) {
	return MeshGenerateOutput{}, &HubError{
		Kind:     ErrorUnsupported,
		Message:  "catalog provider does not support mesh generation",
		Provider: ProviderCatalog,
	}
}

func (a *catalogAdapter) Stream(_ context.Context, _ GenerateInput) (<-chan StreamChunk, error) {
	return nil, &HubError{
		Kind:     ErrorUnsupported,
		Message:  "catalog provider does not support streaming",
		Provider: ProviderCatalog,
	}
}
