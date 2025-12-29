package aikit

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

func FingerprintAPIKey(apiKey string) string {
	trimmed := strings.TrimSpace(apiKey)
	if trimmed == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(trimmed))
	return hex.EncodeToString(sum[:])
}
