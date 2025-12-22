package llmhub

import (
	"strings"
	"sync/atomic"
)

type keyPool struct {
	keys    []string
	counter uint64
}

func newKeyPool(keys []string) *keyPool {
	if len(keys) == 0 {
		return nil
	}
	return &keyPool{keys: keys}
}

func (p *keyPool) Next() string {
	if p == nil || len(p.keys) == 0 {
		return ""
	}
	if len(p.keys) == 1 {
		return p.keys[0]
	}
	idx := atomic.AddUint64(&p.counter, 1) - 1
	return p.keys[idx%uint64(len(p.keys))]
}

func normalizeKeys(primary string, extras []string) []string {
	seen := make(map[string]struct{})
	var keys []string
	appendKey := func(raw string) {
		trimmed := strings.TrimSpace(raw)
		if trimmed == "" {
			return
		}
		if _, ok := seen[trimmed]; ok {
			return
		}
		seen[trimmed] = struct{}{}
		keys = append(keys, trimmed)
	}
	appendKey(primary)
	for _, key := range extras {
		appendKey(key)
	}
	return keys
}
