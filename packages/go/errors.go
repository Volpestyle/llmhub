package llmhub

import "fmt"

type ErrorKind string

const (
	ErrorUnknown             ErrorKind = "unknown_error"
	ErrorProviderAuth        ErrorKind = "provider_auth_error"
	ErrorProviderRateLimit   ErrorKind = "provider_rate_limit"
	ErrorProviderNotFound    ErrorKind = "provider_not_found"
	ErrorProviderUnavailable ErrorKind = "provider_unavailable"
	ErrorValidation          ErrorKind = "validation_error"
	ErrorUnsupported         ErrorKind = "unsupported"
)

type HubError struct {
	Kind           ErrorKind
	Message        string
	Provider       Provider
	UpstreamCode   string
	UpstreamStatus int
	RequestID      string
	Cause          error
}

func (e *HubError) Error() string {
	return fmt.Sprintf("%s: %s", e.Kind, e.Message)
}

func classifyStatus(status int) ErrorKind {
	switch status {
	case 401, 403:
		return ErrorProviderAuth
	case 404:
		return ErrorProviderNotFound
	case 429:
		return ErrorProviderRateLimit
	}
	if status >= 500 {
		return ErrorProviderUnavailable
	}
	return ErrorUnknown
}
