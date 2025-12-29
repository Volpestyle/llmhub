package aikit

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
)

func doJSON(ctx context.Context, client *http.Client, req *http.Request, provider Provider, out interface{}) error {
	resp, err := doRequest(ctx, client, req, provider)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return json.NewDecoder(resp.Body).Decode(out)
}

func doRequest(ctx context.Context, client *http.Client, req *http.Request, provider Provider) (*http.Response, error) {
	req = req.WithContext(ctx)
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, &HubError{
			Kind:           classifyStatus(resp.StatusCode),
			Message:        string(body),
			Provider:       provider,
			UpstreamStatus: resp.StatusCode,
			RequestID:      resp.Header.Get("x-request-id"),
		}
	}
	return resp, nil
}
