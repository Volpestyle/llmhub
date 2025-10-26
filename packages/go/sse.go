package llmhub

import (
	"bufio"
	"context"
	"io"
	"strings"
)

type sseEvent struct {
	Event string
	Data  string
}

func streamSSE(ctx context.Context, body io.ReadCloser) <-chan sseEvent {
	ch := make(chan sseEvent)
	go func() {
		defer close(ch)
		defer body.Close()
		reader := bufio.NewReader(body)
		var eventName string
		var dataBuilder strings.Builder
		flush := func() {
			if dataBuilder.Len() == 0 {
				return
			}
			ch <- sseEvent{
				Event: eventName,
				Data:  strings.TrimSpace(dataBuilder.String()),
			}
			eventName = ""
			dataBuilder.Reset()
		}
		for {
			select {
			case <-ctx.Done():
				return
			default:
			}
			line, err := reader.ReadString('\n')
			if err != nil {
				if err != io.EOF {
					return
				}
			}
			line = strings.TrimRight(line, "\r\n")
			if line == "" {
				flush()
				if err == io.EOF {
					return
				}
				continue
			}
			if strings.HasPrefix(line, "event:") {
				eventName = strings.TrimSpace(line[6:])
				continue
			}
			if strings.HasPrefix(line, "data:") {
				dataBuilder.WriteString(strings.TrimSpace(line[5:]))
				dataBuilder.WriteRune('\n')
				continue
			}
			if err == io.EOF {
				flush()
				return
			}
		}
	}()
	return ch
}
