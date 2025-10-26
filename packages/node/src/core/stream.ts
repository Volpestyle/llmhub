export interface SSEEvent {
  event?: string;
  data: string;
}

export async function* streamSSE(
  response: Response,
): AsyncIterable<SSEEvent> {
  if (!response.body) {
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const event = parseEventBlock(chunk);
      if (event) {
        yield event;
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
  if (buffer.trim().length > 0) {
    const event = parseEventBlock(buffer.trim());
    if (event) {
      yield event;
    }
  }
}

function parseEventBlock(block: string): SSEEvent | undefined {
  if (!block) {
    return undefined;
  }
  const lines = block.split("\n");
  let eventName: string | undefined;
  let data = "";
  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      data += `${line.slice(5).trim()}\n`;
    }
  }
  return {
    event: eventName,
    data: data.trim(),
  };
}

export function parseEventData<T>(event: SSEEvent): T | undefined {
  if (!event.data || event.data === "[DONE]") {
    return undefined;
  }
  try {
    return JSON.parse(event.data) as T;
  } catch {
    return undefined;
  }
}
