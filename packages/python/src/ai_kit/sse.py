from __future__ import annotations

from typing import Dict, Generator, Iterable, Optional


def iter_sse_events(lines: Iterable[str]) -> Generator[Dict[str, str], None, None]:
    event_type: Optional[str] = None
    data_lines: list[str] = []
    for raw in lines:
        line = raw.strip("\r\n")
        if line == "":
            if data_lines:
                yield {
                    "event": event_type or "message",
                    "data": "\n".join(data_lines),
                }
            event_type = None
            data_lines = []
            continue
        if line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
    if data_lines:
        yield {
            "event": event_type or "message",
            "data": "\n".join(data_lines),
        }
