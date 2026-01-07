# Grok Voice Agent in a Python AI/ML Pipeline

Below is a practical, "wire-it-into-a-real-system" guide to integrating the **Grok Voice Agent API** into a **Python AI/ML pipeline** (inference, RAG, analytics, internal tools), based on the WebSocket/event model described in the docs you linked.

## Table of contents

- [How the voice agent fits into a pipeline](#how-the-voice-agent-fits-into-a-pipeline)
- [Step 1: Choose an integration pattern](#step-1-choose-an-integration-pattern)
- [Step 2: Authentication and security](#step-2-authentication-and-security)
- [Step 3: Session configuration (voice, audio format, turn detection)](#step-3-session-configuration-voice-audio-format-turn-detection)
- [Step 4: Realtime event protocol (what you'll actually handle)](#step-4-realtime-event-protocol-what-youll-actually-handle)
- [Step 5: Expose your ML pipeline as function tools](#step-5-expose-your-ml-pipeline-as-function-tools)
- [Step 6: Python pipeline integration skeleton](#step-6-python-pipeline-integration-skeleton)
- [Step 7: Streaming audio in from your pipeline](#step-7-streaming-audio-in-from-your-pipeline)
- [Step 8: Where to plug in RAG + model inference](#step-8-where-to-plug-in-rag-model-inference)
- [Step 9: Production notes that matter in real pipelines](#step-9-production-notes-that-matter-in-real-pipelines)
- [Quick checklist (so it doesn't break at 2am)](#quick-checklist-so-it-doesnt-break-at-2am)

---

## How the voice agent fits into a pipeline

Think of the Voice Agent API as a **real-time speech ↔ reasoning ↔ speech** loop over WebSocket:

```
Audio in (mic / telephony / file)
   -> input_audio_buffer.append (base64 PCM/G711)
      -> Grok Voice Agent (realtime)
         -> events (transcripts, tool calls, audio out)
            -> your pipeline (ML inference, RAG, DB, business logic)
               -> tool outputs back to Grok
                  -> spoken response (audio deltas)
```

You get:

- **Audio + text input** and **audio + text output** in real time over `wss://api.x.ai/v1/realtime`.
- A clean way to connect your pipeline via **function tools** (your code runs) and built-in tools like **web_search**, **x_search**, and **file_search (Collections)** (xAI runs).

---

## Step 1: Choose an integration pattern

### Pattern A: Voice agent as orchestrator (recommended)

Expose your ML pipeline as **function tools** (predict/classify/retrieve/query). The voice agent decides _when_ to call them, and then speaks results naturally.

**Best for:** interactive assistants, agents, copilots, support/triage systems.
**Key mechanism:** `response.function_call_arguments.done` → you run code → `function_call_output` → `response.create`.

### Pattern B: Voice agent for ASR + dialog; pipeline for side analytics

Listen to transcript events like `conversation.item.input_audio_transcription.completed`, run analytics (sentiment, compliance, summaries), store results, maybe show in UI.

### Pattern C: Use it as streaming TTS (text → voice)

Send text via `conversation.item.create`, request audio output, and feed the resulting audio bytes into your downstream system.

---

## Step 2: Authentication and security

### Server-side (Python backend): use your API key

The docs show connecting with an `Authorization: Bearer` header.

### Client-side (browser/mobile): use ephemeral tokens

If you ever connect from a client, xAI recommends minting **ephemeral tokens** from your backend via:

- `POST https://api.x.ai/v1/realtime/client_secrets`
- Example FastAPI endpoint shown in the docs (returns the JSON body from xAI).

---

## Step 3: Session configuration (voice, audio format, turn detection)

You configure the session using a `session.update` event. The guide defines these key knobs:

- **instructions**: your system prompt
- **voice**: Ara | Rex | Sal | Eve | Leo
- **turn_detection.type**:
  - `server_vad` = server does voice activity detection (auto turns)
  - `null` = you control when turns commit (manual)
- **audio**:
  - input/output formats: `audio/pcm` (PCM16 LE), or telephony `audio/pcmu` / `audio/pcma`
  - PCM sample rates supported include 8k–48k (doc lists exact values)

---

## Step 4: Realtime event protocol (what you'll actually handle)

### Sending user input

- Text: `conversation.item.create` with content: `[{"type":"input_text","text":"..."}]`
- Audio chunks: `input_audio_buffer.append` with base64 audio bytes
- Manual turn commit (only when `turn_detection.type` is `null`): `input_audio_buffer.commit`

### Receiving outputs

- Transcript deltas: `response.output_audio_transcript.delta`
- Audio deltas (base64): `response.output_audio.delta`
- Turn done: `response.done`
- Input transcription complete: `conversation.item.input_audio_transcription.completed`

**Note:** If the agent calls a tool, you may receive a `response.done` for the tool-call turn before the follow-up spoken response. After you send `function_call_output` and `response.create`, keep listening for a new `response.created` and `response.output_audio.delta` events.

---

## Step 5: Expose your ML pipeline as function tools

You declare tools inside `session.update`. The voice agent guide shows:

- `file_search` (Collections/RAG)
- `web_search`
- `x_search`
- function tools with JSON schema parameters

When the agent wants your pipeline, it emits `response.function_call_arguments.done`. Then you:

1. parse the arguments
2. run your function
3. send `conversation.item.create` with type: `function_call_output` and the same `call_id`
4. send `response.create` to continue

---

## Step 6: Python pipeline integration skeleton

This example:

- Connects to `wss://api.x.ai/v1/realtime`
- Configures voice/audio/VAD
- Registers a custom tool `classify_intent` that calls your pipeline
- Handles:
  - transcript deltas
  - audio deltas
  - function calls

> Notes:
> - The xAI docs use `additional_headers=...` in `websockets.connect(...)`. Some websockets versions use `extra_headers`. The helper below supports both.
> - This is "pipeline-first": it doesn't include microphone capture; you can feed audio from wherever your pipeline gets it (telephony frames, a stream, a file).

```python
import asyncio
import base64
import inspect
import json
import os
from typing import Any, Callable

import websockets  # pip install -U websockets
# Optional: numpy if you want float<->pcm conversion like the docs demonstrate
# pip install numpy

WS_URL = "wss://api.x.ai/v1/realtime"
XAI_API_KEY = os.environ["XAI_API_KEY"]


def b64encode_bytes(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")


def b64decode_bytes(s: str) -> bytes:
    return base64.b64decode(s)


# ---- Your AI/ML pipeline entrypoints (examples) ----
def classify_intent(text: str) -> dict[str, Any]:
    """
    Replace this with your real pipeline:
      - preprocess
      - embedding / model inference
      - postprocess
    """
    # Dummy logic:
    t = text.lower()
    if "refund" in t or "cancel" in t:
        return {"intent": "billing_refund", "confidence": 0.82}
    if "error" in t or "crash" in t:
        return {"intent": "technical_support", "confidence": 0.78}
    return {"intent": "general_question", "confidence": 0.60}


FUNCTION_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "classify_intent": classify_intent,
}


def ws_connect_kwargs(headers: dict[str, str]) -> dict[str, Any]:
    """
    Make this work across websockets versions:
    - docs show `additional_headers=...`
    - many installs use `extra_headers=...`
    """
    sig = inspect.signature(websockets.connect)
    if "additional_headers" in sig.parameters:
        return {"ssl": True, "additional_headers": headers}
    return {"ssl": True, "extra_headers": headers}


async def send_json(ws, event: dict[str, Any]) -> None:
    await ws.send(json.dumps(event))


async def handle_function_call(ws, event: dict[str, Any]) -> None:
    # Per docs, function-call events include name/call_id/arguments.  [oai_citation:24‡xAI](https://docs.x.ai/docs/guides/voice/agent)
    fn_name = event["name"]
    call_id = event["call_id"]
    args = json.loads(event["arguments"] or "{}")

    handler = FUNCTION_HANDLERS.get(fn_name)
    if handler is None:
        result = {"error": f"Unknown function '{fn_name}'"}
    else:
        # If your model inference is heavy, run it off the event loop:
        # result = await asyncio.to_thread(handler, **args)
        result = handler(**args)

    # Send tool result back to agent (function_call_output) and ask it to continue.  [oai_citation:25‡xAI](https://docs.x.ai/docs/guides/voice/agent)
    await send_json(ws, {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(result),
        },
    })
    await send_json(ws, {"type": "response.create"})


async def receiver_loop(ws) -> None:
    partial_transcript = []
    while True:
        raw = await ws.recv()
        event = json.loads(raw)

        etype = event.get("type")

        if etype == "session.updated":
            print("✅ session.updated")

        elif etype == "conversation.item.input_audio_transcription.completed":
            # Final user transcript for an audio turn.  [oai_citation:26‡xAI](https://docs.x.ai/docs/guides/voice/agent)
            print("👤 user transcript:", event.get("transcript"))

        elif etype == "response.output_audio_transcript.delta":
            # Assistant transcript delta.  [oai_citation:27‡xAI](https://docs.x.ai/docs/guides/voice/agent)
            delta = event.get("delta", "")
            partial_transcript.append(delta)
            print(delta, end="", flush=True)

        elif etype == "response.output_audio_transcript.done":
            print("\n📝 assistant transcript done:", "".join(partial_transcript))
            partial_transcript.clear()

        elif etype == "response.output_audio.delta":
            # Base64 PCM/G711 audio bytes.  [oai_citation:28‡xAI](https://docs.x.ai/docs/guides/voice/agent)
            audio_b64 = event.get("delta")
            audio_bytes = b64decode_bytes(audio_b64)
            # TODO: send to speaker/telephony sink, or store as raw audio stream.
            # e.g. audio_out_queue.put_nowait(audio_bytes)

        elif etype == "response.function_call_arguments.done":
            await handle_function_call(ws, event)

        elif etype == "response.done":
            # End of assistant turn.  [oai_citation:29‡xAI](https://docs.x.ai/docs/guides/voice/agent)
            print("✅ response.done")

        # Optional: log everything for debugging
        # else:
        #     print("event:", etype, event)


async def main():
    headers = {"Authorization": f"Bearer {XAI_API_KEY}"}
    async with websockets.connect(WS_URL, **ws_connect_kwargs(headers)) as ws:
        # 1) Configure session: voice, audio formats, VAD, tools.  [oai_citation:30‡xAI](https://docs.x.ai/docs/guides/voice/agent)
        session_update = {
            "type": "session.update",
            "session": {
                "instructions": (
                    "You are a voice assistant. "
                    "When helpful, call classify_intent(text) to route the request."
                ),
                "voice": "Ara",
                "turn_detection": {"type": "server_vad"},
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": 16000}},
                    "output": {"format": {"type": "audio/pcm", "rate": 16000}},
                },
                "tools": [
                    # Built-in tools (xAI executes these):  [oai_citation:31‡xAI](https://docs.x.ai/docs/guides/voice/agent)
                    {"type": "web_search"},
                    {"type": "x_search", "allowed_x_handles": ["xai"]},

                    # Your client-side tool (you execute this):  [oai_citation:32‡xAI](https://docs.x.ai/docs/guides/voice/agent)
                    {
                        "type": "function",
                        "name": "classify_intent",
                        "description": "Classify the user's intent for routing in our ML system.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "User text to classify"},
                            },
                            "required": ["text"],
                        },
                    },
                ],
            },
        }
        await send_json(ws, session_update)

        # 2) Start receiver loop
        recv_task = asyncio.create_task(receiver_loop(ws))

        # 3) Example: send a text turn, request a response with text+audio.
        # The docs show conversation.item.create for text, then response.create.  [oai_citation:33‡xAI](https://docs.x.ai/docs/guides/voice/agent)
        await send_json(ws, {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "I want a refund for my last invoice."}],
            },
        })
        await send_json(ws, {
            "type": "response.create",
            "response": {"modalities": ["text", "audio"]},
        })

        await recv_task  # run forever (or add your own shutdown conditions)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Step 7: Streaming audio in from your pipeline

If your pipeline already produces **PCM16 mono frames** (or G.711 frames), integration is straightforward:

1. base64 encode frame bytes
2. send `{ "type": "input_audio_buffer.append", "audio": "..." }`
3. If **manual turns** (`turn_detection.type = null`), call `input_audio_buffer.commit` then `response.create`
4. If **server_vad**, the server emits `speech_started` / `speech_stopped` and handles turn boundaries automatically

The docs include helper conversions for PCM16↔base64 and recommend keeping your processing sample rate aligned with session config.

Example: send one audio chunk

```python
# pcm16_chunk: bytes (little-endian int16 samples, mono)
await send_json(ws, {
    "type": "input_audio_buffer.append",
    "audio": b64encode_bytes(pcm16_chunk),
})
```

---

## Step 8: Where to plug in RAG + model inference

### Expose RAG as

- built-in `file_search` (Collections) if your docs are uploaded into xAI Collections
- or your own function tool that hits your vector DB (Pinecone/FAISS/pgvector/etc.)

### Expose ML inference as tools

Examples that work well as tool functions:

- `embed_text(text)` → embeddings
- `classify(text)` → intent/sentiment/compliance label
- `extract_entities(text)` → PII, order id, customer id
- `rank_candidates(query, candidates)` → reranking model
- `predict(features)` → classic ML scoring
- `run_sql(query)` (careful; enforce a safe allowlist)

Then your "pipeline step" function can:

- load models once at process start
- run inference in threads/processes
- return compact JSON (the model will turn it into speech)

---

## Step 9: Production notes that matter in real pipelines

### Concurrency and latency

- Keep the WebSocket receive loop **non-blocking**.
- Run heavy inference via:
  - `await asyncio.to_thread(...)` for CPU work
  - a worker pool (Celery/RQ/Ray Serve) for GPU models

### Backpressure

Audio deltas can arrive fast. Put audio bytes into a queue and have a separate consumer that plays/sends/stores them.

### Cost model

xAI's announcement says **$0.05 per minute of connection time** (per connected session). Design your pipeline to close idle sessions and avoid "hanging" sockets.

### Client architectures (if needed)

The general Voice guide links working examples (web, Twilio, WebRTC-bridge) and shows recommended architectures like:

- Browser ↔ Backend ↔ xAI WebSocket

---

## Quick checklist (so it doesn't break at 2am)

- Session config matches your audio reality (format + sample rate + mono)
- You handle:
  - `response.output_audio.delta` (bytes)
  - `response.output_audio_transcript.delta` (text)
  - function calls: `response.function_call_arguments.done`
- If you disable server VAD: you `commit` + `response.create`
- Your tool outputs are JSON-serializable and small
- You don't expose your API key to clients; use ephemeral tokens when needed

---

If you tell me what your pipeline looks like (batch inference? online GPU service? RAG over internal docs? telephony audio frames?), I can adapt the skeleton into a concrete architecture (queues, worker pool, tool schemas, and the exact "tool router" layout) without changing the core protocol above.
