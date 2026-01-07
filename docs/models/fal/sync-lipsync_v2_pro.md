# Sync Lipsync

> Generate high-quality realistic lipsync animations from audio while preserving unique details like natural teeth and unique facial features using the state-of-the-art Sync Lipsync 2 Pro model.


## Overview

- **Endpoint**: `https://fal.run/fal-ai/sync-lipsync/v2/pro`
- **Model ID**: `fal-ai/sync-lipsync/v2/pro`
- **Category**: video-to-video
- **Kind**: inference
**Tags**: animation, lip sync, high-quality, 



## API Information

This model can be used via our HTTP API or more conveniently via our client libraries.
See the input and output schema below, as well as the usage examples.


### Input Schema

The API accepts the following input parameters:


- **`video_url`** (`string`, _required_):
  URL of the input video
  - Examples: "https://storage.googleapis.com/falserverless/example_inputs/sync_v2_pro_video_input.mp4"

- **`audio_url`** (`string`, _required_):
  URL of the input audio
  - Examples: "https://storage.googleapis.com/falserverless/example_inputs/sync_v2_pro_audio_input.mp3"

- **`sync_mode`** (`SyncModeEnum`, _optional_):
  Lipsync mode when audio and video durations are out of sync. Default value: `"cut_off"`
  - Default: `"cut_off"`
  - Options: `"cut_off"`, `"loop"`, `"bounce"`, `"silence"`, `"remap"`



**Required Parameters Example**:

```json
{
  "video_url": "https://storage.googleapis.com/falserverless/example_inputs/sync_v2_pro_video_input.mp4",
  "audio_url": "https://storage.googleapis.com/falserverless/example_inputs/sync_v2_pro_audio_input.mp3"
}
```

**Full Example**:

```json
{
  "video_url": "https://storage.googleapis.com/falserverless/example_inputs/sync_v2_pro_video_input.mp4",
  "audio_url": "https://storage.googleapis.com/falserverless/example_inputs/sync_v2_pro_audio_input.mp3",
  "sync_mode": "cut_off"
}
```


### Output Schema

The API returns the following output format:

- **`video`** (`File`, _required_):
  The generated video
  - Examples: {"url":"https://storage.googleapis.com/falserverless/example_outputs/sync_v2_pro_output.mp4"}



**Example Response**:

```json
{
  "video": {
    "url": "https://storage.googleapis.com/falserverless/example_outputs/sync_v2_pro_output.mp4"
  }
}
```


## Usage Examples

### cURL

```bash
curl --request POST \
  --url https://fal.run/fal-ai/sync-lipsync/v2/pro \
  --header "Authorization: Key $FAL_KEY" \
  --header "Content-Type: application/json" \
  --data '{
     "video_url": "https://storage.googleapis.com/falserverless/example_inputs/sync_v2_pro_video_input.mp4",
     "audio_url": "https://storage.googleapis.com/falserverless/example_inputs/sync_v2_pro_audio_input.mp3"
   }'
```

### Python

Ensure you have the Python client installed:

```bash
pip install fal-client
```

Then use the API client to make requests:

```python
import fal_client

def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
           print(log["message"])

result = fal_client.subscribe(
    "fal-ai/sync-lipsync/v2/pro",
    arguments={
        "video_url": "https://storage.googleapis.com/falserverless/example_inputs/sync_v2_pro_video_input.mp4",
        "audio_url": "https://storage.googleapis.com/falserverless/example_inputs/sync_v2_pro_audio_input.mp3"
    },
    with_logs=True,
    on_queue_update=on_queue_update,
)
print(result)
```

### JavaScript

Ensure you have the JavaScript client installed:

```bash
npm install --save @fal-ai/client
```

Then use the API client to make requests:

```javascript
import { fal } from "@fal-ai/client";

const result = await fal.subscribe("fal-ai/sync-lipsync/v2/pro", {
  input: {
    video_url: "https://storage.googleapis.com/falserverless/example_inputs/sync_v2_pro_video_input.mp4",
    audio_url: "https://storage.googleapis.com/falserverless/example_inputs/sync_v2_pro_audio_input.mp3"
  },
  logs: true,
  onQueueUpdate: (update) => {
    if (update.status === "IN_PROGRESS") {
      update.logs.map((log) => log.message).forEach(console.log);
    }
  },
});
console.log(result.data);
console.log(result.requestId);
```


## Additional Resources

### Documentation

- [Model Playground](https://fal.ai/models/fal-ai/sync-lipsync/v2/pro)
- [API Documentation](https://fal.ai/models/fal-ai/sync-lipsync/v2/pro/api)
- [OpenAPI Schema](https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=fal-ai/sync-lipsync/v2/pro)

### fal.ai Platform

- [Platform Documentation](https://docs.fal.ai)
- [Python Client](https://docs.fal.ai/clients/python)
- [JavaScript Client](https://docs.fal.ai/clients/javascript)