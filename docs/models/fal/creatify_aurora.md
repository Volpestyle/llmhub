# Creatify Aurora

> Generate high fidelity, studio quality videos of your avatar speaking or singing using the Aurora from Creatify team!


Your request will cost **$0.10** per video second for **480p**, **$0.14** per video second for **720p** generation. Video seconds will be **rounded upwards**, meaning that a generation with an output video of 9.4 seconds will be billed as a 10 second video.

## Overview

- **Endpoint**: `https://fal.run/fal-ai/creatify/aurora`
- **Model ID**: `fal-ai/creatify/aurora`
- **Category**: image-to-video
- **Kind**: inference
**Tags**: lipsync, image-to-video



## API Information

This model can be used via our HTTP API or more conveniently via our client libraries.
See the input and output schema below, as well as the usage examples.


### Input Schema

The API accepts the following input parameters:


- **`image_url`** (`string`, _required_):
  The URL of the image file to be used for video generation.
  - Examples: "https://storage.googleapis.com/falserverless/example_inputs/creatify/aurora/input_.png"

- **`audio_url`** (`string`, _required_):
  The URL of the audio file to be used for video generation.
  - Examples: "https://storage.googleapis.com/falserverless/example_inputs/creatify/aurora/input.wav"

- **`prompt`** (`string`, _optional_):
  A text prompt to guide the video generation process.
  - Examples: "4K studio interview, medium close-up (shoulders-up crop). Solid light-grey seamless backdrop, uniform soft key-light—no lighting change. Presenter faces lens, steady eye-contact. Hands remain below frame, body perfectly still. Ultra-sharp."

- **`guidance_scale`** (`float`, _optional_):
  Guidance scale to be used for text prompt adherence. Default value: `1`
  - Default: `1`
  - Range: `0` to `5`

- **`audio_guidance_scale`** (`float`, _optional_):
  Guidance scale to be used for audio adherence. Default value: `2`
  - Default: `2`
  - Range: `0` to `5`

- **`resolution`** (`ResolutionEnum`, _optional_):
  The resolution of the generated video. Default value: `"720p"`
  - Default: `"720p"`
  - Options: `"480p"`, `"720p"`



**Required Parameters Example**:

```json
{
  "image_url": "https://storage.googleapis.com/falserverless/example_inputs/creatify/aurora/input_.png",
  "audio_url": "https://storage.googleapis.com/falserverless/example_inputs/creatify/aurora/input.wav"
}
```

**Full Example**:

```json
{
  "image_url": "https://storage.googleapis.com/falserverless/example_inputs/creatify/aurora/input_.png",
  "audio_url": "https://storage.googleapis.com/falserverless/example_inputs/creatify/aurora/input.wav",
  "prompt": "4K studio interview, medium close-up (shoulders-up crop). Solid light-grey seamless backdrop, uniform soft key-light—no lighting change. Presenter faces lens, steady eye-contact. Hands remain below frame, body perfectly still. Ultra-sharp.",
  "guidance_scale": 1,
  "audio_guidance_scale": 2,
  "resolution": "720p"
}
```


### Output Schema

The API returns the following output format:

- **`video`** (`VideoFile`, _required_):
  The generated video file.
  - Examples: {"file_name":"output.mp4","content_type":"video/mp4","url":"https://storage.googleapis.com/falserverless/example_outputs/creatify/aurora/output.mp4"}



**Example Response**:

```json
{
  "video": {
    "file_name": "output.mp4",
    "content_type": "video/mp4",
    "url": "https://storage.googleapis.com/falserverless/example_outputs/creatify/aurora/output.mp4"
  }
}
```


## Usage Examples

### cURL

```bash
curl --request POST \
  --url https://fal.run/fal-ai/creatify/aurora \
  --header "Authorization: Key $FAL_KEY" \
  --header "Content-Type: application/json" \
  --data '{
     "image_url": "https://storage.googleapis.com/falserverless/example_inputs/creatify/aurora/input_.png",
     "audio_url": "https://storage.googleapis.com/falserverless/example_inputs/creatify/aurora/input.wav"
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
    "fal-ai/creatify/aurora",
    arguments={
        "image_url": "https://storage.googleapis.com/falserverless/example_inputs/creatify/aurora/input_.png",
        "audio_url": "https://storage.googleapis.com/falserverless/example_inputs/creatify/aurora/input.wav"
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

const result = await fal.subscribe("fal-ai/creatify/aurora", {
  input: {
    image_url: "https://storage.googleapis.com/falserverless/example_inputs/creatify/aurora/input_.png",
    audio_url: "https://storage.googleapis.com/falserverless/example_inputs/creatify/aurora/input.wav"
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

- [Model Playground](https://fal.ai/models/fal-ai/creatify/aurora)
- [API Documentation](https://fal.ai/models/fal-ai/creatify/aurora/api)
- [OpenAPI Schema](https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=fal-ai/creatify/aurora)

### fal.ai Platform

- [Platform Documentation](https://docs.fal.ai)
- [Python Client](https://docs.fal.ai/clients/python)
- [JavaScript Client](https://docs.fal.ai/clients/javascript)
