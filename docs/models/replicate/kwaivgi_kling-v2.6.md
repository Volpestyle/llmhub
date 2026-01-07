## Basic model info

Model name: kwaivgi/kling-v2.6
Model description: Kling 2.6 Pro: Top-tier image-to-video with cinematic visuals, fluid motion, and native audio generation

Criteria

without audio - $0.07 per second of output video or around 14 seconds for $1
with audio - $0.14 per second of output video or around 71 seconds for $10
## Model inputs

- prompt (required): Text prompt for video generation (string)
- negative_prompt (optional): Things you do not want to see in the video (string)
- start_image (optional): First frame of the video (string)
- aspect_ratio (optional): Aspect ratio of the video. Ignored if start_image is provided. (string)
- duration (optional): Duration of the video in seconds (integer)
- generate_audio (optional): Generate audio for the video. When enabled, the model will create synchronized audio based on the video content. (boolean)


## Model output schema

{
  "type": "string",
  "title": "Output",
  "format": "uri"
}

If the input or output schema includes a format of URI, it is referring to a file.


## Example inputs and outputs

Use these example outputs to better understand the types of inputs the model accepts, and the types of outputs the model returns:

### Example (https://replicate.com/p/r4gjh9x961rmw0cvetdrrwz9v0)

#### Input

```json
{
  "prompt": "A cinematic, low-angle tracking shot follows a cyclist from behind as they weave through busy New York City traffic. The camera then smoothly orbits around to the front, capturing the cyclist's determined expression. The cyclist is a young man with a mustache, wearing a white t-shirt, black shorts, white socks, and a blue baseball cap. He is riding a sleek black fixed-gear bicycle. The city streets are filled with iconic yellow taxis and various modern cars, with towering skyscrapers lining the background. Bright, natural sunlight creates sharp contrasts and highlights, enhancing the fast-paced, urban atmosphere. The motion is fluid and dynamic, emphasizing the speed and agility of the cyclist amidst the metropolitan chaos.",
  "duration": 5,
  "aspect_ratio": "16:9",
  "generate_audio": true,
  "negative_prompt": ""
}
```

#### Output

```json
"https://replicate.delivery/xezq/MRjiFeEoaLTsI6YuhqqbVbKid6EO2NQA9MkGuKfE4WZht14VA/tmpt21qn3q_.mp4"
```


### Example (https://replicate.com/p/80cxa81vddrmy0cvetgapjw408)

#### Input

```json
{
  "prompt": "A high-octane cinematic action sequence featuring a man in a black suit desperately fleeing through a series of opulent, classically designed hallways. The camera follows closely behind him in a shaky, first-person-style chase perspective, heightening the tension. As he runs, bullets impact the ornate walls and furniture around him, sending splinters of wood and plaster flying into the air. The man navigates through arched doorways and past large, dark-wood dressers, eventually entering a grand, circular ballroom with a polished marble floor and large windows revealing a twilight blue sky. Without hesitation, he sprints towards a set of double glass doors, crashing through them in a flurry of shattered glass as he leaps out into the open air. The final shot is a dramatic view from below as he sails through the dim evening sky, framed against the silhouettes of distant trees",
  "duration": 5,
  "aspect_ratio": "16:9",
  "generate_audio": true,
  "negative_prompt": ""
}
```

#### Output

```json
"https://replicate.delivery/xezq/oFbcwE74jGZUPdWRhHfjCHRF1yFeHUtKzPahFhdan0WNy14VA/tmpcfjcw14_.mp4"
```


### Example (https://replicate.com/p/y38bgbfwdnrmt0cvetht8fsfkw)

#### Input

```json
{
  "prompt": "A woman walks down a rain-slicked neon street at night, camera slowly tracking behind her. She stops and turns to face the camera, saying \"Let's begin.\" Ambient sound of rain on pavement, distant traffic, soft footsteps",
  "duration": 5,
  "aspect_ratio": "16:9",
  "generate_audio": true,
  "negative_prompt": ""
}
```

#### Output

```json
"https://replicate.delivery/xezq/XyQhfhBIeAi6sELtcv71L1bwrfTsRsaGPagoRuus748hsrxrA/tmp6z_n3c1y.mp4"
```


## Model readme

> # Kling v2.6
> 
> Generate cinematic videos with synchronized audio from text prompts or images. This model creates video and sound together in a single pass—dialogue, ambient effects, and motion all aligned without separate audio production.
> 
> ## What it does
> 
> Kling v2.6 transforms text descriptions or static images into short video clips with native audio. The model generates speech, sound effects, and ambient audio that match the visuals frame-by-frame, so you get lip-synced dialogue and scene-appropriate sound without manual editing.
> 
> You can create videos up to 10 seconds long at 1080p resolution in multiple aspect ratios. The model handles both realistic and stylized content, though it's strongest with photorealistic scenes.
> 
> ## How to use it
> 
> The model works with two input types:
> 
> **Text to video**: Describe what you want to see and hear. The model generates both visuals and audio from your description.
> 
> **Image to video**: Upload a still image and add a text prompt describing the motion and audio you want. The model animates your image with synchronized sound.
> 
> ### Writing effective prompts
> 
> Good prompts guide both the visual content and the audio. Structure your description to include:
> 
> - **Scene setting**: Where and when the action happens, lighting conditions
> - **Subject details**: What characters or objects appear, how they look
> - **Motion**: What happens, how things move, camera behavior
> - **Audio**: Dialogue with quotation marks, ambient sounds, sound effects
> 
> Example: `A woman walks down a rain-slicked neon street at night, camera slowly tracking behind her. She stops and turns to face the camera, saying "Let's begin." Ambient sound of rain on pavement, distant traffic, soft footsteps.`
> 
> For dialogue, put the spoken text in quotes and the model will generate matching lip sync. You can specify voice characteristics like "warm female voice" or "confident male narrator."
> 
> Describe ambient sounds and effects explicitly: "coffee shop chatter, espresso machine hissing, rain on windows" gives better results than just "background noise."
> 
> ### Parameters
> 
> - **Duration**: Choose 5 or 10 seconds per generation
> - **Aspect ratio**: 16:9 (horizontal), 9:16 (vertical), or 1:1 (square)
> - **Audio**: Toggle native audio on or off
> - **Negative prompt**: Specify what to exclude from the generation
> 
> ## What it's good for
> 
> The model works well for:
> 
> - Marketing videos with voiceover narration
> - Social media content with dialogue
> - Product demonstrations with sound effects
> - Character animations with speech
> - Cinematic sequences with ambient audio
> 
> The native audio makes it particularly useful when you need speech synchronized with character mouth movements, or when ambient sound needs to match on-screen action.
> 
> ## Limitations
> 
> - Maximum 10 seconds per generation
> - Audio works best in English and Chinese
> - Character consistency can vary across multiple generations
> - Complex physics interactions may not look fully natural
> - Text overlays in the output can be distorted
> 
> For projects longer than 10 seconds, you'll need to generate multiple clips and edit them together.
> 
> ## Technical details
> 
> The model outputs 1080p video with embedded audio at standard frame rates. It supports vertical video formats for platforms like TikTok and Instagram Reels.
> 
> Audio generation includes multiple layers: dialogue or narration, ambient environmental sound, and specific sound effects. These layers are mixed together in the output.
> 
> ---
> 
> You can try this model on the Replicate Playground at replicate.com/playground