## Basic model info

Model name: bytedance/latentsync
Model description: LatentSync: generate high-quality lip sync animations

$0.072 to run on Replicate, or 13 runs per $1
## Model inputs

- video (optional): Input video (string)
- audio (optional): Input audio to  (string)
- guidance_scale (optional): Guidance scale (number)
- seed (optional): Set to 0 for Random seed (integer)


## Model output schema

{
  "type": "string",
  "title": "Output",
  "format": "uri"
}

If the input or output schema includes a format of URI, it is referring to a file.


## Example inputs and outputs

Use these example outputs to better understand the types of inputs the model accepts, and the types of outputs the model returns:

### Example (https://replicate.com/p/n77pj48a9hrm80cnc7qv584apw)

#### Input

```json
{
  "seed": 0,
  "audio": "https://replicate.delivery/pbxt/MGZuENopzAwWcpFsZ7SwoZ7itP4gvqasswPeEJwbRHTxtkwF/demo2_audio.wav",
  "video": "https://replicate.delivery/pbxt/MGZuEgzJZh6avv1LDEMppJZXLP9avGXqRuH7iAb7MBAz0Wu4/demo2_video.mp4",
  "guidance_scale": 1
}
```

#### Output

```json
"https://replicate.delivery/xezq/M5PsudldGc6LMRB6zlgmDeOtgc2VcW9fzhzb7qeRdPVeX2URB/output-8992e038.mp4"
```


### Example (https://replicate.com/p/2xpg1fw5fdrmc0cnc7s8t5jp7g)

#### Input

```json
{
  "seed": 0,
  "audio": "https://replicate.delivery/pbxt/MGZqqL5XKmosNskbrueTdmzZg8zaTrKKAXDbf8XnVfnZYbK0/demo1_audio.wav",
  "video": "https://replicate.delivery/pbxt/MGZqpusSGKYxoRz4rzfcTToV3Ubj7g4MItu7TBIxVzNH4JSE/demo1_video.mp4",
  "guidance_scale": 1
}
```

#### Output

```json
"https://replicate.delivery/xezq/b4S3LkuxZVpwGlv6xtS1ctsZTqGRheQ6CXD7sW4ZOSFa1mKKA/video_out.mp4"
```


### Example (https://replicate.com/p/3h5r1vk4vxrmc0cnc7tbm5p9qm)

#### Input

```json
{
  "seed": 0,
  "audio": "https://replicate.delivery/pbxt/MGZz9C8etlAJvI4AHx2N35GR2e7Pgjm5Gu8v3QwdQhxwvKgx/demo3_audio.wav",
  "video": "https://replicate.delivery/pbxt/MGZz8viAVAEEousqPFv7A4zDvIQY26a3tg2JjwnxILppKxgJ/demo3_video.mp4",
  "guidance_scale": 1
}
```

#### Output

```json
"https://replicate.delivery/xezq/aK3JC1amtSKgHZ6B4DLUMJDUOe5QeF3fnFWq4DViQqCPcbqoA/video_out.mp4"
```


### Example (https://replicate.com/p/725sr45pwdrme0cnm7et7kt62r)

#### Input

```json
{
  "seed": 0,
  "audio": "https://replicate.delivery/pbxt/MGZuENopzAwWcpFsZ7SwoZ7itP4gvqasswPeEJwbRHTxtkwF/demo2_audio.wav",
  "video": "https://replicate.delivery/pbxt/MGZuEgzJZh6avv1LDEMppJZXLP9avGXqRuH7iAb7MBAz0Wu4/demo2_video.mp4",
  "guidance_scale": 1
}
```

#### Output

```json
"https://replicate.delivery/xezq/GWI5d7MZWTqhPdQ7CJOOSRxLz17p1VvkUKevLWrexuU0cTZUA/output-5e1d1818.mp4"
```


## Model readme

> ## About
> 
> This is a Cog implementation of [bytedance/LatentSync](https://github.com/bytedance/LatentSync). Supports mp4 for video input and mp3/aac/wav/m4a audio files for the audio input
> 
> _Note: Do not include spaces in your filenames_
> 
> ## Abstract
> 
> We present LatentSync, an end-to-end lip sync framework based on audio conditioned latent diffusion models without any intermediate motion representation, diverging from previous diffusion-based lip sync methods based on pixel space diffusion or two-stage generation. Our framework can leverage the powerful capabilities of Stable Diffusion to directly model complex audio-visual correlations. Additionally, we found that the diffusion-based lip sync methods exhibit inferior temporal consistency due to the inconsistency in the diffusion process across different frames. We propose Temporal REPresentation Alignment (TREPA) to enhance temporal consistency while preserving lip-sync accuracy. TREPA uses temporal representations extracted by large-scale self-supervised video models to align the generated frames with the ground truth frames.
> 
> ## 🏗️ Framework
> 
> <p align="center">
> <img src="https://github.com/lucataco/cog-LatentSync/raw/main/assets/framework.png" width=100%>
> <p>
> 
> LatentSync uses the Whisper to convert melspectrogram into audio embeddings, which are then integrated into the U-Net via cross-attention layers. The reference and masked frames are channel-wise concatenated with noised latents as the input of U-Net. In the training process, we use one-step method to get estimated clean latents from predicted noises, which are then decoded to obtain the estimated clean frames. The TREPA, LPIPS and SyncNet loss are added in the pixel space.