## Basic model info

Model name: pixverse/lipsync
Model description: Generate realistic lipsync animations from audio for high-quality synchronization

$0.04 per second of output video
or 25 seconds for $1
## Model inputs

- video (required): Video file to upload to PixVerse as media (string)
- audio (required): Audio file to upload to PixVerse as media (string)


## Model output schema

{
  "type": "string",
  "title": "Output",
  "format": "uri"
}

If the input or output schema includes a format of URI, it is referring to a file.


## Example inputs and outputs

Use these example outputs to better understand the types of inputs the model accepts, and the types of outputs the model returns:

### Example (https://replicate.com/p/91jhw5yzp5rme0cse8hty4qj5g)

#### Input

```json
{
  "audio": "https://replicate.delivery/pbxt/Nl3OAXz8mkiA0KFCoL1m7d6hWdrEGlXjN3hLFPpM7giMlvwi/every-person.mp3",
  "video": "https://replicate.delivery/pbxt/Nl3OB4gPYVqgF271A7Tk2Fm8ACFnNWmcnju4o1pIHlCD0V4B/woman-cafe.mp4"
}
```

#### Output

```json
"https://replicate.delivery/xezq/IFUeAG1KsQWCHKURDyq182XIeJDVq5l0qVNGkksLttbvPyXVA/tmp__afyey0.mp4"
```


### Example (https://replicate.com/p/mtn4cyy4a9rme0csebxv1jq9zg)

#### Input

```json
{
  "audio": "https://replicate.delivery/pbxt/Nl6zanmgCGxtFBSEisZDBDVDEBzaYp5k6ZwYYNPP07A1mkgS/growth.mp3",
  "video": "https://replicate.delivery/pbxt/Nl6zasIdckyfnApP8Mltr6Obykv6puAuKKSn7zlsTrW7rP23/business-man.mp4"
}
```

#### Output

```json
"https://replicate.delivery/xezq/0k3hoAL4195uOFZxfObFhuYhivMyfzj63tnz7LpXeUQdZrvqA/tmpr4rehw37.mp4"
```


### Example (https://replicate.com/p/kv4snc4525rmc0csec1808mzwr)

#### Input

```json
{
  "audio": "https://replicate.delivery/pbxt/Nl76bQBoS5O7IJ6XaI6SkWdkMgey89UwzqZCTaGXGGgcp2Em/crown.mp3",
  "video": "https://replicate.delivery/pbxt/Nl76aYYgpAdV6114HKeM5IZJtLeQwF5xBgaXxSIxWFPDPlE0/woman-portrait.mp4"
}
```

#### Output

```json
"https://replicate.delivery/xezq/EUoLquC0YAKaIB1QuQp6zdxz9pa6NmLo3mVqqNiRN9p8c9VF/tmpte7skn1j.mp4"
```


## Model readme

> No readme available for this model.