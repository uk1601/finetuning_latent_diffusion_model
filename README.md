# Flickr30k Fine-tuned Stable Diffusion Model - Usage Guide

This repository contains a U-Net component fine-tuned on the Flickr30k dataset using the Stable Diffusion v1-4 model. Only the U-Net was updated during training; the CLIP text encoder and VAE components remain frozen and use the original pretrained weights.

---

## Model Details

- **Model Name**: `flickr30k-fine-tuned-unet`
- **Repository**: `dinaaaaaa/flickr30k-fine-tuned-unet`
- **Base Model**: `CompVis/stable-diffusion-v1-4`
- **Fine-tuned Component**: U-Net (CLIP and VAE remain unchanged)

---

## Backend Usage

```python

from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch

def load_model():
    # Load base model
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16  # Use fp16 for faster inference and less memory
    )
    
    # Replace with fine-tuned UNet
    pipe.unet = UNet2DConditionModel.from_pretrained(
        "dinaaaaaa/flickr30k-fine-tuned-unet",
        torch_dtype=torch.float16
    )
    
    # Move to GPU and enable memory optimization
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()  # Reduces memory usage
    
    return pipe

def generate_image(pipe, prompt, negative_prompt=None, guidance_scale=7.5, num_inference_steps=50):
    """
    Generate an image from a text prompt
    
    Args:
        pipe: StableDiffusionPipeline object
        prompt: Text description of the desired image
        negative_prompt: Text description of what to avoid (optional)
        guidance_scale: How strictly to follow the prompt (7-9 is a good range)
        num_inference_steps: Number of denoising steps (more = higher quality but slower)
        
    Returns:
        PIL Image object
    """
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]
    
    return image

# Example usage
pipe = load_model()
image = generate_image(pipe, "a person walking on the beach at sunset")
image.save("generated_image.png")
```

---

## API Implementation Example

```python

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from io import BytesIO
import base64

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = None
    guidance_scale: float = 7.5
    num_inference_steps: int = 50

# Load model at startup
@app.on_event("startup")
async def startup_event():
    global pipe
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16
    )
    
    pipe.unet = UNet2DConditionModel.from_pretrained(
        "dinaaaaaa/flickr30k-fine-tuned-unet",
        torch_dtype=torch.float16
    )
    
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()

@app.post("/api/generate")
async def generate_image(request: GenerationRequest):
    try:
        image = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps
        ).images[0]
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {"image": img_str}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Frontend Usage Example

```javascript

import React, { useState } from 'react';
import './App.css';

function App() {
  const [prompt, setPrompt] = useState('');
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const generateImage = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });
      
      if (!response.ok) {
        throw new Error('Image generation failed');
      }
      
      const data = await response.json();
      setImage(data.image);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Text-to-Image Generator</h1>
      <div className="input-container">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter a description..."
        />
        <button onClick={generateImage} disabled={loading || !prompt}>
          {loading ? 'Generating...' : 'Generate Image'}
        </button>
      </div>
      
      {error && <div className="error">{error}</div>}
      
      {image && (
        <div className="image-container">
          <img src={`data:image/png;base64,${image}`} alt={prompt} />
        </div>
      )}
    </div>
  );
}

export default App;
```

## Notes and Tips

1. **Memory Requirements**: The model requires ~6–8GB of GPU memory for inference.

2. **Optimization Options**:
   - Use `torch.float16` for faster inference
   - Enable `attention_slicing` to reduce memory usage
   - For lower memory GPUs, add `enable_xformers_memory_efficient_attention()` if xformers is installed

3. **Model Characteristics**:
   - The model was trained on Flickr30k dataset at 128x128 resolution
   - It performs better on photorealistic descriptions
   - It works well with descriptive prompts that match the style of Flickr30k captions

4. **Advanced Configuration**:
   - Try different `guidance_scale` values: 7–8 for balanced results, higher for more prompt adherence
   - Adjust `num_inference_steps`: higher values (50–100) for quality, lower (20–30) for speed
