# backend_app.py
import torch
import os
import gc
import io
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
import uvicorn
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
BASE_MODEL_ID = "CompVis/stable-diffusion-v1-4"
FINE_TUNED_UNET_ID = "dinaaaaaa/flickr30k-fine-tuned-unet" # Or local path
DEVICE = "cpu" # Default, will be updated
TORCH_DTYPE = torch.float32 # Default, will be updated
ENABLE_ATTENTION_SLICING = False # Default, will be updated

# Inference parameters (can be overridden by request)
DEFAULT_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_SEED = 42 # Or use -1 for random seed per request

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
        # Optionally add logging.FileHandler("backend.log")
    ]
)
logger = logging.getLogger(__name__)

# --- Model Loading and State ---
# Use a dictionary to hold models/components in app state
app_state = {}

# Function to select device
def get_device():
    global TORCH_DTYPE, ENABLE_ATTENTION_SLICING
    if torch.cuda.is_available():
        selected_device = "cuda"
        TORCH_DTYPE = torch.float16 # Keep float16 for CUDA if desired
        ENABLE_ATTENTION_SLICING = True
        logger.info(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    elif torch.backends.mps.is_available():
        selected_device = "mps"
        # !!! --- DEBUGGING STEP --- !!!
        # Force float32 on MPS to check for stability issues
        TORCH_DTYPE = torch.float32
        # ENABLE_ATTENTION_SLICING = True # May not be needed/as effective with float32
        ENABLE_ATTENTION_SLICING = False # Disable slicing when debugging with float32
        # !!! --- END DEBUGGING STEP --- !!!
        logger.info(f"Using Apple Silicon MPS (Forced dtype: {TORCH_DTYPE})") # Log the forced dtype
    else:
        selected_device = "cpu"
        TORCH_DTYPE = torch.float32
        ENABLE_ATTENTION_SLICING = False
        logger.info("CUDA and MPS not available, using CPU")
    return selected_device

# Helper to free memory
def free_memory():
    gc.collect()
    if app_state.get("device", "").startswith("cuda"):
        torch.cuda.empty_cache()
    logger.debug("Memory freed.")

# Function to load all models
def load_models():
    device = app_state["device"]
    logger.info(f"Starting model loading process on device '{device}' with dtype {TORCH_DTYPE}...")
    free_memory()

    try:
        # --- Load Shared Components ---
        logger.info("Loading shared components...")
        app_state["tokenizer"] = CLIPTokenizer.from_pretrained(BASE_MODEL_ID, subfolder="tokenizer")
        app_state["text_encoder"] = CLIPTextModel.from_pretrained(
            BASE_MODEL_ID, subfolder="text_encoder", torch_dtype=TORCH_DTYPE
        ).to(device)
        app_state["vae"] = AutoencoderKL.from_pretrained(
            BASE_MODEL_ID, subfolder="vae", torch_dtype=TORCH_DTYPE
        ).to(device)
        app_state["scheduler"] = DDPMScheduler.from_pretrained(BASE_MODEL_ID, subfolder="scheduler")
        logger.info("Shared components loaded.")
        free_memory()

        # --- Load U-Nets ---
        logger.info("Loading U-Nets...")
        app_state["original_unet"] = UNet2DConditionModel.from_pretrained(
            BASE_MODEL_ID, subfolder="unet", torch_dtype=TORCH_DTYPE
        ).to(device)
        logger.info("Original UNet loaded.")

        try:
            app_state["fine_tuned_unet"] = UNet2DConditionModel.from_pretrained(
                FINE_TUNED_UNET_ID, torch_dtype=TORCH_DTYPE
            ).to(device)
            logger.info("Fine-tuned UNet loaded.")
        except Exception as e:
            logger.error(f"Failed to load fine-tuned UNet from {FINE_TUNED_UNET_ID}: {e}", exc_info=True)
            app_state["fine_tuned_unet"] = None # Mark as unavailable

        free_memory()

        # --- Assemble Pipelines ---
        logger.info("Assembling pipelines...")
        shared_components = {
            "vae": app_state["vae"],
            "text_encoder": app_state["text_encoder"],
            "tokenizer": app_state["tokenizer"],
            "scheduler": app_state["scheduler"],
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
        }

        if app_state.get("original_unet"):
            app_state["original_pipe"] = StableDiffusionPipeline(
                unet=app_state["original_unet"], **shared_components
            )
            if ENABLE_ATTENTION_SLICING:
                app_state["original_pipe"].enable_attention_slicing()
            logger.info("Original pipeline assembled.")
        else:
             app_state["original_pipe"] = None
             logger.error("Original UNet not loaded, cannot assemble original pipeline.")

        if app_state.get("fine_tuned_unet"):
             app_state["fine_tuned_pipe"] = StableDiffusionPipeline(
                unet=app_state["fine_tuned_unet"], **shared_components
            )
             if ENABLE_ATTENTION_SLICING:
                app_state["fine_tuned_pipe"].enable_attention_slicing()
             logger.info("Fine-tuned pipeline assembled.")
        else:
             app_state["fine_tuned_pipe"] = None
             logger.warning("Fine-tuned UNet not loaded or failed to load, fine-tuned pipeline unavailable.")

        logger.info("Model loading and pipeline assembly complete.")

    except Exception as e:
        logger.error(f"Critical error during model loading: {e}", exc_info=True)
        # Mark pipelines as unavailable if loading failed fundamentally
        app_state["original_pipe"] = None
        app_state["fine_tuned_pipe"] = None

# --- FastAPI Lifespan Management ---
# Use lifespan context manager for startup/shutdown logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Backend starting up...")
    app_state["device"] = get_device()
    # Run model loading in a separate thread to avoid blocking startup if it's long
    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(load_models)
    executor.shutdown(wait=False) # Don't wait here, let it load in background
    logger.info("Model loading initiated in background.")
    yield
    # Shutdown
    logger.info("Backend shutting down...")
    free_memory()
    # Cleanup any resources if needed

app = FastAPI(lifespan=lifespan)

# --- Request/Response Models ---
class GenerateRequest(BaseModel):
    prompt: str
    model_type: str = 'fine-tuned' # 'original' or 'fine-tuned'
    num_inference_steps: int = DEFAULT_INFERENCE_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    seed: int = DEFAULT_SEED # Use -1 or None for random seed

# --- Image Generation Logic (Synchronous) ---
# This function runs the actual inference and needs to be run in a thread
def generate_image_sync(request: GenerateRequest):
    logger.info(f"Received generation request: prompt='{request.prompt}', model='{request.model_type}'")

    # Select pipeline
    if request.model_type == 'original':
        pipe = app_state.get("original_pipe")
        model_name = "Original"
    elif request.model_type == 'fine-tuned':
        pipe = app_state.get("fine_tuned_pipe")
        model_name = "Fine-tuned"
    else:
        raise ValueError("Invalid model_type specified.")

    if pipe is None:
        logger.error(f"{model_name} pipeline is not available.")
        raise RuntimeError(f"{model_name} model pipeline is not loaded or failed to load.")

    # Prepare generator
    if request.seed is None or request.seed < 0:
         generator = None # Use random seed
         logger.info("Using random seed for generation.")
    else:
         generator = torch.Generator(device=app_state["device"]).manual_seed(request.seed)
         logger.info(f"Using fixed seed: {request.seed}")

    try:
        logger.info(f"Starting generation with {model_name} model...")
        with torch.inference_mode():
            output = pipe(
                prompt=request.prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
                output_type='pil' # Ensure PIL output
            )
        image = output.images[0]

        # Check for black image (basic check)
        if image and image.mode == 'RGB' and image.getextrema() == ((0, 0), (0, 0), (0, 0)):
             logger.warning(f"{model_name} model generated a potentially black image.")

        logger.info(f"Generation successful with {model_name} model.")

        # Encode image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)  # Go to the beginning of the BytesIO buffer

        return img_byte_arr

    except Exception as e:
        logger.error(f"Error during image generation with {model_name} model: {e}", exc_info=True)
        # Re-raise a more specific error or handle based on type
        if "out of memory" in str(e).lower():
             free_memory()
             raise RuntimeError(f"CUDA Out of Memory error during generation. Try a shorter prompt or restart.") from e
        else:
             raise RuntimeError(f"An unexpected error occurred during generation: {e}") from e
    finally:
        # Ensure memory is freed even if errors occur
        free_memory()


# --- API Endpoint ---
@app.post("/generate/")
async def generate_image_endpoint(request: GenerateRequest):
    try:
        # Run the synchronous generation function in a thread pool executor
        # FastAPI automatically uses anyio's threadpool for standard functions,
        # but being explicit can sometimes be clearer or necessary for long tasks.
        # loop = asyncio.get_running_loop()
        # img_bytes_io = await loop.run_in_executor(None, generate_image_sync, request)

        # Simpler: Directly call the sync function, FastAPI handles threading
        img_bytes_io = generate_image_sync(request)

        return StreamingResponse(img_bytes_io, media_type="image/png")

    except ValueError as e: # Handle invalid input like model_type
        logger.warning(f"Bad request for /generate: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e: # Handle errors from generation (OOM, loading issues)
         logger.error(f"Server error during /generate: {e}")
         # Check if it's a model loading issue
         if "pipeline is not loaded" in str(e):
              raise HTTPException(status_code=503, detail=str(e)) # Service Unavailable
         else:
              raise HTTPException(status_code=500, detail=str(e)) # Internal Server Error
    except Exception as e: # Catch any other unexpected errors
         logger.exception(f"Unexpected critical error during /generate") # Log full traceback
         raise HTTPException(status_code=500, detail="An unexpected critical error occurred.")

# --- Basic Root Endpoint ---
@app.get("/")
async def read_root():
    return {"message": "Stable Diffusion FastAPI Backend"}

# --- Main Execution (for testing) ---
# if __name__ == "__main__":
#     # Note: Running this way won't use the lifespan manager correctly for model loading on startup
#     # It's better to run using: uvicorn backend_app:app --host 0.0.0.0 --port 8000
#     uvicorn.run(app, host="127.0.0.1", port=8000)