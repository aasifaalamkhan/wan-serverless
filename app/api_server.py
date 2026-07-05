from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import logging
import base64
from model_downloader import ModelDownloader
from cli_video_generator import create_cli_generator
from check_cuda import is_cuda_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Wan Video Generation API", version="1.0")

import threading

# Global variables for model
generator = None
model_info = None
generation_lock = threading.Lock()


class GenerationRequest(BaseModel):
    prompt: str
    image: Optional[str] = None
    negative_prompt: Optional[str] = ""
    resolution_preset: Optional[str] = None
    width: Optional[int] = 1280
    height: Optional[int] = 720
    duration_seconds: Optional[float] = 5.0
    fps: Optional[int] = 24
    guidance_scale: Optional[float] = 5.0
    num_inference_steps: Optional[int] = 50
    seed: Optional[int] = None

@app.on_event("startup")
def initialize_models():
    """Download models and initialize the video generator on startup"""
    global generator, model_info
    
    if not is_cuda_available():
        logger.error("❌ CUDA is not available on this system.")
    
    try:
        logger.info("Initializing WAN models...")
        
        # Use /workspace/models and /workspace/loras for persistent Network Volume storage
        models_dir = os.getenv("MODELS_DIR", "/workspace/models")
        loras_dir = os.getenv("LORAS_DIR", "/workspace/loras")
        
        downloader = ModelDownloader(models_dir=models_dir, loras_dir=loras_dir)
        model_info = downloader.setup_models()
        
        logger.info(f"Model setup complete: {model_info}")
        
        use_mock = os.getenv("USE_MOCK_GENERATOR", "false").lower() == "true"
        if use_mock:
            from cli_video_generator import create_cli_generator
            wan_repo_path = os.getenv("WAN_REPO_PATH", "/workspace/wan-serverless/wan")
            generator = create_cli_generator("", "I2V-14B-480P", use_mock=True, wan_repo_path=wan_repo_path)
        else:
            from in_memory_video_generator import InMemoryVideoGenerator
            generator = InMemoryVideoGenerator(
                model_path=model_info["model_path"],
                model_type=model_info["model_type"]
            )
        logger.info("✅ Video generator initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        logger.info("🔄 Falling back to mock CLI generator...")
        from cli_video_generator import create_cli_generator
        wan_repo_path = os.getenv("WAN_REPO_PATH", "/workspace/wan-serverless/wan")
        generator = create_cli_generator("", "I2V-14B-480P", use_mock=True, wan_repo_path=wan_repo_path)

@app.get("/health")
def health_check():
    """Simple health check endpoint"""
    global generator
    return {
        "status": "healthy" if generator is not None else "initializing",
        "gpu_available": is_cuda_available(),
        "model_type": model_info.get("model_type", "unknown") if model_info else None
    }

@app.post("/generate")
def generate_video(req: GenerationRequest):
    """POST endpoint to generate video from prompt/image"""
    global generator
    
    if generator is None:
        raise HTTPException(status_code=503, detail="Model is still initializing. Please try again in a few moments.")
    
    logger.info(f"Received generation request for prompt: {req.prompt[:50]}. Waiting for lock...")
    with generation_lock:
        logger.info(f"Lock acquired. Starting generation for prompt: {req.prompt[:50]}")
        try:
            # Validate inputs
            if req.duration_seconds <= 0:
                raise HTTPException(status_code=400, detail="duration_seconds must be greater than 0")
            
            if req.fps <= 0 or req.fps > 60:
                raise HTTPException(status_code=400, detail="fps must be between 1 and 60")
            
            # Generate video
            video_base64 = generator.generate_video(
                prompt=req.prompt,
                image=req.image,
                negative_prompt=req.negative_prompt,
                width=req.width,
                height=req.height,
                duration_seconds=req.duration_seconds,
                fps=req.fps,
                guidance_scale=req.guidance_scale,
                num_inference_steps=req.num_inference_steps,
                seed=req.seed,
                resolution_preset=req.resolution_preset
            )
            
            return {
                "video_base64": video_base64,
                "prompt": req.prompt,
                "duration_seconds": req.duration_seconds,
                "fps": req.fps,
                "resolution": f"{req.width}x{req.height}",
                "has_input_image": req.image is not None,
                "model_type": model_info.get("model_type", "unknown") if model_info else "mock",
                "loras_loaded": len(model_info.get("lora_paths", [])) if model_info else 0
            }
            
        except Exception as e:
            logger.error(f"generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
