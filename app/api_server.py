from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import logging
import base64
import threading
import uuid
import queue
import time
import requests
from model_downloader import ModelDownloader
from check_cuda import is_cuda_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Wan Video Generation API", version="1.0")

# Global variables for model
generators = []
model_info = None

# Job tracking
jobs = {}
jobs_lock = threading.Lock()
task_queue = queue.Queue()

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
    sample_solver: Optional[str] = "dpm++"
    task_id: Optional[str] = None
    callback_url: Optional[str] = None
    webhook_secret: Optional[str] = None

def cleanup_old_jobs():
    """Delete completed/failed jobs that are older than 5 minutes to free RAM"""
    with jobs_lock:
        now = time.time()
        to_delete = []
        for j_id, j in jobs.items():
            if j["status"] in ("completed", "failed"):
                if now - j.get("completed_at", 0) > 300:
                    to_delete.append(j_id)
        for j_id in to_delete:
            jobs.pop(j_id, None)

def worker_loop(gpu_idx: int, gen):
    """Background worker thread that serializes generation requests for a specific GPU generator"""
    while True:
        try:
            job_id, req = task_queue.get()
            logger.info(f"Worker {gpu_idx} picked up job {job_id} for prompt: {req.prompt[:50]}")
            
            with jobs_lock:
                jobs[job_id]["status"] = "running"
                
            try:
                # Validate inputs inside thread
                if req.duration_seconds <= 0:
                    raise ValueError("duration_seconds must be greater than 0")
                if req.fps <= 0 or req.fps > 60:
                    raise ValueError("fps must be between 1 and 60")
                
                # Generate video (no global lock needed since each generator runs on a different GPU)
                video_base64 = gen.generate_video(
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
                    resolution_preset=req.resolution_preset,
                    sample_solver=req.sample_solver
                )
                
                with jobs_lock:
                    jobs[job_id]["status"] = "completed"
                    jobs[job_id]["completed_at"] = time.time()
                    jobs[job_id]["result"] = {
                        "video_base64": video_base64,
                        "prompt": req.prompt,
                        "duration_seconds": req.duration_seconds,
                        "fps": req.fps,
                        "resolution": f"{req.width}x{req.height}",
                        "has_input_image": req.image is not None,
                        "model_type": model_info.get("model_type", "unknown") if model_info else "mock",
                        "loras_loaded": len(model_info.get("lora_paths", [])) if model_info else 0
                    }
                    logger.info(f"Job {job_id} completed successfully")
                
                # Send webhook callback if callback_url is set
                if req.callback_url:
                    try:
                        headers = {}
                        if req.webhook_secret:
                            headers["Authorization"] = f"Bearer {req.webhook_secret}"
                        payload = {
                            "taskId": req.task_id or job_id,
                            "status": "completed",
                            "video_base64": video_base64
                        }
                        logger.info(f"Sending success callback to {req.callback_url} for task {req.task_id or job_id}")
                        res = requests.post(req.callback_url, json=payload, headers=headers, timeout=60)
                        logger.info(f"Callback response status: {res.status_code}")
                    except Exception as cb_err:
                        logger.error(f"Failed to send success callback: {cb_err}")

            except Exception as e:
                logger.exception(f"Job {job_id} failed")
                with jobs_lock:
                    jobs[job_id]["status"] = "failed"
                    jobs[job_id]["completed_at"] = time.time()
                    jobs[job_id]["error"] = str(e)
                
                # Send failure webhook callback if callback_url is set
                if req.callback_url:
                    try:
                        headers = {}
                        if req.webhook_secret:
                            headers["Authorization"] = f"Bearer {req.webhook_secret}"
                        payload = {
                            "taskId": req.task_id or job_id,
                            "status": "failed",
                            "error": str(e)
                        }
                        logger.info(f"Sending failure callback to {req.callback_url} for task {req.task_id or job_id}")
                        res = requests.post(req.callback_url, json=payload, headers=headers, timeout=60)
                        logger.info(f"Callback response status: {res.status_code}")
                    except Exception as cb_err:
                        logger.error(f"Failed to send failure callback: {cb_err}")

            finally:
                task_queue.task_done()
        except Exception as e:
            logger.error(f"Error in worker loop: {e}")
            time.sleep(1)

@app.on_event("startup")
def initialize_models():
    """Download models and initialize the video generator on startup"""
    global generators, model_info
    
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
            generators.append(create_cli_generator("", "I2V-14B-480P", use_mock=True, wan_repo_path=wan_repo_path))
        else:
            import torch
            num_gpus = torch.cuda.device_count() if is_cuda_available() else 1
            logger.info(f"Initializing {num_gpus} generator(s)...")
            from in_memory_video_generator import InMemoryVideoGenerator
            for gpu_idx in range(num_gpus):
                logger.info(f"Loading model on GPU {gpu_idx}...")
                generators.append(InMemoryVideoGenerator(
                    model_path=model_info["model_path"],
                    model_type=model_info["model_type"],
                    device_id=gpu_idx
                ))
        logger.info(f"✅ {len(generators)} Video generator(s) initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        logger.info("🔄 Falling back to mock CLI generator...")
        from cli_video_generator import create_cli_generator
        wan_repo_path = os.getenv("WAN_REPO_PATH", "/workspace/wan-serverless/wan")
        generators.append(create_cli_generator("", "I2V-14B-480P", use_mock=True, wan_repo_path=wan_repo_path))
    
    # Start background worker thread for each generator
    for gpu_idx, gen in enumerate(generators):
        worker_thread = threading.Thread(target=worker_loop, args=(gpu_idx, gen), daemon=True)
        worker_thread.start()
        logger.info(f"✅ Background worker thread {gpu_idx} started successfully on GPU {gpu_idx}")

@app.get("/health")
def health_check():
    """Simple health check endpoint"""
    global generators
    return {
        "status": "healthy" if len(generators) > 0 else "initializing",
        "gpu_available": is_cuda_available(),
        "model_type": model_info.get("model_type", "unknown") if model_info else None
    }

@app.post("/generate")
def generate_video(req: GenerationRequest):
    """POST endpoint to queue video generation from prompt/image"""
    global generators
    
    if not generators:
        raise HTTPException(status_code=503, detail="Model is still initializing. Please try again in a few moments.")
    
    # Run cleanup of old completed jobs
    cleanup_old_jobs()
    
    job_id = str(uuid.uuid4())
    logger.info(f"Queuing generation request: {job_id} for prompt: {req.prompt[:50]}")
    
    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "result": None,
            "error": None,
            "created_at": time.time()
        }
    
    task_queue.put((job_id, req))
    return {
        "job_id": job_id,
        "status": "queued"
    }

@app.get("/status/{job_id}")
def get_status(job_id: str):
    """GET endpoint to poll status of video generation job"""
    cleanup_old_jobs()
    
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return jobs[job_id]
