import os
import torch
import logging
from PIL import Image
import io
import tempfile
import base64
from pathlib import Path

import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import save_video

logger = logging.getLogger(__name__)

class InMemoryVideoGenerator:
    def __init__(self, model_path, model_type="I2V-14B-480P", wan_repo_path=None):
        self.model_path = model_path
        self.model_type = model_type
        
        # Map our model type to official task
        if "I2V-14B" in model_type or "i2v-A14B" in model_type:
            self.task = "i2v-A14B"
        elif "TI2V-5B" in model_type:
            self.task = "ti2v-5B"
        else:
            self.task = "i2v-A14B"
            
        self.cfg = WAN_CONFIGS[self.task]
        
        # Determine offloading based on VRAM capacity
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # We enable CPU offloading for L40S (48GB) to prevent VRAM OOM, but keep it in a single process
        self.use_offload = total_memory <= 60 * 1024 * 1024 * 1024
        
        logger.info(f"Initializing InMemoryVideoGenerator for model: {model_type}")
        logger.info(f"Detected GPU VRAM: {total_memory / (1024**3):.2f} GB. Use CPU offloading: {self.use_offload}")
        
        # Load the Wan pipeline in-memory
        logger.info(f"Loading WanI2V pipeline from {self.model_path}...")
        self.pipeline = wan.WanI2V(
            config=self.cfg,
            checkpoint_dir=self.model_path,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            convert_model_dtype=True
        )
        logger.info("✅ WanI2V pipeline loaded successfully inside memory!")

    def generate_video(self, prompt, image, negative_prompt="", width=832, height=480, duration_seconds=3.0, fps=16, guidance_scale=5.0, num_inference_steps=20, seed=None, resolution_preset=None):
        """Generates video from prompt and image directly in-memory"""
        
        if not image:
            raise ValueError("An input image is required for Image-to-Video generation.")
        
        # 1. Decode base64 image if it is a string
        if isinstance(image, str):
            if "," in image:
                image = image.split(",")[1]
            image_bytes = base64.b64decode(image)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            img = image.convert("RGB")
            
        # 2. Setup Seed
        if seed is None or seed < 0:
            seed = torch.randint(0, 2**31 - 1, (1,)).item()
            
        # 3. Configure frame counts (must be 4n+1)
        frame_num = int(fps * duration_seconds)
        if frame_num % 4 != 1:
            frame_num = (frame_num // 4) * 4 + 1
            
        # 4. Map size
        size_str = f"{width}*{height}"
        if resolution_preset == "480p":
            size_str = "832*480"
        elif resolution_preset == "720p":
            size_str = "1280*720"
        elif resolution_preset == "square":
            size_str = "832*832"
            
        max_area = MAX_AREA_CONFIGS[size_str]
        
        logger.info(f"🚀 Running in-memory generation (Steps: {num_inference_steps}, Size: {size_str}, Seed: {seed})...")
        
        # 5. Execute pipeline in inference mode
        with torch.inference_mode():
            video = self.pipeline.generate(
                prompt,
                img,
                max_area=max_area,
                frame_num=frame_num,
                shift=5.0,
                sample_solver="unipc",
                sampling_steps=num_inference_steps,
                guide_scale=(guidance_scale, guidance_scale),
                seed=seed,
                offload_model=self.use_offload
            )
            
        # 6. Save video to a temp file and encode to base64
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        try:
            save_video(
                tensor=video[None],
                save_file=tmp_path,
                fps=self.cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
            
            with open(tmp_path, "rb") as f:
                video_bytes = f.read()
                
            return base64.b64encode(video_bytes).decode("utf-8")
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            # Cleanup CUDA/GPU caches
            del video
            torch.cuda.empty_cache()
