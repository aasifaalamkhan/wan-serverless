import os
import torch
import logging
from PIL import Image
import io
import tempfile
import base64
import gc
from functools import partial

import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import save_video
from wan.image2video import WanI2V
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.modules.model import WanModel

logger = logging.getLogger(__name__)

class OOMSafeWanI2V(WanI2V):
    """Subclass of WanI2V that loads weights sequentially to prevent exceeding 62GB CPU RAM limits"""
    def __init__(self,
                 config,
                 checkpoint_dir,
                 device_id=0,
                 rank=0,
                 t5_fsdp=False,
                 dit_fsdp=False,
                 use_sp=False,
                 t5_cpu=False,
                 init_on_cpu=True,
                 convert_model_dtype=False):
        
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        # 1. Load T5 Text Encoder on CPU
        logger.info("OOMSafeWanI2V: Loading T5 encoder...")
        from wan.distributed.fsdp import shard_model
        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )
        gc.collect()

        # 2. Load VAE on GPU
        logger.info("OOMSafeWanI2V: Loading VAE on GPU...")
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)
        gc.collect()

        # 3. Load Low-Noise DiT model on CPU
        logger.info("OOMSafeWanI2V: Loading low_noise_model...")
        self.low_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.low_noise_checkpoint)
        self.low_noise_model = self._configure_model(
            model=self.low_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)
            
        # 4. MOVE Low-Noise Model to GPU immediately to free CPU memory
        logger.info("OOMSafeWanI2V: Moving low_noise_model to GPU VRAM...")
        self.low_noise_model.to(self.device)
        gc.collect()
        torch.cuda.empty_cache()

        # 5. Load High-Noise DiT model on CPU (leaving it on CPU for offloading)
        logger.info("OOMSafeWanI2V: Loading high_noise_model...")
        self.high_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.high_noise_checkpoint)
        self.high_noise_model = self._configure_model(
            model=self.high_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)
            
        # Now, low_noise_model is on GPU VRAM, high_noise_model is on CPU RAM.
        # This divides the 67.8GB memory usage between CPU (39GB) and GPU (28GB) perfectly.
        logger.info("✅ OOMSafeWanI2V: Sequentially loaded and placed both models successfully!")
        
        if use_sp:
            from wan.distributed.util import get_world_size
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt


class InMemoryVideoGenerator:
    def __init__(self, model_path, model_type="I2V-14B-480P", wan_repo_path=None, device_id=0):
        self.model_path = model_path
        self.model_type = model_type
        self.device_id = device_id
        
        # Map our model type to official task
        if "I2V-14B" in model_type or "i2v-A14B" in model_type:
            self.task = "i2v-A14B"
        elif "TI2V-5B" in model_type:
            self.task = "ti2v-5B"
        else:
            self.task = "i2v-A14B"
            
        self.cfg = WAN_CONFIGS[self.task]
        
        # Determine offloading based on VRAM capacity, allowing override via environment variable FORCE_OFFLOAD
        force_offload = os.getenv("FORCE_OFFLOAD", "False").lower() in ("true", "1", "yes")
        total_memory = torch.cuda.get_device_properties(self.device_id).total_memory
        self.use_offload = force_offload or (total_memory <= 60 * 1024 * 1024 * 1024)
        
        logger.info(f"Initializing InMemoryVideoGenerator for model: {model_type} on GPU {self.device_id}")
        logger.info(f"Detected GPU VRAM on GPU {self.device_id}: {total_memory / (1024**3):.2f} GB. Use CPU offloading: {self.use_offload}")
        
        # Load the Wan pipeline in-memory. If high VRAM is available, use the official loader directly.
        logger.info(f"Loading WanI2V pipeline from {self.model_path} on GPU {self.device_id}...")
        if not self.use_offload:
            logger.info(f"High VRAM and RAM detected. Using official WanI2V loader directly on GPU {self.device_id}.")
            self.pipeline = WanI2V(
                config=self.cfg,
                checkpoint_dir=self.model_path,
                device_id=self.device_id,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_sp=False,
                t5_cpu=False,
                init_on_cpu=False,
                convert_model_dtype=False
            )
        else:
            logger.info(f"Limited VRAM/RAM detected. Using OOM-safe sequential loader on GPU {self.device_id}.")
            self.pipeline = OOMSafeWanI2V(
                config=self.cfg,
                checkpoint_dir=self.model_path,
                device_id=self.device_id,
                rank=rank if 'rank' in locals() else 0,
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
            
        max_area = MAX_AREA_CONFIGS.get(size_str, width * height)
        
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
