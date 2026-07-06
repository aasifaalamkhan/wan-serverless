import os
import torch
import logging
from PIL import Image
import io
import tempfile
import base64
import gc
from functools import partial
import math
import random
import sys
from contextlib import contextmanager
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import save_video, best_output_size, masks_like
from wan.image2video import WanI2V
from wan.textimage2video import WanTI2V
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.modules.model import WanModel
from wan.distributed.fsdp import shard_model
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

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


class Wan21I2V(WanTI2V):
    """Pipeline for Wan 2.1 Image-to-Video models using single-model checkpoint and VAE 2.1"""
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
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        
        # 1. Load T5 Text Encoder on CPU
        logger.info("Wan21I2V: Loading T5 encoder...")
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)
        gc.collect()

        # 2. Load VAE2.1 on GPU
        logger.info("Wan21I2V: Loading VAE2.1...")
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)
        gc.collect()

        # 3. Load Main Transformer Model directly from checkpoint_dir
        logger.info("Wan21I2V: Loading main model...")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)
        
        if use_sp:
            from wan.distributed.util import get_world_size
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def i2v(self,
            input_prompt,
            img,
            size=(1280, 704),
            max_area=704 * 1280,
            frame_num=81,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=40,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True):
        
        # preprocess
        ih, iw = img.height, img.width
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

        # center-crop
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        assert img.width == ow and img.height == oh

        # to tensor
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess T5
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # Prepare msk and y for Image-to-Video model_type == 'i2v'
        lat_h = oh // self.vae_stride[1]
        lat_w = ow // self.vae_stride[2]
        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(oh, ow), mode='bicubic').transpose(0, 1),
                torch.zeros(3, F - 1, oh, ow)
            ], dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise
            mask1, mask2 = masks_like([noise], zero=True)
            latent = (1. - mask2[0]) * y[4:] + mask2[0] * latent

            arg_c = {
                'context': [context[0]],
                'seq_len': seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
                'y': [y],
            }

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]
                timestep = torch.stack(timestep).to(self.device)

                # Predict noise
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                # step
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t.to(self.device),
                    latent.to(self.device).unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
                
                # Blend back conditional first frame
                latent = (1. - mask2[0]) * y[4:] + mask2[0] * latent
                
                del latent_model_input, timestep

            x0 = [latent]
            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos


class InMemoryVideoGenerator:
    def __init__(self, model_path, model_type="I2V-14B-480P", wan_repo_path=None, device_id=0):
        self.model_path = model_path
        self.model_type = model_type
        self.device_id = device_id
        
        # Map our model type to official task
        if "I2V-14B-480P" in model_type:
            self.task = "i2v-A14B"  # Use i2v-A14B config parameters (VAE stride, T5 files) for Wan 2.1 I2V 480P
        elif "I2V-14B" in model_type or "i2v-A14B" in model_type:
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
        if model_type == "I2V-14B-480P":
            logger.info(f"Loading Wan2.1 I2V 14B 480P pipeline from {self.model_path} on GPU {self.device_id}...")
            self.pipeline = Wan21I2V(
                config=self.cfg,
                checkpoint_dir=self.model_path,
                device_id=self.device_id,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_sp=False,
                t5_cpu=False,
                init_on_cpu=self.use_offload,
                convert_model_dtype=True
            )
        elif not self.use_offload:
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

    def generate_video(self, prompt, image, negative_prompt="", width=832, height=480, duration_seconds=3.0, fps=16, guidance_scale=3.5, num_inference_steps=30, seed=None, resolution_preset=None, sample_solver="dpm++"):
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
        
        logger.info(f"🚀 Running in-memory generation (Steps: {num_inference_steps}, Solver: {sample_solver}, Size: {size_str}, Seed: {seed})...")
        
        # 5. Execute pipeline in inference mode
        # guide_scale is a float for single-model (Wan21I2V/WanTI2V), and a tuple for Wan2.2 dual-model (WanI2V)
        g_scale = guidance_scale if isinstance(self.pipeline, Wan21I2V) else (guidance_scale, guidance_scale)
        
        with torch.inference_mode():
            if isinstance(self.pipeline, Wan21I2V):
                # Wan21I2V uses i2v() and returns a list of tensors
                videos = self.pipeline.i2v(
                    prompt,
                    img,
                    max_area=max_area,
                    frame_num=frame_num,
                    shift=3.0 if max_area <= 832 * 480 else 5.0,
                    sample_solver=sample_solver,
                    sampling_steps=num_inference_steps,
                    guide_scale=g_scale,
                    seed=seed,
                    offload_model=self.use_offload
                )
                video = videos[0]  # list -> tensor [C, T, H, W]
            else:
                video = self.pipeline.generate(
                    prompt,
                    img,
                    max_area=max_area,
                    frame_num=frame_num,
                    shift=3.0 if max_area <= 832 * 480 else 5.0,
                    sample_solver=sample_solver,
                    sampling_steps=num_inference_steps,
                    guide_scale=g_scale,
                    seed=seed,
                    offload_model=self.use_offload
                )
            
        # 6. Save video to a temp file and encode to base64
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        try:
            save_video(
                tensor=video[None],  # [C,T,H,W] -> [1,C,T,H,W]
                save_file=tmp_path,
                fps=fps,
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
