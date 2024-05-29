import os
import torch
import torchvision.transforms as transforms

try:
    from diffusers import MarigoldDepthPipeline, MarigoldNormalsPipeline, AutoencoderTiny
except:
    MarigoldDepthPipeline = None

from diffusers.schedulers import (
        DDIMScheduler,
        LCMScheduler
    )

import comfy.utils
import model_management 
import folder_paths

class MarigoldModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {            
            "model": (
            ['marigold-v1-0',
             'marigold-lcm-v1-0',
             'marigold-normals-v0-1',
             'marigold-normals-lcm-v0-1',], 
            {
               "default": 'marigold-lcm-v1-0'
            }),
            },
            }
    
    RETURN_TYPES = ("MARIGOLDMODEL",)
    RETURN_NAMES =("marigold_model",)
    FUNCTION = "load"
    CATEGORY = "Marigold"
    DESCRIPTION = """
Diffusion-based monocular depth estimation:  
https://github.com/prs-eth/Marigold  
  
Uses Diffusers 0.28.0 Marigold pipelines.  
"""

    def load(self, model):
        device = model_management.get_torch_device()
        diffusers_model_path = os.path.join(folder_paths.models_dir,'diffusers')
        checkpoint_path = os.path.join(diffusers_model_path, model)

        if not os.path.exists(checkpoint_path):
            print(f"Selected model: {checkpoint_path} not found, downloading...")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=f"prs-eth/{model}", 
                                allow_patterns=["*.json", "*.txt","*fp16*"],
                                ignore_patterns=["*.bin"],
                                local_dir=checkpoint_path, 
                                local_dir_use_symlinks=False
                                )
        if "normals" in model:
            modeltype = "normals"
            self.marigold_pipeline = MarigoldNormalsPipeline.from_pretrained(
            checkpoint_path, 
            variant="fp16", 
            torch_dtype=torch.float16).to(device)
        else:
            modeltype = "depth"
            self.marigold_pipeline = MarigoldDepthPipeline.from_pretrained(
            checkpoint_path, 
            variant="fp16", 
            torch_dtype=torch.float16).to(device)

        marigold_model = {
            "pipeline": self.marigold_pipeline,
            "modeltype": modeltype
        }
        return (marigold_model,)
    
class MarigoldDepthEstimation_v2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "marigold_model": ("MARIGOLDMODEL",),
            "image": ("IMAGE", ),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "denoise_steps": ("INT", {"default": 4, "min": 1, "max": 4096, "step": 1}),
            "ensemble_size": ("INT", {"default": 3, "min": 1, "max": 4096, "step": 1}),
            "processing_resolution": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
            "scheduler": (
            ["DDIMScheduler", "LCMScheduler",], 
            {
               "default": 'LCMScheduler'
            }),
            "use_taesd_vae": ("BOOLEAN", {"default": False}),
            },
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("ensembled_image",)
    FUNCTION = "process"
    CATEGORY = "Marigold"
    DESCRIPTION = """
Diffusion-based monocular depth estimation:  
https://github.com/prs-eth/Marigold  
  
Uses Diffusers 0.28.0 Marigold pipelines.  
"""

    def process(self, marigold_model, image, seed, denoise_steps, processing_resolution, ensemble_size, scheduler, use_taesd_vae):
        batch_size = image.shape[0]
        device = model_management.get_torch_device()
        torch.manual_seed(seed)

        image = image.permute(0, 3, 1, 2).to(device)

        pipeline = marigold_model['pipeline']
        pred_type = marigold_model['modeltype']

        if use_taesd_vae:
            pipeline.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to(device)
            
        pbar = comfy.utils.ProgressBar(batch_size)

        scheduler_kwargs = {
            DDIMScheduler: {
                "num_inference_steps": denoise_steps,
                "ensemble_size": ensemble_size,
            },
            LCMScheduler: {
                "num_inference_steps": denoise_steps,
                "ensemble_size": ensemble_size,
            },	
        }
        if scheduler == 'DDIMScheduler':
            pipe_kwargs = scheduler_kwargs[DDIMScheduler]
        elif scheduler == 'LCMScheduler':
            pipe_kwargs = scheduler_kwargs[LCMScheduler]

        generator = torch.Generator(device).manual_seed(seed)

        processed_out = []

        for i in range(batch_size):
            processed = pipeline(
                image[i],
                output_type = "pt",
                generator = generator,
                processing_resolution = processing_resolution,
                **pipe_kwargs
                )
            
            pbar.update(1)
            if pred_type == "normals":
                normals = pipeline.image_processor.visualize_normals(processed.prediction)
                normals_tensor = transforms.ToTensor()(normals[0])
                processed_out.append(normals_tensor)
            else:
                processed_out.append(processed[0])
        
        if pred_type == "normals":
            processed_out = torch.stack(processed_out, dim=0)
            processed_out = processed_out.permute(0, 2, 3, 1).cpu().float()
        else:
            processed_out = torch.cat(processed_out, dim=0)
            processed_out = processed_out.permute(0, 2, 3, 1).repeat(1, 1, 1, 3).cpu().float()
            processed_out = 1.0 - processed_out

        return (processed_out,)

class MarigoldDepthEstimation_v2_video:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "marigold_model": ("MARIGOLDMODEL",),  
            "images": ("IMAGE", ),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "denoise_steps": ("INT", {"default": 4, "min": 1, "max": 4096, "step": 1}),
            "processing_resolution": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
            "scheduler": (
            ["DDIMScheduler", "LCMScheduler",], 
            {
               "default": 'LCMScheduler'
            }),
            
            "blend_factor": ("FLOAT", {"default": 0.1,"min": 0.0, "max": 1.0, "step": 0.01}),
            "use_taesd_vae": ("BOOLEAN", {"default": True}),
            },
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("ensembled_image",)
    FUNCTION = "process"
    CATEGORY = "Marigold"
    DESCRIPTION = """
Diffusion-based monocular depth estimation:  
https://github.com/prs-eth/Marigold  
  
Uses Diffusers 0.28.0 Marigold pipelines.  
"""

    def process(self, marigold_model, images, seed, denoise_steps, processing_resolution, blend_factor, scheduler, use_taesd_vae):
       
        device = model_management.get_torch_device()
        
        pipeline = marigold_model['pipeline']
        pred_type = marigold_model['modeltype']

        if use_taesd_vae:
            pipeline.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to(device)

        scheduler_kwargs = {
            DDIMScheduler: {
                "num_inference_steps": denoise_steps,
                "ensemble_size": 1,
            },
            LCMScheduler: {
                "num_inference_steps": denoise_steps,
                "ensemble_size": 1,
            },	
        }
        if scheduler == 'DDIMScheduler':
            pipe_kwargs = scheduler_kwargs[DDIMScheduler]
        elif scheduler == 'LCMScheduler':
            pipe_kwargs = scheduler_kwargs[LCMScheduler]

        
        B, H, W, C  = images.shape
        size = [W, H]
        images = images.permute(0, 3, 1, 2).to(device)

        last_frame_latent = None
        torch.manual_seed(seed)
        latent_common = torch.randn((1, 4, processing_resolution * size[1] // (8 * max(size)), processing_resolution * size[0] // (8 * max(size)))).to(device=device, dtype=torch.float16)
        print("latent_common shape: ",latent_common.shape)
        pbar = comfy.utils.ProgressBar(B)
        processed_out = []
        for img in images:
            
            print(img.shape)
            latents = latent_common
            if last_frame_latent is not None:
                latents = (1 - blend_factor) * latents + blend_factor * last_frame_latent

            processed = pipeline(
                img,
                processing_resolution = processing_resolution,
                match_input_resolution=False, 
                latents=latents,
                output_latent=True,
                output_type = "pt",
                **pipe_kwargs
                )
            last_frame_latent = processed.latent
            print("last frame latent shape: ",last_frame_latent.shape)
            pbar.update(1)
            if pred_type == "normals":
                normals = pipeline.image_processor.visualize_normals(processed.prediction)
                normals_tensor = transforms.ToTensor()(normals[0])
                processed_out.append(normals_tensor)
            else:
                processed_out.append(processed[0])
        
        if pred_type == "normals":
            processed_out = torch.stack(processed_out, dim=0)
            processed_out = processed_out.permute(0, 2, 3, 1).cpu().float()
        else:
            processed_out = torch.cat(processed_out, dim=0)
            processed_out = processed_out.permute(0, 2, 3, 1).repeat(1, 1, 1, 3).cpu().float()
            processed_out = 1.0 - processed_out

        return (processed_out,)
    
NODE_CLASS_MAPPINGS = {
    "MarigoldModelLoader": MarigoldModelLoader,
    "MarigoldDepthEstimation_v2": MarigoldDepthEstimation_v2,
    "MarigoldDepthEstimation_v2_video": MarigoldDepthEstimation_v2_video,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MarigoldModelLoader": MarigoldModelLoader,
    "MarigoldDepthEstimation_v2": "MarigoldDepthEstimation_v2",
    "MarigoldDepthEstimation_v2_video": "MarigoldDepthEstimation_v2_video",
}