import os
import torch

from .marigold.model.marigold_pipeline import MarigoldPipeline
from .marigold.util.ensemble import ensemble_depths
from .marigold.util.image_util import chw2hwc, colorize_depth_maps, resize_max_res

import comfy.utils
class MarigoldDepthEstimation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "image": ("IMAGE", ),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "denoise_steps": ("INT", {"default": 10, "min": 1, "max": 4096, "step": 1}),
            "n_repeat": ("INT", {"default": 2, "min": 2, "max": 4096, "step": 1}),
            "regularizer_strength": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 4096, "step": 0.001}),
            "reduction_method": (
            [   
                'median',
                'mean',  
            ], {
               "default": 'median'
            }),
            "max_iter": ("INT", {"default": 5, "min": 1, "max": 4096, "step": 1}),
            "tol": ("FLOAT", {"default": 1e-3, "min": 1e-6, "max": 1e-1, "step": 1e-6}),
            
            "invert": ("BOOLEAN", {"default": True}),
            },
            
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("ensembled_image",)
    FUNCTION = "process"

    CATEGORY = "Marigold"

    def process(self, image, seed, denoise_steps, n_repeat, regularizer_strength, reduction_method, max_iter, tol,invert):
        batch_size = image.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        image = image.permute(0, 3, 1, 2).to(device).to(dtype=torch.float16)

        #load the diffusers model
        script_directory = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(script_directory, "checkpoints/Marigold_v1_merged")
        # Check if the first path exists
        if not os.path.exists(checkpoint_path):
            # If it doesn't exist, construct the alternative path
            checkpoint_path = os.path.join(script_directory, "checkpoints/Marigold")

        self.marigold_pipeline = MarigoldPipeline.from_pretrained(checkpoint_path, enable_xformers=False)
        self.marigold_pipeline = self.marigold_pipeline.to(device).half()
  
        self.marigold_pipeline.unet.eval()  # Set the model to evaluation mode

        pbar = comfy.utils.ProgressBar(batch_size * n_repeat)

        out = []
        for i in range(batch_size):
            depth_maps = []

            with torch.no_grad():
                for _ in range(n_repeat):
                    depth_map = self.marigold_pipeline(image[i].unsqueeze(0), num_inference_steps=denoise_steps, show_pbar=False)  # Process the image tensor to get the depth map
                    depth_map = torch.clip(depth_map, -1.0, 1.0)
                    depth_map = (depth_map + 1.0) / 2.0
                    depth_maps.append(depth_map)
                    pbar.update(1)
            depth_predictions = torch.cat(depth_maps, dim=0).squeeze()
            
            torch.cuda.empty_cache()  # clear vram cache for ensembling

            # Test-time ensembling
            if n_repeat > 1:
                depth_map, pred_uncert = ensemble_depths(
                    depth_predictions,
                    regularizer_strength=regularizer_strength,
                    max_iter=max_iter,
                    tol=tol,
                    reduction=reduction_method,
                    max_res=None,
                    device=device,
                )
            depth_map = depth_map.unsqueeze(2).repeat(1, 1, 3)
            out.append(depth_map)
            

        if invert:
            outstack = 1.0 - torch.stack(out, dim=0)
        else:
            outstack = torch.stack(out, dim=0)
        
        return (outstack,)

NODE_CLASS_MAPPINGS = {
    "MarigoldDepthEstimation": MarigoldDepthEstimation,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MarigoldDepthEstimation": "MarigoldDepthEstimation",
}