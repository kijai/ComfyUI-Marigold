import os
from glob import glob

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .marigold.model.marigold_pipeline import MarigoldPipeline
from .marigold.util.ensemble import ensemble_depths
from .marigold.util.image_util import chw2hwc, colorize_depth_maps, resize_max_res
from .marigold.util.seed_all import seed_all
from .marigold.util.batchsize import find_batch_size

class MarigoldDepthEstimation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "image": ("IMAGE", ),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "denoise_steps": ("INT", {"default": 10, "min": 0, "max": 4096, "step": 1}),
            "n_repeat": ("INT", {"default": 2, "min": 2, "max": 4096, "step": 1}),
            "invert": ("BOOLEAN", {"default": True}),
            },
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("image",)
    FUNCTION = "process"

    CATEGORY = "Marigold"

    def process(self, image, seed, denoise_steps, n_repeat, invert):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        image = image.permute(0, 3, 1, 2).to(device)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(script_directory, "checkpoints/Marigold_v1_merged")

        self.marigold_pipeline = MarigoldPipeline.from_pretrained(checkpoint_path, enable_xformers=False)
        self.marigold_pipeline = self.marigold_pipeline.to(device)
        self.marigold_pipeline.unet.eval()  # Set the model to evaluation mode
        
        print(image.shape)
        depth_maps = []

        with torch.no_grad():
            for _ in range(n_repeat):
                depth_map = self.marigold_pipeline(image, num_inference_steps=denoise_steps)  # Process the image tensor to get the depth map
                depth_map = torch.clip(depth_map, -1.0, 1.0)
                depth_map = (depth_map + 1.0) / 2.0
                depth_maps.append(depth_map)
        
        depth_predictions = torch.concat(depth_maps, axis=0).squeeze()
        
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        #ensemble parameters
        regularizer_strength = 0.02
        max_iter = 5
        tol = 1e-3
        reduction_method = "median"
        merging_max_res = None

        # Test-time ensembling
        if n_repeat > 1:
            depth_map, pred_uncert = ensemble_depths(
                depth_predictions,
                regularizer_strength=regularizer_strength,
                max_iter=max_iter,
                tol=tol,
                reduction=reduction_method,
                max_res=merging_max_res,
                device=device,
            )
        
        depth_map = depth_map.unsqueeze_(0).to(dtype=torch.float32)
       
        if invert:
            depth_map = 1 - depth_map
        return (depth_map, )

NODE_CLASS_MAPPINGS = {
    "MarigoldDepthEstimation": MarigoldDepthEstimation,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MarigoldDepthEstimation": "MarigoldDepthEstimation",
}