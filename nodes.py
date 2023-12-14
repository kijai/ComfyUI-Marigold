import os
import torch
import numpy as np

from .marigold.model.marigold_pipeline import MarigoldPipeline
from .marigold.util.ensemble import ensemble_depths
from .marigold.util.image_util import chw2hwc, colorize_depth_maps, resize_max_res

import comfy.utils

def colorizedepth(depth_map, colorize_method):
    depth_map = depth_map.cpu().numpy()
    percentile = 0.03
    min_depth_pct = np.percentile(depth_map, percentile)
    max_depth_pct = np.percentile(depth_map, 100 - percentile)
    
    depth_colored = colorize_depth_maps(
        depth_map, min_depth_pct, max_depth_pct, cmap=colorize_method
    ).squeeze()  # [3, H, W], value in (0, 1)
    depth_colored = (depth_colored * 255).astype(np.uint8)
    depth_colored_hwc = chw2hwc(depth_colored)
    return depth_colored_hwc

empty_text_embed = torch.load("empty_text_embed.pt", map_location="cpu")

class MarigoldDepthEstimation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "image": ("IMAGE", ),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "denoise_steps": ("INT", {"default": 10, "min": 1, "max": 4096, "step": 1}),
            "n_repeat": ("INT", {"default": 10, "min": 2, "max": 4096, "step": 1}),
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
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
            "n_repeat_batch_size": ("INT", {"default": 2, "min": 1, "max": 4096, "step": 1}),           
            },
            
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("ensembled_image",)
    FUNCTION = "process"

    CATEGORY = "Marigold"

    def process(self, image, seed, denoise_steps, n_repeat, regularizer_strength, reduction_method, max_iter, tol,invert, keep_model_loaded, n_repeat_batch_size):
        batch_size = image.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        image = image.permute(0, 3, 1, 2).to(device).to(dtype=torch.float16)

        #load the diffusers model
        script_directory = os.path.dirname(os.path.abspath(__file__))
        folders_to_check = [
            "checkpoints/Marigold_v1_merged",
            "checkpoints/Marigold",
            "../../models/diffusers/Marigold_v1_merged",
            "../../models/diffusers/Marigold",
        ]

        checkpoint_path = None
        for folder in folders_to_check:
            potential_path = os.path.join(script_directory, folder)
            if os.path.exists(potential_path):
                checkpoint_path = potential_path
                break

        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint directory found.")

        self.marigold_pipeline = MarigoldPipeline.from_pretrained(checkpoint_path, enable_xformers=False, empty_text_embed=empty_text_embed)
        self.marigold_pipeline = self.marigold_pipeline.to(device).half()
        self.marigold_pipeline.unet.eval()  # Set the model to evaluation mode

        pbar = comfy.utils.ProgressBar(batch_size * n_repeat)

        out = []
        # Set the number of images to process in a batch
        batch_process_size = n_repeat_batch_size 

        with torch.no_grad():
            for i in range(batch_size):
                # Duplicate the current image n_repeat times
                duplicated_batch = image[i].unsqueeze(0).repeat(n_repeat, 1, 1, 1)
                
                # Process the duplicated batch in sub-batches
                depth_maps = []
                for j in range(0, n_repeat, batch_process_size):
                    # Get the current sub-batch
                    sub_batch = duplicated_batch[j:j + batch_process_size]
                    
                    # Process the sub-batch
                    depth_maps_sub_batch = self.marigold_pipeline(sub_batch, num_inference_steps=denoise_steps, show_pbar=False)
                    
                    # Process each depth map in the sub-batch if necessary
                    for depth_map in depth_maps_sub_batch:
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
            outstack = 1.0 - torch.stack(out, dim=0).cpu()
        else:
            outstack = torch.stack(out, dim=0).cpu()
        if not keep_model_loaded:
            self.marigold_pipeline = None
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return (outstack,)

class ColorizeDepthmap:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "image": ("IMAGE", ),
            "colorize_method": (
            [   
                'Spectral',
                'terrain', 
                'viridis',
                'plasma',
                'inferno',
                'magma',
                'cividis',
                'twilight',
                'rainbow',
                'gist_rainbow',
                'gist_ncar',
                'gist_earth',
                'turbo',
                'jet',
                'afmhot',
                'copper',
                'seismic',
                'hsv',
                'brg',

            ], {
               "default": 'Spectral'
            }),
            },
            
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("image",)
    FUNCTION = "color"

    CATEGORY = "Marigold"

    def color(self, image, colorize_method):
        colored_images = []
        for i in range(image.shape[0]):  # Iterate over the batch dimension
            depth_map = image[i].squeeze().permute(2, 0, 1)
            depth_map = depth_map[0]
            depth_map = colorizedepth(depth_map, colorize_method)
            depth_map = torch.from_numpy(depth_map) / 255
            depth_map = depth_map.unsqueeze(0)
            colored_images.append(depth_map)
        
        # Stack the list of tensors along a new dimension
        colored_images = torch.cat(colored_images, dim=0)
        return (colored_images,)

import folder_paths

class SaveImageOpenEXR:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "images": ("IMAGE", ),
            "filename_prefix": ("STRING", {"default": "ComfyUI_EXR"})
            },
            
            }
    
    RETURN_TYPES = ()
    FUNCTION = "saveexr"
    OUTPUT_NODE = True
    CATEGORY = "Marigold"

    def saveexr(self, images, filename_prefix):
        import OpenEXR
        import Imath 
        import re
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        def file_counter():
            max_counter = 0
            # Loop through the existing files
            for existing_file in os.listdir(full_output_folder):
                # Check if the file matches the expected format
                match = re.fullmatch(f"{filename}_(\d+)_?\.[a-zA-Z0-9]+", existing_file)
                if match:
                    # Extract the numeric portion of the filename
                    file_counter = int(match.group(1))
                    # Update the maximum counter value if necessary
                    if file_counter > max_counter:
                        max_counter = file_counter
            return max_counter
        
        for image in images:
            # Ensure the tensor is on the CPU and convert it to a numpy array
            image_np = image.cpu().numpy()
            #image_np = image_np[0]
            image_np = image_np.astype(np.float32)

            # Assuming the image is in the format of floating point 32 bit (change PIXEL_TYPE if not)
            PIXEL_TYPE = Imath.PixelType(Imath.PixelType.FLOAT)
            height, width, channels = image_np.shape

            # Prepare the EXR header
            header = OpenEXR.Header(width, height)
            half_chan = Imath.Channel(PIXEL_TYPE)
            header['channels'] = dict([(c, half_chan) for c in "RGB"])

            # Split the channels for OpenEXR
            R = image_np[:, :, 0].tostring()
            G = image_np[:, :, 1].tostring()
            B = image_np[:, :, 2].tostring()

            # Increment the counter by 1 to get the next available value
            counter = file_counter() + 1
            file = f"{filename}_{counter:05}.exr"

            # Write the EXR file
            exr_file = OpenEXR.OutputFile(os.path.join(full_output_folder, file), header)
            exr_file.writePixels({'R': R, 'G': G, 'B': B})
            exr_file.close()

        return ()
    
NODE_CLASS_MAPPINGS = {
    "MarigoldDepthEstimation": MarigoldDepthEstimation,
    "ColorizeDepthmap": ColorizeDepthmap,
    "SaveImageOpenEXR": SaveImageOpenEXR,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MarigoldDepthEstimation": "MarigoldDepthEstimation",
    "ColorizeDepthmap": "ColorizeDepthmap",
    "SaveImageOpenEXR": "SaveImageOpenEXR",
}