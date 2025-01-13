from PIL import Image
import torch
from core.ComfyUI_UltimateSDUpscale.utils import tensor_to_pil, pil_to_tensor
#from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from core.ComfyUI_UltimateSDUpscale.modules import shared
import  core.Imagen.model_management
import numpy as np
if (not hasattr(Image, 'Resampling')):  # For older versions of Pillow
    Image.Resampling = Image

class ImageUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "upscale_model": ("UPSCALE_MODEL",),
                              "image": ("IMAGE",),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model, image):
        if isinstance (image, np.ndarray):
            print( "image is numpy")
        if isinstance (image, torch.Tensor):
            print("image is a tensor")
        print("image type", image.dtype)
        print("image -------gqzdfén", image.shape)
        
        print("size of image", image.size())
      
        device =core.Imagen.model_management.get_torch_device()
        
        memory_required = core.Imagen.model_management.module_size(upscale_model.model)
        memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0 #The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
        memory_required += image.nelement() * image.element_size()
        core.Imagen.model_management.free_memory(memory_required, device)
        print("image.element_size()", image.element_size())
        
        
        upscale_model.to(device)
        in_img = image.movedim(-1,-3).to(device)
        #in_img = image.to(device)
        
        #in_img=list(in_img)
        print("in-img", in_img.size())
        tile = 512
        overlap = 32

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * core.Imagen.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = core.Imagen.utils.ProgressBar(steps)
                s = core.Imagen.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                oom = False
                print("ssssss", s.size())
            except core.Imagen.model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        upscale_model.to("cpu")
        print("-------------torch s", s)
        s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
        return s
    
    
class Upscaler:

    def _upscale(self, img: Image, scale):
        if scale == 1.0:
            return img
        print("-------upscaler-------------")
        #if (shared.actual_upscaler is None):
          #  return img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
        
        tensor = pil_to_tensor(img)
        print("img",tensor.shape)
        print("image mode", tensor.mode)
        image_upscale_node = ImageUpscaleWithModel()
        upscaled = image_upscale_node.upscale(shared.actual_upscaler, tensor)
        print("upscaled", upscaled.size())
        # Permuter les dimensions [H, W, C] -> [C, H, W]
        #upscaled = upscaled.permute(2, 0, 1)
        print("tensor_to_pil(upscaled)",tensor_to_pil(upscaled))
        
        return tensor_to_pil(upscaled)

    def upscale(self, img: Image, scale, selected_model: str = None):
        shared.batch = [self._upscale(img, scale) for img in shared.batch]
        return shared.batch[0]


class UpscalerData:
    name = ""
    data_path = ""

    def __init__(self):
        self.scaler = Upscaler()
