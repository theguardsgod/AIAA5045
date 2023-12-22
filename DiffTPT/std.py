import os

import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline
def pil_loader(path):
    """Image Loader
    """
    with open(path, "rb") as afile:
        img = Image.open(afile)
        return img.convert("RGB")
device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url,verify=False)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))
path = "/home/ubuntu22/dataset/ISIC2018/ISIC2018_Task3_Training_Input/"
save_path = "/home/ubuntu22/dataset/ISIC2018/ISIC2018_Task3_Training_Input_aug/"
for file in os.listdir(path):
    if file.endswith(".jpg"):
        image = pil_loader(os.path.join(path,file))
        prompt = "do strong data augmentation for deep learning training for this skin lesion image"

        images = pipe(prompt=prompt, image=image, strength=0.05, guidance_scale=2.5).images

        images[0].save(os.path.join(save_path,"strong_"+file))

