import os
import torch
import argparse
from PIL import Image
from torchvision import transforms
from accelerate import Accelerator
from diffusers import StableDiffusionImageVariationPipeline
from torch.utils.data import Dataset
import random
import shutil
import numpy as np
import pandas as pd
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, default = 5)
parser.add_argument("--data_dir", type = str)
parser.add_argument("--save_image_gen", type = str)
parser.add_argument("--dfu_times", type = int, default = 2)
args = parser.parse_args()


accelerator = Accelerator()
os.makedirs(args.save_image_gen, exist_ok = True)

class Skin7(Dataset):
    """SKin Lesion"""
    def __init__(self, root="./data", train='train', transform=None):
        self.root = root
        self.transform = transform
        self.train = train

        self.data, self.targets = self.get_data(self.root)
        self.classes_name = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        self.classes = list(range(len(self.classes_name)))
        self.target_img_dict = {}
        targets = np.array(self.targets)
        for target in self.classes:
            indexes = np.nonzero(targets == target)[0]
            self.target_img_dict.update({target: indexes})

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the
                   target class.
        """
        path = self.data[index]
        target = self.targets[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return self.data[index], img

    def __len__(self):
        return len(self.data)

    def get_data(self, data_dir):

        if self.train == 'train':
            csv = '/home/ubuntu22/code/AIAA5045/data/converted_label/train.csv'
        elif self.train == 'val':
            csv = './data/converted_label/val.csv'
        elif self.train == 'test':
            csv = './data/converted_label/test.csv'
        elif self.train == 'labeled':
            csv = './data/converted_label/labeled_20_split.csv'
        elif self.train == 'unlabeled':
            csv = './data/converted_label/unlabeled_80_split.csv'

        fn = csv
        csvfile = pd.read_csv(fn)
        raw_data = csvfile.values

        data = []
        targets = []
        for path, label in raw_data:
            data.append(os.path.join(self.root, path))
            targets.append(label)

        return data, targets

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def pil_loader(path):
    """Image Loader
    """
    with open(path, "rb") as afile:
        img = Image.open(afile)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def print_dataset(dataset, print_time):
    print(len(dataset))
    from collections import Counter
    counter = Counter()
    labels = []
    for index, (img, label) in enumerate(dataset):
        if index % print_time == 0:
            print(img.size(), label)
        labels.append(label)
    counter.update(labels)
    print(counter)

class Dataset_ImageNetR(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.folders = os.listdir(self.root)
        self.folders.sort()
        self.images = []
        for folder in self.folders:
            if not os.path.isdir(os.path.join(self.root, folder)):
                continue
            class_images = os.listdir(os.path.join(self.root, folder))
            class_images = list(map(lambda x: os.path.join(folder, x), class_images))
            random.shuffle(class_images)
            class_image = class_images[0:5]
            self.images  = self.images + class_image

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(self.root, self.images[idx])).convert('RGB'))
        return self.images[idx], image
    

def generate_images(pipe, dataloader, args):
    pipe, dataloader = accelerator.prepare(pipe, dataloader)
    pipe.safety_checker = lambda images, clip_input: (images, False)
    pipe = pipe.to(accelerator.device)
    with torch.no_grad():
        for count, (image_locations, original_images) in enumerate(dataloader):
            print(f'{count} / {len(dataloader)}, {image_locations[0]}.')

            for image_lo in image_locations:
                os.makedirs(os.path.join(args.save_image_gen, os.path.dirname(image_lo)), exist_ok = True)
                source_path = os.path.join(args.data_dir, image_lo)
                dist_path = os.path.join(args.save_image_gen, image_lo)
                
                if not os.path.exists(dist_path):
                    shutil.copyfile(source_path, dist_path)
                    with open(os.path.join(args.save_image_gen, 'selected_data_list.txt'), 'a+') as f:
                        f.write(dist_path+'\n')

            for time_ in range(args.dfu_times):
                prompt = "Generate augmentations"
                print(original_images.shape)
                images = pipe(prompt=prompt, image=original_images, guidance_scale = 3).images
                for index in range(len(images)):
                    # print(image_locations[index].split('.')[0]+'_'+str(126+time_)+'.'+image_locations[index].split('.')[1])
                    images[index].save(os.path.join(args.save_image_gen, image_locations[index].split('.')[0]+'_'+str(time_)+'.'+image_locations[index].split('.')[1]))


def main():
    model_name_path = "lambdalabs/sd-image-variations-diffusers"
    pipe = StableDiffusionImageVariationPipeline.from_pretrained(model_name_path, revision = "v2.0")
    text2img = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]),
    ])
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # dataset = Dataset_ImageNetR(args.data_dir, tform)
    dataset = Skin7(root=args.data_dir, train='train',
                             transform=val_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    generate_images(img2img, dataloader, args)



if __name__ == "__main__":
    main()