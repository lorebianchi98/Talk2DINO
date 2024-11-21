import os
import sys

import torch
from torchvision import transforms

sys.path.append(os.getcwd())
sys.path.append("../../..")

import torch.utils.data as data
from PIL import Image


class COCODiff(data.Dataset):

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.coco_image_ids = sorted(os.listdir(self.dataset_dir))
        self.pil_to_tensor = transforms.PILToTensor()

    def __getitem__(self, index):
        coco_image_index = index // 5
        coco_image_id = self.coco_image_ids[coco_image_index]
        coco_image_dir = os.path.join(self.dataset_dir, coco_image_id)
        coco_caption_offset = index % 5
        all_files = os.listdir(coco_image_dir)
        png_files = sorted([file for file in all_files if file.endswith(".png")])
        gen_image = self.pil_to_tensor(Image.open(os.path.join(coco_image_dir, png_files[coco_caption_offset])))
        gen_image_id = png_files[coco_caption_offset].split(".")[0]
        with open(os.path.join(coco_image_dir, f"{gen_image_id}.txt"), "r") as f:
            caption = f.read()
        heatmaps_files = [file for file in all_files if file.startswith(gen_image_id) and file.endswith(".pt")]
        heatmaps = torch.stack([torch.load(os.path.join(coco_image_dir, heatmap_file)) for heatmap_file in heatmaps_files])
        classnames = [heatmap_file.replace(".pt", "").replace(f"{gen_image_id}_", "") for heatmap_file in heatmaps_files]
        return gen_image, caption, heatmaps, classnames, gen_image_id

    def __len__(self):
        return len(self.coco_image_ids) * 5