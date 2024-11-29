import torch
import numpy as np
from omegaconf import OmegaConf
from torchvision.io import read_image
import matplotlib.pyplot as plt
import sys
import argparse

sys.path.insert(0, "src/open_vocabulary_segmentation")
from models.dinotext import DINOText
from models import build_model

device = "cuda"

def plot_qualitative(image, sim, output_path, palette):
    qualitative_plot = np.zeros((sim.shape[0], sim.shape[1], 3)).astype(np.uint8)

    for j in list(np.unique(sim)):
        qualitative_plot[sim == j] = np.array(palette[j])
    plt.axis('off')
    plt.imshow(image)
    plt.imshow(qualitative_plot, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

parser = argparse.ArgumentParser(description="Demo for Talk2DINO")
parser.add_argument("--config", type=str, default="src/open_vocabulary_segmentation/configs/cityscapes/dinotext_cityscapes_vitb_mlp_infonce.yml")
parser.add_argument("--output", type=str, default="pikachu_seg.png")
parser.add_argument("--input", type=str, default="assets/pikachu.png")
parser.add_argument("--with_background", action="store_true")
parser.add_argument("--textual_categories", type=str, default="pikachu,traffic_sign,forest,road")
args = parser.parse_args()

config_file = args.config
output_file = args.output
input_file = args.input
with_background = args.with_background
text = args.textual_categories.replace("_", " ").split(",")

cfg = OmegaConf.load(config_file)

model = build_model(cfg.model)
model.to(device).eval()

img = read_image(input_file).to(device).float().unsqueeze(0)
# text = ["pikachu", "traffic sign", "forest", "road"]
palette = [
    [255, 0, 0],
    [255, 255, 0],
    [0, 255, 0],
    [0, 255, 255],
    [0, 0, 255],
    [128, 128, 128]
]
if len(text) > len(palette):
    for _ in range(len(text) - len(palette)):
        palette.append([np.random.randint(0, 255) for _ in range(3)])
        
if with_background:
    palette.insert(0, [0, 0, 0])
    model.with_bg_clean = True

with torch.no_grad():
    text_emb = model.build_dataset_class_tokens("sub_imagenet_template", text)
    text_emb = model.build_text_embedding(text_emb)
    
    mask, _ = model.generate_masks(img, img_metas=None, text_emb=text_emb, classnames=text, apply_pamr=True)
    if with_background:
        background = torch.ones_like(mask[:, :1]) * 0.55
        mask = torch.cat([background, mask], dim=1)
    
    mask = mask.argmax(dim=1)
    
plot_qualitative(img.cpu()[0].permute(1,2,0).int().numpy(), mask.cpu()[0].numpy(), output_file, palette)
