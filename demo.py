import torch
import numpy as np
from omegaconf import OmegaConf
from torchvision.io import read_image
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "src/open_vocabulary_segmentation")
from models.dinotext import DINOText
from models import build_model

def plot_qualitative(image, sim, output_path, palette):
    qualitative_plot = np.zeros((sim.shape[0], sim.shape[1], 3)).astype(np.uint8)

    for j in list(np.unique(sim)):
        qualitative_plot[sim == j] = np.array(palette[j])
    plt.axis('off')
    plt.imshow(image)
    plt.imshow(qualitative_plot, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)


device = "cuda"
config_file = "src/open_vocabulary_segmentation/configs/cityscapes/dinotext_cityscapes_vitb_mlp_infonce.yml"
output_file = "pikachu_seg.png"
cfg = OmegaConf.load(config_file)

model = build_model(cfg.model)
model.to(device).eval()

img = read_image("assets/pikachu.png").to(device).float().unsqueeze(0)
text = ["pikachu", "traffic sign", "forest", "road"]
palette = [
    [255, 0, 0],
    [255, 255, 0],
    [0, 255, 0],
    [0, 255, 255],
    # [0, 0, 255],
    # [128, 128, 128]
]

with torch.no_grad():
    text_emb = model.build_dataset_class_tokens("sub_imagenet_template", text)
    text_emb = model.build_text_embedding(text_emb)
    
    mask, _ = model.generate_masks(img, img_metas=None, text_emb=text_emb, classnames=text, apply_pamr=True)
    # background = torch.ones_like(mask[:, :1]) * 0.55
    # mask = torch.cat([background, mask], dim=1)
    
    mask = mask.argmax(dim=1)
    
plot_qualitative(img.cpu()[0].permute(1,2,0).int().numpy(), mask.cpu()[0].numpy(), output_file, palette)
