# Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation
## Installation
```bash
conda create --name dino-text python=3.9
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Feature Extraction
To speed up training, we use pre-extracted features. Follow these steps:

1. Download the 2014 images and annotations from the [COCO website](https://cocodataset.org/#download).
2. Run the following commands to extract features:
    ```bash
    mkdir ../coco2014_b14
    python dino_extraction_v2.py --ann_path ../coco/captions_val2014.json --out_path ../coco2014_b14/val.pth --model dinov2_vitb14_reg --resize_dim 448 --crop_dim 448 --extract_avg_self_attn --extract_disentangled_self_attn
    python dino_extraction_v2.py --ann_path ../coco/captions_train2014.json --out_path ../coco2014_b14/train.pth --model dinov2_vitb14_reg --resize_dim 448 --crop_dim 448 --extract_avg_self_attn --extract_disentangled_self_attn
    python text_features_extraction.py --ann_path ../coco2014_b14/train.pth
    python text_features_extraction.py --ann_path ../coco2014_b14/val.pth
    ```
## Training

To train the model, use the following command (this example runs training for the ViT-Base configuration):

```bash
python train.py --model configs/vitb_mlp_infonce.yaml --train_dataset ../coco2014_b14/train.pth --val_dataset ../coco2014_b14/val.pth
```
## Evaluation

To evaluate the model on open-vocabulary segmentation benchmarks, use the `src/open_vocabulary_segmentation/main.py` script. Select the appropriate configuration based on the model, benchmark, and PAMR settings. Below is an example to evaluate the ViT-Base model on Cityscapes without PAMR:

```bash
python -m torch.distributed.run src/open_vocabulary_segmentation/main.py --eval --eval_cfg src/open_vocabulary_segmentation/configs/cityscapes/dinotext_cityscapes_vitb_mlp_infonce.yml --eval_base src/open_vocabulary_segmentation/configs/cityscapes/eval_cityscapes.yml
```