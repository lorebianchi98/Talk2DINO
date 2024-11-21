import webdataset as wds
import os
import torch
from tqdm import tqdm

from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset

class DinoClipDataset(Dataset):
    def __init__(self, features_file, features_name='dino_features', text_features='ann_feats', load_attn_maps=False, is_wds=False):
        if is_wds:
            self.__load_wds_dataset(features_file, features_name, text_features, load_attn_maps)
        else:
            self.__load_pth_dataset(features_file, features_name, text_features, load_attn_maps)
            
            
    def __getitem__(self, idx):
        annotation = self.data[idx]['annotation']
        image = self.data[idx]['image']
        metadata = {
            'annotation_id': self.data[idx]['annotation_id'],
            'image_id': self.data[idx]['image_id']
        }
        
        to_ret = {
            'annotation': annotation,
            'image': image,
            'metadata': metadata,
        }
        
        if 'self_attn_maps' in self.data[idx]:
            to_ret['self_attn_maps'] = self.data[idx]['self_attn_maps']
            to_ret['dino_features'] = self.data[idx]['dino_features']
            
        if 'text_input_mask' in self.data[idx]:
            to_ret['text_input_mask'] = self.data[idx]['text_input_mask']
            
        if 'text_argmax' in self.data[idx]:
            to_ret['text_argmax'] = self.data[idx]['text_argmax']
        
        return to_ret
    
    def __len__(self):
        return len(self.data)
    
    def __load_pth_dataset(self, features_file, features_name='dino_features', text_features='ann_feats', load_attn_maps=False):
        print("Loading dataset...")
        data = torch.load(features_file, map_location='cpu')    
        print("Dataset loaded!")
        
        images = {imm['id']: imm for imm in data['images']}
        del data['images']
        self.data = {}
        
        for idx, ann in enumerate(data['annotations']):
            ann_id = ann['id']
            imm_id = ann['image_id']
            self.data[idx] = {}
            if text_features != 'clip_txt_out_tokens_avg':
                self.data[idx]['annotation'] = ann[text_features] 
            else:
                mask = ann['text_input_mask']
                mask[mask.sum() - 1] = False # excluding end of sequence
                mask[0] = False # excluding CLS token
                self.data[idx]['annotation'] = ann['clip_txt_out_tokens'][mask].mean(dim=0)
            if text_features == 'clip_second_last_out':
                self.data[idx]['text_argmax'] = ann['text_argmax']
            self.data[idx]['image'] = images[imm_id][features_name]
            if load_attn_maps:
                self.data[idx]['self_attn_maps'] = images[imm_id]['self_attn_maps']
                self.data[idx]['dino_features'] = images[imm_id]['dino_features']
            if text_features == 'clip_txt_out_tokens':
                self.data[idx]['text_input_mask'] = ann['text_input_mask']
            self.data[idx]['image_id'] = imm_id
            self.data[idx]['annotation_id'] = ann_id
            
    def __load_wds_dataset(self, features_file, features_name='dino_features', text_features='ann_feats', load_attn_maps=False):
        print("Loading dataset...")
        def my_decoder(key, value):
            if not key.endswith(".pth"):
                return None
            return torch.load(BytesIO(value))
        dataset = wds.WebDataset(features_file).decode(my_decoder)
        
        self.data = {}
        for idx, obj in enumerate(dataset):
            self.data[idx] = {}
            if text_features != 'clip_txt_out_tokens_avg':
                self.data[idx]['annotation'] = obj['pth'][text_features]
            else:
                mask = obj['pth']['text_input_mask']
                mask[mask.sum() - 1] = False # excluding end of sequence
                mask[0] = False # excluding CLS token
                self.data[idx]['annotation'] = obj['pth']['clip_txt_out_tokens'][mask].mean(dim=0)
            self.data[idx]['image'] = obj['pth'][features_name]
            if load_attn_maps:
                self.data[idx]['self_attn_maps'] = obj['pth']['self_attn_maps']
                self.data[idx]['dino_features'] = obj['pth']['dino_features']
            if text_features == 'clip_txt_out_tokens':
                self.data[idx]['text_input_mask'] = obj['pth']['text_input_mask']
            self.data[idx]['image_id'] = obj['pth']['image_id']
            self.data[idx]['annotation_id'] = obj['pth']['id']
        print("Dataset loaded!")
 
class COCOCaptions(Dataset):
    def __init__(self, ann_path, data_dir, split="train", image_transform=None, text_transform=None, device="cuda"):
        self.data = torch.load(ann_path)
        self.data_dir = data_dir
        self.split = split
        images = {imm['id']: imm for imm in self.data['images']}
        self.samples = []
        for ann in self.data['annotations']:
            if split not in images[ann['image_id']]['file_name']:
                continue
            self.samples.append({
                'annotation': ann['caption'],
                'image_path': images[ann['image_id']]['file_name']
            })
        self.n_imgs = len(self.samples)
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.device = device
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        annotation = self.samples[idx]['annotation']
        image_path = self.samples[idx]['image_path']
        image = Image.open(os.path.join(self.data_dir, image_path))
        if image.mode == 'L':
            image = image.convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)
        if self.text_transform:
            annotation = self.text_transform(annotation)[0]
        
        return {"image": image, "annotation": annotation}
    
    def __len__(self):
        return self.n_imgs
    
