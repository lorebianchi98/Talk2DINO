import itertools
import os
import pickle
from math import sqrt
import re
import yaml

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from transformers import BertModel, AutoTokenizer
import torchvision.transforms as T
import clip
import importlib

from models.builder import MODELS
from models.dinotext.pamr import PAMR
from models.dinotext.masker import DINOTextMasker
import us
from datasets import get_template

from src.model import ProjectionLayer, VisualProjectionLayer, CLIPLastLayer, DoubleMLP
from src.loss import Contrastive
from src.hooks import average_text_tokens, get_vit_out, feats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@MODELS.register_module()
class DINOText(nn.Module):
    
    def get_self_attention(self, module, input, output):
        self.feats['self_attn'] = output
        
    def get_clip_second_last_dense_out(self, model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        self.feats['clip_second_last_out'] = output
        self.feats['clip_second_last_out'].to(dtype=torch.float32)
    
    def get_all_out_tokens(self, model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        self.feats['clip_txt_out_tokens'] = output
        
    def __init__(
            self, model_name, resize_dim, clip_model_name, proj_class, proj_name, proj_model, avg_self_attn_token=False, disentangled_self_attn_token=True, loss=None, pre_trained=True,
            unfreeze_last_text_layer=False, unfreeze_last_image_layer=False, is_eval=True, use_avg_text_token=False, keep_cls=False, keep_end_seq=False, **kwargs
    ):
        super().__init__()
        self.feats = {}
        self.model_name = model_name
        # loading the model
        
        if 'dinov2' in model_name:
            self.model_family = 'facebookresearch/dinov2' if 'dinov2' in model_name else 'facebookresearch/dino:main'
            self.model = torch.hub.load(self.model_family, model_name)
            
            
        elif 'mae' in model_name or 'sam' in model_name or 'clip' in model_name or 'dino' in model_name:
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,  # remove classifier nn.Linear
                img_size=resize_dim
            )
            
            if 'sam' in model_name:
                self.model.blocks[-1].register_forward_hook(get_vit_out)
        else:
            raise Exception("Unknown ViT model")
        # self.model.eval()
        mean = (0.485, 0.456, 0.406) if not 'clip' in model_name else (0.4815, 0.4578, 0.4082)
        std = (0.229, 0.224, 0.225) if not 'clip' in model_name else (0.2686, 0.2613, 0.2758)
        self.image_transforms = T.Compose([
                T.Resize((resize_dim, resize_dim)),
                lambda x: x / 255.0,
                T.Normalize(mean, std),
        ])
        
        self.model.to(device)
        self.model.requires_grad_(False)
        
        self.clip_model_name = clip_model_name
        if 'bert' in self.clip_model_name:
            self.clip_model = BertModel.from_pretrained(self.clip_model_name, output_hidden_states = False)
            # load the corresponding wordtokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.clip_model_name)
        else:
            self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        if unfreeze_last_text_layer:
            for param in self.clip_model.transformer.resblocks[-1].parameters():
                param.requires_grad = True
            for param in self.clip_model.ln_final.parameters():
                param.requires_grad = True
            self.clip_model.text_projection.requires_grad = True
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        with open(os.path.join('configs', f"{proj_class}.yaml"), 'r') as config_file:
            config = yaml.safe_load(config_file)['model']
            
        ProjClass = getattr(importlib.import_module('src.model'), proj_model)
        self.proj = ProjClass.from_config(config)
        if type(self.proj) == CLIPLastLayer:
            self.clip_model.transformer.resblocks[-2].register_forward_hook(self.get_clip_second_last_dense_out)
        
            
        # if proj_model == 'ProjectionLayer':
        #     self.proj = ProjectionLayer.from_config(config)
        # elif proj_model == 'VisualProjectionLayer':
        #     self.proj = VisualProjectionLayer.from_config(config)
        # else:
        #     raise ValueError(f"Unknown projection model: {config['proj_model']}")
        if pre_trained:
            self.proj.load_state_dict(torch.load(os.path.join("weights", f"{proj_name}.pth"), 'cpu'))
        self.proj.to(device)
        
        self.masker = DINOTextMasker(similarity_type="cosine")
        self.masker = self.masker.eval()
        
        self.pamr = None
                
        self.avg_self_attn_token = avg_self_attn_token
        self.disentangled_self_attn_token = disentangled_self_attn_token
        
        if self.avg_self_attn_token or self.disentangled_self_attn_token or is_eval:
            self.model.blocks[-1].attn.qkv.register_forward_hook(self.get_self_attention)
            self.num_global_tokens = 5 if 'reg' in model_name else 1
            if 'sam' in self.model_name:
                self.num_global_tokens = 0
            self.num_attn_heads = 16
            self.scale = 0.125
        
        self.use_avg_text_token = use_avg_text_token
        if self.use_avg_text_token:
            self.feats = {}
            # in this case we register a forward hook with the aim of getting all the tokens and not only the cls
            self.clip_model.ln_final.register_forward_hook(self.get_all_out_tokens)
            self.keep_cls = keep_cls
            self.keep_end_seq = keep_end_seq
        
                        
        # TODO
        # if loss is not None:
        #     self.contrastive_loss = Contrastive(ltype=loss['ltype'],
        #                                         margin=loss['margin'] if 'margin' in loss else 0.1,
        #                                         max_violation=loss['max_violation'] if 'max_violation' in loss else False
        #                                         )
            

    
    def process_self_attention(self, output, batch_size, num_tokens, num_attn_heads, embed_dim, scale, num_global_tokens, ret_self_attn_maps=False):
        qkv = output.reshape(batch_size, num_tokens, 3, num_attn_heads, embed_dim // num_attn_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        self_attn_maps = attn[:, : , 0, num_global_tokens:]
        self_attn = self_attn_maps.mean(dim=1)
        self_attn = self_attn.softmax(dim=-1)
        if ret_self_attn_maps:
            return self_attn, self_attn_maps
        else:
            return self_attn
    
    def encode_text(self, tokenized_texts):
        if type(self.proj) == CLIPLastLayer:
            self.clip_model.encode_text(tokenized_texts)
            x = self.feats['clip_second_last_out']
            x = x.to(dtype=torch.float32)
        else:
            x = self.clip_model.encode_text(tokenized_texts)
        # x = self.clip2dino_proj.project_clip_txt(x)
        return x
    
    def encode_image(self, images):
        batch_size, _, _, _ = images.shape
        self_attn_maps = None
        x = self.model(images, is_training=(self.avg_self_attn_token or self.disentangled_self_attn_token))
        batch_size, num_tokens, embed_dim = x['x_norm_patchtokens'].shape
        num_tokens = num_tokens + self.num_global_tokens
        if self.avg_self_attn_token or self.disentangled_self_attn_token:
            self_attn, self_attn_maps = self.process_self_attention(self.feats['self_attn'], batch_size, num_tokens, self.num_attn_heads, embed_dim, self.scale, self.num_global_tokens, ret_self_attn_maps=True)
        if self.avg_self_attn_token:
            x = (self_attn.unsqueeze(-1) * x['x_norm_patchtokens']).mean(dim=1)
        elif self.disentangled_self_attn_token:
            self_attn_maps = self_attn_maps.softmax(dim=-1)
            x = (x['x_norm_patchtokens'].unsqueeze(1) * self_attn_maps.unsqueeze(-1)).mean(dim=2)

        # if self.avg_self_attn_token:
        #     batch_size, num_tokens, embed_dim = x['x_norm_patchtokens'].shape
        #     num_tokens = num_tokens + self.num_global_tokens
        #     self_attn = self.process_self_attention(self.feats['self_attn'], batch_size, num_tokens, self.num_attn_heads, embed_dim, self.scale, self.num_global_tokens)
        #     x = (self_attn.unsqueeze(-1) * x['x_norm_patchtokens']).mean(dim=1)
        return x, self_attn_maps

    def forward(self, image, text, return_logit_scale=False):
        with torch.no_grad():
            # text_emb = clip.tokenize(text).to(device)
            txt_embed = self.encode_text(text)
        # txt_embed = self.proj.project_clip_txt(txt_embed)
            
        img_embed, self_attn_maps = self.encode_image(image)
        
        if type(self.proj) == CLIPLastLayer:
            img_embed, txt_embed = self.proj(img_embed, txt_embed, ret_embeds=True, self_attn_maps=self_attn_maps, text_argmax=text.argmax(dim=-1))
        else:
            img_embed, txt_embed = self.proj(img_embed, txt_embed, ret_embeds=True, self_attn_maps=self_attn_maps)
        
        if return_logit_scale:
            return txt_embed, img_embed, self.logit_scale

        return txt_embed, img_embed
        
    def compute_loss(self, image, text, cosine=True, ret_similarity_matrix=True):
        ret = {}
        if cosine:
            img_embed = F.normalize(img_embed, p=2, dim=1)
            txt_embed = F.normalize(txt_embed, p=2, dim=1)
        sim = img_embed @ txt_embed.transpose(1, 0)
        if not ret_similarity_matrix:
            sim = sim[torch.eye(len(sim)) > 0.5] # only diagonal elements
        
        ret['contrastive_loss'] = self.contrastive_loss.compute_contrastive_loss(sim)
        
        return ret


    @torch.no_grad()
    def build_dataset_class_tokens(self, template_set, classnames):
        tokens = []
        templates = get_template(template_set)
        for classname in classnames:
            if 'bert' not in self.clip_model_name:
                tokens.append(
                    clip.tokenize([template.format(classname) for template in templates])
                )
            else:
                tokens.append(self.tokenizer([template.format(classname) for template in templates], return_tensors='pt', padding='max_length')['input_ids'])
        # [N, T, L], N: number of instance, T: number of captions (including ensembled), L: sequence length
        tokens = torch.stack(tokens)

        return tokens

    @torch.no_grad()
    def build_text_embedding(self, text):
        """
        Args:
            text (torch.Tensor): [NUM_CLASSES, NUM_TEMPLATES, CONTEXT_LENGTH] text tokens

        Returns:
            text_embs
        """
        text = text.to(device)
        num_classes, num_templates = text.shape[:2]
        text_argmax = text.argmax(dim=-1)
        text_argmax = rearrange(text_argmax, 'n t -> (n t)', n=num_classes, t=num_templates)
        text = rearrange(text, 'n t l -> (n t) l', n=num_classes, t=num_templates)
        # chunked inference for memory limitation
        chunk_size = 32
        N = text.size(0)
        if type(self.proj) == CLIPLastLayer:
            text_embs = torch.cat([
            self.proj.project_clip_txt(self.encode_text(text[i:i + chunk_size]).permute(1, 0, 2), text_argmax=text_argmax[i:i + chunk_size])
            for i in range(0, N, chunk_size)
        ])
        else:
            if not self.use_avg_text_token:
                # performing classification using CLS textual token
                if 'bert' not in self.clip_model_name:
                    text_embs = torch.cat([
                        self.clip_model.encode_text(text[i:i + chunk_size])
                        for i in range(0, N, chunk_size)
                    ])
                else:
                    # encoding with BERT
                    text_embs = []
                    for i in range(0, N, chunk_size):
                        outputs = self.clip_model(text[i:i + chunk_size])
                        text_embs.append(outputs['pooler_output'])
                    text_embs = torch.cat(text_embs)
            else:
                # using text token average
                text_embs = []
                for i in range(0, N, chunk_size):
                    self.clip_model.encode_text(text[i:i + chunk_size])
                    text_embs.append(average_text_tokens(self.feats['clip_txt_out_tokens'] @ self.clip_model.text_projection, text[i:i + chunk_size] > 0, self.keep_cls, self.keep_end_seq))
                text_embs = torch.cat(text_embs)
        # [N, T, C]
        text_embs = rearrange(text_embs, '(n t) c -> n t c', n=num_classes, t=num_templates)
        # [N, C]
        text_embs = text_embs.mean(dim=1).float()
        if type(self.proj) == ProjectionLayer or type(self.proj) == DoubleMLP:
            text_embs = self.proj.project_clip_txt(text_embs)
        text_embs = us.normalize(text_embs, dim=-1)

        return text_embs

    def apply_pamr(self, image, mask):
        image = F.interpolate(image, mask.shape[-2:], mode="bilinear", align_corners=True)
        if self.pamr is None:
            pamr_iter = 10
            pamr_kernel = [1, 2, 4, 8, 12, 24]
            self.pamr = PAMR(pamr_iter, pamr_kernel)
            self.pamr.eval()
            self.pamr.to(next(self.parameters()).device)

        mask = self.pamr(image, mask)
        return mask

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b
    
    @torch.no_grad()
    def generate_masks(
            self, image, img_metas, text_emb, classnames, text_is_token=False, apply_pamr=False, with_bg=False, background_func="weighted_average_sigmoid", lambda_bg=0.2,
            # kp_w=0.3,
    ):
        """Generate masks for each text embeddings

        Args:
            image [B, 3, H, W]

        Returns:
            softmask [B, N, H, W]: softmasks for each text embeddings
        """

        H, W = image.shape[2:]  # original image shape

        # padded image size
        pH, pW = image.shape[2:]
        num_classes = text_emb.shape[0]
        batch_size = image.shape[0]

        image = image[:, [2, 1, 0], :, :]  # BGR to RGB
        ori_image = image.clone()
        
        img_preprocessed = self.image_transforms(image).to(device)
        if 'dinov2' in self.model_name:
            image_feat = self.model.forward_features(img_preprocessed)['x_norm_patchtokens']
        elif 'mae' in self.model_name or 'clip' in self.model_name or 'dino' in self.model_name:
            image_feat = self.model.forward_features(img_preprocessed)[:, 1:, :]
        elif 'sam' in self.model_name:
            self.model.forward_features(img_preprocessed)
            image_feat = feats['vit_out'].reshape(feats['vit_out'].shape[0], feats['vit_out'].shape[1]**2, feats['vit_out'].shape[-1]) # BS x N_PATCHES x EMBED_DIM
              
        batch_size, num_tokens, embed_dim = image_feat.shape
        if type(self.proj) == VisualProjectionLayer:
            image_feat = self.proj.project_dino(image_feat.float())
        if type(self.proj) == DoubleMLP:
            image_feat = self.proj.project_visual(image_feat.float())
        b, np, c = image_feat.shape
        np_h = np_w = int(sqrt(np))
        image_feat = image_feat.reshape(b, np_h, np_w, c).permute(0, 3, 1, 2)
        
        self_attn, self_attn_maps = self.process_self_attention(self.feats['self_attn'], batch_size, num_tokens + self.num_global_tokens, self.num_attn_heads, embed_dim, self.scale, self.num_global_tokens, ret_self_attn_maps=True)
        # normalize self_attn_maps, which has shape [B, N, P], in range min(self_attn_maps) ~ 1
        # self_attn_maps = self_attn_maps - self_attn_maps.min()
        # self_attn_maps = self_attn_maps / self_attn_maps.max()
        # self_attn_maps = self_attn_maps.mean(dim=1)
        # self_attn_maps = torch.nn.Sigmoid()(self_attn_maps)
        
        # proj_text_emb = self.proj.project_clip_txt(text_emb.float())

        ############### Generate mask ################
        # soft mask
        mask, simmap = self.masker.forward_seg(image_feat, text_emb, hard=False)  # [B, N, H', W']
        
        if with_bg:
            # mask = mask * self_attn_maps.reshape(mask.shape[0], 1, mask.shape[2], mask.shape[3])
            # mask = mask + lambda_bg * self_attn.reshape(mask.shape[0], 1, mask.shape[2], mask.shape[3])
            # rescale self_attn in range [min(mask), max(mask)]
            if background_func == "weighted_average":
                mask = self.weighted_average_with_self_attn(mask, self_attn, lambda_bg)
            if background_func == "weighted_average_sigmoid":
                mask = self.weighted_average_with_self_attn_sigmoid(mask, self_attn, lambda_bg)
            elif background_func == "weighted_average_head":
                mask = self.weighted_average_with_self_attn_heads(mask, self_attn_maps, lambda_bg)
            elif background_func == "similarity_assignment":
                mask = self.similarity_assignment(mask, image_feat, self_attn_maps, text_emb, lambda_bg)
            elif background_func == "similarity_assignment_sigmoid":
                mask = self.similarity_assignment_sigmoid(mask, image_feat, self_attn_maps, text_emb, lambda_bg)


        # resize
        mask = F.interpolate(mask, (pH, pW), mode='bilinear', align_corners=True)  # [B, N, H, W]


        if apply_pamr:
            for c in range(0, mask.shape[1], 30):
                mask[:, c:c + 30] = self.apply_pamr(ori_image, mask[:, c:c + 30])

        assert mask.shape[2] == H and mask.shape[3] == W, f"shape mismatch: ({H}, {W}) / {mask.shape}"

        return mask, simmap

    def weighted_average_with_self_attn(self, mask, self_attn, lambda_bg=0.2):
        # mask = mask * self_attn
        # mask = mask + lambda_bg * self_attn
        # rescale self_attn in range [min(mask), max(mask)]
        bs, num_classes, h, w = mask.shape
        min_ = self_attn.min().item()
        max_ = max(self_attn.max().item(), self_attn.max().item() - min_)
        self_attn = self_attn - min_
        self_attn = self_attn / max_
        min_ = mask.min().item()
        max_ = max(mask.max().item(), mask.max().item() - min_)
        self_attn = self_attn * (max_ - min_) + min_
        self_attn = self_attn.reshape(bs, 1, h, w)
        mask = (mask + lambda_bg * self_attn) / (1 + lambda_bg)
        return mask
    
    def weighted_average_with_self_attn_sigmoid(self, mask, self_attn, lambda_bg=0.2):
        # mask = mask * self_attn
        # mask = mask + lambda_bg * self_attn
        # rescale self_attn in range [min(mask), max(mask)]
        bs, num_classes, h, w = mask.shape
        # min_ = self_attn.min().item()
        # max_ = max(self_attn.max().item(), self_attn.max().item() - min_)
        # self_attn = self_attn - min_
        # self_attn = self_attn / max_
        # min_ = mask.min().item()
        # max_ = max(mask.max().item(), mask.max().item() - min_)
        # self_attn = self_attn * (max_ - min_) + min_
        self_attn = torch.sigmoid(self_attn.reshape(bs, 1, h, w))
        mask = (mask + lambda_bg * self_attn) / (1 + lambda_bg)
        return mask
    
    def weighted_average_with_self_attn_heads(self, mask, self_attn_maps, lambda_bg=0.2):
        _, num_heads, hw = self_attn_maps.shape
        bs, num_classes, h, w = mask.shape
        assert bs == 1, "Batch size must be 1"
        self_attn_maps = self_attn_maps.reshape(num_heads, hw) # [M, P]
        mask = mask.reshape(num_classes, hw) # [N, P]
        self_attn_maps_repeat = self_attn_maps.unsqueeze(0).repeat(num_classes, 1, 1) # [N, M, P]
        mask_repeat = mask.unsqueeze(1).repeat(1, num_heads, 1) # [N, M, P]
        min_mask = mask_repeat.min(dim=-1).values.unsqueeze(-1) # [N, M, 1]
        max_mask = mask_repeat.max(dim=-1).values.unsqueeze(-1) # [N, M, 1]
        
        min_self_attn = self_attn_maps_repeat.min(dim=-1).values.unsqueeze(-1) # [N, M, 1]
        max_self_attn = self_attn_maps_repeat.max(dim=-1).values.unsqueeze(-1) # [N, M, 1]
        max_self_attn = torch.cat([max_self_attn, max_self_attn - min_self_attn], dim=-1).max(dim=-1).values.unsqueeze(-1)
        self_attn_maps_repeat = self_attn_maps_repeat - min_self_attn
        self_attn_maps_repeat = self_attn_maps_repeat / max_self_attn
        
        self_attn_maps_repeat = self_attn_maps_repeat * (max_mask - min_mask) + min_mask
        squared_diff = ((mask_repeat - self_attn_maps_repeat) ** 2).sum(dim=-1)
        self_attn_maps_argmin = squared_diff.argmin(dim=-1)
        mask_output = (mask_repeat[torch.arange(num_classes), self_attn_maps_argmin] + lambda_bg * self_attn_maps_repeat[torch.arange(num_classes), self_attn_maps_argmin]).reshape(bs, num_classes, h, w) / (1 + lambda_bg)
        return mask_output
    
    def similarity_assignment(self, mask, image_feat, self_attn_maps, text_emb, lambda_bg=0.2):
        bs, c, h, w = image_feat.shape
        bs, num_classes, h, w = mask.shape
        bs, num_heads, hw = self_attn_maps.shape
        image_feat = image_feat.reshape(bs, c, hw)
        num_classes, c = text_emb.shape
        avg_head_embed = (self_attn_maps.unsqueeze(1) * image_feat.unsqueeze(2)).sum(dim=-1) # [B, C, M]
        avg_head_embed = avg_head_embed.permute(0, 2, 1) # [B, M, C]
        avg_head_embed = avg_head_embed / avg_head_embed.norm(dim=-1, keepdim=True)
        avg_head_embed = avg_head_embed.permute(0, 2, 1) # [B, C, M]
        head_text_sim = text_emb.unsqueeze(0) @ avg_head_embed # [B, M, N]
        head_text_sim = head_text_sim.argmax(dim=-1) # [B, M]
        
        assert bs == 1, "Batch size must be 1"
        self_attn_maps = self_attn_maps.reshape(num_heads, hw)
        self_attn_maps_repeat = self_attn_maps.unsqueeze(0).repeat(num_classes, 1, 1)
        mask = mask.reshape(num_classes, hw) # [N, P]
        mask_repeat = mask.unsqueeze(1).repeat(1, num_heads, 1)
        
        min_mask = mask.min().item()
        max_mask = mask.max().item()
        
        min_self_attn = self_attn_maps_repeat.min(dim=-1).values.unsqueeze(-1) # [N, M, 1]
        max_self_attn = self_attn_maps_repeat.max(dim=-1).values.unsqueeze(-1) # [N, M, 1]
        max_self_attn = torch.cat([max_self_attn, max_self_attn - min_self_attn], dim=-1).max(dim=-1).values.unsqueeze(-1)
        self_attn_maps_repeat = self_attn_maps_repeat - min_self_attn
        self_attn_maps_repeat = self_attn_maps_repeat / max_self_attn
        self_attn_maps_repeat = self_attn_maps_repeat * (max_mask - min_mask) + min_mask

        # mask_output = (mask_repeat[torch.arange(num_classes), head_text_sim] + lambda_bg * self_attn_maps_repeat[torch.arange(num_classes), head_text_sim]).reshape(bs, num_classes, h, w) / (1 + lambda_bg)
        mask_output = mask_repeat[torch.arange(num_classes), head_text_sim]
        mask_output[(mask_output <= 0.55) * (self_attn_maps_repeat[torch.arange(num_classes), head_text_sim] <= 0.55)] = 0
        mask_output = mask_output.reshape(bs, num_classes, h, w)
        return mask_output
    
    def similarity_assignment_sigmoid(self, mask, image_feat, self_attn_maps, text_emb, lambda_bg=0.2):
        bs, c, h, w = image_feat.shape
        bs, num_classes, h, w = mask.shape
        bs, num_heads, hw = self_attn_maps.shape
        image_feat = image_feat.reshape(bs, c, hw)
        num_classes, c = text_emb.shape
        avg_head_embed = (self_attn_maps.unsqueeze(1) * image_feat.unsqueeze(2)).sum(dim=-1) # [B, C, M]
        avg_head_embed = avg_head_embed.permute(0, 2, 1) # [B, M, C]
        avg_head_embed = avg_head_embed / avg_head_embed.norm(dim=-1, keepdim=True)
        avg_head_embed = avg_head_embed.permute(0, 2, 1) # [B, C, M]
        head_text_sim = text_emb.unsqueeze(0) @ avg_head_embed # [B, M, N]
        head_text_sim = head_text_sim.argmax(dim=-1) # [B, M]
        
        assert bs == 1, "Batch size must be 1"
        self_attn_maps = self_attn_maps.reshape(num_heads, hw)
        self_attn_maps_repeat = self_attn_maps.unsqueeze(0).repeat(num_classes, 1, 1)
        mask = mask.reshape(num_classes, hw) # [N, P]
        mask_repeat = mask.unsqueeze(1).repeat(1, num_heads, 1)
        
        # mask_output = (mask_repeat[torch.arange(num_classes), head_text_sim] + lambda_bg * self_attn_maps_repeat[torch.arange(num_classes), head_text_sim]).reshape(bs, num_classes, h, w) / (1 + lambda_bg)
        mask_output = mask_repeat[torch.arange(num_classes), head_text_sim]
        mask_output[(mask_output <= 0.55) * (torch.sigmoid(self_attn_maps_repeat[torch.arange(num_classes), head_text_sim]) <= 0.8)] = 0
        mask_output = mask_output.reshape(bs, num_classes, h, w)
        return mask_output