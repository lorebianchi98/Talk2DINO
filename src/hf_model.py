import torch
import clip
from huggingface_hub import PyTorchModelHubMixin
import sys
sys.path.append("src/open_vocabulary_segmentation")
from models.dinotext import DINOText


class Talk2DINO(DINOText, PyTorchModelHubMixin):
    def encode_text(self, texts):
        """ texts: string or list of strings
         returns: text embeddings (N, D) where N is the number of texts, D is the embedding dimension
        """
        text_tokens = clip.tokenize(texts).to(self.parameters().__next__().device)
        txt_embed = self.clip_model.encode_text(text_tokens)
        txt_embed = self.proj.project_clip_txt(txt_embed)
        return txt_embed
    
    def encode_image(self, images):
        """ images: PIL image or list of PIL images
         returns: image embeddings (N, L, D) where N is the number of images, L is the number of patches, D is the embedding dimension
        """
        if type(images) is not list:
            images = [images]
        img_preprocessed = [self.image_transforms(img).to(next(self.parameters()).device) for img in images]
        img_preprocessed = torch.stack(img_preprocessed)
        if 'dinov2' in self.model_name or 'dinov3' in self.model_name:
            img_embed = self.model.forward_features(img_preprocessed)['x_norm_patchtokens']
        elif 'mae' in self.model_name or 'clip' in self.model_name or 'dino' in self.model_name:
            img_embed = self.model.forward_features(img_preprocessed)[:, 1:, :]
              
        return img_embed