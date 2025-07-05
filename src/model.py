import clip
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.hooks import get_self_attention, process_self_attention, feats


class VisualProjectionLayer(nn.Module):
    """
    Creates a projection layer on top of the DINO encoder.
    The forward method calculate the similarity between the projected DINO token and the CLIP textual CLS token. 
    """
    def __init__(self, act=nn.Tanh(), hidden_layer=False, cosine=True, hidden_embed_dim=None, dino_embed_dim=1024, clip_embed_dim=512):
        # mlp_dims list of mlp dimensions
        super().__init__()
        if hidden_embed_dim is None:
            hidden_embed_dim = clip_embed_dim
        
        self.linear_layer = nn.Linear(dino_embed_dim, hidden_embed_dim)
        if hidden_layer:
            self.linear_layer2 = nn.Linear(hidden_embed_dim, clip_embed_dim) 
        self.act = act
        self.cosine = cosine
    
    @classmethod
    def from_config(cls, config):
        if type(config) is str:
            # if the configuration is a string, we treat it as a file path
            with open(config, 'r') as f:
                config = yaml.safe_load(f)['model']
        
        # loading the activation function
        act = config.get('act', None)
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'sigmoid':
            act = nn.Sigmoid()
        elif act is not None:
            raise Exception("Unknown activation function")
        
        model = cls(
            act=act,
            hidden_layer=config.get('hidden_layer', False),
            cosine=config.get('cosine', True),
            hidden_embed_dim=config.get('hidden_embed_dim', None) if config.get('hidden_layer', False) else None,
            dino_embed_dim=config.get('dino_embed_dim', 1024),
            clip_embed_dim=config.get('clip_embed_dim', 512)

        )
        return model
        
    
    def forward(self, visual_embedding, textual_embedding, ret_similarity_matrix=True, ret_embeds=False):
        visual_embedding = self.project_dino(visual_embedding)
        textual_embedding = textual_embedding.float()
        
        if self.cosine:
            textual_embedding = F.normalize(textual_embedding, p=2, dim=1)
            visual_embedding = F.normalize(visual_embedding, p=2, dim=1)
        if ret_embeds:
            return textual_embedding, visual_embedding
        x = textual_embedding @ visual_embedding.transpose(1, 0)
        if not ret_similarity_matrix:
            x = x[torch.eye(len(x)) > 0.5] # only diagonal elements
        
        return x
    
    def project_dino(self, visual_embedding):
        visual_embedding = visual_embedding.float()
        
        x = self.linear_layer(visual_embedding)
        if self.act:
            x = self.act(x)
        if hasattr(self, 'linear_layer2'):
            x = self.linear_layer2(x)
        
        return x
    
    def __len__(self):
        return sum(p.numel() for p in self.parameters())   



class ProjectionLayer(nn.Module):
    """
    Creates a projection layer on top of the CLIP-text encoder.
    The forward method calculate the similarity between the DINO CLS token and the projected CLIP textual CLS token. 
    """
    def __init__(self, act=nn.Tanh(), hidden_layer=False, cosine=True, dino_embed_dim=1024, clip_embed_dim=512, num_attn_head=16, weight_attn_heads=None,
                 alignment_strategy='max_score', alpha=0.6, keep_cls=False, keep_end_seq=False):
        # mlp_dims list of mlp dimensions
        super().__init__()
        self.num_attn_head = num_attn_head      
        
        self.linear_layer = nn.Linear(clip_embed_dim, dino_embed_dim)
        if hidden_layer:
            hidden_layer = 1 if hidden_layer is True else hidden_layer # ensuring compatibility with old code
            # self.linear_layer2 = nn.Linear(dino_embed_dim, dino_embed_dim) 
            self.hidden_layers = nn.ModuleList([nn.Linear(dino_embed_dim, dino_embed_dim) for _ in range(hidden_layer)])
        self.act = act
        self.cosine = cosine
        
        self.weight_attn_heads = weight_attn_heads
        if weight_attn_heads == 'static':
            self.attn_weights = nn.Parameter(torch.rand(self.num_attn_head))
        elif weight_attn_heads == 'conditioned':
            self.weight_layer1 = nn.Linear(dino_embed_dim, dino_embed_dim)
            self.weight_layer2 = nn.Linear(dino_embed_dim, self.num_attn_head)
            
        self.alignment_strategy = alignment_strategy # relevant only if we use disentangled_self_attn
        self.keep_cls = keep_cls # relevant only if we use clip_txt_tokens_out
        self.keep_end_seq = keep_end_seq # relevant only if we use clip_txt_tokens_out
        self.alpha = alpha
    
    @classmethod
    def from_config(cls, config):
        if type(config) is str:
            # if the configuration is a string, we treat it as a file path
            with open(config, 'r') as f:
                config = yaml.safe_load(f)['model']
        
        # loading the activation function
        act = config.get('act', None)
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'sigmoid':
            act = nn.Sigmoid()
        elif act is not None:
            raise Exception("Unknown activation function")
        
        model = cls(
            act=act,
            hidden_layer=config.get('hidden_layer', False),
            cosine=config.get('cosine', True),
            dino_embed_dim=config.get('dino_embed_dim', 1024),
            num_attn_head=config.get('num_attn_head', 16),
            clip_embed_dim=config.get('clip_embed_dim', 512),
            weight_attn_heads=config.get('weight_attn_heads', None),
            alignment_strategy=config.get('alignment_strategy', 'max_score'),
            alpha=config.get('alpha', 0.6),
            keep_cls=config.get('keep_cls', None),
            keep_end_seq=config.get('keep_end_seq', None),
        )
        if config.get('starting_checkpoint', None) is not None:
            model.load_state_dict(torch.load(config['starting_checkpoint'], 'cpu'))
        
        return model
    
    def compute_similarity(self, visual_embedding, textual_embedding, text_input_mask=None, return_index=False):
        if len(visual_embedding.shape) == 3 or len(textual_embedding.shape) == 3:
            # at least one embedding is decomposed: either we have all textual tokens or we have all the attention head tokens
            
            if self.alignment_strategy == 'weighted_avg':
                if len(visual_embedding.shape) != 3 or len(textual_embedding.shape) != 2:
                    raise Exception("Alignment strategy not implemented for this type of embeddings!")
                sims = torch.einsum('ik,ijk->ij', textual_embedding, visual_embedding)
                sims = sims.softmax(dim=-1)
                # in this case, we keep as visual_embedding the averaged token weighted by the text similarities
                visual_embedding = (visual_embedding * sims.unsqueeze(dim=-1)).mean(dim=1)
                sims = textual_embedding @ visual_embedding.transpose(1, 0)
                
            # in this case we sample the visual embedding from the softmax similarities of attention heads tokens and the textual tokens
            elif self.alignment_strategy == 'sampled_attn_map':
                if len(visual_embedding.shape) != 3 or len(textual_embedding.shape) != 2:
                    raise Exception("Alignment strategy not implemented for this type of embeddings!")
                sims = torch.einsum('ik,ijk->ij', textual_embedding, visual_embedding)
                sims = sims.softmax(dim=-1)
                # in this case, we sample from the distribution given byt text2attn-maps similarities the attention map to align
                index = torch.multinomial(sims, 1).view(-1, 1, 1).expand(-1, 1, visual_embedding.shape[-1])
                visual_embedding = torch.gather(visual_embedding, 1, index).squeeze(1)
                sims = textual_embedding @ visual_embedding.transpose(1, 0)
            
            elif self.alignment_strategy == 'max_score':
                sims = torch.einsum('ik,ijk->ij', textual_embedding, visual_embedding)
                sims = sims.softmax(dim=-1)
                index = sims.argmax(dim=-1)
                index_reshaped = sims.argmax(dim=-1).view(-1, 1, 1).expand(-1, 1, visual_embedding.shape[-1])
                visual_embedding = torch.gather(visual_embedding, 1, index_reshaped).squeeze(1)
                sims = textual_embedding @ visual_embedding.transpose(1, 0)
            else:
                # in this case we construct a similarity matrix between attention head tokens and textual tokens
                
                # we ensure that both the batch embeddings have the same number of dimensions
                textual_embedding = textual_embedding.unsqueeze(1) if len(textual_embedding.shape) == 2 else textual_embedding
                visual_embedding = visual_embedding.unsqueeze(1) if len(visual_embedding.shape) == 2 else visual_embedding
                if textual_embedding.shape[1] > 1:
                    assert text_input_mask is not None, "If we use all the textual embeddings, we need the input mask"
                    if not self.keep_end_seq:
                        # we take the last True value of the mask and we set it to False
                        text_input_mask[torch.arange(text_input_mask.shape[0]), torch.sum(text_input_mask, dim=1) - 1] = False
                    if not self.keep_cls:
                        text_input_mask[:, 0] = False

                # do not consider cls and eos tokens
                im_set = visual_embedding
                s_seq = textual_embedding

                im_set_batch = im_set.size(0)
                im_set_len = im_set.size(1)
                s_seq_batch = s_seq.size(0)
                s_seq_len = s_seq.size(1)

                im_set = im_set.unsqueeze(1).expand(-1, s_seq_batch, -1, -1)  # B x B x S_im x dim
                s_seq = s_seq.unsqueeze(0).expand(im_set_batch, -1, -1, -1) # B x B x S_s x dim
                alignments = torch.matmul(im_set, s_seq.permute(0, 1, 3, 2))  # B x B x S_im x S_s

                # compute mask for the alignments tensor
                if text_input_mask is not None:
                    alignment_mask = text_input_mask.unsqueeze(1).unsqueeze(0).expand(im_set_batch, -1, im_set_len, -1).logical_not()

                    alignments.masked_fill_(alignment_mask, value=0)
                # alignments = F.relu(alignments)
                # alignments = F.normalize(alignments,p=2, dim=2)

                if self.alignment_strategy == 'sum':
                    sims = alignments.sum(dim=(2,3))
                elif self.alignment_strategy == 'mean':
                    sims = alignments.mean(dim=(2,3))
                elif self.alignment_strategy == 'max-row_sum':
                    sims = alignments.max(2)[0].sum(2)
                elif self.alignment_strategy == 'nucleus-sampling':
                    max_alignments = alignments.max(2)[0]
                    sorted_alignments = max_alignments.sort(dim=2, descending=True)[0]
                    # min-max normalization
                    mins = sorted_alignments.min(2)[0].unsqueeze(-1).expand(-1, -1, s_seq_len)
                    maxs = sorted_alignments.max(2)[0].unsqueeze(-1).expand(-1, -1, s_seq_len)
                    norm_alignments = ((sorted_alignments - mins) / (maxs - mins)) 
                    # transform values in percentage
                    sums = norm_alignments.sum(dim=-1).unsqueeze(-1).expand(-1, -1, s_seq_len)
                    norm_alignments = norm_alignments / sums
                    # finding the element indices which surpasses alpha
                    cumsums = norm_alignments.cumsum(2)
                    indices = torch.argmax((cumsums > self.alpha).int() + 1, dim=2)

                    mask = torch.arange(s_seq_len).unsqueeze(0).unsqueeze(0).expand(s_seq_batch, s_seq_batch, s_seq_len).to(indices.device) < indices.unsqueeze(-1).expand(-1, -1, s_seq_len) + 1
                    relevant_alignments = (sorted_alignments * mask)
                    sims = relevant_alignments.sum(dim=2)
        else:
            # default case: dot-product
            sims = textual_embedding @ visual_embedding.transpose(1, 0)
        
        if not return_index:
            return sims
        else:
            return sims, index
        
        
    
    def forward(self, visual_embedding, textual_embedding, ret_similarity_matrix=True, ret_embeds=False, self_attn_maps=None, cls=None, text_input_mask=None, return_index=False):
        if self.weight_attn_heads is not None:
            assert self_attn_maps is not None, "In case we have attention maps weights, we have to weight patch tokens mean by the weighted self-attention maps"
            visual_embedding = self.get_visual_embed(visual_embedding, self_attn_maps=self_attn_maps, cls=cls)    
        
        textual_embedding = self.project_clip_txt(textual_embedding)
        
        if self.cosine:
            textual_embedding = F.normalize(textual_embedding, p=2, dim=-1)
            visual_embedding = F.normalize(visual_embedding, p=2, dim=-1)
            
            
        if ret_embeds:
            return textual_embedding, visual_embedding
        
        if not return_index:
            x = self.compute_similarity(visual_embedding, textual_embedding, text_input_mask, return_index)
        else:
            x, index = self.compute_similarity(visual_embedding, textual_embedding, text_input_mask, return_index)
            
        if not ret_similarity_matrix:
            x = x[torch.eye(len(x)) > 0.5] # only diagonal elements
        
        if not return_index:
            return x
        else:
            return x, index
    
    def get_visual_embed(self, visual_embedding, self_attn_maps=None, cls=None):
        if self_attn_maps is not None:
            # we weight each attention head to obtain a weighted self-attention map 
            assert len(visual_embedding.shape) == 3, "In case we have attention maps weights, the visual_embedding should contain patch embeddings, with shape BS x NUM_PATCHES x EMBED_DIM"
            if self.weight_attn_heads == 'conditioned':
                assert cls is not None, "cls must be setted in case of dinamic attention weighting"
                x = self.weight_layer1(cls)
                x = self.act(x)
                x = self.weight_layer2(x)
                normalized_attn_weights = x.softmax(dim=1)
                self_attn = (self_attn_maps * normalized_attn_weights.unsqueeze(dim=-1)).mean(dim=1)
            else:
                normalized_attn_weights = self.attn_weights.softmax(dim=0)
                self_attn = (self_attn_maps * normalized_attn_weights.view(1, normalized_attn_weights.shape[0], 1)).mean(dim=1)
            self_attn = self_attn.softmax(dim=-1)   
            
            # then we perform the weighted mean of patches
            visual_embedding = (self_attn.unsqueeze(-1) * visual_embedding).mean(dim=1)   
        return visual_embedding
    
    def project_clip_txt(self, textual_embedding):
        textual_embedding = textual_embedding.float()
        x = self.linear_layer(textual_embedding)
        
        if hasattr(self, 'hidden_layers'):
            for hidden_layer in self.hidden_layers:
                if self.act:
                    x = self.act(x)
                x = hidden_layer(x)
            
        return x
    def load_state_dict(self, state_dict, strict=True):
        # compatibility with old code
        if 'linear_layer2.weight' in state_dict:
            state_dict['hidden_layers.0.weight'] = state_dict.pop('linear_layer2.weight')
            state_dict['hidden_layers.0.bias'] = state_dict.pop('linear_layer2.bias')
        # Call the parent class's load_state_dict with the modified state_dict
        super(ProjectionLayer, self).load_state_dict(state_dict, strict)
    
    def set_alignment_strategy(self, alignment_strategy):
        self.alignment_strategy = alignment_strategy
        return
    
    def __len__(self):
        return sum(p.numel() for p in self.parameters())   
    
class DoubleMLP(nn.Module):
    def __init__(self, act=nn.Tanh(), hidden_layer=False, cosine=True, dino_embed_dim=1024, clip_embed_dim=512, num_attn_head=16, weight_attn_heads=None,
                 alignment_strategy='max_score', alpha=0.6, keep_cls=False, keep_end_seq=False):
        super().__init__()
        self.num_attn_head = num_attn_head
        
        self.linear_layer = nn.Linear(clip_embed_dim, dino_embed_dim)
        if hidden_layer:
            hidden_layer = 1 if hidden_layer is True else hidden_layer # ensuring compatibility with old code
            # self.linear_layer2 = nn.Linear(dino_embed_dim, dino_embed_dim) 
            self.hidden_layers = nn.ModuleList([nn.Linear(dino_embed_dim, dino_embed_dim) for _ in range(hidden_layer)])
        self.act = act
        self.cosine = cosine
        
        self.weight_attn_heads = weight_attn_heads
        if weight_attn_heads == 'static':
            self.attn_weights = nn.Parameter(torch.rand(self.num_attn_head))
        elif weight_attn_heads == 'conditioned':
            self.weight_layer1 = nn.Linear(dino_embed_dim, dino_embed_dim)
            self.weight_layer2 = nn.Linear(dino_embed_dim, self.num_attn_head)
            
        self.alignment_strategy = alignment_strategy # relevant only if we use disentangled_self_attn
        self.keep_cls = keep_cls # relevant only if we use clip_txt_tokens_out
        self.keep_end_seq = keep_end_seq # relevant only if we use clip_txt_tokens_out
        self.alpha = alpha
        
        self.visual_linear = nn.Linear(dino_embed_dim, dino_embed_dim)
        if hidden_layer:
            hidden_layer = 1 if hidden_layer is True else hidden_layer # ensuring compatibility with old code
            self.visual_hidden_layers = nn.ModuleList([nn.Linear(dino_embed_dim, dino_embed_dim) for _ in range(hidden_layer)])
        
    @classmethod
    def from_config(cls, config):
        if type(config) is str:
            # if the configuration is a string, we treat it as a file path
            with open(config, 'r') as f:
                config = yaml.safe_load(f)['model']
        
        # loading the activation function
        act = config.get('act', None)
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'sigmoid':
            act = nn.Sigmoid()
        elif act is not None:
            raise Exception("Unknown activation function")
        
        model = cls(
            act=act,
            hidden_layer=config.get('hidden_layer', False),
            cosine=config.get('cosine', True),
            dino_embed_dim=config.get('dino_embed_dim', 1024),
            num_attn_head=config.get('num_attn_head', 16),
            clip_embed_dim=config.get('clip_embed_dim', 512),
            weight_attn_heads=config.get('weight_attn_heads', None),
            alignment_strategy=config.get('alignment_strategy', 'max_score'),
            alpha=config.get('alpha', 0.6),
            keep_cls=config.get('keep_cls', None),
            keep_end_seq=config.get('keep_end_seq', None),
        )
        if config.get('starting_checkpoint', None) is not None:
            model.load_state_dict(torch.load(config['starting_checkpoint'], 'cpu'))
        
        return model
    
    def compute_similarity(self, visual_embedding, textual_embedding, text_input_mask=None):
        if len(visual_embedding.shape) == 3 or len(textual_embedding.shape) == 3:
            # at least one embedding is decomposed: either we have all textual tokens or we have all the attention head tokens
            
            if self.alignment_strategy == 'weighted_avg':
                if len(visual_embedding.shape) != 3 or len(textual_embedding.shape) != 2:
                    raise Exception("Alignment strategy not implemented for this type of embeddings!")
                sims = torch.einsum('ik,ijk->ij', textual_embedding, visual_embedding)
                sims = sims.softmax(dim=-1)
                # in this case, we keep as visual_embedding the averaged token weighted by the text similarities
                visual_embedding = (visual_embedding * sims.unsqueeze(dim=-1)).mean(dim=1)
                sims = textual_embedding @ visual_embedding.transpose(1, 0)
                
            # in this case we sample the visual embedding from the softmax similarities of attention heads tokens and the textual tokens
            elif self.alignment_strategy == 'sampled_attn_map':
                if len(visual_embedding.shape) != 3 or len(textual_embedding.shape) != 2:
                    raise Exception("Alignment strategy not implemented for this type of embeddings!")
                sims = torch.einsum('ik,ijk->ij', textual_embedding, visual_embedding)
                sims = sims.softmax(dim=-1)
                # in this case, we sample from the distribution given byt text2attn-maps similarities the attention map to align
                index = torch.multinomial(sims, 1).view(-1, 1, 1).expand(-1, 1, visual_embedding.shape[-1])
                visual_embedding = torch.gather(visual_embedding, 1, index).squeeze(1)
                sims = textual_embedding @ visual_embedding.transpose(1, 0)
            
            elif self.alignment_strategy == 'max_score':
                sims = torch.einsum('ik,ijk->ij', textual_embedding, visual_embedding)
                sims = sims.softmax(dim=-1)
                index = sims.argmax(dim=-1).view(-1, 1, 1).expand(-1, 1, visual_embedding.shape[-1])
                visual_embedding = torch.gather(visual_embedding, 1, index).squeeze(1)
                sims = textual_embedding @ visual_embedding.transpose(1, 0)
            else:
                # in this case we construct a similarity matrix between attention head tokens and textual tokens
                
                # we ensure that both the batch embeddings have the same number of dimensions
                textual_embedding = textual_embedding.unsqueeze(1) if len(textual_embedding.shape) == 2 else textual_embedding
                visual_embedding = visual_embedding.unsqueeze(1) if len(visual_embedding.shape) == 2 else visual_embedding
                if textual_embedding.shape[1] > 1:
                    assert text_input_mask is not None, "If we use all the textual embeddings, we need the input mask"
                    if not self.keep_end_seq:
                        # we take the last True value of the mask and we set it to False
                        text_input_mask[torch.arange(text_input_mask.shape[0]), torch.sum(text_input_mask, dim=1) - 1] = False
                    if not self.keep_cls:
                        text_input_mask[:, 0] = False

                # do not consider cls and eos tokens
                im_set = visual_embedding
                s_seq = textual_embedding

                im_set_batch = im_set.size(0)
                im_set_len = im_set.size(1)
                s_seq_batch = s_seq.size(0)
                s_seq_len = s_seq.size(1)

                im_set = im_set.unsqueeze(1).expand(-1, s_seq_batch, -1, -1)  # B x B x S_im x dim
                s_seq = s_seq.unsqueeze(0).expand(im_set_batch, -1, -1, -1) # B x B x S_s x dim
                alignments = torch.matmul(im_set, s_seq.permute(0, 1, 3, 2))  # B x B x S_im x S_s

                # compute mask for the alignments tensor
                if text_input_mask is not None:
                    alignment_mask = text_input_mask.unsqueeze(1).unsqueeze(0).expand(im_set_batch, -1, im_set_len, -1).logical_not()

                    alignments.masked_fill_(alignment_mask, value=0)
                # alignments = F.relu(alignments)
                # alignments = F.normalize(alignments,p=2, dim=2)

                if self.alignment_strategy == 'sum':
                    sims = alignments.sum(dim=(2,3))
                elif self.alignment_strategy == 'mean':
                    sims = alignments.mean(dim=(2,3))
                elif self.alignment_strategy == 'max-row_sum':
                    sims = alignments.max(2)[0].sum(2)
                elif self.alignment_strategy == 'nucleus-sampling':
                    max_alignments = alignments.max(2)[0]
                    sorted_alignments = max_alignments.sort(dim=2, descending=True)[0]
                    # min-max normalization
                    mins = sorted_alignments.min(2)[0].unsqueeze(-1).expand(-1, -1, s_seq_len)
                    maxs = sorted_alignments.max(2)[0].unsqueeze(-1).expand(-1, -1, s_seq_len)
                    norm_alignments = ((sorted_alignments - mins) / (maxs - mins)) 
                    # transform values in percentage
                    sums = norm_alignments.sum(dim=-1).unsqueeze(-1).expand(-1, -1, s_seq_len)
                    norm_alignments = norm_alignments / sums
                    # finding the element indices which surpasses alpha
                    cumsums = norm_alignments.cumsum(2)
                    indices = torch.argmax((cumsums > self.alpha).int() + 1, dim=2)

                    mask = torch.arange(s_seq_len).unsqueeze(0).unsqueeze(0).expand(s_seq_batch, s_seq_batch, s_seq_len).to(indices.device) < indices.unsqueeze(-1).expand(-1, -1, s_seq_len) + 1
                    relevant_alignments = (sorted_alignments * mask)
                    sims = relevant_alignments.sum(dim=2)
        else:
            # default case: dot-product
            sims = textual_embedding @ visual_embedding.transpose(1, 0)
        
        return sims
        
        
    
    def forward(self, visual_embedding, textual_embedding, ret_similarity_matrix=True, ret_embeds=False, self_attn_maps=None, cls=None, text_input_mask=None):
        if self.weight_attn_heads is not None:
            assert self_attn_maps is not None, "In case we have attention maps weights, we have to weight patch tokens mean by the weighted self-attention maps"
            visual_embedding = self.get_visual_embed(visual_embedding, self_attn_maps=self_attn_maps, cls=cls) 
        
        visual_embedding = self.project_visual(visual_embedding)   
        
        textual_embedding = self.project_clip_txt(textual_embedding)
        
        if self.cosine:
            textual_embedding = F.normalize(textual_embedding, p=2, dim=-1)
            visual_embedding = F.normalize(visual_embedding, p=2, dim=-1)
            
            
        if ret_embeds:
            return textual_embedding, visual_embedding
        
        x = self.compute_similarity(visual_embedding, textual_embedding, text_input_mask)
        if not ret_similarity_matrix:
            x = x[torch.eye(len(x)) > 0.5] # only diagonal elements
        
        return x
    
    def get_visual_embed(self, visual_embedding, self_attn_maps=None, cls=None):
        if self_attn_maps is not None:
            # we weight each attention head to obtain a weighted self-attention map 
            assert len(visual_embedding.shape) == 3, "In case we have attention maps weights, the visual_embedding should contain patch embeddings, with shape BS x NUM_PATCHES x EMBED_DIM"
            if self.weight_attn_heads == 'conditioned':
                assert cls is not None, "cls must be setted in case of dinamic attention weighting"
                x = self.weight_layer1(cls)
                x = self.act(x)
                x = self.weight_layer2(x)
                normalized_attn_weights = x.softmax(dim=1)
                self_attn = (self_attn_maps * normalized_attn_weights.unsqueeze(dim=-1)).mean(dim=1)
            else:
                normalized_attn_weights = self.attn_weights.softmax(dim=0)
                self_attn = (self_attn_maps * normalized_attn_weights.view(1, normalized_attn_weights.shape[0], 1)).mean(dim=1)
            self_attn = self_attn.softmax(dim=-1)   
            
            # then we perform the weighted mean of patches
            visual_embedding = (self_attn.unsqueeze(-1) * visual_embedding).mean(dim=1)   
        return visual_embedding
    
    def project_clip_txt(self, textual_embedding):
        textual_embedding = textual_embedding.float()
        x = self.linear_layer(textual_embedding)
        
        for hidden_layer in self.hidden_layers:
            if self.act:
                x = self.act(x)
            x = hidden_layer(x)
            
        return x
    
    def project_visual(self, visual_embedding):
        visual_embedding = visual_embedding.float()
        x = self.visual_linear(visual_embedding)
        
        for hidden_layer in self.visual_hidden_layers:
            if self.act:
                x = self.act(x)
            x = hidden_layer(x)
            
        return x
    
    def load_state_dict(self, state_dict, strict=True):
        # compatibility with old code
        if 'linear_layer2.weight' in state_dict:
            state_dict['hidden_layers.0.weight'] = state_dict.pop('linear_layer2.weight')
            state_dict['hidden_layers.0.bias'] = state_dict.pop('linear_layer2.bias')
        # Call the parent class's load_state_dict with the modified state_dict
        super(DoubleMLP, self).load_state_dict(state_dict, strict)
    
    def set_alignment_strategy(self, alignment_strategy):
        self.alignment_strategy = alignment_strategy
        return
    
    def __len__(self):
        return sum(p.numel() for p in self.parameters())   

    
class CLIPLastLayer(nn.Module):
    def __init__(self,  act=nn.Tanh(), hidden_layer=False, cosine=True, dino_embed_dim=1024, clip_embed_dim=512, weight_attn_heads=None, alignment_strategy='max_score', clip_model='ViT-B/16', text_input_mask=None, projection_weights=None):
        import clip
        super().__init__()
        self.clip_model, _ = clip.load(clip_model)
        self.clip_model.to(dtype=torch.float32)
        # self.last_resblock = copy.deepcopy(self.clip_model.transformer.resblocks[-1])
        self.last_resblock = self.clip_model.transformer.resblocks[-1]
        # self.last_resblock.requires_grad_(False)
        # self.last_ln = copy.deepcopy(self.clip_model.ln_final)
        self.last_ln = self.clip_model.ln_final
        # self.last_ln.requires_grad_(False)
        # self.clip_text_proj = copy.deepcopy(self.clip_model.text_projection)
        self.clip_text_proj = self.clip_model.text_projection
        # self.clip_text_proj.requires_grad_(False)
        self.clip_dtype = self.clip_model.dtype
        del self.clip_model
                
        self.projection_layer = ProjectionLayer(act=act, hidden_layer=hidden_layer, cosine=cosine, dino_embed_dim=dino_embed_dim,
                                                clip_embed_dim=clip_embed_dim, weight_attn_heads=weight_attn_heads, alignment_strategy=alignment_strategy)
        
        if projection_weights is not None:
            self.projection_layer.load_state_dict(torch.load(projection_weights, 'cpu'))
        
    def forward(self, visual_embedding, textual_embedding, ret_similarity_matrix=True, ret_embeds=False, self_attn_maps=None, cls=None, text_argmax=None, text_input_mask=None):
        x = self.last_resblock(textual_embedding.permute(1, 0, 2))
        x = x.permute(1, 0, 2)
        x = self.last_ln(x).type(self.clip_dtype)
        x = x[torch.arange(x.shape[0]), text_argmax] @ self.clip_text_proj
        if ret_embeds:
            textual_embedding, visual_embedding = self.projection_layer(visual_embedding, x, ret_similarity_matrix=ret_similarity_matrix, ret_embeds=ret_embeds, self_attn_maps=self_attn_maps, cls=cls)
            return textual_embedding, visual_embedding
        x = self.projection_layer(visual_embedding, x, ret_similarity_matrix=ret_similarity_matrix, ret_embeds=ret_embeds, self_attn_maps=self_attn_maps, cls=cls)
        return x
    
    def project_clip_txt(self, textual_embedding, text_argmax):
        x = self.last_resblock(textual_embedding.permute(1, 0, 2))
        x = x.permute(1, 0, 2)
        x = self.last_ln(x).type(self.clip_dtype)
        x = x[torch.arange(x.shape[0]), text_argmax] @ self.clip_text_proj
        x = self.projection_layer.project_clip_txt(x)
        return x
    
    @classmethod
    def from_config(cls, config):
        if type(config) is str:
            # if the configuration is a string, we treat it as a file path
            with open(config, 'r') as f:
                config = yaml.safe_load(f)['model']
        
        # loading the activation function
        act = config.get('act', None)
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'sigmoid':
            act = nn.Sigmoid()
        elif act is not None:
            raise Exception("Unknown activation function")
        
        model = cls(
            act=act,
            hidden_layer=config.get('hidden_layer', False),
            cosine=config.get('cosine', True),
            dino_embed_dim=config.get('dino_embed_dim', 1024),
            clip_embed_dim=config.get('clip_embed_dim', 512),
            weight_attn_heads=config.get('weight_attn_heads', None),
            alignment_strategy=config.get('alignment_strategy', 'max_score'),
            clip_model=config.get('clip_model', 'ViT-B/16'),
            projection_weights=config.get('projection_weights', None),
            
        )
        if config.get('starting_checkpoint', None) is not None:
            model.load_state_dict(torch.load(config['starting_checkpoint'], 'cpu'))
        
        return model
    
    def __len__(self):
        return sum(p.numel() for p in self.parameters())   
    
class DinoText(nn.Module):
    """
    Project images and texts into DINOv2 latent space.
    """
    def __init__(self, dino_cfg="dinov2_vitl14_reg", clip_cfg="ViT-B/16", projection_cfg="configs/linear.yaml", projection_weights="weights/linear_avg_self_attn_out.pth", freeze_text_encoder=True, avg_self_attn_token=True, use_disentangled_self_attn=False):
        super().__init__()
        # DINO parameters
        self.num_global_tokens = 1 if "reg" not in dino_cfg else 5
        self.embed_dim = 1024 if "vitl" in dino_cfg else 768
        self.num_attn_heads = 16
        self.scale = 0.125
        
        self.visual_backbone = torch.hub.load('facebookresearch/dinov2', dino_cfg)
        self.text_backbone, _ = clip.load(clip_cfg)
        self.clip2dino_proj = ProjectionLayer.from_config(projection_cfg)
        if projection_weights is not None:
            self.clip2dino_proj.load_state_dict(torch.load(projection_weights, 'cpu'))
        self.use_avg_self_attn = avg_self_attn_token
        self.use_disentangled_self_attn = use_disentangled_self_attn
        if self.use_avg_self_attn or self.use_disentangled_self_attn:
            self.visual_backbone.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)
        if self.use_disentangled_self_attn:
            self.visual_backbone.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)
        if freeze_text_encoder:
            self.text_backbone.eval()
            self.text_backbone.requires_grad_(False)
        self.avg_self_attn_token = avg_self_attn_token
        if self.avg_self_attn_token or self.use_disentangled_self_attn:
            self.visual_backbone.blocks[-1].attn.qkv.register_forward_hook(self.get_self_attention)
            self.feats = {}
            self.num_global_tokens = 1 if "reg" not in dino_cfg else 5
            self.num_attn_heads = 16
            self.scale = 0.125

    
    @classmethod
    def from_config(cls, cfg):
        if type(cfg) is str:
            # if the configuration is a string, we treat it as a file path
            with open(cfg, 'r') as f:
                cfg = yaml.safe_load(f)['model']
        
        model = cls(
            dino_cfg=cfg.get('dino_cfg', "dinov2_vitl14_reg"),
            clip_cfg=cfg.get('clip_cfg', "ViT-B/16"),
            projection_cfg=cfg.get('projection_cfg', "configs/linear.yaml"),
            projection_weights=cfg.get('projection_weights', None),
            avg_self_attn_token=cfg.get('use_avg_self_attn', False),
            use_disentangled_self_attn=cfg.get('use_disentangled_self_attn', False),
        )
        return model
    
    def encode_text(self, tokenized_texts):
        x = self.text_backbone.encode_text(tokenized_texts)
        x = self.clip2dino_proj.project_clip_txt(x)
        return x
    
    def encode_image(self, images):
        batch_size, _, _, _ = images.shape
        x = self.visual_backbone(images, is_training=self.avg_self_attn_token or self.use_disentangled_self_attn)
        if self.avg_self_attn_token:
            batch_size, num_tokens, embed_dim = x['x_norm_patchtokens'].shape
            num_tokens = num_tokens + self.num_global_tokens
            self_attn = self.process_self_attention(self.feats['self_attn'], batch_size, num_tokens, self.num_attn_heads, embed_dim, self.scale, self.num_global_tokens)
            x = (self_attn.unsqueeze(-1) * x['x_norm_patchtokens']).mean(dim=1)
        if self.use_disentangled_self_attn:
            batch_size, num_tokens, embed_dim = x['x_norm_patchtokens'].shape
            num_tokens = num_tokens + self.num_global_tokens
            self_attn, self_attn_maps = self.process_self_attention(self.feats['self_attn'], batch_size, num_tokens, self.num_attn_heads, embed_dim, self.scale, self.num_global_tokens, ret_self_attn_maps=True)
            self_attn_maps = self_attn_maps.softmax(dim=-1)
            x = (x['x_norm_patchtokens'].unsqueeze(1) * self_attn_maps.unsqueeze(-1)).mean(dim=2)
        return x
    
    def get_self_attention(self, module, input, output):
        self.feats['self_attn'] = output
        
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
        
    def forward(self, images, tokenized_texts, cosine=True, ret_similarity_matrix=True):
        img_embed = self.encode_image(images)
        txt_embed = self.encode_text(tokenized_texts)
        
        if cosine:
            img_embed = F.normalize(img_embed, p=2, dim=1)
            txt_embed = F.normalize(txt_embed, p=2, dim=1)
        x = img_embed @ txt_embed.transpose(1, 0)
        if not ret_similarity_matrix:
            x = x[torch.eye(len(x)) > 0.5] # only diagonal elements
        
        return x
    
    def __len__(self):
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())
        return count_parameters(self.visual_backbone) + count_parameters(self.clip2dino_proj) + count_parameters(self.text_backbone.transformer)

