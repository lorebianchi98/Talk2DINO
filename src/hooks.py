import torch
feats = {}
def get_self_attention(module, input, output):
    feats['self_attn'] = output
    
def process_self_attention(output, batch_size, num_tokens, num_attn_heads, embed_dim, scale, num_global_tokens, ret_self_attn_maps=False):
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
    
def get_vit_out(model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
    feats['vit_out'] = output
    
def get_second_last_out(model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
    feats['second_last_out'] = output

def get_all_out_tokens(model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
    feats['clip_txt_out_tokens'] = output
    
def get_clip_second_last_dense_out(model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
    feats['clip_second_last_out'] = output.permute(1,0,2)

def get_dinov1_patches(model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
    feats['dinov1_patches'] = output

def get_all_out_tokens(model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
    feats['clip_txt_out_tokens'] = output
    
def average_text_tokens(text_embeddings, mask, keep_cls=False, keep_end_seq=False):
    if not keep_end_seq:
        mask[torch.arange(mask.shape[0]), mask.sum(dim=1) - 1] = False # excluding end of sequence
    if not keep_cls:
        mask[:, 0] = False # excluding CLS token
    
    
    masked_embeddings = text_embeddings * mask.unsqueeze(-1)  # shape: [BS, SEQ_LEN, 512]

    sum_embeddings = masked_embeddings.sum(dim=1)  # shape: [BS, 512]

    valid_elements = mask.sum(dim=1, keepdim=True)  # shape: [BS, 1]

    mean_embeddings = sum_embeddings / valid_elements  # shape: [BS, 512]
    
    return mean_embeddings
    