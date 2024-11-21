from copy import deepcopy
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.loss import ContrastiveLoss
import os
import matplotlib.pyplot as plt
import wandb
import numpy as np
import random
import json

def set_seed(seed):
    print(f'Setting seed {seed}...')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def const_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def train(model, train_dataloader, contrastive_loss, optimizer, scheduler=None, wandb=False, save_head_attivations=None, n_epochs=0):
    """train the model for one epoch"""
    train_batch_losses = []
    device = next(model.parameters()).device
    prev_iter = n_epochs * len(train_dataloader)
    
    head_attivations = []
    ann_ids = []
    img_ids = []
    for n_batch, batch in enumerate(tqdm(train_dataloader)):
        annotations = batch['annotation'].to(device, dtype=torch.float32)
        images = batch['image'].to(device)
        if 'text_argmax' in batch:
            text_argmax = batch['text_argmax'].to(device)
        else:
            text_argmax = None
        if 'self_attn_maps' in batch:
            self_attn_maps = batch['self_attn_maps'].to(device)
            cls = batch['dino_features'].to(device)
        else: 
            self_attn_maps = None
            cls = None
            
        if 'text_input_mask' in batch:
            text_input_mask = batch['text_input_mask'].to(device)
        else:
            text_input_mask = None
            
        if scheduler is not None:
            scheduler(n_batch + prev_iter)
                    
        if not save_head_attivations:
            loss = contrastive_loss(images, annotations, return_similarity_mat=False, self_attn_maps=self_attn_maps, cls=cls, text_input_mask=text_input_mask, text_argmax=text_argmax)
        else:
            loss, batch_head_attivations = contrastive_loss(images, annotations, return_similarity_mat=False, self_attn_maps=self_attn_maps, cls=cls, text_input_mask=text_input_mask, text_argmax=text_argmax, return_index=True)
            head_attivations.append(batch_head_attivations)
            ann_ids.append(batch['metadata']['annotation_id'])
            img_ids.append(batch['metadata']['image_id'])
        train_batch_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, norm_type=2.0)
        optimizer.step()
        if wandb:
            wandb.log({'train_loss': loss.item()})
            
    if save_head_attivations is not None:
        head_attivations = torch.cat(head_attivations)
        ann_ids = torch.cat(ann_ids)
        img_ids = torch.cat(img_ids)
        act_dict = {}
        for act, ann, img in zip(head_attivations, ann_ids, img_ids):
            act_dict[ann.item()] = {
                'image_id': img.item(),
                'activation_head': act.item()
            }
        with open(save_head_attivations, 'w') as f:
            json.dump(act_dict, f)
            print(f"Saved activation heads summary at {save_head_attivations}")
        
    return torch.mean(torch.tensor(train_batch_losses)).item()

def validate(model, val_dataloader, contrastive_loss, verbose=False):
    # evaluate the model in the validation set
    device = next(model.parameters()).device
    val_batch_losses = []
    
    val_dataloader = tqdm(val_dataloader) if verbose else val_dataloader
    for n_batch, batch in enumerate(val_dataloader):
        annotations = batch['annotation'].to(device, dtype=torch.float32)
        if 'text_argmax' in batch:
            text_argmax = batch['text_argmax'].to(device)
        else:
            text_argmax = None

        images = batch['image'].to(device)
        if 'self_attn_maps' in batch:
            self_attn_maps = batch['self_attn_maps'].to(device)
            cls = batch['dino_features'].to(device)
        else: 
            self_attn_maps = None
            cls = None
            
        if 'text_input_mask' in batch:
            text_input_mask = batch['text_input_mask'].to(device)
        else:
            text_input_mask = None
        
        with torch.no_grad():
            loss = contrastive_loss(images, annotations, return_similarity_mat=False, self_attn_maps=self_attn_maps, cls=cls, text_input_mask=text_input_mask, text_argmax=text_argmax)
    
        val_batch_losses.append(loss.item())
    return torch.mean(torch.tensor(val_batch_losses)).item()
    

def do_train(model, train_dataset, val_dataset, train_cfg, seed=123, optimizer_name="Adam", weight_decay=0.05, scheduler_name='linear', warmup=0, save_head_attivations=None):
    device = next(model.parameters()).device
    # setting manual seed
    # torch.manual_seed(seed)
    set_seed(seed)
    
    # mandatory parameters
    lr, ltype, num_epochs, batch_size = train_cfg['lr'], train_cfg['ltype'], train_cfg['num_epochs'], train_cfg['batch_size']
    # optional parameters
    margin = train_cfg.get('margin', 0.2)
    max_violation = train_cfg.get('max_violation', True)
    shuffle = train_cfg.get('shuffle', True)
    save_best_model = train_cfg.get('save_best_model', True)
    # early_stopping = train_cfg.get('early_stopping', 0) # 0 means no early-stopping
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    criterion = ContrastiveLoss(model, margin=margin, max_violation=max_violation, ltype=ltype)
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not implemented")
    total_steps = len(train_dataloader) * num_epochs
    if scheduler_name == 'linear' and warmup == 0:
        scheduler = None
    elif scheduler_name == 'linear' and warmup > 0:
        scheduler = const_lr(optimizer, lr, warmup, total_steps)
    elif scheduler_name == 'cosine':
        scheduler = cosine_lr(optimizer, lr, warmup, total_steps)
    
    # losses declaration
    train_losses = torch.zeros(num_epochs)
    val_losses = torch.zeros(num_epochs)
    for epoch in range(num_epochs):
        # train loss
        model.train()
        train_loss = train(model, train_dataloader, criterion, optimizer, scheduler, save_head_attivations=None if epoch < num_epochs - 1 else save_head_attivations, n_epochs=epoch)
        train_losses[epoch] = train_loss
        
        # eval loop
        model.eval()
        print("Performing Evaluation...")
        val_loss = validate(model, val_dataloader, criterion)
        val_losses[epoch] = val_loss
        
        print(f"Epoch {epoch}: train_loss={train_losses[epoch]} - val_loss={val_losses[epoch]}")
        # evaluating if best model and check early stopping
        if save_best_model and (epoch == 0 or val_losses[epoch] < min(val_losses[:epoch]).item()):
            print(f"Best validation loss, saving the model")
            best_model = deepcopy(model)
    
    model = model if not save_best_model else best_model

    return model, train_losses, val_losses