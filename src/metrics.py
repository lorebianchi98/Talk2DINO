import argparse
import numpy
from tqdm import tqdm
import torch
import json
import os

from src.model import ProjectionLayer


def i2t(images, captions, npts=None, return_ranks=False, model=None):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if model is not None:
        device = next(model.parameters()).device
    
    
    if npts is None:
        npts = images.shape[0] // 5
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in tqdm(range(npts)):

        # Get query image
        im = images[5 * index].reshape((1,) + images.shape[1:])

        # Compute scores
        if model is not None:
            ims_tensor = torch.tensor(im).to(device)
            queries_tensor = torch.tensor(captions).to(device)
            with torch.no_grad():
                d = model(ims_tensor.expand((queries_tensor.shape[0],) + (ims_tensor.shape[1:])), queries_tensor, ret_similarity_matrix=False).cpu().detach().numpy()
        else:
            captions = captions.astype(numpy.float32)
            im = im.astype(numpy.float32)
            captions = captions / numpy.linalg.norm(captions, axis=0)
            im = im / numpy.linalg.norm(im, axis=0)
            d = numpy.dot(im, captions.T).flatten()
        
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, return_ranks=False, model=None):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if model is not None:
        device = next(model.parameters()).device
    
    if npts is None:
        npts = images.shape[0] // 5
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)
    for index in tqdm(range(npts)):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
        if model is not None:
            ims_tensor = torch.tensor(ims).to(device)
            queries_tensor = torch.tensor(queries).to(device)
            with torch.no_grad():
                d = numpy.array([model(ims_tensor, query.unsqueeze(0).expand(ims_tensor.shape[0], -1), ret_similarity_matrix=False).cpu().detach().numpy() for query in queries_tensor])
        else:
            queries = queries.astype(numpy.float32)
            ims = ims.astype(numpy.float32)
            queries = queries / numpy.linalg.norm(queries, axis=0)
            ims = ims / numpy.linalg.norm(ims, axis=0)
            d = numpy.dot(queries, ims.T)
        
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def get_image_and_text_tensor(path, feature_name='dino_features', text_features='ann_feats', model=None, return_capts_and_imms=False):
    # model can be used in the case of feature_name == 'patch_tokens', in order to get the weights of the attention maps 
    data = torch.load(path)
        
    if model is None:
        images = {imm['id']: imm[feature_name] for imm in data['images']}
    else:
        
        device = next(model.parameters()).device
        images = {imm['id']: model.get_visual_embed(imm[feature_name].unsqueeze(0).to(device),
                                                    imm['self_attn_maps'].unsqueeze(0).to(device),
                                                    imm['dino_features'].unsqueeze(0).to(device) if model.weight_attn_heads == 'conditioned' else None
                                                    ).squeeze(0).detach().cpu()
                                                    for imm in data['images']}
    imm_paths = {imm['id']: imm['file_name'] for imm in data['images']}
    annotations = {}
    capts = {}
    
    def get_text_features(ann, text_features_name):
        if text_features_name == 'clip_txt_out_tokens_avg':
            mask = ann['text_input_mask']
            mask[mask.sum() - 1] = False # excluding end of sequence
            mask[0] = False # excluding CLS token
            return ann['clip_txt_out_tokens'][mask].mean(dim=0)
        else:
            return ann[text_features_name]
    
    for ann in data['annotations']:
        annotations[ann['image_id']] = [get_text_features(ann, text_features)] + annotations.get(ann['image_id'], [])
        capts[ann['image_id']] = [ann['caption']] + capts.get(ann['image_id'], [])
    
    imm_feats, ann_feats = None, None
    imm_file_names = []
    ann_texts = []
    for imm_id in tqdm(annotations.keys()):
        depth =  1 if len(images[imm_id].shape) == 1 else images[imm_id].shape[0]
        imm_feat = images[imm_id].expand(len(annotations[imm_id]), depth, -1)
        imm_file_names += [imm_paths[imm_id]] * 5
        ann_texts += capts[imm_id]
        
        if depth == 1:
            imm_feat = imm_feat.squeeze(dim=1)
        if ann_feats is None:
            ann_feats = torch.stack(annotations[imm_id])
            imm_feats = imm_feat
        else:
            ann_feats = torch.cat((ann_feats, torch.stack(annotations[imm_id])))
            imm_feats = torch.cat((imm_feats, imm_feat))
    
    if not return_capts_and_imms: 
        return imm_feats, ann_feats
    else:
        return imm_feats, ann_feats, imm_file_names, ann_texts
    
    
def main():
    # Usage example:
    # PYTHONPATH=. python src/metrics.py --config configs/vitl_mlp_infonce.yaml --weights weights/vitl_mlp_infonce.pth --test_data ../coco2014_l14_448/test.pth
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_alignment', default=False, action="store_true", help="If setted the alignment strategy will be used at test time") 
    parser.add_argument('--config', type=str, default=None, help="Config of the model") 
    parser.add_argument('--weights', type=str, default=None, help="Weights of the model. If the weights are None, the input features will not be projected")
    parser.add_argument('--img_features', type=str, default='avg_self_attn_out', help="Name of the field of the image features")
    parser.add_argument('--text_features', type=str, default='ann_feats', help="Name of the field of the text features")
    parser.add_argument('--test_data', type=str, default="../coco2014_b14_448/test.pth", help="Path of the test data")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    
    images, texts = get_image_and_text_tensor(args.test_data, args.img_features, text_features=args.text_features)
    
    print(f"Images {len(images)} -------- Texts {len(texts)}")
    print("Model results (t2i, i2t):")
    if not args.custom_alignment and args.weights is not None:
        proj = ProjectionLayer.from_config(args.config)
        proj.load_state_dict(torch.load(args.weights, 'cpu'))
        proj.to(device)
        texts = proj.project_clip_txt(texts.to(device).float()).detach().cpu()
    alignment = proj if args.custom_alignment else None
    t2i_res = t2i(images.numpy(), texts.numpy(), model=alignment)
    print(" & ".join(f"{x:.1f}" if i != 3 else f"{int(x)}" for i, x in enumerate(t2i_res)))
    i2t_res = i2t(images.numpy(), texts.numpy(), model=alignment)
    print(" & ".join(f"{x:.1f}" if i != 3 else f"{int(x)}" for i, x in enumerate(i2t_res)))
    
if __name__ == '__main__':
    main()