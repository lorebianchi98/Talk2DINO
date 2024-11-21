import argparse
import os
import json
import webdataset as wds
import re
import tarfile
import torch

from io import BytesIO
from tqdm import tqdm

def create_webdataset_single_shard(data, output_tar_path):
# Function to create WebDataset tar file from COCO format
    if os.path.exists(output_tar_path):
        os.remove(output_tar_path)
        print(f"Old {output_tar_path} has been deleted successfully.")
    # Open tarfile for writing
    with tarfile.open(output_tar_path, "w") as tar:
        # Create a dictionary mapping image_id to annotations
        annotations_map = {ann['image_id']: ann for ann in data['annotations']}
        
        # Iterate through the images
        for image in data['images']:
            image_id = image['id']
            
            # Get the corresponding annotation for the image
            if image_id in annotations_map:
                annotation = annotations_map[image_id]
                # filling annotation field with the ones coming from the image
                for field in image.keys():
                    if field == 'id':
                        continue
                    annotation[field] = image[field]
                
                # saving the file as pth file
                buffer = BytesIO()
                torch.save(annotation, buffer)
                buffer.seek(0)
                
                annotation_tarinfo = tarfile.TarInfo(name=f"{image_id}.pth")
                annotation_tarinfo.size = buffer.getbuffer().nbytes
                tar.addfile(annotation_tarinfo, buffer)
    print(f"{output_tar_path} created succesfully!")
    
def create_webdataset_tar(data, output_tar_dir, n_shards=1, offset=0):
    """
    n_shards: number of shards in which we want to split the tar output file
    offset: index of the first shard to save
    """
    step_ann = len(data['annotations']) // n_shards
    step_imm = len(data['images']) // n_shards
    for i in range(n_shards):
        start_ann = step_ann * i
        end_ann = (step_ann * (i + 1) - 1) if (i + 1) < n_shards else len(data['annotations'])
        start_imm = step_imm * i
        end_imm = (step_imm * (i + 1) - 1) if (i + 1) < n_shards else len(data['images'])
        
        new_data = {
            'annotations': data['annotations'][start_ann:end_ann],
            'images': data['images'][start_imm:end_imm],
        }
        
        output_tar_path = os.path.join(output_tar_dir, f"shard{str(i + offset).zfill(5)}.tar")
        create_webdataset_single_shard(new_data, output_tar_path)

def cc2coco_format(shards_dir, n_splits=4, batch_offset=0):
    """
    shards_dir: Directory with the shards which contain the .tar files
    n_splits: Number of splits in which we want to divide the tar files. For example, with 4 n_split we elaborate 332 // 4 = 83 tar files.
    batch_offset: Of the n_splits in which we have divided tars, we decide which of them elaborate 
    """
    tar_list = [filename for filename in os.listdir(shards_dir) if 'tar' in filename]
    n_files = max([int(re.findall(r'\d+', filename)[0]) for filename in tar_list]) + 1
    match = re.search(r'([a-zA-Z]+)\d+', tar_list[0])
    prefix = '' if match is None else match.group(1)

    batch_dim = n_files // n_splits
    start = batch_dim * batch_offset
    end = batch_dim * (batch_offset + 1) - 1
    in_path = os.path.join(shards_dir, f"{prefix}{{{str(start).zfill(5)}..{str(end).zfill(5)}}}.tar")
    
    print(f"Reading webdataset {in_path}")
    dataset = wds.WebDataset(in_path).decode("pil")
    
    data = {
        'annotations': [],
        'images': []
    }

    for elem in tqdm(dataset):
        if 'json' in elem:
            obj = elem['json']
            ann = {
                'image_id': obj['key'],
                'id': obj['key'],
                'caption': obj['caption'],
            }
            imm = {
                'id': obj['key'],
                'file_name': obj['url'],
                'height': obj['height'],
                'width': obj['width'],
                'jpg': elem['jpg'],
            }
        else:
            obj = elem['pth']
            ann_field = ['image_id', 'id', 'caption', 'text_features']
            imm_field = ['id', 'file_name', 'height', 'width', 'dino_features', 'avg_self_attn_out',
                         'second_last_out', 'patch_tokens', 'self_attn_maps', 'disentangled_self_attn']
            ann = {field: obj[field] for field in ann_field if field in obj}
            imm = {field: obj[field] for field in imm_field if field in obj}
            
            
        data['annotations'].append(ann)
        data['images'].append(imm)

    return data

def read_coco_format_wds(ann_path):
    print(f"Reading webdataset {ann_path}")
    dataset = wds.WebDataset(ann_path).decode("pil")
    
    anns = []
    imgs = []
    for elem in tqdm(dataset):
        obj = elem['pth']
        ann_fields = ['id', 'image_id', 'caption', 'ann_feats']
        img_fields = ['id', 'file_name', 'height', 'width']
        anns.append({field: obj[field] for field in ann_fields})
        imgs.append({field: obj[field] for field in img_fields})
        
    return {
        'images': imgs,
        'annotations': anns
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, required=True, help="Directory of the output files") 
    parser.add_argument('--out_shards', type=int, default=16, help="Number of splits of the output") 
    parser.add_argument('--shards_dir', type=str, required=True, help="Directory of the webdataset") 
    args = parser.parse_args()

    
    data = cc2coco_format(args.shards_dir)
    
    
    print(f"Dataset composed by {len(data['images'])} couple (text, image)")
    if args.out_shards == 1:
        torch.save(data, args.out_path)
    else:
        os.makedirs(args.out_path, exist_ok=True)
        step = len(data['annotations']) // args.out_shards
        for i in range(args.out_shards):
            start = step * i
            end = (step * (i + 1) - 1) if i < args.out_shards else len(data['annotations']) - 1
            
            new_data = {
                'annotations': data['annotations'][start:end],
                'images': data['images'][start:end],
            }
            
            save_path = os.path.join(args.out_path, f"shard{i}.pth")
            torch.save(new_data, save_path)
            print(f"Saved elements between {start} and {end} at {save_path}")
        
if __name__ == '__main__':
    main()