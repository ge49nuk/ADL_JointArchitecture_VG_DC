"""
REFERENCE TO https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
"""

import os
import csv
import json
import torch
import hydra
import numpy as np
import sys
import open3d as o3d
from functools import partial
from tqdm.contrib.concurrent import process_map
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def get_semantic_mapping_file(file_path):
    label_mapping = {}
    with open(file_path, "r") as f:
        tsv_file = csv.reader(f, delimiter="\t")
        next(tsv_file)  # skip the header
        for line in tsv_file:
            label_mapping[line[1]] = int(line[4])  # use nyu40 label set
    return label_mapping


def read_mesh_file(mesh_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    return np.asarray(mesh.vertices, dtype=np.float32), \
           np.rint(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8), \
           np.asarray(mesh.vertex_normals, dtype=np.float32)


def get_semantic_labels(obj_name_to_segs, seg_to_verts, num_verts, label_map, filtered_label_map):
    semantic_labels = np.full(shape=num_verts, fill_value=-1, dtype=np.int16)
    for label, segs in obj_name_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            if label not in label_map or label_map[label] not in filtered_label_map:
                semantic_label = -1
            else:
                semantic_label = filtered_label_map[label_map[label]]
            semantic_labels[verts] = semantic_label
    return semantic_labels


def read_agg_file(file_path):
    object_id_to_segs = {}
    obj_name_to_segs = {}
    with open(file_path, "r") as f:
        data = json.load(f)
    for group in data['segGroups']:
        object_name = group['label']
        segments = group['segments']
        object_id_to_segs[group["objectId"]] = segments
        if object_name in obj_name_to_segs:
            obj_name_to_segs[object_name].extend(segments)
        else:
            obj_name_to_segs[object_name] = segments.copy()
    return object_id_to_segs, obj_name_to_segs


def read_seg_file(seg_file):
    seg2verts = {}
    vert2seg = {}
    with open(seg_file, 'r') as json_data:
        data = json.load(json_data)
    for vert, seg in enumerate(data['segIndices']):
        if seg not in seg2verts:
            seg2verts[seg] = []
        vert2seg[vert] = seg
        seg2verts[seg].append(vert)
    return seg2verts, vert2seg

def read_descr_file(desc_file, agg_file, iou_0_set, scan, split):
    with open(desc_file, 'r') as json_data:
        data = json.load(json_data)
    inst_descr = []
    scan_data = data[scan]
    seen_desc = {}
    for obj_id in scan_data:
        instance = {}
        ignored_obj = 0
        with open(agg_file, "r") as f:
            data = json.load(f)
            for k, group in enumerate(data['segGroups']):
                if k > int(obj_id):
                    break
                if 'floor' in group['label'] or 'wall' in group['label']:
                    ignored_obj += 1
            instance["object_id"] = [int(obj_id) - ignored_obj]
            instance["obj_id"] = obj_id # Original obj id
            if split == "train" and scan + str(instance["object_id"][0]) in iou_0_set:
                continue
        for ann_id in scan_data[obj_id]:
            instance_cpy = instance.copy()
            instance_cpy["ann_id"] = ann_id
            instance_cpy["object_name"] = scan_data[obj_id][ann_id]["object_name"]
            descr = "[CLS] " + scan_data[obj_id][ann_id]["description"]
            descr = descr.replace('.',' [SEP]')
            if descr in seen_desc:
                inst_descr[seen_desc[descr]]["object_id"].append(instance["object_id"][0])
                continue
            tokenized_descr = np.array(["[PAD]"] * 134, dtype=np.dtype('U15'))
            tokens = np.array(tokenizer.tokenize(descr))
            np.put(tokenized_descr, range(len(tokens)), tokens)
            instance_cpy["tokens"] = np.array(tokenizer.convert_tokens_to_ids(tokenized_descr))
            instance_cpy["num_tokens"] = len(tokens)
            seen_desc[descr] = len(inst_descr)
            inst_descr.append(instance_cpy)
    
    return inst_descr

def get_instance_ids(object_id2segs, seg2verts, sem_labels, invalid_ids):
    object_id2label_id = {}
    instance_ids = np.full(shape=len(sem_labels), fill_value=-1, dtype=np.int16)
    instance_descr = np.full(shape=len(sem_labels), fill_value="", dtype=str)
    new_object_id = 0
    real_id = 0
    for _, segs in object_id2segs.items():
        for seg in segs:
            verts = seg2verts[seg]
            if sem_labels[verts][0] in invalid_ids:
                # skip room architecture and point with invalid semantic labels
                new_object_id -= 1
                break
            instance_ids[verts] = new_object_id
        new_object_id += 1
        real_id += 1
    return instance_ids


def process_one_scan(scan, cfg, split, label_map, iou_0_set):
    mesh_file_path = os.path.join(cfg.data.raw_scene_path, scan, scan + '_vh_clean_2.ply')
    agg_file_path = os.path.join(cfg.data.raw_scene_path, scan, scan + '.aggregation.json')
    seg_file_path = os.path.join(cfg.data.raw_scene_path, scan, scan + '_vh_clean_2.0.010000.segs.json')
    descr_file_path = os.path.join(cfg.data.scanrefer_path, 'ScanRefer_filtered_organized.json')

    # read mesh_file
    xyz, rgb, normal = read_mesh_file(mesh_file_path)
    num_verts = len(xyz)

    if os.path.exists(agg_file_path):
        # read seg_file
        seg2verts,vert2seg = read_seg_file(seg_file_path)
        # read agg_file
        object_id2segs, label2segs = read_agg_file(agg_file_path)

        object_descr = read_descr_file(descr_file_path, agg_file_path, iou_0_set, scan, split)

        # get semantic labels
        # create a map, skip invalid labels to make the final semantic labels consecutive
        filtered_label_map = {}
        invalid_ids = []
        for i, sem_id in enumerate(cfg.data.mapping_classes_ids):
            filtered_label_map[sem_id] = i
            if sem_id in cfg.data.ignore_classes:
                invalid_ids.append(filtered_label_map[sem_id])
        sem_labels = get_semantic_labels(label2segs, seg2verts, num_verts, label_map, filtered_label_map)
        # get instance labels
        instance_ids = get_instance_ids(object_id2segs, seg2verts, sem_labels, invalid_ids)
    else:
        # use zero as placeholders for the test scene
        sem_labels = np.full(shape=num_verts, fill_value=-1, dtype=np.int16)
        instance_ids = np.full(shape=num_verts, fill_value=-1, dtype=np.int16)
    torch.save({'num_descr':len(object_descr), 'xyz': xyz, 'rgb': rgb, 'normal': normal, 'sem_labels': sem_labels, 'instance_ids': instance_ids},#'vert2seg':vert2seg},
               os.path.join(cfg.data.dataset_path, split, f"{scan}.pth"))
    for i, desc in enumerate(object_descr):
        torch.save({'object_descr' : desc},
               os.path.join(cfg.data.dataset_path, split, f"{scan}_{i}.pth"))


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    label_map = get_semantic_mapping_file(cfg.data.metadata.combine_file)
    print("\nDefault: using all CPU cores.")
    iou_0_file_path = cfg.data.metadata.iou_list
    with open(iou_0_file_path) as file:
            iou_0_set = set([line.rstrip() for line in file])
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(cfg.data.dataset_path, split), exist_ok=True)
        with open(getattr(cfg.data.metadata, f"{split}_list")) as f:
            id_list = [line.strip() for line in f]
        print(f"==> Processing {split} split ...")
        process_map(partial(process_one_scan, cfg=cfg, split=split, label_map=label_map, iou_0_set=iou_0_set), id_list, chunksize=1)


if __name__ == '__main__':
    main()
