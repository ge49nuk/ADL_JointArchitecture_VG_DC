import os
import open3d as o3d
import numpy as np
import json


def get_bbox(predicted_verts, points):
    x_min = None
    y_min = None
    z_min = None
    x_max = None
    y_max = None
    z_max = None
    for i in predicted_verts:
        if x_min is None or points[i][0] < x_min:
            x_min = points[i][0]
        if y_min is None or points[i][1] < y_min:
            y_min = points[i][1]
        if z_min is None or points[i][2] < z_min:
            z_min = points[i][2]
        if x_max is None or points[i][0] > x_max:
            x_max = points[i][0]
        if y_max is None or points[i][1] > y_max:
            y_max = points[i][1]
        if z_max is None or points[i][2] > z_max:
            z_max = points[i][2]
    return (x_min, y_min, z_min), (x_max, y_max, z_max)


def get_scene_info(scans_dir, scan_id):
    ply_file = os.path.join(scans_dir, scan_id, f'{scan_id}_vh_clean_2.ply')
    # get mesh
    scannet_data = o3d.io.read_triangle_mesh(ply_file)
    scannet_data.compute_vertex_normals()
    points = np.asarray(scannet_data.vertices)
    colors = np.asarray(scannet_data.vertex_colors)
    indices = np.asarray(scannet_data.triangles)
    colors = colors * 255.0
    
    return points, colors, indices

def calculate_iou(bbox1, bbox2):
    intersection_min = [max(bbox1[0][0], bbox2[0][0]), max(bbox1[0][1], bbox2[0][1]), max(bbox1[0][2], bbox2[0][2])]
    intersection_max = [min(bbox1[1][0], bbox2[1][0]), min(bbox1[1][1], bbox2[1][1]), min(bbox1[1][2], bbox2[1][2])]
    intersection_volume = max(0, intersection_max[0] - intersection_min[0]) * \
                          max(0, intersection_max[1] - intersection_min[1]) * \
                          max(0, intersection_max[2] - intersection_min[2])
    bbox1_volume = (bbox1[1][0] - bbox1[0][0]) * (bbox1[1][1] - bbox1[0][1]) * (bbox1[1][2] - bbox1[0][2])
    bbox2_volume = (bbox2[1][0] - bbox2[0][0]) * (bbox2[1][1] - bbox2[0][1]) * (bbox2[1][2] - bbox2[0][2])
    union_volume = bbox1_volume + bbox2_volume - intersection_volume
    iou = intersection_volume / union_volume

    return iou


def _get_raw2label(cfg):
    type2class = type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
            'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}  
    # mapping
    scannet_labels = type2class.keys()
    scannet2label = {label: i for i, label in enumerate(scannet_labels)}

    # data path
    SCANNET_V2_TSV = os.path.join(cfg.data.dataset_root_path, "scannetv2/metadata/scannetv2-labels.combined.tsv")
    lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
    lines = lines[1:]
    raw2label = {}
    for i in range(len(lines)):
        label_classes_set = set(scannet_labels)
        elements = lines[i].split('\t')
        raw_name = elements[1]
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2label[raw_name] = scannet2label['others']
        else:
            raw2label[raw_name] = scannet2label[nyu40_name]
    raw2label["shower_curtain"] = 13

    return raw2label

def _get_unique_multiple_lookup(cfg):
    descr_file_path = os.path.join(cfg.data.scanrefer_path, 'ScanRefer_filtered_organized.json')
    with open(descr_file_path, 'r') as json_data:
        scanrefer = json.load(json_data)
    raw2label = _get_raw2label(cfg)
    all_sem_labels = {}
    cache = {}
    for s_id in scanrefer:
        for o_id in scanrefer[s_id]:
            for ann_id in scanrefer[s_id][o_id]:
                data = scanrefer[s_id][o_id][ann_id]
                scene_id = data["scene_id"]
                object_id = data["object_id"]
                object_name = " ".join(data["object_name"].split("_"))
                ann_id = data["ann_id"]

                if scene_id not in all_sem_labels:
                    all_sem_labels[scene_id] = []

                if scene_id not in cache:
                    cache[scene_id] = {}

                if object_id not in cache[scene_id]:
                    cache[scene_id][object_id] = {}
                    try:
                        all_sem_labels[scene_id].append(raw2label[object_name])
                    except KeyError:
                        all_sem_labels[scene_id].append(17)

    # convert to numpy array
    all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

    unique_multiple_lookup = {}
    for s_id in scanrefer:
        for o_id in scanrefer[s_id]:
            for ann_id in scanrefer[s_id][o_id]:
                data = scanrefer[s_id][o_id][ann_id]
                scene_id = data["scene_id"]
                object_id = data["object_id"]
                object_name = " ".join(data["object_name"].split("_"))
                ann_id = data["ann_id"]

                try:
                    sem_label = raw2label[object_name]
                except KeyError:
                    sem_label = 17

                unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

                # store
                if scene_id not in unique_multiple_lookup:
                    unique_multiple_lookup[scene_id] = {}

                if object_id not in unique_multiple_lookup[scene_id]:
                    unique_multiple_lookup[scene_id][object_id] = {}

                if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                    unique_multiple_lookup[scene_id][object_id][ann_id] = None

                unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

    return unique_multiple_lookup
