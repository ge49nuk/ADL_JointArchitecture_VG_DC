import os
import open3d as o3d
import numpy as np


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
