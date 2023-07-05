import os
import torch
import numpy as np
import sys
from tqdm import tqdm
import MinkowskiEngine as ME
from torch.utils.data import Dataset
from minsu3d.util.transform import jitter, flip, rotz, elastic, crop
import random

rand_descr_seed = 42



class GeneralDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.max_num_point = cfg.data.max_num_point
         
        self.augs_per_scene = 6
        self.aug_memory = {}
        self._load_from_disk()

    def _load_from_disk(self):
        with open(getattr(self.cfg.data.metadata, f"{self.split}_list")) as f:
            self.scene_names = [line.strip() for line in f]
        self.scenes = []
        for scene_name in tqdm(self.scene_names, desc=f"Loading {self.split} data from disk"):
            scene_path = os.path.join(self.cfg.data.dataset_path, self.split, f"{scene_name}.pth")
            scene_info = torch.load(scene_path)
            scene_info["xyz"] -= scene_info["xyz"].mean(axis=0)
            scene_info["rgb"] = scene_info["rgb"].astype(np.float32) / 127.5 - 1
            scene_info["scene_id"] = scene_name
            for i in range(scene_info['num_descr']): # scene_info['num_descr']
                scene = scene_info.copy()
                scene_path = os.path.join(self.cfg.data.dataset_path, self.split, f"{scene_name}_{i}.pth")
                scene_descr = torch.load(scene_path)
                scene["object_descr"] = scene_descr["object_descr"]
                scene["descr_id"] = i
                self.scenes.append(scene)

    def __len__(self):
        return len(self.scenes)

    def _get_augmentation_matrix(self):
        m = np.eye(3)
        if self.cfg.data.augmentation.jitter_xyz:
            m = np.matmul(m, jitter())
        if self.cfg.data.augmentation.flip:
            flip_m = flip(0, random=True)
            m *= flip_m
        if self.cfg.data.augmentation.rotation:
            t = np.random.rand() * 2 * np.pi
            rot_m = rotz(t)
            m = np.matmul(m, rot_m)  # rotation around z
        return m.astype(np.float32)

    def _get_cropped_inst_ids(self, instance_ids, valid_idxs):
        """
        Postprocess instance_ids after cropping
        """
        instance_ids = instance_ids[valid_idxs]
        j = 0
        while j < instance_ids.max():
            if np.count_nonzero(instance_ids == j) == 0:
                instance_ids[instance_ids == instance_ids.max()] = j
            j += 1
        return instance_ids

    def _get_inst_info(self, xyz, instance_ids, sem_labels):
        instance_num_point = []
        unique_instance_ids = np.unique(instance_ids)
        # print(unique_instance_ids)
        unique_instance_ids = unique_instance_ids[unique_instance_ids != -1]
        num_instance = unique_instance_ids.shape[0]
        instance_center_xyz = np.empty(shape=(xyz.shape[0], 3), dtype=np.float32)
        instance_cls = np.full(shape=unique_instance_ids.shape[0], fill_value=-1, dtype=np.int16)
        for index, i in enumerate(unique_instance_ids):
            inst_i_idx = np.where(instance_ids == i)[0]

            # instance center
            instance_center_xyz[inst_i_idx] = xyz[inst_i_idx].mean(0)

            # instance_num_point
            instance_num_point.append(inst_i_idx.size)

            # semantic label
            cls_idx = inst_i_idx[0]
            instance_cls[index] = sem_labels[cls_idx] - len(self.cfg.data.ignore_classes) \
                if sem_labels[cls_idx] != -1 else sem_labels[cls_idx]
            # bounding boxes
        return num_instance, instance_center_xyz, instance_num_point, instance_cls

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        scene_id =scene["scene_id"]

        # Saving augmented scenes
        aug_matrix = self._get_augmentation_matrix()
        if scene_id not in self.aug_memory:
            self.aug_memory[scene_id] = 0 # times_seen, aug_id 
        else:
            augs_created = self.aug_memory[scene_id]
            if augs_created < self.augs_per_scene-1 and self.split == "train": 
                self.aug_memory[scene_id] = augs_created + 1
        
        aug_scene_id = scene["scene_id"] + ":" + str(self.aug_memory[scene_id])

        descr_id = scene["descr_id"]

        point_xyz = scene["xyz"].astype(np.float32)  # (N, 3)
        colors = scene["rgb"].astype(np.float32)  # (N, 3)
        normals = scene["normal"].astype(np.float32)  # (N, 3)

        instance_ids = scene["instance_ids"].astype(np.int16)  # (N, )
        sem_labels = scene["sem_labels"].astype(np.int16)  # (N, )
        descr_dict = scene["object_descr"]
        descr_tokens = descr_dict["tokens"]   
        num_descr_tokens = descr_dict["num_tokens"] 
        object_name = descr_dict["object_name"]
        object_id = descr_dict["obj_id"] # Original obj id
        queried_obj = descr_dict["object_id"]
        ann_id = descr_dict["ann_id"]
        

        data = {"scan_id": scene_id}
        data["aug_id"] = aug_scene_id 


        #augment
        if self.split == "train":
            point_xyz = np.matmul(point_xyz, aug_matrix)
            normals = np.matmul(normals, np.transpose(np.linalg.inv(aug_matrix)))
            if self.cfg.data.augmentation.jitter_rgb:
                # jitter rgb
                colors += np.random.randn(3) * 0.1

        # elastic
        scale = (1 / self.cfg.data.voxel_size)
        if self.split == "train" and self.cfg.data.augmentation.elastic:
            point_xyz_elastic = elastic(point_xyz * scale, 6 * scale // 50, 40 * scale / 50)
            point_xyz_elastic = elastic(point_xyz_elastic, 20 * scale // 50, 160 * scale / 50)
        else:
            point_xyz_elastic = point_xyz * scale

        point_xyz_elastic -= point_xyz_elastic.min(axis=0)

        # crop
        # if self.split == "train":
        #     # HACK, in case there are few points left
        #     max_tries = 20
        #     valid_idxs_count = 0
        #     valid_idxs = np.ones(shape=point_xyz.shape[0], dtype=bool)
        #     if valid_idxs.shape[0] > self.max_num_point:
        #         print("TOO MANY VERTS",valid_idxs.shape[0])
        #         while max_tries > 0:
        #             points_tmp, valid_idxs = crop(point_xyz_elastic, self.max_num_point, self.cfg.data.full_scale[1])
        #             valid_idxs_count = np.count_nonzero(valid_idxs)
        #             if valid_idxs_count >= (self.max_num_point // 2) and np.any(sem_labels[valid_idxs] != -1) \
        #                     and np.any(instance_ids[valid_idxs] != -1):
        #                 point_xyz_elastic = points_tmp
        #                 break
        #             max_tries -= 1
        #         if valid_idxs_count < (self.max_num_point // 2) or np.all(sem_labels[valid_idxs] == -1) \
        #                 and np.all(instance_ids[valid_idxs] == -1):
        #             raise Exception("Over-cropped!")

        #     point_xyz_elastic = point_xyz_elastic[valid_idxs]
        #     point_xyz = point_xyz[valid_idxs]
        #     normals = normals[valid_idxs]
        #     colors = colors[valid_idxs]
        #     sem_labels = sem_labels[valid_idxs]
        #     instance_ids = self._get_cropped_inst_ids(instance_ids, valid_idxs)

        point_xyz_elastic /= (1 / self.cfg.data.voxel_size)  # TODO

        num_instance, instance_center_xyz, instance_num_point, instance_semantic_cls = self._get_inst_info(
            point_xyz, instance_ids, sem_labels
        )
        
        # Check if querried object index is correct
        # b = instance_ids == queried_obj
        # verts = b.nonzero()[0]
        # segs = []
        # for vert in verts:
        #     seg = scene["vert2seg"][vert]
        #     if seg not in segs:
        #         segs.append(seg)
        # print("SCENE:",scene_id, "QUERRIED", queried_obj, "SEGS", segs)

        point_features = np.zeros(shape=(len(point_xyz), 0), dtype=np.float32)
        if self.cfg.model.network.use_color:
            point_features = np.concatenate((point_features, colors), axis=1)
        if self.cfg.model.network.use_normal:
            point_features = np.concatenate((point_features, normals), axis=1)

        point_features = np.concatenate((point_features, point_xyz), axis=1)  # add xyz to point features

        data["point_xyz"] = point_xyz  # (N, 3)
        data["sem_labels"] = sem_labels  # (N, )
        data["instance_ids"] = instance_ids  # (N, )
        data["num_instance"] = np.array(num_instance, dtype=np.int32)
        data["instance_center_xyz"] = instance_center_xyz
        data["instance_num_point"] = np.array(instance_num_point, dtype=np.int32)
        data["instance_semantic_cls"] = instance_semantic_cls
        data["descr_tokens"] = descr_tokens
        data["num_descr_tokens"] = num_descr_tokens
        data["object_name"] = object_name
        data["object_id"] = object_id
        data["descr_id"] = descr_id
        data["queried_obj"] = queried_obj
        data["ann_id"] = ann_id

        data["voxel_xyz"], data["voxel_features"], _, data["voxel_point_map"] = ME.utils.sparse_quantize(
            coordinates=point_xyz_elastic, features=point_features,
            return_index=True,
            return_inverse=True, quantization_size=self.cfg.data.voxel_size
        )

        return data
