import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class JointDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        # self.max_num_point = cfg.data.max_num_point
        self.instance_size = cfg.data.instance_size
        self._load_from_disk()

    def _load_from_disk(self):
        folder = os.path.join(self.cfg.data.dataset_path, self.split)
        filenames = os.listdir(folder)
        self.joint_scenes = []
        for filename in tqdm(filenames, desc=f"Loading joint_{self.split} data from disk"):
            file = os.path.join(folder, filename)
            joint_data = torch.load(file)
            self.joint_scenes.append(joint_data)

    def __len__(self):
        return len(self.joint_scenes)
   
    
    def __getitem__(self, idx):
        joint_scene = self.joint_scenes[idx]
        
        # For light training
        data = {"point_features": joint_scene["point_features"]}
        data["instance_splits"] = joint_scene["instance_splits"]
        data["target_proposals"] = joint_scene['target_proposals']
        data["num_target_proposals"] = data["target_proposals"].shape[0]
        data["text_embedding"] = joint_scene['text_embedding'] 
        data["target_word_ids"] = joint_scene['target_word_ids'] 
        data["num_tokens"] = joint_scene['num_tokens'] 
        data["target_class"] = joint_scene['target_class']
        
        # For testing
        data["queried_objs"] = joint_scene['proposals_idx']
        data["proposals_idx"] = joint_scene['proposals_idx']
        data["instance_ids"] = joint_scene['instance_ids']
        data["scan_desc_id"] = joint_scene['scan_desc_id']

        # with open('queried_objs.txt', 'w') as f:
        #     print(joint_scene['proposals_idx'], file=f)
        # with open('proposals_idx.txt', 'w') as f:
        #     print(joint_scene["proposals_idx"], file=f)
        # with open('instance_ids.txt', 'w') as f:
        #     print(joint_scene["instance_ids"], file=f)
        # with open('scan_desc_id.txt', 'w') as f:
        #     print(joint_scene["scan_desc_id"], file=f)
        # sys.exit()

        return data