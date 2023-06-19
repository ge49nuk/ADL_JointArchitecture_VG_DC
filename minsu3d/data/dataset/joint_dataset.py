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
        split_folder = os.path.join(self.cfg.data.dataset_path, self.split)
        scan_ids = os.listdir(split_folder)
        self.scans = []
        self.scans_idx = []
        self.descrs = []
        for scan_id in tqdm(scan_ids, desc=f"Loading joint_{self.split} data from disk"):
            scan_folder = os.path.join(split_folder, scan_id)
            scan_file = os.path.join(scan_folder, f"{scan_id}.pth")
            self.scans.append(torch.load(scan_file))
            descr_folder = os.path.join(scan_folder, "descr")
            descr_fns = os.listdir(descr_folder)
            for descr_fn in descr_fns:
                self.scans_idx.append(len(self.scans) - 1) # Mapping: idx -> scan
                descr_file = os.path.join(descr_folder, descr_fn)
                self.descrs.append(torch.load(descr_file))

    def __len__(self):
        return len(self.scans_idx)
   
    
    def __getitem__(self, idx):
        scan = self.scans[self.scans_idx[idx]]
        descr = self.descrs[idx]
        
        # For light training
        data = {"point_features": scan["point_features"]}
        data["instance_splits"] = scan["instance_splits"]
        data["target_proposals"] = scan['target_proposals']
        data["num_target_proposals"] = scan["target_proposals"].shape[0]
        data["text_embedding"] = descr['text_embedding'] 
        data["target_word_ids"] = descr['target_word_ids'] 
        data["num_tokens"] = descr['num_tokens'] 
        data["target_class"] = descr['target_class']
        
        # For testing
        data["queried_objs"] = scan['proposals_idx']
        data["proposals_idx"] = scan['proposals_idx']
        data["instance_ids"] = scan['instance_ids']
        data["scan_desc_id"] = descr['scan_desc_id']

        return data