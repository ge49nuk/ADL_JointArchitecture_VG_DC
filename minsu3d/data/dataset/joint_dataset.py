import os
import random
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
        
        curr_scan = 0
        for scan_id in tqdm(scan_ids, desc=f"Loading joint_{self.split} data from disk"):
            if scan_id == "_ignore":
                continue
            scan_folder = os.path.join(split_folder, scan_id)
            scan_fns = os.listdir(scan_folder)
            num_augs = len(scan_fns)-1
            for scan_fn in scan_fns:
                if scan_fn == "descr":
                    continue
                scan_file = os.path.join(scan_folder, f"{scan_fn}")
                self.scans.append(torch.load(scan_file))
            descr_folder = os.path.join(scan_folder, "descr")
            descr_fns = os.listdir(descr_folder)
            for descr_fn in descr_fns:
                descr_file = os.path.join(descr_folder, descr_fn)
                self.descrs.append(torch.load(descr_file))
                self.scans_idx.append((curr_scan, curr_scan + num_augs)) # Mapping: idx -> scan
            curr_scan += num_augs

    def __len__(self):
        return len(self.scans_idx)
   
    
    def __getitem__(self, idx):
        scan_id_range = self.scans_idx[idx]

        scan = self.scans[random.randint(scan_id_range[0], scan_id_range[1]-1)]
        descr = self.descrs[idx]
        # print("Loaded scan", scan["aug_id"], "for description", descr["scan_desc_id"])

        # Calculate best proposals
        best_proposals = []
        for o in descr['queried_objs']:
            ious_queried_obj = scan['ious_on_cluster'][:,o]
            best_proposals.append((torch.argmax(ious_queried_obj)).cpu().numpy())
        best_proposals = np.asarray(best_proposals)
        
        # For light training
        data = {"point_features": scan["point_features"]}
        data["instance_splits"] = scan["instance_splits"]
        data["target_proposals"] = best_proposals
        
        data["num_target_proposals"] = descr["target_proposals"].shape[0]
        data["text_embedding"] = descr['text_embedding'] 
        data["target_word_ids"] = descr['target_word_ids'] 
        data["num_tokens"] = descr['num_tokens'] 
        data["target_class"] = descr['target_class']
        
        # For testing
        data["proposals_idx"] = scan['proposals_idx']
        data["queried_objs"] = np.array(descr['queried_objs'])
        data["instance_ids"] = scan['instance_ids']
        data["scan_desc_id"] = descr['scan_desc_id']

        return data