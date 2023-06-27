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
        self.augs_per_scene = 4
        self.num_descrs = 12
        self._load_from_disk()

    def _load_from_disk(self):
        split_folder = os.path.join(self.cfg.data.dataset_path, self.split)
        scan_ids = os.listdir(split_folder)
        self.scans = []
        self.descrs = []
        self.scan_and_descr_idx = []
        
        for scan_id in tqdm(scan_ids, desc=f"Loading joint_{self.split} data from disk"):
            if scan_id == "_ignore":
                continue
            scan_folder = os.path.join(split_folder, scan_id)
            scan_fns = os.listdir(scan_folder)
            num_augs = 0
            for scan_fn in scan_fns:
                if scan_fn == "descr":
                    continue
                if num_augs >= self.augs_per_scene:
                    break
                num_augs += 1
                scan_file = os.path.join(scan_folder, f"{scan_fn}")
                self.scans.append(torch.load(scan_file))

                descr_folder = os.path.join(scan_folder, "descr")
                descr_fns = os.listdir(descr_folder)
                num_descrs = min(self.num_descrs, len(descr_fns))
                selected_idx = np.random.choice(len(descr_fns) , num_descrs, replace=False)
                for i in selected_idx:
                    descr_fn = descr_fns[i]
                    descr_file = os.path.join(descr_folder, descr_fn)
                    self.descrs.append(torch.load(descr_file))
                    self.scan_and_descr_idx.append((len(self.scans)-1, len(self.descrs)-1)) # Mapping: idx -> (scan_idx, descr_idx)

    def __len__(self):
        return len(self.scan_and_descr_idx)
   
    
    def __getitem__(self, idx):
        scan = self.scans[self.scan_and_descr_idx[idx][0]]
        descr = self.descrs[self.scan_and_descr_idx[idx][1]]

        # Calculate best proposals
        best_proposals = []
        for o in descr['queried_objs']:
            ious_queried_obj = scan['ious_on_cluster'][:,o]
            best_proposals.append((np.argmax(ious_queried_obj)))
        best_proposals = np.asarray(best_proposals)
        
        # For light training
        data = {"point_features": scan["point_features"]}
        data["instance_splits"] = scan["instance_splits"]
        data["target_proposals"] = best_proposals
        data["ious_on_cluster"] = scan['ious_on_cluster']
        
        data["num_target_proposals"] = data["target_proposals"].shape[0]
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