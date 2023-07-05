import os
import random
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class JointDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.instance_size = cfg.data.instance_size
        
        self._load_from_disk()

    
    def _load_single_scan(self, scan_id, scan_fn):
        split_folder = os.path.join(self.cfg.data.dataset_path, self.split)
        scan_folder = os.path.join(split_folder, scan_id)
        scan_file = os.path.join(scan_folder, f"{scan_fn}")
        scan = torch.load(scan_file)
        scan["scan_fn"] = scan_fn

        # Sampling point features
        samples = self.cfg.model.samples_per_proposal
        point_features = scan["point_features"]
        insts = np.split(point_features, scan["instance_splits"], axis=0) # (#instances, #pts, dim_pt_feats)
        scan["point_features"] = torch.stack([torch.from_numpy\
        (
            np.append(
                inst[np.random.choice(inst.shape[0] , min(samples, inst.shape[0]), replace=False)].flatten(),\
                #filling up with zeros, in case: #samples > #proposal_feats
                torch.zeros((samples - min(samples, inst.shape[0]), inst.shape[1]))
            )
        ) for inst in insts])

        return scan


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
            # Load scene augmentations
            num_augs = 0
            for scan_fn in scan_fns:
                if scan_fn == "descr":
                    continue
                if num_augs >= self.cfg.model.augs_per_scene or (self.split == 'val' and num_augs >= 1):
                    break
                self.scans.append({"scan_id":scan_id, "scan_fn":scan_fn})
                num_augs += 1

            # Load descriptions
            descr_folder = os.path.join(scan_folder, "descr")
            descr_fns = os.listdir(descr_folder)
            num_descrs = min(self.cfg.model.num_descriptions, len(descr_fns))
            if self.split == "val":
                num_descrs = len(descr_fns)
            for i in range(num_descrs):
                descr_file = os.path.join(descr_folder, descr_fns[i])
                self.descrs.append(torch.load(descr_file))
                self.scans_idx.append((curr_scan, curr_scan + num_augs)) # Mapping: idx -> scan
            curr_scan += num_augs

    def __len__(self):
        return len(self.scans_idx)
   
    
    def __getitem__(self, idx):
        # Load scene from disc
        scan_id_range = self.scans_idx[idx]
        scan_info = self.scans[random.randint(scan_id_range[0], scan_id_range[1]-1)]
        scan = self._load_single_scan(scan_info["scan_id"], scan_info["scan_fn"])

        descr = self.descrs[idx]


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
        
        if self.split == "val":
            # For testing
            data["proposals_idx"] = scan['proposals_idx']
            data["queried_objs"] = np.array(descr['queried_objs'])
            data["instance_ids"] = scan['instance_ids']
            data["scan_desc_id"] = descr['scan_desc_id']
            data["ann_id"] = descr['ann_id']

            # For corpus generation
            data["scene_id"] = descr['scene_id']
            data["object_id"] = descr['object_id']
            data["object_name"] = descr['object_name']

        return data