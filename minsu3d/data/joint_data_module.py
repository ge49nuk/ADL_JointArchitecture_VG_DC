import numpy as np
import torch
import sys
import pytorch_lightning as pl
from importlib import import_module
from torch.utils.data import DataLoader
from minsu3d.data import DataModule


class JointDataModuleT(DataModule):
    def __init__(self, data_cfg):
        super().__init__(data_cfg)
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_set = self.dataset(self.data_cfg, "train")
            self.val_set = self.dataset(self.data_cfg, "val")
        if stage == "test" or stage is None:
            self.val_set = self.dataset(self.data_cfg, self.data_cfg.model.inference.split)
        if stage == "predict" or stage is None:
            self.test_set = self.dataset(self.data_cfg, "test")

class JointDataModuleV(DataModule):
    def __init__(self, data_cfg):
        super().__init__(data_cfg)
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_set = self.dataset(self.data_cfg, "val")
            self.val_set = self.dataset(self.data_cfg, "val")
        if stage == "test" or stage is None:
            self.val_set = self.dataset(self.data_cfg, self.data_cfg.model.inference.split)
        if stage == "predict" or stage is None:
            self.test_set = self.dataset(self.data_cfg, "test")

class JointDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.dataset = getattr(import_module('minsu3d.data.dataset'), data_cfg.data.dataset)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_set = self.dataset(self.data_cfg, "train")
            self.val_set = self.dataset(self.data_cfg, "val")
        if stage == "test" or stage is None:
            self.val_set = self.dataset(self.data_cfg, self.data_cfg.model.inference.split)
        if stage == "predict" or stage is None:
            self.test_set = self.dataset(self.data_cfg, "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.data_cfg.data.batch_size, shuffle=True,
            pin_memory=True, collate_fn=_collate_fn, num_workers=self.data_cfg.data.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.data_cfg.data.batch_size, pin_memory=True,
            collate_fn=_collate_fn, num_workers=self.data_cfg.data.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.data_cfg.data.batch_size, pin_memory=True,
            collate_fn=_collate_fn, num_workers=self.data_cfg.data.num_workers
        )


def _collate_fn(batch):
    instances = []
    scene_splits = [0]
    target_proposals = []
    target_proposal_splits = [0]
    text_embeddings = []
    target_word_ids = []
    num_tokens = []
    target_classes = []
    ious_on_cluster = []

    queried_objs = []
    proposals_idx = []
    instance_ids = []
    scan_desc_id = []
    ann_id = []

    scene_ids = []
    object_ids = []
    object_names = []

    for b in batch:
        point_features = b["point_features"]
        scene_insts = point_features #(#instances, dim_inst_feats= samples_per_proposal * 32)
        instances.append(scene_insts)

        scene_splits.append(scene_insts.size()[0] + scene_splits[-1])
        target_proposals.append(torch.from_numpy(b["target_proposals"]))
        target_proposal_splits.append(b["num_target_proposals"] + target_proposal_splits[-1])
        text_embeddings.append(torch.from_numpy(b["text_embedding"]))
        target_word_ids.append(torch.from_numpy(b["target_word_ids"]))
        num_tokens.append(b["num_tokens"])
        target_classes.append(torch.from_numpy(b["target_class"]))
        ious_on_cluster.append(b["ious_on_cluster"])
        
        if 'queried_objs' in b.keys():
            queried_objs.append(b["queried_objs"])
            proposals_idx.append(torch.from_numpy(b["proposals_idx"]))
            instance_ids.append(torch.from_numpy(b["instance_ids"]))
            scan_desc_id.append(b["scan_desc_id"])
            ann_id.append(b["ann_id"])

            scene_ids.append(b["scene_id"])
            object_ids.append(b["object_id"])
            object_names.append(b["object_name"])
        
    
    data = {'instances': torch.cat(instances, dim=0)}
    data['scene_splits'] = scene_splits # need to extract [1:-1]
    data['target_proposals'] = torch.cat(target_proposals, dim=0)
    data['target_proposal_splits'] = target_proposal_splits # need to extract [1:-1]
    data['text_embeddings'] = torch.cat(text_embeddings, dim=0)
    data['target_word_ids'] = torch.stack(target_word_ids, dim=0)
    data['num_tokens'] = num_tokens
    data['target_classes'] = target_classes
    data["ious_on_cluster"] = ious_on_cluster

    if 'queried_objs' in b.keys():
        data['queried_objs'] = queried_objs
        data['proposals_idx'] = proposals_idx
        data['instance_ids'] = instance_ids
        data['scan_desc_id'] = scan_desc_id
        data["ann_ids"] = ann_id

        data['scene_ids'] = scene_ids
        data['object_ids'] = object_ids
        data['object_names'] = object_names

    return data