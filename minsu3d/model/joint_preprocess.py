import numpy as np
import torch
import torch.nn as nn
import hydra
import os
import pytorch_lightning as pl
from minsu3d.common_ops.functions import common_ops
from transformers import BertConfig, BertModel
from minsu3d.model.softgroup import SoftGroup

class_names = [ 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
               'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
               'bathtub', 'otherfurniture' ]

class JointPreprocessModelT(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
 
        configuration = BertConfig()
        self.bert = BertModel(configuration)

        self.softgroup = SoftGroup(cfg=cfg)
        checkpoint = torch.load(cfg.model.ckpt_path_SG)
        self.softgroup.load_state_dict(checkpoint['state_dict'])
        
        for param in self.softgroup.parameters():
            param.requires_grad = False
        for param in self.bert.parameters():
            param.requires_grad = False

        self.lin = nn.Linear(1, 1)
        self.split = 'train'
    
    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.cfg.model.optimizer, params=self.parameters())

    def forward(self, data_dict):
        scene = data_dict["scan_ids"][0]
        
        # SoftGroup
        self.softgroup.eval()
        output_dict = self.softgroup(data_dict)
 
        # Compute instance point features
        proposals_idx = output_dict["proposals_idx"]
        point_features = (output_dict["point_features"])[proposals_idx[:, 1]]

        _, counts = proposals_idx[:, 0].unique(return_counts=True)

        for i, count in enumerate(counts[:-1]):
            counts[i+1] += count
        instance_splits = counts[:-1]
        # instance_fatures = torch.tensor_split(point_features, instance_splits.cpu(), dim=0) # (#instances, #pts, dim_pt_feats)

        # Compute most accurate object proposal
        queried_objs = data_dict["queried_objs"][0]
        proposals_idx = output_dict["proposals_idx"][:, 1].int().contiguous()
        proposals_offset = output_dict["proposals_offset"]
        # calculate iou of clustered instance
        ious_on_cluster = common_ops.get_mask_iou_on_cluster(
            proposals_idx, proposals_offset, data_dict["instance_ids"], data_dict["instance_num_point"]
        )
        # Collect proposals of highest IoU with GT
        best_proposals = []
        for o in queried_objs:
            ious_queried_obj = ious_on_cluster[:,o]
            best_proposals.append((torch.argmax(ious_queried_obj)).cpu().numpy())
        best_proposals = np.asarray(best_proposals)

        # BERT
        self.bert.eval()
        text_embedding = self.bert(data_dict["descr_tokens"])[0]
        target_word_ids = data_dict["descr_tokens"][0]
        num_tokens = data_dict["num_descr_tokens"][0]
        
        # Mapping for descrption object: object name -> class id
        obj_name = data_dict["object_names"][0]
        obj_name = np.repeat(obj_name, len(class_names))
        target_class = torch.tensor(class_names == obj_name).float() # One-hot vector

        # Get descr_id for naming the file
        descr_id = data_dict["descr_ids"][0]
        scan_desc_id = scene + ":" + str(descr_id)

        
        output_path = os.path.join(self.cfg.data.dataset_root_path, "joint_data")
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, self.split), exist_ok=True)
        torch.save({'point_features': point_features.cpu().numpy(), 'instance_splits': instance_splits.cpu().numpy(), 'target_proposals': best_proposals,
                    'text_embedding': text_embedding.cpu().numpy(), 'target_word_ids': target_word_ids.cpu().numpy(), 'num_tokens': num_tokens, 'target_class': target_class.cpu().numpy(),
                    'queried_objs': queried_objs, 'proposals_idx': output_dict["proposals_idx"].cpu().numpy(), 'instance_ids': data_dict["instance_ids"].cpu().numpy(), 'scan_desc_id': scan_desc_id},
               os.path.join(output_path, self.split, f"{scene}_{descr_id}.pth"))

        return {}


    def _loss(self, data_dict, output_dict):
        return {}
    
    def training_step(self, data_dict, idx):
        self.forward(data_dict)

    def validation_step(self, data_dict, idx):
        pass

    def test_step(self, data_dict, idx):
        pass


class JointPreprocessModelV(JointPreprocessModelT):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.split = 'val'