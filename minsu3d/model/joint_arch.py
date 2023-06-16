import numpy as np
import torch
import torch.nn as nn
import hydra
import os
import sys
import time
import psutil
import pytorch_lightning as pl
from minsu3d.evaluation.instance_segmentation import get_gt_instances, rle_encode
from minsu3d.evaluation.object_detection import get_gt_bbox
from minsu3d.util.lr_decay import cosine_lr_decay
from minsu3d.util.io import save_prediction_joint_arch
from minsu3d.common_ops.functions import softgroup_ops, common_ops

from transformers import BertConfig, BertModel
from minsu3d.model.transformer import Transformer
from minsu3d.model.softgroup import SoftGroup

RAM_THRESHOLD = 2000 * 1024 * 1024 # leave at least 2GB of ram free
class_names = [ 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
               'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
               'bathtub', 'otherfurniture' ]

class Joint_Arch(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.val_test_step_outputs = []
        self.transformer = Transformer()
        self.transformer.to("cuda")
        configuration = BertConfig()
        self.bert = BertModel(configuration).to("cuda")
        self.softgroup = SoftGroup(cfg=cfg)
        self.softgroup_past = {}
        checkpoint = torch.load(cfg.model.ckpt_path_SG)
        self.softgroup.load_state_dict(checkpoint['state_dict'])
        for param in self.softgroup.parameters():
            param.requires_grad = False
        for param in self.bert.parameters():
            param.requires_grad = False
        output_channel = cfg.model.network.m
        self.instance_classes = cfg.data.classes - len(cfg.data.ignore_classes)
    
    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.cfg.model.optimizer, params=self.parameters())

    def forward(self, data_dict):
        scene = data_dict["scan_ids"][0]
        # with open('output.txt', 'w') as f:
        #     print(data_dict["scan_ids"], file=f)
        #     sys.exit()
        #SoftGroup
        self.softgroup.eval()
        if scene in self.softgroup_past:
            output_dict = self.softgroup_past[scene].copy()
            output_dict["proposals_idx"] = output_dict["proposals_idx"].to("cuda")
            output_dict["point_features"] = output_dict["point_features"].to("cuda")
            output_dict["proposals_offset"] = output_dict["proposals_offset"].to("cuda")
        else:
            output_dict = self.softgroup(data_dict)
            if psutil.virtual_memory().available > RAM_THRESHOLD:
                out_cpy = {}
                if(self.hparams.cfg.model.quick_training):
                    out_cpy["proposals_idx"] = output_dict["proposals_idx"].to("cpu")
                    out_cpy["point_features"] = output_dict["point_features"].to("cpu")
                    out_cpy["proposals_offset"] = output_dict["proposals_offset"].to("cpu")
                else:
                    for k,v in output_dict.items():
                        out_cpy[k] = v.to("cpu")
                self.softgroup_past[scene] = out_cpy

        # BERT
        with torch.no_grad():
            self.bert.eval()
            bert_out = self.bert(data_dict["descr_tokens"]) 
        output_dict["descr_embedding"] = bert_out[0]                # Shape: [tensor(batch_size, seq_len, emb_dim)]


        # COMPUTE MOST ACCURATE OBJECT PROPOSAL
        queried_obj = data_dict["queried_objs"][0]
        proposals_idx = output_dict["proposals_idx"][:, 1].int().contiguous()
        proposals_offset = output_dict["proposals_offset"]
        # calculate iou of clustered instance
        ious_on_cluster = common_ops.get_mask_iou_on_cluster(
            proposals_idx, proposals_offset, data_dict["instance_ids"], data_dict["instance_num_point"]
        )
        # Collect proposals of highest IoU with GT
        best_proposals = []
        for o in queried_obj:
            ious_queried_obj = ious_on_cluster[:,o]
            best_proposals.append(torch.argmax(ious_queried_obj))
        output_dict["target_proposals"] = best_proposals


        # Mapping for descrption object: object name -> class id
        obj_name = data_dict["object_names"][0]
        obj_name = np.repeat(obj_name, len(class_names))
        target_class = torch.tensor(class_names == obj_name).float()
        data_dict["target_class"] = target_class # One-hot vector

        # Transformer
        transformer_out = self.transformer(data_dict, output_dict)
        output_dict["Match_scores"] = transformer_out["Match_scores"]
        output_dict["CLS_scores"] = transformer_out["CLS_scores"]
        output_dict["DC_scores"] = transformer_out["DC_scores"]
        output_dict["Match_loss"] = transformer_out["Match_loss"]
        output_dict["CLS_loss"] = transformer_out["CLS_loss"]
        output_dict["DC_loss"] = transformer_out["DC_loss"]
        
        return output_dict


    def _loss(self, data_dict, output_dict):
        losses = {}
        losses["Match_loss"] = output_dict["Match_loss"]
        losses["CLS_loss"] = output_dict["CLS_loss"]
        losses["DC_loss"] = output_dict["DC_loss"]
        return losses
    
    
    def training_step(self, data_dict, idx):
        output_dict = self(data_dict)
        losses = self._loss(data_dict, output_dict)
        total_loss = 0
        for loss_name, loss_value in losses.items():
            total_loss += loss_value
            self.log(
                f"train/{loss_name}", loss_value, on_step=False, sync_dist=True,
                on_epoch=True, batch_size=len(data_dict["scan_ids"])
            )
        self.log(
            "train/total_loss", total_loss, on_step=False, sync_dist=True,
            on_epoch=True, batch_size=len(data_dict["scan_ids"])
        )
        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True,  batch_size=len(data_dict["scan_ids"]))
        return total_loss
    
    # def on_train_epoch_end(self):
    #     cosine_lr_decay(
    #         self.trainer.optimizers[0], self.hparams.cfg.model.optimizer.lr, self.current_epoch,
    #         self.hparams.cfg.model.lr_decay.decay_start_epoch, self.hparams.cfg.model.trainer.max_epochs, 1e-6
    #     )

    def validation_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self(data_dict)
        losses = self._loss(data_dict, output_dict)
        # output_dict = output_dict.copy()
        # for k,v in output_dict.items():
        #     if k in ["proposals_idx", "point_features"]:
        #         output_dict[k] = v.to("cuda")

        # log losses
        total_loss = 0
        for loss_name, loss_value in losses.items():
            total_loss += loss_value
            self.log(f"val/{loss_name}", loss_value, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("val/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)


    def on_validation_epoch_end(self):
        pass
        # evaluate instance predictions
        # if self.current_epoch > self.hparams.cfg.model.network.prepare_epochs:
        #     all_pred_insts = []
        #     all_gt_insts = []
        #     all_gt_insts_bbox = []
        #     for pred_instances, gt_instances, gt_instances_bbox in self.val_test_step_outputs:
        #         all_gt_insts_bbox.append(gt_instances_bbox)
        #         all_pred_insts.append(pred_instances)
        #         all_gt_insts.append(gt_instances)
        #     self.val_test_step_outputs.clear()
        #     inst_seg_evaluator = GeneralDatasetEvaluator(
        #         self.hparams.cfg.data.class_names, -1, self.hparams.cfg.data.ignore_classes
        #     )
        #     inst_seg_eval_result = inst_seg_evaluator.evaluate(all_pred_insts, all_gt_insts, print_result=False)

        #     obj_detect_eval_result = evaluate_bbox_acc(
        #         all_pred_insts, all_gt_insts_bbox, self.hparams.cfg.data.class_names,
        #         self.hparams.cfg.data.ignore_classes, print_result=False
        #     )

        #     self.log("val_eval/AP", inst_seg_eval_result["all_ap"], sync_dist=True)
        #     self.log("val_eval/AP 50%", inst_seg_eval_result['all_ap_50%'], sync_dist=True)
        #     self.log("val_eval/AP 25%", inst_seg_eval_result["all_ap_25%"], sync_dist=True)
        #     self.log("val_eval/BBox AP 25%", obj_detect_eval_result["all_bbox_ap_0.25"]["avg"], sync_dist=True)
        #     self.log("val_eval/BBox AP 50%", obj_detect_eval_result["all_bbox_ap_0.5"]["avg"], sync_dist=True)

    def test_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self(data_dict)
            
        #prepare data for visualization
        queried_obj = data_dict["queried_objs"][0]
        predicted_proposal_ids = []
        vg_scores_cpy = output_dict["VG_scores"]
        for _ in queried_obj:
            predicted_proposal_ids.append(torch.argmax(vg_scores_cpy))
            vg_scores_cpy[torch.argmax(vg_scores_cpy)] = -9999
            
        predicted_proposal_ids = [id.item() for id in predicted_proposal_ids]
        predicted_verts = output_dict["proposals_idx"].cpu()
        predicted_verts_arr = []
        for id in predicted_proposal_ids:
            predicted_verts_arr.append(np.array(predicted_verts[predicted_verts[:,0] == id][:,1]).tolist())

        GT_verts_arr = []
        for o in queried_obj:
            b = data_dict["instance_ids"] == o
            GT_verts = b.nonzero()
            GT_verts_arr.append([tensor.item() for tensor in GT_verts])
        
        scan_desc_id = data_dict["scan_ids"][0]+":"+str(data_dict["descr_ids"][0])

        self.val_test_step_outputs.append(
            (predicted_verts_arr, GT_verts_arr, scan_desc_id)
        )

    def on_test_epoch_end(self):
        all_pred_verts = []
        all_gt_verts = []
        all_scan_desc_ids = []
        for predicted_verts, gt_verts, scan_desc_id in self.val_test_step_outputs:
            all_pred_verts.append(predicted_verts)
            all_gt_verts.append(gt_verts)
            all_scan_desc_ids.append(scan_desc_id)

        self.val_test_step_outputs.clear()
        if self.hparams.cfg.model.inference.save_predictions:
            save_dir = os.path.join(
                self.hparams.cfg.exp_output_root_path, 'inference', self.hparams.cfg.model.inference.split,
                'predictions'
            )
            save_prediction_joint_arch(
                save_dir, all_pred_verts, all_gt_verts, all_scan_desc_ids
            )
            self.print(f"\nPredictions saved at {os.path.abspath(save_dir)}")
