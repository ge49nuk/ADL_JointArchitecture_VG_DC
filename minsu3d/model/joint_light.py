import numpy as np
import torch
import torch.nn as nn
import math
import hydra
import os
import sys
import pytorch_lightning as pl
from minsu3d.model.transformer_light import Transformer_Light
from minsu3d.util.io import save_prediction_joint_arch


class Joint_Light(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.val_test_step_outputs = []
        self.transformer = Transformer_Light()
        
    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.cfg.model.optimizer, params=self.parameters())

    def forward(self, data_dict):
        # Transformer
        transformer_out = self.transformer(data_dict)
        
        return transformer_out


    def _loss(self, output_dict):
        losses = {}
        losses["Match_loss"] = output_dict["Match_loss"]
        losses["CLS_loss"] = output_dict["CLS_loss"]
        losses["DC_loss"] = output_dict["DC_loss"]
        return losses
    
    
    def training_step(self, data_dict, idx):
        output_dict = self(data_dict)
        losses = self._loss(output_dict)
        total_loss = 0
        batch_size = len(data_dict['scan_desc_id'])
        for loss_name, loss_value in losses.items():
            total_loss += loss_value
            self.log(
                f"train/{loss_name}", loss_value, on_step=False, sync_dist=True,
                on_epoch=True, batch_size=batch_size
            )
        self.log(
            "train/total_loss", total_loss, on_step=False, sync_dist=True,
            on_epoch=True, batch_size=batch_size
        )
        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True,  batch_size=batch_size)
        return total_loss
    
    # def on_train_epoch_end(self):
    #     cosine_lr_decay(
    #         self.trainer.optimizers[0], self.hparams.cfg.model.optimizer.lr, self.current_epoch,
    #         self.hparams.cfg.model.lr_decay.decay_start_epoch, self.hparams.cfg.model.trainer.max_epochs, 1e-6
    #     )

    def validation_step(self, data_dict, idx):
        output_dict = self(data_dict)
        losses = self._loss(output_dict)

        # log losses
        total_loss = 0
        for loss_name, loss_value in losses.items():
            total_loss += loss_value
            self.log(f"val/{loss_name}", loss_value, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("val/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)


    def on_validation_epoch_end(self):
        pass
        #     self.log("val_eval/AP", inst_seg_eval_result["all_ap"], sync_dist=True)
        #     self.log("val_eval/AP 50%", inst_seg_eval_result['all_ap_50%'], sync_dist=True)
        #     self.log("val_eval/AP 25%", inst_seg_eval_result["all_ap_25%"], sync_dist=True)
        #     self.log("val_eval/BBox AP 25%", obj_detect_eval_result["all_bbox_ap_0.25"]["avg"], sync_dist=True)
        #     self.log("val_eval/BBox AP 50%", obj_detect_eval_result["all_bbox_ap_0.5"]["avg"], sync_dist=True)

    def test_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self(data_dict)
        
        batch_size = len(data_dict['scan_desc_id'])
        for i in range(batch_size):
            #prepare data for visualization
            queried_obj = data_dict["queried_objs"][i]
            predicted_proposal_ids = []
            match_scores_cpy = output_dict["Match_scores"][i]
            for _ in queried_obj:
                predicted_proposal_ids.append(torch.argmax(match_scores_cpy))
                match_scores_cpy[torch.argmax(match_scores_cpy)] = -9999
                
            predicted_proposal_ids = [id.item() for id in predicted_proposal_ids]
            predicted_verts = data_dict["proposals_idx"][i].cpu()
            predicted_verts_arr = []
            for id in predicted_proposal_ids:
                predicted_verts_arr.append(np.array(predicted_verts[predicted_verts[:,0] == id][:,1]).tolist())

            GT_verts_arr = []
            for o in queried_obj:
                b = data_dict["instance_ids"][i] == o
                GT_verts = b.nonzero()
                GT_verts_arr.append([tensor.item() for tensor in GT_verts])
            
            scan_desc_id = data_dict["scan_desc_id"][i]

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
