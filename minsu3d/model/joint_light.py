import numpy as np
import torch
import torch.nn as nn
import math
import hydra
import os
import sys
import json
import pytorch_lightning as pl
from transformers import BertTokenizer
from minsu3d.model.transformer_light import Transformer_Light
from minsu3d.util.io import save_prediction_joint_arch
from minsu3d.util.eval_utils import get_scene_info, calculate_iou, get_bbox, _get_unique_multiple_lookup
from minsu3d.capeval.eval_helper import eval_cap


class Joint_Light(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.val_test_step_outputs = []
        self.transformer = Transformer_Light()
        self.batch_size = cfg.data.batch_size

        self.correct_guesses_train = [0,0]

        self.correct_guesses_val = [0,0]
        self.iou_val = [0,0,0] # <- iou25 iou50 total

        self.candidates_iou25 = {}
        self.candidates_iou50 = {}

        self.epoch_count = 0

        self.unique_multiple_lookup = _get_unique_multiple_lookup(self.hparams.cfg)
        self.unique_iou = [0,0,0] # <- iou25 iou50 total
        self.multiple_iou = [0,0,0] # <- iou25 iou50 total
        
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.cfg.model.optimizer, params=filter(lambda p: p.requires_grad, self.parameters()), weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=20,
            gamma=0.8
        )

        return [optimizer], [scheduler]

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer=optimizer,
        #     mode='min',
        #     factor=0.8,
        #     patience=20,
        #     min_lr=0.0000001
        # )

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val/total_loss",
        #         "frequency": 30
        #     },
        # }

    def forward(self, data_dict):
        # Transformer
        transformer_out = self.transformer(data_dict)
        # print(torch.max(transformer_out["DC_scores"][0], dim=1))
        return transformer_out


    def _loss(self, output_dict):
        losses = {}
        losses["Match_loss"] = output_dict["Match_loss"]
        losses["CLS_loss"] = output_dict["CLS_loss"]
        losses["DC_loss"] = output_dict["DC_loss"]
        losses["Cider_loss"] = output_dict["Cider_loss"]
        return losses
    
    
    def training_step(self, data_dict, idx):
        output_dict = self(data_dict)
        losses = self._loss(output_dict)
        total_loss = 0
        batch_size = len(output_dict["Match_scores"])
        for loss_name, loss_value in losses.items():
            total_loss += loss_value
            self.log(f"train/{loss_name}", loss_value, on_step=False, sync_dist=True, on_epoch=True, batch_size=batch_size)
        self.log("train/total_loss", total_loss, on_step=False, sync_dist=True, on_epoch=True, batch_size=batch_size)
        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True,  batch_size=batch_size)
    
        # Log accuracy 
        target_proposals = data_dict['target_proposals']
        target_proposal_splits = data_dict['target_proposal_splits']
        best_proposals = torch.tensor_split(target_proposals, target_proposal_splits[1:-1], dim=0)

        if self.batch_size == batch_size: # check if we have a full batch, dont log if not
            for bi in range(self.batch_size):
                if self.batch_size > len(output_dict["Match_scores"]):
                    break
                # Log correct guesses
                self.correct_guesses_train[1] += 1
                if torch.argmax(output_dict["Match_scores"][bi]) in best_proposals[bi]:
                    self.correct_guesses_train[0] += 1
        
        return total_loss
    
    # def on_train_epoch_end(self):
    #     cosine_lr_decay(
    #         self.trainer.optimizers[0], self.hparams.cfg.model.optimizer.lr, self.current_epoch,
    #         self.hparams.cfg.model.lr_decay.decay_start_epoch, self.hparams.cfg.model.trainer.max_epochs, 1e-6
    #     )

    def validation_step(self, data_dict, idx):
        if self.transformer.disturb:
            self.transformer.disable_disturb()
        output_dict = self(data_dict)
        losses = self._loss(output_dict)
        batch_size = len(output_dict["Match_scores"])

        # log losses
        total_loss = 0
        for loss_name, loss_value in losses.items():
            total_loss += loss_value
            self.log(f"val/{loss_name}", loss_value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("val/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        # Log correct guesses
        target_proposals = data_dict['target_proposals']
        target_proposal_splits = data_dict['target_proposal_splits']
        best_proposals = torch.tensor_split(target_proposals, target_proposal_splits[1:-1], dim=0)
        
        if self.batch_size == batch_size: # check if we have a full batch, dont log if not
            self.correct_guesses_val[1] += self.batch_size
            self.iou_val[2] += self.batch_size
            for bi in range(self.batch_size): 
                guess = torch.argmax(output_dict["Match_scores"][bi])
                if guess in best_proposals[bi]:
                    self.correct_guesses_val[0] += 1
                for o in data_dict["queried_objs"][bi]:
                    if data_dict["ious_on_cluster"][bi][guess][o] >= 0.25:
                        self.iou_val[0] += 1
                        if data_dict["ious_on_cluster"][bi][guess][o] >= 0.5:
                            self.iou_val[1] += 1


    def on_train_epoch_end(self):
        if self.correct_guesses_train[1] != 0:
            train_acc = self.correct_guesses_train[0]/self.correct_guesses_train[1]
            self.log("train/acc", train_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.correct_guesses_train = [0,0]
        self.epoch_count += 1
        if self.epoch_count == self.hparams.cfg.model.disturbation_start_epoch:
            self.transformer.start_disturb()
    
    def on_validation_epoch_end(self):
        if not self.correct_guesses_val[1] == 0.0:
            val_acc = self.correct_guesses_val[0]/self.correct_guesses_val[1]
            iou25 = self.iou_val[0]/self.iou_val[2]
            iou50 = self.iou_val[1]/self.iou_val[2]
            self.log("val/acc", val_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            print("\nIOU25(val):", iou25, "IOU50(val):", iou50, "\n")
        
        # bleu4, cider, rouge, meteor = eval_cap(self.hparams.cfg, self.candidates_iou25, self.candidates_iou50)

        # folder_path = os.path.join(self.hparams.cfg.exp_output_root_path, 'val_scores')
        # os.makedirs(folder_path, exist_ok=True)
        # score_path = os.path.join(folder_path, "epoch_{}.json".format(self.epoch_count))
        # self.epoch_count += 10

        # scores = {
        #         "iou25": iou25,
        #         "iou50": iou50,
        #         "bleu4_iou25": bleu4[0],
        #         "cider_iou25": cider[0],
        #         "rouge_iou25": rouge[0],
        #         "meteor_iou25": meteor[0],
        #         "bleu4_iou50": bleu4[1],
        #         "cider_iou50": cider[1],
        #         "rouge_iou50": rouge[1],
        #         "meteor_iou50": meteor[1]
        #         }
        # with open(score_path, "w") as f:
        #     json.dump(scores, f, indent=4)

        # self.log("val/bleu4_iou25", bleu4[0], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("val/cider_iou25", cider[0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("val/rouge_iou25", rouge[0], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("val/meteor_iou25", meteor[0], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("val/bleu4_iou50", bleu4[1], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("val/cider_iou50", cider[1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("val/rouge_iou50", rouge[1], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("val/meteor_iou50", meteor[1], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        self.correct_guesses_val = [0, 0]
        self.iou_val = [0, 0, 0]
        # self.candidates_iou25 = {}
        # self.candidates_iou50 = {}
        if self.epoch_count >= self.hparams.cfg.model.disturbation_start_epoch:
            self.transformer.start_disturb()

    def test_step(self, data_dict, idx):
        if self.transformer.disturb:
            self.transformer.disable_disturb()
        # prepare input and forward
        Match_scores = self.transformer.feed_VG(data_dict)     
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        batch_size = len(data_dict['scan_desc_id'])
        for i in range(batch_size):
            #prepare data for visualization
            queried_obj = data_dict["queried_objs"][i]
            predicted_proposal_ids = []
            match_scores_cpy = Match_scores[i]
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
            scan_id, _ = scan_desc_id.split("::")

            #Calculate bbox iou
            points, colors, indices = get_scene_info(self.hparams.cfg.data.scans_path, scan_id)

            bboxes_pred = [get_bbox(verts, points) for verts in predicted_verts_arr]
            bboxes_gt = [get_bbox(verts, points) for verts in GT_verts_arr]
            iou = calculate_iou(bboxes_pred[0], bboxes_gt[0])
                
            if self.unique_multiple_lookup[scan_id][data_dict["object_ids"][i]][data_dict["ann_ids"][i]] == 0:
                self.unique_iou[2] += 1.
                if iou >= 0.25:
                    self.unique_iou[0] += 1.
                    key = "{}|{}|{}".format(data_dict["scene_ids"][i], data_dict["object_ids"][i], data_dict["object_names"][i])
                    if key not in self.candidates_iou25:
                        candidate_descr = self.transformer.feed_DC(data_dict, idx=i)
                        self.candidates_iou25[key] = [candidate_descr]
                    if iou >= 0.5:
                        self.unique_iou[1] += 1.
                        if key not in self.candidates_iou50:
                            self.candidates_iou50[key] = self.candidates_iou25[key]
            else:
                self.multiple_iou[2] += 1.
                if iou >= 0.25:
                    self.multiple_iou[0] += 1.
                    key = "{}|{}|{}".format(data_dict["scene_ids"][i], data_dict["object_ids"][i], data_dict["object_names"][i])
                    if key not in self.candidates_iou25:
                        candidate_descr = self.transformer.feed_DC(data_dict, idx=i)
                        self.candidates_iou25[key] = [candidate_descr]
                    if iou >= 0.5:
                        self.multiple_iou[1] += 1.
                        if key not in self.candidates_iou50:
                            self.candidates_iou50[key] = self.candidates_iou25[key]
            

            gt_descr = tokenizer.decode(data_dict["target_word_ids"][i][1:data_dict['num_tokens'][i]-1])

            self.val_test_step_outputs.append(
                (predicted_verts_arr, GT_verts_arr, scan_desc_id, gt_descr, bboxes_pred, bboxes_gt)
            )

    def on_test_epoch_end(self):
        # print("Unique IOU25:", self.unique_iou[0] / self.unique_iou[2])
        # print("Unique IOU50:", self.unique_iou[1] / self.unique_iou[2])
        # print("Multiple IOU25:", self.multiple_iou[0] / self.multiple_iou[2])
        # print("Multiple IOU50:", self.multiple_iou[1] / self.multiple_iou[2])
        # print("Overall IOU25:", (self.unique_iou[0]+self.multiple_iou[0]) / (self.unique_iou[2]+self.multiple_iou[2]))
        # print("Overall IOU50:", (self.unique_iou[1]+self.multiple_iou[1]) / (self.unique_iou[2]+self.multiple_iou[2]))
        all_pred_verts = []
        all_gt_verts = []
        all_scan_desc_ids = []
        all_desc = []
        all_bboxes_gt = []
        all_bboxes_pred = []
        for predicted_verts, gt_verts, scan_desc_id, gt_descr, bboxes_pred, bboxes_gt in self.val_test_step_outputs:
            all_pred_verts.append(predicted_verts)
            all_gt_verts.append(gt_verts)
            all_scan_desc_ids.append(scan_desc_id)
            all_desc.append(gt_descr)
            all_bboxes_pred.append(bboxes_pred)
            all_bboxes_gt.append(bboxes_gt)

        self.val_test_step_outputs.clear()
        if self.hparams.cfg.model.inference.save_predictions:
            save_dir = os.path.join(
                self.hparams.cfg.exp_output_root_path, 'inference', self.hparams.cfg.model.inference.split,
                'predictions'
            )
            save_prediction_joint_arch(
                save_dir, all_pred_verts, all_gt_verts, all_scan_desc_ids, all_desc, all_bboxes_pred, all_bboxes_gt
            )
            self.print(f"\nPredictions saved at {os.path.abspath(save_dir)}")

        # F1 scores
        bleu4, cider, rouge, meteor = eval_cap(self.hparams.cfg, self.candidates_iou25, self.candidates_iou50)

        folder_path = os.path.join(self.hparams.cfg.exp_output_root_path, 'F1_scores')
        os.makedirs(folder_path, exist_ok=True)
        score_path = os.path.join(folder_path, "F1_scores.json")

        scores = {
                "bleu4_iou25": bleu4[0],
                "cider_iou25": cider[0],
                "rouge_iou25": rouge[0],
                "meteor_iou25": meteor[0],
                "bleu4_iou50": bleu4[1],
                "cider_iou50": cider[1],
                "rouge_iou50": rouge[1],
                "meteor_iou50": meteor[1]
                }
        with open(score_path, "w") as f:
            json.dump(scores, f, indent=4)