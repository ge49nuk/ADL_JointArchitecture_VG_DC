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
from minsu3d.common_ops.functions import softgroup_ops, common_ops

from transformers import BertConfig, BertModel
from minsu3d.model.transformer import Transformer
from minsu3d.model.softgroup import SoftGroup

RAM_THRESHOLD = 2000 * 1024 * 1024 # leave at least 2GB of ram free

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
        checkpoint = torch.load(cfg.model.ckpt_path)
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
                out_cpy["proposals_idx"] = output_dict["proposals_idx"].to("cpu")
                out_cpy["point_features"] = output_dict["point_features"].to("cpu")
                out_cpy["proposals_offset"] = output_dict["proposals_offset"].to("cpu")
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

        # Transformer
        transformer_out = self.transformer(data_dict, output_dict)
        output_dict["VG_scores"] = transformer_out["VG_scores"]
        output_dict["DC_scores"] = transformer_out["DC_scores"]
        output_dict["VG_loss"] = transformer_out["VG_loss"]
        output_dict["DC_loss"] = transformer_out["DC_loss"]
        
        return output_dict


    # def global_pool(self, x, expand=False):
    #     indices = x.coordinates[:, 0]
    #     batch_offset = torch.cumsum(torch.bincount(indices + 1), dim=0).int()
    #     x_pool = softgroup_ops.global_avg_pool(x.features, batch_offset)
    #     if not expand:
    #         return x_pool
    #     x_pool_expand = x_pool[indices.long()]
    #     x.features = torch.cat((x.features, x_pool_expand), dim=1)
    #     return x

    def _loss(self, data_dict, output_dict):
        losses = {}

        # word_ids = data_dict["descr_token"][0]
        #dc_scores = output_dict["DC_scores"]
        
        losses["VG_loss"] = output_dict["VG_loss"]
        losses["DC_loss"] = output_dict["DC_loss"]
        
        # print("PREDICTED",vg_scores, "TRUTH", VG_target, "QUERIED", data_dict["queried_obj"])
        # word_ids = nn.functional.one_hot(word_ids, self.transformer.size_vocab)
        # word_ids = torch.tensor(word_ids, dtype=torch.float32)
        #DC_loss = self.transformer.loss_criterion(dc_scores, word_ids)

        #losses["DC_loss"]  = DC_loss
        # print("VG LOSS: {:.4f}".format(VG_loss.item()))
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


        # # log semantic prediction accuracy
        # semantic_predictions = output_dict["semantic_scores"].max(1)[1]
        # semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions, data_dict["sem_labels"], ignore_label=-1)
        # semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, data_dict["sem_labels"], ignore_label=-1)
        # self.log(
        #     "val_eval/semantic_accuracy", semantic_accuracy, on_step=False, on_epoch=True, sync_dist=True, batch_size=1
        # )
        # self.log(
        #     "val_eval/semantic_mean_iou", semantic_mean_iou, on_step=False, on_epoch=True, sync_dist=True, batch_size=1
        # )

        # if self.current_epoch > self.hparams.cfg.model.network.prepare_epochs:
        #     point_xyz_cpu = data_dict["point_xyz"].cpu().numpy()
        #     instance_ids_cpu = data_dict["instance_ids"].cpu()
        #     sem_labels = data_dict["sem_labels"].cpu()

        #     pred_instances = self._get_pred_instances(
        #         data_dict["scan_ids"][0], point_xyz_cpu, output_dict["proposals_idx"].cpu(),
        #         output_dict["semantic_scores"].size(0), output_dict["cls_scores"].cpu(),
        #         output_dict["iou_scores"].cpu(), output_dict["mask_scores"].cpu(),
        #         len(self.hparams.cfg.data.ignore_classes)
        #     )
        #     gt_instances = get_gt_instances(
        #         sem_labels, instance_ids_cpu, self.hparams.cfg.data.ignore_classes
        #     )
        #     gt_instances_bbox = get_gt_bbox(
        #         point_xyz_cpu, instance_ids_cpu.numpy(),
        #         sem_labels.numpy(), -1, self.hparams.cfg.data.ignore_classes
        #     )

        #     self.val_test_step_outputs.append((pred_instances, gt_instances, gt_instances_bbox))

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
        pass
        # prepare input and forward
        # output_dict = self(data_dict)
        # semantic_accuracy = None
        # semantic_mean_iou = None
        # if self.hparams.cfg.model.inference.evaluate:
        #     semantic_predictions = output_dict["semantic_scores"].max(1)[1]
        #     semantic_accuracy = evaluate_semantic_accuracy(
        #         semantic_predictions, data_dict["sem_labels"], ignore_label=-1
        #     )
        #     semantic_mean_iou = evaluate_semantic_miou(
        #         semantic_predictions, data_dict["sem_labels"], ignore_label=-1
        #     )

        # if self.current_epoch > self.hparams.cfg.model.network.prepare_epochs:
        #     point_xyz_cpu = data_dict["point_xyz"].cpu().numpy()
        #     instance_ids_cpu = data_dict["instance_ids"].cpu()
        #     sem_labels = data_dict["sem_labels"].cpu()

        #     pred_instances = self._get_pred_instances(
        #         data_dict["scan_ids"][0], point_xyz_cpu, output_dict["proposals_idx"].cpu(),
        #         output_dict["semantic_scores"].size(0), output_dict["cls_scores"].cpu(), output_dict["iou_scores"].cpu(),
        #         output_dict["mask_scores"].cpu(), len(self.hparams.cfg.data.ignore_classes)
        #     )
        #     gt_instances = None
        #     gt_instances_bbox = None
        #     if self.hparams.cfg.model.inference.evaluate:
        #         gt_instances = get_gt_instances(
        #             sem_labels, instance_ids_cpu, self.hparams.cfg.data.ignore_classes
        #         )

        #         gt_instances_bbox = get_gt_bbox(
        #             point_xyz_cpu, instance_ids_cpu.numpy(),
        #             sem_labels.numpy(), -1, self.hparams.cfg.data.ignore_classes
        #         )

        #     self.val_test_step_outputs.append(
        #         (semantic_accuracy, semantic_mean_iou, pred_instances, gt_instances, gt_instances_bbox)
        #     )

    def on_test_epoch_end(self):
        pass
        # evaluate instance predictions
        # if self.current_epoch > self.hparams.cfg.model.network.prepare_epochs:
        #     all_pred_insts = []
        #     all_gt_insts = []
        #     all_gt_insts_bbox = []
        #     all_sem_acc = []
        #     all_sem_miou = []
        #     for semantic_accuracy, semantic_mean_iou, pred_instances, gt_instances, gt_instances_bbox in self.val_test_step_outputs:
        #         all_sem_acc.append(semantic_accuracy)
        #         all_sem_miou.append(semantic_mean_iou)
        #         all_gt_insts_bbox.append(gt_instances_bbox)
        #         all_gt_insts.append(gt_instances)
        #         all_pred_insts.append(pred_instances)

        #     self.val_test_step_outputs.clear()

        #     if self.hparams.cfg.model.inference.evaluate:
        #         inst_seg_evaluator = GeneralDatasetEvaluator(
        #             self.hparams.cfg.data.class_names, -1, self.hparams.cfg.data.ignore_classes
        #         )
        #         self.print("Evaluating instance segmentation ...")
        #         inst_seg_eval_result = inst_seg_evaluator.evaluate(all_pred_insts, all_gt_insts, print_result=True)
        #         obj_detect_eval_result = evaluate_bbox_acc(
        #             all_pred_insts, all_gt_insts_bbox,
        #             self.hparams.cfg.data.class_names, self.hparams.cfg.data.ignore_classes, print_result=True
        #         )

        #         sem_miou_avg = np.mean(np.array(all_sem_miou))
        #         sem_acc_avg = np.mean(np.array(all_sem_acc))
        #         self.print(f"Semantic Accuracy: {sem_acc_avg}")
        #         self.print(f"Semantic mean IoU: {sem_miou_avg}")

        #     if self.hparams.cfg.model.inference.save_predictions:
        #         save_dir = os.path.join(
        #             self.hparams.cfg.exp_output_root_path, 'inference', self.hparams.cfg.model.inference.split,
        #             'predictions'
        #         )
        #         save_prediction(
        #             save_dir, all_pred_insts, self.hparams.cfg.data.mapping_classes_ids,
        #             self.hparams.cfg.data.ignore_classes
        #         )
        #         self.print(f"\nPredictions saved at {os.path.abspath(save_dir)}")


    # def _get_pred_instances(self, scan_id, gt_xyz, proposals_idx, num_points, cls_scores, iou_scores, mask_scores,
    #                         num_ignored_classes):
    #     num_instances = cls_scores.size(0)
    #     cls_scores = cls_scores.softmax(1)
    #     cls_pred_list, score_pred_list, mask_pred_list = [], [], []
    #     for i in range(self.instance_classes):
    #         cls_pred = cls_scores.new_full((num_instances,), i + 1, dtype=torch.long)
    #         cur_cls_scores = cls_scores[:, i]
    #         cur_iou_scores = iou_scores[:, i]
    #         cur_mask_scores = mask_scores[:, i]
    #         score_pred = cur_cls_scores * cur_iou_scores.clamp(0, 1)
    #         mask_pred = torch.zeros((num_instances, num_points), dtype=torch.bool, device="cpu")
    #         mask_inds = cur_mask_scores > self.hparams.cfg.model.network.test_cfg.mask_score_thr
    #         cur_proposals_idx = proposals_idx[mask_inds]
    #         mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = True

    #         # filter low score instance
    #         inds = cur_cls_scores > self.hparams.cfg.model.network.test_cfg.cls_score_thr
    #         cls_pred = cls_pred[inds]
    #         score_pred = score_pred[inds]
    #         mask_pred = mask_pred[inds]

    #         # filter too small instances
    #         npoint = torch.count_nonzero(mask_pred, dim=1)
    #         inds = npoint >= self.hparams.cfg.model.network.test_cfg.min_npoint
    #         cls_pred = cls_pred[inds]
    #         score_pred = score_pred[inds]
    #         mask_pred = mask_pred[inds]

    #         cls_pred_list.append(cls_pred)
    #         score_pred_list.append(score_pred)
    #         mask_pred_list.append(mask_pred)

    #     cls_pred = torch.cat(cls_pred_list).numpy()
    #     score_pred = torch.cat(score_pred_list).numpy()
    #     mask_pred = torch.cat(mask_pred_list).numpy()

    #     pred_instances = []
    #     for i in range(cls_pred.shape[0]):
    #         pred = {'scan_id': scan_id, 'label_id': cls_pred[i], 'conf': score_pred[i],
    #                 'pred_mask': rle_encode(mask_pred[i])}
    #         pred_xyz = gt_xyz[mask_pred[i]]
    #         pred['pred_bbox'] = np.concatenate((pred_xyz.min(0), pred_xyz.max(0)))
    #         pred_instances.append(pred)
    #     return pred_instances
