import numpy as np
import torch.nn as nn
import sys
from minsu3d.evaluation.instance_segmentation import get_gt_instances, rle_encode
from minsu3d.evaluation.object_detection import get_gt_bbox
from minsu3d.common_ops.functions import softgroup_ops, common_ops
from minsu3d.evaluation.semantic_segmentation import *
from minsu3d.model.module import TinyUnet
from minsu3d.model.general_model import GeneralModel, clusters_voxelization
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from minsu3d.model.transformer import Transformer
from minsu3d.model.softgroup import SoftGroup


class Joint_Arch(GeneralModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.transformer = Transformer(max_text_len=134)
        self.transformer.to("cuda")
        self.bert = BertModel.from_pretrained('bert-base-uncased').to("cuda")
        self.bert.eval()
        self.softgroup = SoftGroup(cfg=cfg)
        checkpoint = torch.load(cfg.model.ckpt_path)
        self.softgroup.load_state_dict(checkpoint['state_dict'])
        for param in self.softgroup.parameters():
            param.requires_grad = False
        output_channel = cfg.model.network.m
        self.instance_classes = cfg.data.classes - len(cfg.data.ignore_classes)

        """
            Top-down Refinement Block
        """
        self.tiny_unet = TinyUnet(output_channel)

        self.classification_branch = nn.Linear(output_channel, self.instance_classes + 1)

        self.mask_scoring_branch = nn.Sequential(
            nn.Linear(output_channel, output_channel),
            nn.ReLU(inplace=True),
            nn.Linear(output_channel, self.instance_classes + 1)
        )

        self.iou_score = nn.Linear(output_channel, self.instance_classes + 1)

    def forward(self, data_dict):
        output_dict = self.softgroup(data_dict)

        # BERT
        with torch.no_grad():
            encoded_layers, _ = self.bert(data_dict["descr_token"]) 
        
        output_dict["descr_embedding"] = encoded_layers[-1]  # Shape: [tensor(batch_size, seq_len, emb_dim)]
        transformer_out = self.transformer(data_dict, output_dict)
        output_dict["VG_scores"] = transformer_out["VG_scores"]
        #output_dict["DC_scores"] = transformer_out["DC_scores"]

        return output_dict


    def global_pool(self, x, expand=False):
        indices = x.coordinates[:, 0]
        batch_offset = torch.cumsum(torch.bincount(indices + 1), dim=0).int()
        x_pool = softgroup_ops.global_avg_pool(x.features, batch_offset)
        if not expand:
            return x_pool
        x_pool_expand = x_pool[indices.long()]
        x.features = torch.cat((x.features, x_pool_expand), dim=1)
        return x

    def _loss(self, data_dict, output_dict):
        losses = self.softgroup._loss(data_dict, output_dict)

        word_ids = data_dict["descr_token"][0]
        vg_scores = output_dict["VG_scores"]
        #dc_scores = output_dict["DC_scores"]
        queried_obj = data_dict["queried_obj"][0]


        # COMPUTE MOST ACCURATE OBJECT PROPOSAL
        proposals_idx = output_dict["proposals_idx"][:, 1].int().contiguous()
        proposals_offset = output_dict["proposals_offset"]
        # calculate iou of clustered instance
        ious_on_cluster = common_ops.get_mask_iou_on_cluster(
            proposals_idx, proposals_offset, data_dict["instance_ids"], data_dict["instance_num_point"]
        )

        ious_queried_obj = ious_on_cluster[:,queried_obj]
        best_proposal = torch.argmax(ious_queried_obj)

        VG_target = torch.zeros(vg_scores.shape)
        VG_target[best_proposal] = 1.0
        VG_loss = self.transformer.loss_criterion(vg_scores, VG_target.to("cuda"))
        # print("PREDICTED",vg_scores, "TRUTH", VG_target)
        # word_ids = nn.functional.one_hot(word_ids, self.transformer.size_vocab)
        # word_ids = torch.tensor(word_ids, dtype=torch.float32)
        #DC_loss = self.transformer.loss_criterion(dc_scores, word_ids)

        losses["VG_loss"] = VG_loss
        #losses["DC_loss"] = DC_loss

        print("VG LOSS:",VG_loss)
        return losses

    def validation_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self(data_dict)
        losses = self._loss(data_dict, output_dict)

        # log losses
        total_loss = 0
        for loss_name, loss_value in losses.items():
            total_loss += loss_value
            self.log(f"val/{loss_name}", loss_value, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)

        # log semantic prediction accuracy
        semantic_predictions = output_dict["semantic_scores"].max(1)[1]
        semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions, data_dict["sem_labels"], ignore_label=-1)
        semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, data_dict["sem_labels"], ignore_label=-1)
        self.log(
            "val_eval/semantic_accuracy", semantic_accuracy, on_step=False, on_epoch=True, sync_dist=True, batch_size=1
        )
        self.log(
            "val_eval/semantic_mean_iou", semantic_mean_iou, on_step=False, on_epoch=True, sync_dist=True, batch_size=1
        )

        if self.current_epoch > self.hparams.cfg.model.network.prepare_epochs:
            point_xyz_cpu = data_dict["point_xyz"].cpu().numpy()
            instance_ids_cpu = data_dict["instance_ids"].cpu()
            sem_labels = data_dict["sem_labels"].cpu()

            pred_instances = self._get_pred_instances(
                data_dict["scan_ids"][0], point_xyz_cpu, output_dict["proposals_idx"].cpu(),
                output_dict["semantic_scores"].size(0), output_dict["cls_scores"].cpu(),
                output_dict["iou_scores"].cpu(), output_dict["mask_scores"].cpu(),
                len(self.hparams.cfg.data.ignore_classes)
            )
            gt_instances = get_gt_instances(
                sem_labels, instance_ids_cpu, self.hparams.cfg.data.ignore_classes
            )
            gt_instances_bbox = get_gt_bbox(
                point_xyz_cpu, instance_ids_cpu.numpy(),
                sem_labels.numpy(), -1, self.hparams.cfg.data.ignore_classes
            )

            self.val_test_step_outputs.append((pred_instances, gt_instances, gt_instances_bbox))

    def test_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self(data_dict)
        semantic_accuracy = None
        semantic_mean_iou = None
        if self.hparams.cfg.model.inference.evaluate:
            semantic_predictions = output_dict["semantic_scores"].max(1)[1]
            semantic_accuracy = evaluate_semantic_accuracy(
                semantic_predictions, data_dict["sem_labels"], ignore_label=-1
            )
            semantic_mean_iou = evaluate_semantic_miou(
                semantic_predictions, data_dict["sem_labels"], ignore_label=-1
            )

        if self.current_epoch > self.hparams.cfg.model.network.prepare_epochs:
            point_xyz_cpu = data_dict["point_xyz"].cpu().numpy()
            instance_ids_cpu = data_dict["instance_ids"].cpu()
            sem_labels = data_dict["sem_labels"].cpu()

            pred_instances = self._get_pred_instances(
                data_dict["scan_ids"][0], point_xyz_cpu, output_dict["proposals_idx"].cpu(),
                output_dict["semantic_scores"].size(0), output_dict["cls_scores"].cpu(), output_dict["iou_scores"].cpu(),
                output_dict["mask_scores"].cpu(), len(self.hparams.cfg.data.ignore_classes)
            )
            gt_instances = None
            gt_instances_bbox = None
            if self.hparams.cfg.model.inference.evaluate:
                gt_instances = get_gt_instances(
                    sem_labels, instance_ids_cpu, self.hparams.cfg.data.ignore_classes
                )

                gt_instances_bbox = get_gt_bbox(
                    point_xyz_cpu, instance_ids_cpu.numpy(),
                    sem_labels.numpy(), -1, self.hparams.cfg.data.ignore_classes
                )

            self.val_test_step_outputs.append(
                (semantic_accuracy, semantic_mean_iou, pred_instances, gt_instances, gt_instances_bbox)
            )

    def _get_pred_instances(self, scan_id, gt_xyz, proposals_idx, num_points, cls_scores, iou_scores, mask_scores,
                            num_ignored_classes):
        num_instances = cls_scores.size(0)
        cls_scores = cls_scores.softmax(1)
        cls_pred_list, score_pred_list, mask_pred_list = [], [], []
        for i in range(self.instance_classes):
            cls_pred = cls_scores.new_full((num_instances,), i + 1, dtype=torch.long)
            cur_cls_scores = cls_scores[:, i]
            cur_iou_scores = iou_scores[:, i]
            cur_mask_scores = mask_scores[:, i]
            score_pred = cur_cls_scores * cur_iou_scores.clamp(0, 1)
            mask_pred = torch.zeros((num_instances, num_points), dtype=torch.bool, device="cpu")
            mask_inds = cur_mask_scores > self.hparams.cfg.model.network.test_cfg.mask_score_thr
            cur_proposals_idx = proposals_idx[mask_inds]
            mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = True

            # filter low score instance
            inds = cur_cls_scores > self.hparams.cfg.model.network.test_cfg.cls_score_thr
            cls_pred = cls_pred[inds]
            score_pred = score_pred[inds]
            mask_pred = mask_pred[inds]

            # filter too small instances
            npoint = torch.count_nonzero(mask_pred, dim=1)
            inds = npoint >= self.hparams.cfg.model.network.test_cfg.min_npoint
            cls_pred = cls_pred[inds]
            score_pred = score_pred[inds]
            mask_pred = mask_pred[inds]

            cls_pred_list.append(cls_pred)
            score_pred_list.append(score_pred)
            mask_pred_list.append(mask_pred)

        cls_pred = torch.cat(cls_pred_list).numpy()
        score_pred = torch.cat(score_pred_list).numpy()
        mask_pred = torch.cat(mask_pred_list).numpy()

        pred_instances = []
        for i in range(cls_pred.shape[0]):
            pred = {'scan_id': scan_id, 'label_id': cls_pred[i], 'conf': score_pred[i],
                    'pred_mask': rle_encode(mask_pred[i])}
            pred_xyz = gt_xyz[mask_pred[i]]
            pred['pred_bbox'] = np.concatenate((pred_xyz.min(0), pred_xyz.max(0)))
            pred_instances.append(pred)
        return pred_instances
