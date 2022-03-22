import torch
from torch import nn
from loss.losses import *
from loss.utils import *
import numpy as np
from config.hw_config import Config

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.down_stride = cfg.down_stride

        self.focal_loss = modified_focal_loss
        self.iou_loss = DIOULoss
        self.l1_loss = F.l1_loss
        self.feature_loss = torch.nn.BCEWithLogitsLoss()
        self.alpha = cfg.loss_alpha
        self.beta = cfg.loss_beta
        self.gamma = cfg.loss_gamma

    def forward(self, pred, gt):
        pred_hm, pred_wh, pred_offset = pred
        imgs, gt_boxes, gt_classes, gt_hm, infos = gt

        cls_loss = self.focal_loss(pred_hm, gt_hm)

        wh_loss = cls_loss.new_tensor(0.)
        offset_loss = cls_loss.new_tensor(0.)
        num = 0

        for batch in range(imgs.size(0)):
            ct = infos[batch]['ct'].cuda()
            ct_int = ct.long()
            num += len(ct_int)
            batch_pos_pred_wh = pred_wh[batch, :, ct_int[:, 1], ct_int[:, 0]].view(-1)
            batch_pos_pred_offset = pred_offset[batch, :, ct_int[:, 1], ct_int[:, 0]].view(-1)

            batch_boxes = gt_boxes[batch]
            wh = torch.stack([
                batch_boxes[:, 2] - batch_boxes[:, 0],
                batch_boxes[:, 3] - batch_boxes[:, 1]
            ]).view(-1) / self.down_stride
            offset = (ct - ct_int.float()).T.contiguous().view(-1)
            wh_loss += self.l1_loss(batch_pos_pred_wh, wh, reduction='sum')
            offset_loss += self.l1_loss(batch_pos_pred_offset, offset, reduction='sum')

        regr_loss = wh_loss * self.beta + offset_loss * self.gamma

        # + center_loss / (num + 1e-6)
        return cls_loss * self.alpha , regr_loss / (num + 1e-6), cls_loss.new_tensor(0.)

class JointLoss(nn.Module):
    def __init__(self, cfg):
        super(JointLoss, self).__init__()
        self.down_stride = cfg.down_stride

        self.focal_loss = modified_focal_loss
        self.iou_loss = DIOULoss
        self.l1_loss = F.l1_loss

        self.alpha = cfg.loss_alpha
        self.beta = cfg.loss_beta
        self.gamma = cfg.loss_gamma

    def forward(self, pred, gt):
        pred_hm, pred_keypoints, pred_offset = pred
        imgs, gt_keypoints, gt_classes, gt_hm, infos = gt

        cls_loss = self.focal_loss(pred_hm, gt_hm)

        points_loss = cls_loss.new_tensor(0.)
        offset_loss = cls_loss.new_tensor(0.)
        num = 0
        for batch in range(imgs.size(0)):
            ct = infos[batch]['ct'].cuda()

            ct_int = ct.long()
            num += len(ct_int)

            batch_pos_pred_keypoints = pred_keypoints[batch, :, ct_int[0, 1], ct_int[0, 0]].view(-1)
            batch_pos_pred_offset = pred_offset[batch, :, ct_int[0, 1], ct_int[0, 0]].view(-1)

            batch_keypoints = gt_keypoints[batch]
            points = batch_keypoints.view(-1) / self.down_stride
            offset = (ct - ct_int.float()).T.contiguous().view(-1)
            #print(batch_pos_pred_keypoints, points)
            points_loss += self.l1_loss(batch_pos_pred_keypoints, points, reduction='sum')
            offset_loss += self.l1_loss(batch_pos_pred_offset, offset, reduction='sum')

        regr_loss = points_loss * self.beta + offset_loss * self.gamma
        return cls_loss * self.alpha, regr_loss / (num + 1e-6), cls_loss.new_tensor(0.)


