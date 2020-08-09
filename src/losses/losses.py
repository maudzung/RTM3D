# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# Modified by Nguyen Mau Dung (2020.08.09)
# ------------------------------------------------------------------------------

import sys

import torch.nn as nn
import torch
import torch.nn.functional as F

sys.path.append('../')

from utils.torch_utils import to_cpu


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _neg_loss(pred, gt, alpha=2, beta=4):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, beta)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class Vertexes_Coor_L1Loss(nn.Module):
    def __init__(self):
        super(Vertexes_Coor_L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.mse_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _transpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss


def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')


def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res


class Compute_Loss(nn.Module):
    def __init__(self):
        super(Compute_Loss, self).__init__()
        self.focal_loss = FocalLoss()
        self.l1_loss = L1Loss()
        self.vercoor_l1_loss = Vertexes_Coor_L1Loss()
        self.l2_loss = L2Loss()
        self.rot_loss = BinRotLoss()
        self.weight_mc, self.weight_ver, self.weight_dim, self.weight_rot, self.weight_depth = 1., 1., 1., 0.5, 0.1
        self.weight_vercoor, self.weight_cenoff, self.weight_veroff = 1., 0.5, 0.5

    def forward(self, outputs, targets):
        l_mc = self.focal_loss(F.sigmoid(outputs['hm_mc']), targets['hm_mc'])
        l_ver = self.focal_loss(F.sigmoid(outputs['hm_ver']), targets['hm_ver'])
        # output, mask, ind, target
        l_vercoor = self.vercoor_l1_loss(outputs['hm_vercoor'], targets['ver_coor_mask'], targets['indices_center'],
                                         targets['ver_coor'])
        l_cenoff = self.l1_loss(outputs['hm_cenoff'], targets['obj_mask'], targets['indices_center'],
                                targets['cen_offset'])
        l_veroff = self.l1_loss(outputs['hm_veroff'], targets['ver_offset_mask'], targets['indices_vertexes'],
                                targets['ver_offset'])
        l_dim = self.l2_loss(outputs['hm_dim'], targets['obj_mask'], targets['indices_center'], targets['dimension'])
        # output, mask, ind, rotbin, rotres
        l_rot = self.rot_loss(outputs['hm_rot'], targets['obj_mask'], targets['indices_center'], targets['rotbin'],
                              targets['rotres'])
        # TODO: What happend if the norm_dim < 0, we can't apply the log operator
        # l_depth = self.l2_loss(torch.log(outputs['hm_depth']), targets['obj_mask'], targets['indices_center'],
        #                        torch.log(targets['depth']))
        l_depth = self.l2_loss(outputs['hm_depth'], targets['obj_mask'], targets['indices_center'], targets['depth'])

        total_loss = l_mc * self.weight_mc + l_ver * self.weight_ver + l_vercoor * self.weight_vercoor + \
                     l_cenoff * self.weight_cenoff + l_veroff * self.weight_veroff + l_dim * self.weight_dim + \
                     l_rot * self.weight_rot + l_depth * self.weight_depth

        loss_stats = {
            'total_loss': to_cpu(total_loss).item(),
            'hm_mc_loss': to_cpu(l_mc).item(),
            'hm_ver_loss': to_cpu(l_ver).item(),
            'ver_coor_loss': to_cpu(l_vercoor).item(),
            'cen_offset_loss': to_cpu(l_cenoff).item(),
            'ver_offset_loss': to_cpu(l_veroff).item(),
            'dim_loss': to_cpu(l_dim).item(),
            'rot_loss': to_cpu(l_rot).item(),
            'depth_loss': to_cpu(l_depth).item()
        }

        return total_loss, loss_stats
