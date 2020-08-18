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

from utils.torch_utils import to_cpu, _sigmoid


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
    def __init__(self, device):
        super(Compute_Loss, self).__init__()
        self.device = device
        self.focal_loss = FocalLoss()
        self.l1_loss = L1Loss()
        self.vercoor_l1 = Vertexes_Coor_L1Loss()
        self.l2_loss = L2Loss()
        self.rot_loss = BinRotLoss()
        self.weight_hm_mc, self.weight_hm_ver, self.weight_dim, self.weight_rot, self.weight_depth = 1., 1., 1., 0.5, 0.1
        self.weight_vercoor, self.weight_cenoff, self.weight_veroff, self.weight_wh = 1., 0.5, 0.5, 0.5
        self.mean_dim = torch.tensor([1.53, 1.62, 3.89], device=device, dtype=torch.float).view(1, 1, 3)
        self.std_dim = torch.tensor([0.13, 0.1, 0.41], device=device, dtype=torch.float).view(1, 1, 3)

    def normalize_dim(self, dim):
        """
        dim: (batch, max_objects, 3)
        return: normalized dimension
        """
        return (dim - self.mean_dim) / self.std_dim

    def forward(self, outputs, tg):
        # tg: targets
        outputs['hm_mc'] = _sigmoid(outputs['hm_mc'])
        outputs['hm_ver'] = _sigmoid(outputs['hm_ver'])

        # Normalize dimension
        # TODO: What happend if the norm_dim < 0, we can't apply the log operator
        # tg['dim'] = self.normalize_dim(tg['dim'])
        # tg['dim'] = F.log(tg['dim'])  # take the log of the normalized dimension

        # Follow depth loss in CenterNet
        outputs['depth'] = 1. / (_sigmoid(outputs['depth']) + 1e-9) - 1.

        l_hm_mc = self.focal_loss(outputs['hm_mc'], tg['hm_mc'])
        l_hm_ver = self.focal_loss(outputs['hm_ver'], tg['hm_ver'])
        # output, mask, ind, target
        l_vercoor = self.vercoor_l1(outputs['vercoor'], tg['ver_coor_mask'], tg['indices_center'], tg['ver_coor'])
        l_cenoff = self.l1_loss(outputs['cenoff'], tg['obj_mask'], tg['indices_center'], tg['cen_offset'])
        l_veroff = self.l1_loss(outputs['veroff'], tg['ver_offset_mask'], tg['indices_vertexes'], tg['ver_offset'])
        # TODO: What happend if the norm_dim < 0, we can't apply the log operator
        # Apply dimension loss (l1_loss) in the CenterNet instead of the l2_loss in the paper
        l_dim = self.l1_loss(outputs['dim'], tg['obj_mask'], tg['indices_center'], tg['dim'])
        # output, mask, ind, rotbin, rotres
        l_rot = self.rot_loss(outputs['rot'], tg['obj_mask'], tg['indices_center'], tg['rotbin'], tg['rotres'])
        # Apply depth loss (l1_loss) in the CenterNet instead of the l2_loss in the paper
        # l_depth = self.l2_loss(torch.log(outputs['depth']), tg['obj_mask'], tg['indices_center'], torch.log(tg['depth']))
        l_depth = self.l1_loss(outputs['depth'], tg['obj_mask'], tg['indices_center'], tg['depth'])
        l_boxwh = self.l1_loss(outputs['wh'], tg['obj_mask'], tg['indices_center'], tg['wh'])

        total_loss = l_hm_mc * self.weight_hm_mc + l_hm_ver * self.weight_hm_ver + l_vercoor * self.weight_vercoor + \
                     l_cenoff * self.weight_cenoff + l_veroff * self.weight_veroff + l_dim * self.weight_dim + \
                     l_rot * self.weight_rot + l_depth * self.weight_depth + l_boxwh * self.weight_wh

        loss_stats = {
            'total_loss': to_cpu(total_loss).item(),
            'hm_mc_loss': to_cpu(l_hm_mc).item(),
            'hm_ver_loss': to_cpu(l_hm_ver).item(),
            'ver_coor_loss': to_cpu(l_vercoor).item(),
            'cen_offset_loss': to_cpu(l_cenoff).item(),
            'ver_offset_loss': to_cpu(l_veroff).item(),
            'dim_loss': to_cpu(l_dim).item(),
            'rot_loss': to_cpu(l_rot).item(),
            'depth_loss': to_cpu(l_depth).item(),
            'wh_loss': to_cpu(l_boxwh).item()
        }

        return total_loss, loss_stats
