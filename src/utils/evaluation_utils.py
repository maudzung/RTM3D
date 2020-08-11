"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.10
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: The utils for evaluation
# Refer from: https://github.com/xingyizhou/CenterNet
"""

from __future__ import division
import sys

import torch
import numpy as np
import torch.nn.functional as F
import cv2

sys.path.append('../')


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep


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


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def rtm3d_decode(hm_mc, hm_ver, ver_coor, cen_off, ver_off, wh, rot, depth, dim, K=40):
    batch_size, num_classes, height, width = hm_mc.size()
    num_vertexes = hm_ver.size(1)

    hm_mc = _nms(hm_mc)
    scores, inds, clses, ys, xs = _topk(hm_mc, K=K)
    if cen_off is not None:
        cen_off = _transpose_and_gather_feat(cen_off, inds)
        cen_off = cen_off.view(batch_size, K, 2)
        xs = xs.view(batch_size, K, 1) + cen_off[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + cen_off[:, :, 1:2]
    else:
        xs = xs.view(batch_size, K, 1) + 0.5
        ys = ys.view(batch_size, K, 1) + 0.5

    rot = _transpose_and_gather_feat(rot, inds)
    rot = rot.view(batch_size, K, 8)
    depth = _transpose_and_gather_feat(depth, inds)
    depth = depth.view(batch_size, K, 1)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch_size, K, 3)
    clses = clses.view(batch_size, K, 1).float()
    scores = scores.view(batch_size, K, 1)

    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch_size, K, 2)

    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)

    ver_coor = _transpose_and_gather_feat(ver_coor, inds)
    ver_coor = ver_coor.view(batch_size, K, num_vertexes * 2)
    ver_coor[..., ::2] += xs.view(batch_size, K, 1).expand(batch_size, K, num_vertexes)
    ver_coor[..., 1::2] += ys.view(batch_size, K, 1).expand(batch_size, K, num_vertexes)
    ver_coor = ver_coor.view(batch_size, K, num_vertexes, 2).permute(0, 2, 1, 3).contiguous()  # b x J x K x 2
    reg_kps = ver_coor.unsqueeze(3).expand(batch_size, num_vertexes, K, K, 2)

    hm_ver = _nms(hm_ver)
    thresh = 0.1
    hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_ver, K=K)  # b x J x K
    if ver_off is not None:
        ver_off = _transpose_and_gather_feat(ver_off, hm_inds.view(batch_size, -1))
        ver_off = ver_off.view(batch_size, num_vertexes, K, 2)
        hm_xs = hm_xs + ver_off[:, :, :, 0]
        hm_ys = hm_ys + ver_off[:, :, :, 1]
    else:
        hm_xs = hm_xs + 0.5
        hm_ys = hm_ys + 0.5

    mask = (hm_score > thresh).float()
    hm_score = (1 - mask) * -1 + mask * hm_score
    hm_ys = (1 - mask) * (-10000) + mask * hm_ys
    hm_xs = (1 - mask) * (-10000) + mask * hm_xs
    hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(2).expand(batch_size, num_vertexes, K, K, 2)
    dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
    min_dist, min_ind = dist.min(dim=3)  # b x J x K
    hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
    min_dist = min_dist.unsqueeze(-1)
    min_ind = min_ind.view(batch_size, num_vertexes, K, 1, 1).expand(batch_size, num_vertexes, K, 1, 2)
    hm_kps = hm_kps.gather(3, min_ind)
    hm_kps = hm_kps.view(batch_size, num_vertexes, K, 2)
    l = bboxes[:, :, 0].view(batch_size, 1, K, 1).expand(batch_size, num_vertexes, K, 1)
    t = bboxes[:, :, 1].view(batch_size, 1, K, 1).expand(batch_size, num_vertexes, K, 1)
    r = bboxes[:, :, 2].view(batch_size, 1, K, 1).expand(batch_size, num_vertexes, K, 1)
    b = bboxes[:, :, 3].view(batch_size, 1, K, 1).expand(batch_size, num_vertexes, K, 1)
    mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
           (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
           (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
    mask = (mask > 0).float().expand(batch_size, num_vertexes, K, 2)
    ver_coor = (1 - mask) * hm_kps + mask * ver_coor
    ver_coor = ver_coor.permute(0, 2, 1, 3).contiguous().view(batch_size, K, num_vertexes * 2)

    # (scores x 1, xs x 1, ys x 1, wh x 2, bboxes x 4, ver_coor x 16, rot x 8, depth x 1, dim x 3, clses x 1)
    # (scores-0:1, xs-1:2, ys-2:3, wh-3:5, bboxes-5:9, ver_coor-9:25, rot-25:33, depth-33:34, dim-34:37, clses-37:38)
    # detections: [batch_size, K, 38]
    detections = torch.cat([scores, xs, ys, wh, bboxes, ver_coor, rot, depth, dim, clses], dim=2)

    return detections


def get_alpha(rot):
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # return rot[:, 0]
    idx = rot[:, 1] > rot[:, 5]
    alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


def get_pred_depth(depth):
    return depth


def post_processing_2d(detections):
    """

    :param detections: [batch_size, K, 38]
    # (scores x 1, xs x 1, ys x 1, wh x 2, bboxes x 4, ver_coor x 16, rot x 8, depth x 1, dim x 3, clses x 1)
    # (scores-0:1, xs-1:2, ys-2:3, wh-3:5, bboxes-5:9, ver_coor-9:25, rot-25:33, depth-33:34, dim-34:37, clses-37:38)
    :param conf_thresh:
    :return:
    """
    # TODO: Need to consider rescale to the original scale: bbox, xs, ys, and ver_coor - 1:25
    num_classes = 3
    down_ratio = 4
    ret = []
    for i in range(detections.shape[0]):
        top_preds = {}
        classes = detections[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j] = np.concatenate([
                detections[i, inds, :1].astype(np.float32),
                detections[i, inds, 1:25].astype(np.float32) * down_ratio,
                get_alpha(detections[i, inds, 25:33])[:, np.newaxis].astype(np.float32),
                get_pred_depth(detections[i, inds, 33:34]).astype(np.float32),
                detections[i, inds, 34:37].astype(np.float32)], axis=1)
        ret.append(top_preds)

    return ret


def get_final_pred(detections, num_classes=3, peak_thresh=0.2):
    for j in range(num_classes):
        if len(detections[j] > 0):
            keep_inds = (detections[j][:, 0] > peak_thresh)
            detections[j] = detections[j][keep_inds]

    return detections


def draw_predictions(img, detections, colors, num_classes=3):
    for j in range(num_classes):
        if len(detections[j] > 0):
            for det in detections[j]:
                # (scores-0:1, xs-1:2, ys-2:3, wh-3:5, bboxes-5:9, ver_coor-9:25, rot-25:26, depth-26:27, dim-27:30)
                _score = det[0]
                _x, _y, _wh, _bbox, _ver_coor = det[1], det[2], det[3:5], det[5:9], det[9:25]
                _rot, _depth, _dim = det[25], det[26], det[27:30]
                _bbox = np.array(_bbox, dtype=np.int)
                img = cv2.rectangle(img, (_bbox[0], _bbox[1]), (_bbox[2], _bbox[3]), colors[j], 2)

    return img


def post_processing_3d(detections, conf_thresh=0.95):
    """

    """
    pass
