"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.09
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for the KITTI dataset
# Refer from: https://github.com/xingyizhou/CenterNet
"""

import sys
import os
import random

import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import cv2

sys.path.append('../')

from data_process.kitti_data_utils import gen_2d_gaussian_hm, compute_box_3d, draw_box_3d, draw_box_3d_v2, \
    project_to_image, Calibration
import config.kitti_config as cnf


class KittiDataset(Dataset):
    def __init__(self, dataset_dir, input_size=(384, 1280), hm_size=(96, 320), mode='train', aug_transforms=None,
                 hflip_prob=0., num_samples=None):
        self.dataset_dir = dataset_dir
        self.input_size = input_size
        self.hm_size = hm_size
        self.down_ratio = 4
        self.hflip_prob = hflip_prob

        self.mean_rgb = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        self.std_rgb = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
        self.mean_dim = np.array([1.53, 1.62, 3.89], np.float32)
        self.std_dim = np.array([0.13, 0.1, 0.41], np.float32)

        assert mode in ['train', 'val', 'test'], 'Invalid mode: {}'.format(mode)
        self.mode = mode
        self.is_test = (self.mode == 'test')
        sub_folder = 'testing' if self.is_test else 'training'

        self.aug_transforms = aug_transforms

        self.image_dir = os.path.join(self.dataset_dir, sub_folder, "image_2")
        self.calib_dir = os.path.join(self.dataset_dir, sub_folder, "calib")
        self.label_dir = os.path.join(self.dataset_dir, sub_folder, "label_2")
        split_txt_path = os.path.join(self.dataset_dir, 'ImageSets', '{}.txt'.format(mode))
        self.sample_id_list = [int(x.strip()) for x in open(split_txt_path).readlines()]

        if num_samples is not None:
            self.sample_id_list = self.sample_id_list[:num_samples]
        self.num_samples = len(self.sample_id_list)

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        if self.is_test:
            return self.load_img_only(index)
        else:
            return self.load_img_with_targets(index)

    def load_img_only(self, index):
        """Load only image for the testing phase"""

        sample_id = int(self.sample_id_list[index])
        img_path = os.path.join(self.image_dir, '{:06d}.png'.format(sample_id))
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = self.normalize_img(img)

        return img_path, img.transpose(2, 0, 1)

    def load_img_with_targets(self, index):
        """Load images and targets for the training and validation phase"""

        hflipped = False
        sample_id = int(self.sample_id_list[index])
        img_path = os.path.join(self.image_dir, '{:06d}.png'.format(sample_id))
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img, pad_size = self.padding_img(img, input_size=self.input_size)
        if np.random.random() < self.hflip_prob:
            hflipped = True
            img = img[:, ::-1, :]
        img = self.normalize_img(img)

        calib = self.get_calib(sample_id)
        labels = self.get_label(sample_id)
        targets = self.build_targets(labels, calib, pad_size, hflipped)

        return img_path, img.transpose(2, 0, 1), targets

    def padding_img(self, img, input_size):
        h, w, c = img.shape
        ret_img = np.zeros((input_size[0], input_size[1], c))
        pad_y = (input_size[0] - h) // 2
        pad_x = (input_size[1] - w) // 2
        ret_img[pad_y: pad_y + h, pad_x: pad_x + w] = img
        pad_size = np.array([pad_x, pad_y])

        return ret_img, pad_size

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '{:06d}.png'.format(idx))
        # assert os.path.isfile(img_file)
        return cv2.imread(img_file)  # (H, W, C) -> (H, W, 3) OpenCV reads in BGR mode

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '{:06d}.txt'.format(idx))
        # assert os.path.isfile(calib_file)
        return Calibration(calib_file)

    def get_label(self, idx):
        labels = []
        label_path = os.path.join(self.label_dir, '{:06d}.txt'.format(idx))
        for line in open(label_path, 'r'):
            line = line.rstrip()
            line_parts = line.split(' ')
            obj_name = line_parts[0]  # 'Car', 'Pedestrian', ...
            cat_id = cnf.CLASS_NAME_TO_ID[obj_name]
            if cat_id <= -99:  # ignore Tram and Misc
                continue
            truncated = int(float(line_parts[1]))  # truncated pixel ratio [0..1]
            occluded = int(line_parts[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
            alpha = float(line_parts[3])  # object observation angle [-pi..pi]
            # xmin, ymin, xmax, ymax
            bbox = np.array([float(line_parts[4]), float(line_parts[5]), float(line_parts[6]), float(line_parts[7])])
            # height, width, length (h, w, l)
            dim = np.array([float(line_parts[8]), float(line_parts[9]), float(line_parts[10])])
            # location (x,y,z) in camera coord.
            location = np.array([float(line_parts[11]), float(line_parts[12]), float(line_parts[13])])
            ry = float(line_parts[14])  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

            object_label = [cat_id, bbox, dim, location, ry, alpha]
            labels.append(object_label)

        return labels

    def normalize_img(self, img):
        return (img / 255. - self.mean_rgb) / self.std_rgb

    def normalize_dim(self, dim):
        return (dim - self.mean_dim) / self.std_dim

    def build_targets(self, labels, calib, pad_size, hflipped):
        num_classes = 3
        num_vertexes = 8
        max_objects = 50
        num_objects = min(len(labels), max_objects)
        hm_h, hm_w = self.hm_size
        input_h, input_w = self.input_size

        # Settings for computing dynamic sigma
        max_sigma = 19
        min_sigma = 3
        Amax = 129682
        Amin = 13

        hm_main_center = np.zeros((num_classes, hm_h, hm_w), dtype=np.float32)
        hm_ver = np.zeros((num_vertexes, hm_h, hm_w), dtype=np.float32)

        cen_offset = np.zeros((max_objects, 2), dtype=np.float32)
        indices_center = np.zeros((max_objects), dtype=np.int64)
        obj_mask = np.zeros((max_objects), dtype=np.uint8)

        ver_coor = np.zeros((max_objects, num_vertexes * 2), dtype=np.float32)
        ver_coor_mask = np.zeros((max_objects, num_vertexes * 2), dtype=np.uint8)
        ver_offset = np.zeros((max_objects * num_vertexes, 2), dtype=np.float32)
        ver_offset_mask = np.zeros((max_objects * num_vertexes), dtype=np.uint8)
        indices_vertexes = np.zeros((max_objects * num_vertexes), dtype=np.int64)

        dimension = np.zeros((max_objects, 3), dtype=np.float32)

        rotbin = np.zeros((max_objects, 2), dtype=np.int64)
        rotres = np.zeros((max_objects, 2), dtype=np.float32)

        depth = np.zeros((max_objects, 1), dtype=np.float32)

        for k in range(num_objects):
            cls_id, bbox, dim, location, ry, alpha = labels[k]
            bbox[[0, 2]] = bbox[[0, 2]] + pad_size[0]  # pad_x
            bbox[[1, 3]] = bbox[[1, 3]] + pad_size[1]  # pad_x

            if hflipped:
                bbox[[0, 2]] = input_w - bbox[[0, 2]] - 1
                # Need to consider the ry, alpha values

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                bbox_area = w * h
                # Compute sigma value based on bbox size
                sigma = bbox_area * (max_sigma - min_sigma) / (Amax - Amin)

                bbox = bbox / self.down_ratio  # on the heatmap
                center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                center_int = center.astype(np.int32)
                if cls_id < 0:
                    ignore_id = [_ for _ in range(num_classes)] if cls_id == - 1 else [- cls_id - 2]
                    # Consider to make mask ignore
                    continue

                # Generate heatmaps for main center
                gen_2d_gaussian_hm(hm_main_center[cls_id], center, sigma)  # dynamic sigma
                # Index of the center
                indices_center[k] = center_int[1] * hm_w + center_int[0]

                # Generate heatmaps for 8 vertexes
                vertexes_3d = compute_box_3d(dim, location, ry)
                vertexes_2d = project_to_image(vertexes_3d, calib.P)
                vertexes_2d += pad_size.reshape(-1, 2)  # pad_x, pad_y

                if hflipped:
                    vertexes_2d[:, 0] = input_w - vertexes_2d[:, 0] - 1
                    # Need to swap vertexes' index

                vertexes_2d = vertexes_2d / self.down_ratio  # on the heatmap
                for ver_idx, ver in enumerate(vertexes_2d):
                    ver_int = ver.astype(np.int32)
                    if (0 <= ver_int[0] < hm_w) and (0 <= ver_int[1] < hm_h):
                        gen_2d_gaussian_hm(hm_ver[ver_idx], ver, sigma)
                        # targets for vertexes coordinates
                        ver_coor[k, ver_idx * 2: (ver_idx + 1) * 2] = ver - center
                        ver_coor_mask[k, ver_idx * 2: (ver_idx + 1) * 2] = 1
                        # targets for vertexes offset
                        ver_offset[k * num_vertexes + ver_idx] = ver - ver_int
                        ver_offset_mask[k * num_vertexes + ver_idx] = 1
                        # Indices of vertexes
                        indices_vertexes[k * num_vertexes + ver_idx] = ver_int[1] * hm_w + ver_int[0]

                # targets for center offset
                cen_offset[k] = center - center_int

                # targets for dimension
                # Normalize dimension
                norm_dim = self.normalize_dim(dim)
                # TODO: What happend if the norm_dim < 0, we can't apply the log operator
                # dimension[k] = np.log(norm_dim)  # take the log of the normalized dimension
                dimension[k] = norm_dim

                # targets for orientation
                if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
                    rotbin[k, 0] = 1
                    rotres[k, 0] = alpha - (-0.5 * np.pi)
                if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
                    rotbin[k, 1] = 1
                    rotres[k, 1] = alpha - (0.5 * np.pi)

                # targets for depth
                depth[k] = location[2]

                # targets for 2d bbox

                # Generate masks
                obj_mask[k] = 1

        targets = {
            'hm_mc': hm_main_center,
            'hm_ver': hm_ver,
            'ver_coor': ver_coor,
            'cen_offset': cen_offset,
            'ver_offset': ver_offset,
            'dimension': dimension,
            'rotbin': rotbin,
            'rotres': rotres,
            'depth': depth,
            'indices_center': indices_center,
            'indices_vertexes': indices_vertexes,
            'obj_mask': obj_mask,
            'ver_coor_mask': ver_coor_mask,
            'ver_offset_mask': ver_offset_mask
        }

        return targets

    def draw_img_with_label(self, index):
        sample_id = int(self.sample_id_list[index])
        img_path = os.path.join(self.image_dir, '{:06d}.png'.format(sample_id))
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img, pad_size = self.padding_img(img, input_size=self.input_size)

        calib = self.get_calib(sample_id)
        labels = self.get_label(sample_id)
        for label in labels:
            cat_id, bbox, dim, location, ry, alpha = label
            if cat_id < 0:
                continue

            bbox[[0, 2]] = bbox[[0, 2]] + pad_size[0]  # pad_x
            bbox[[1, 3]] = bbox[[1, 3]] + pad_size[1]  # pad_x

            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            center_int = center.astype(np.int32)
            cv2.circle(img, (center_int[0], center_int[1]), 5, (255, 0, 0), -1)  # draw the center box

            vertexes_3d = compute_box_3d(dim, location, ry)
            vertexes_2d = project_to_image(vertexes_3d, calib.P)
            vertexes_2d += pad_size.reshape(-1, 2)  # pad_x, pad_y
            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            img = draw_box_3d(img, vertexes_2d, color=cnf.colors[cat_id])
            # img = draw_box_3d_v2(img, conners_2d, color=cnf.colors[cat_id])

        return img.astype(np.uint8)


if __name__ == '__main__':
    from easydict import EasyDict as edict

    configs = edict()
    configs.distributed = False  # For testing
    configs.pin_memory = False
    configs.num_samples = None
    configs.dataset_dir = os.path.join('../../', 'dataset', 'kitti')
    dataset = KittiDataset(configs.dataset_dir, mode='val', aug_transforms=None, num_samples=configs.num_samples)

    print('\n\nPress n to see the next sample >>> Press Esc to quit...')
    for idx in range(len(dataset)):
        img = dataset.draw_img_with_label(idx)
        cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(0) & 0xff == 27:
            break
