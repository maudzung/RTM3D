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
import math

import numpy as np
from torch.utils.data import Dataset
import cv2

sys.path.append('../')

from data_process.kitti_data_utils import gen_hm_dynamic_sigma, gen_hm_radius, compute_radius, compute_box_3d, \
    draw_box_3d, draw_box_3d_v2, project_to_image, Calibration
import config.kitti_config as cnf


class KittiDataset(Dataset):
    def __init__(self, configs, mode='train', aug_transforms=None, hflip_prob=0., use_left_cam_prob=1.,
                 num_samples=None):
        self.dataset_dir = configs.dataset_dir
        self.input_size = configs.input_size
        self.hm_size = configs.hm_size
        self.down_ratio = configs.down_ratio

        self.num_classes = configs.num_classes
        self.num_vertexes = configs.num_vertexes
        self.max_objects = configs.max_objects

        self.hflip_prob = hflip_prob
        self.hflip_indices = [[0, 1], [2, 3], [4, 5], [6, 7]]
        self.dynamic_sigma = configs.dynamic_sigma
        self.use_left_cam_prob = use_left_cam_prob

        self.mean_rgb = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        self.std_rgb = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        assert mode in ['train', 'val', 'test'], 'Invalid mode: {}'.format(mode)
        self.mode = mode
        self.is_test = (self.mode == 'test')
        sub_folder = 'testing' if self.is_test else 'training'

        self.aug_transforms = aug_transforms

        self.image_dir_left = os.path.join(self.dataset_dir, sub_folder, "image_2")
        self.image_dir_right = os.path.join(self.dataset_dir, sub_folder, "image_3")
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

        use_left_cam = False
        if np.random.random() < self.use_left_cam_prob:
            use_left_cam = True
        sample_id = int(self.sample_id_list[index])
        img_path, img = self.get_image(sample_id, use_left_cam)
        img, pad_size = self.padding_img(img, input_size=self.input_size)

        hflipped = False
        if np.random.random() < self.hflip_prob:
            hflipped = True
            img = img[:, ::-1, :]
        img = self.normalize_img(img)

        metadata = {
            'use_left_cam': use_left_cam,
            'hflipped': hflipped
        }

        return img_path, img.transpose(2, 0, 1), metadata

    def load_img_with_targets(self, index):
        """Load images and targets for the training and validation phase"""

        use_left_cam = False
        if np.random.random() < self.use_left_cam_prob:
            use_left_cam = True
        sample_id = int(self.sample_id_list[index])
        img_path, img = self.get_image(sample_id, use_left_cam)

        # Apply the augmentation for the raw image
        if self.aug_transforms:
            img = self.aug_transforms(image=img)['image']

        img, pad_size = self.padding_img(img, input_size=self.input_size)

        hflipped = False
        if np.random.random() < self.hflip_prob:
            hflipped = True
            img = img[:, ::-1, :]
        img = self.normalize_img(img)

        calib = self.get_calib(sample_id)
        labels = self.get_label(sample_id)
        targets = self.build_targets(labels, calib, pad_size, hflipped, use_left_cam)
        metadata = {
            'use_left_cam': use_left_cam,
            'hflipped': hflipped
        }

        return img_path, img.transpose(2, 0, 1), targets, metadata

    def padding_img(self, img, input_size):
        h, w, c = img.shape
        ret_img = np.zeros((input_size[0], input_size[1], c))
        pad_y = (input_size[0] - h) // 2
        pad_x = (input_size[1] - w) // 2
        ret_img[pad_y: pad_y + h, pad_x: pad_x + w] = img
        pad_size = np.array([pad_x, pad_y])

        return ret_img, pad_size

    def get_image(self, idx, use_left_cam):
        if use_left_cam:
            img_path = os.path.join(self.image_dir_left, '{:06d}.png'.format(idx))
        else:
            img_path = os.path.join(self.image_dir_right, '{:06d}.png'.format(idx))

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        return img_path, img

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

    def build_targets(self, labels, calib, pad_size, hflipped, use_left_cam):

        num_objects = min(len(labels), self.max_objects)
        hm_h, hm_w = self.hm_size

        hm_main_center = np.zeros((self.num_classes, hm_h, hm_w), dtype=np.float32)
        hm_ver = np.zeros((self.num_vertexes, hm_h, hm_w), dtype=np.float32)

        cen_offset = np.zeros((self.max_objects, 2), dtype=np.float32)
        indices_center = np.zeros((self.max_objects), dtype=np.int64)
        obj_mask = np.zeros((self.max_objects), dtype=np.uint8)

        ver_coor = np.zeros((self.max_objects, self.num_vertexes * 2), dtype=np.float32)
        ver_coor_mask = np.zeros((self.max_objects, self.num_vertexes * 2), dtype=np.uint8)
        ver_offset = np.zeros((self.max_objects * self.num_vertexes, 2), dtype=np.float32)
        ver_offset_mask = np.zeros((self.max_objects * self.num_vertexes), dtype=np.uint8)
        indices_vertexes = np.zeros((self.max_objects * self.num_vertexes), dtype=np.int64)

        dimension = np.zeros((self.max_objects, 3), dtype=np.float32)

        rotbin = np.zeros((self.max_objects, 2), dtype=np.int64)
        rotres = np.zeros((self.max_objects, 2), dtype=np.float32)

        depth = np.zeros((self.max_objects, 1), dtype=np.float32)
        whs = np.zeros((self.max_objects, 2), dtype=np.float32)

        for k in range(num_objects):
            cls_id, bbox, dim, location, ry, alpha = labels[k]
            bbox[[0, 2]] = bbox[[0, 2]] + pad_size[0]  # pad_x
            bbox[[1, 3]] = bbox[[1, 3]] + pad_size[1]  # pad_x

            bbox = bbox / self.down_ratio  # on the heatmap
            bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if bbox_h > 0 and bbox_w > 0:
                sigma = 1.  # Just dummy
                radius = 1  # Just dummy
                if self.dynamic_sigma:
                    # Settings for computing dynamic sigma
                    max_sigma = 19
                    min_sigma = 3
                    Amax = 129682 / self.down_ratio
                    Amin = 13 / self.down_ratio
                    # Compute sigma value based on bbox size
                    bbox_area = bbox_w * bbox_h
                    sigma = bbox_area * (max_sigma - min_sigma) / (Amax - Amin)
                else:
                    radius = compute_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
                    radius = max(0, int(radius))

                # Generate heatmaps for 8 vertexes
                vertexes_3d = compute_box_3d(dim, location, ry)
                if use_left_cam:
                    vertexes_2d = project_to_image(vertexes_3d, calib.P2)
                    vertexes_2d += pad_size.reshape(-1, 2)  # pad_x, pad_y
                else:
                    center_3d = np.mean(vertexes_3d, axis=0, keepdims=True)
                    vertexes_2d = project_to_image(vertexes_3d, calib.P3)
                    vertexes_2d += pad_size.reshape(-1, 2)  # pad_x, pad_y
                    # TODO: Carefully check the translation's center of 2D box
                    translate_center = project_to_image(center_3d, calib.P3) - project_to_image(center_3d, calib.P2)
                    translate_center = translate_center.squeeze()
                    bbox[[0, 2]] += translate_center[0] / self.down_ratio
                    bbox[[1, 3]] += translate_center[1] / self.down_ratio

                vertexes_2d = vertexes_2d / self.down_ratio  # on the heatmap
                if hflipped:
                    # Don't need to consider the ry, alpha values
                    bbox[[0, 2]] = hm_w - bbox[[0, 2]] - 1
                    vertexes_2d[:, 0] = hm_w - vertexes_2d[:, 0] - 1
                    # Need to swap vertexes' index
                    for e in self.hflip_indices:
                        vertexes_2d[e[0]], vertexes_2d[e[1]] = vertexes_2d[e[1]].copy(), vertexes_2d[e[0]].copy()

                center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                center_int = center.astype(np.int32)
                if cls_id < 0:
                    ignore_ids = [_ for _ in range(self.num_classes)] if cls_id == - 1 else [- cls_id - 2]
                    # Consider to make mask ignore
                    for cls_ig in ignore_ids:
                        if self.dynamic_sigma:
                            gen_hm_dynamic_sigma(hm_main_center[cls_ig], center, sigma)
                        else:
                            gen_hm_radius(hm_main_center[cls_ig], center_int, radius)
                    hm_main_center[ignore_ids, center_int[1], center_int[0]] = 0.9999
                    continue

                # Generate heatmaps for main center
                if self.dynamic_sigma:
                    gen_hm_dynamic_sigma(hm_main_center[cls_id], center, sigma)  # dynamic sigma
                else:
                    gen_hm_radius(hm_main_center[cls_id], center, radius)
                # Index of the center
                indices_center[k] = center_int[1] * hm_w + center_int[0]

                for ver_idx, ver in enumerate(vertexes_2d):
                    ver_int = ver.astype(np.int32)
                    if (0 <= ver_int[0] < hm_w) and (0 <= ver_int[1] < hm_h):
                        if self.dynamic_sigma:
                            gen_hm_dynamic_sigma(hm_ver[ver_idx], ver, sigma)
                        else:
                            gen_hm_radius(hm_ver[ver_idx], ver, radius)
                        # targets for vertexes coordinates
                        ver_coor[k, ver_idx * 2: (ver_idx + 1) * 2] = ver - center  # Don't take the absolute values
                        ver_coor_mask[k, ver_idx * 2: (ver_idx + 1) * 2] = 1
                        # targets for vertexes offset
                        ver_offset[k * self.num_vertexes + ver_idx] = ver - ver_int
                        ver_offset_mask[k * self.num_vertexes + ver_idx] = 1
                        # Indices of vertexes
                        indices_vertexes[k * self.num_vertexes + ver_idx] = ver_int[1] * hm_w + ver_int[0]

                # targets for center offset
                cen_offset[k] = center - center_int

                # targets for dimension
                dimension[k] = dim

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
                whs[k, 0] = bbox_w
                whs[k, 1] = bbox_h

                # Generate masks
                obj_mask[k] = 1

        targets = {
            'hm_mc': hm_main_center,
            'hm_ver': hm_ver,
            'ver_coor': ver_coor,
            'cen_offset': cen_offset,
            'ver_offset': ver_offset,
            'dim': dimension,
            'rotbin': rotbin,
            'rotres': rotres,
            'depth': depth,
            'indices_center': indices_center,
            'indices_vertexes': indices_vertexes,
            'obj_mask': obj_mask,
            'ver_coor_mask': ver_coor_mask,
            'ver_offset_mask': ver_offset_mask,
            'wh': whs
        }

        return targets

    def draw_img_with_label(self, index):
        use_left_cam = False
        if np.random.random() < self.use_left_cam_prob:
            use_left_cam = True

        input_h, input_w = self.input_size
        sample_id = int(self.sample_id_list[index])
        img_path, img = self.get_image(sample_id, use_left_cam)
        # Apply the augmentation for the raw image
        if self.aug_transforms:
            img = self.aug_transforms(image=img)['image']

        img, pad_size = self.padding_img(img, input_size=self.input_size)
        hflipped = False
        if np.random.random() < self.hflip_prob:
            hflipped = True
            img = (img[:, ::-1, :]).astype(np.uint8)

        calib = self.get_calib(sample_id)
        labels = self.get_label(sample_id)
        for label_idx, label in enumerate(labels):
            cat_id, bbox, dim, location, ry, alpha = label
            if cat_id < 0:
                continue

            bbox[[0, 2]] = bbox[[0, 2]] + pad_size[0]  # pad_x
            bbox[[1, 3]] = bbox[[1, 3]] + pad_size[1]  # pad_x

            if hflipped:
                bbox[[0, 2]] = input_w - bbox[[0, 2]] - 1

            vertexes_3d = compute_box_3d(dim, location, ry)
            if use_left_cam:
                vertexes_2d = project_to_image(vertexes_3d, calib.P2)
                vertexes_2d += pad_size.reshape(-1, 2)  # pad_x, pad_y
            else:
                center_3d = np.mean(vertexes_3d, axis=0, keepdims=True)
                vertexes_2d = project_to_image(vertexes_3d, calib.P3)
                vertexes_2d += pad_size.reshape(-1, 2)  # pad_x, pad_y
                translate_center = project_to_image(center_3d, calib.P3) - project_to_image(center_3d, calib.P2)
                translate_center = translate_center.squeeze()
                bbox[[0, 2]] += translate_center[0]
                bbox[[1, 3]] += translate_center[1]
                # TODO: Carefully check the translation's center

            if hflipped:
                vertexes_2d[:, 0] = input_w - vertexes_2d[:, 0] - 1
                for e in self.hflip_indices:
                    vertexes_2d[e[0]], vertexes_2d[e[1]] = vertexes_2d[e[1]].copy(), vertexes_2d[e[0]].copy()

            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            center_int = center.astype(np.int32)
            cv2.circle(img, (center_int[0], center_int[1]), 5, (0, 255, 0), -1)  # draw the center box

            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            img = draw_box_3d(img, vertexes_2d, color=cnf.colors[cat_id])
            # Put text that indicate the object index
            img = cv2.putText(img, 'obj {}'.format(label_idx), tuple(center_int), cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (0, 255, 0), 2, cv2.LINE_AA)
            # for cn_idx, cn in enumerate(vertexes_2d):
            #     img = cv2.putText(img, '{}'.format(cn_idx), tuple(cn), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #                       (0, 255, 0), 2, cv2.LINE_AA)
            # img = draw_box_3d_v2(img, conners_2d, color=cnf.colors[cat_id])

        return img.astype(np.uint8)


if __name__ == '__main__':
    from easydict import EasyDict as edict
    import albumentations as album

    configs = edict()
    configs.distributed = False  # For testing
    configs.pin_memory = False
    configs.num_samples = None
    configs.input_size = (384, 1280)
    configs.hm_size = (96, 320)
    configs.down_ratio = 4
    configs.max_objects = 50
    configs.num_classes = 3
    configs.num_vertexes = 8
    configs.dynamic_sigma = False

    configs.dataset_dir = os.path.join('../../', 'dataset', 'kitti')

    aug_transforms = album.Compose([
        album.RandomBrightnessContrast(p=0.5),
        album.GaussNoise(p=0.5)
    ], p=1.)

    dataset = KittiDataset(configs, mode='val', aug_transforms=aug_transforms, num_samples=configs.num_samples,
                           hflip_prob=0., use_left_cam_prob=1.)

    print('\n\nPress n to see the next sample >>> Press Esc to quit...')
    for idx in range(len(dataset)):
        img = dataset.draw_img_with_label(idx)
        cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(0) & 0xff == 27:
            break
