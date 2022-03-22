import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.utils import gaussian_radius, draw_umich_gaussian
import os
import cv2
from dataset.transform import Transform
opj = os.path.join

def set_features(num_rect,
            noise_key,
            small_lines_key,
            shift_key,
            big_lines_key,
            ct_int,
            radius,
            features):
    draw_umich_gaussian(features[0], ct_int, radius, k=num_rect)
    draw_umich_gaussian(features[1], ct_int, radius, k=noise_key)
    draw_umich_gaussian(features[2], ct_int, radius, k=small_lines_key)
    draw_umich_gaussian(features[3], ct_int, radius, k=shift_key)
    draw_umich_gaussian(features[4], ct_int, radius, k=big_lines_key)
    return features

def get_empty_features(h, w):
    return np.zeros((5, h, w))

class BCDataset(Dataset):
    def __init__(self, data_packs=None, resize_size=(512, 512),
                 mean=(0.40789654, 0.44719302, 0.47026115), std=(0.28863828, 0.27408164, 0.27809835)):
        self.data_packs = data_packs
        self.transform = Transform('pascal_voc')
        self.resize_size = resize_size
        self.down_stride = 4
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.data_packs)

    def __getitem__(self, idx):
        img = np.array(self.data_packs[idx].image)
        #cv2.imshow("test", img)
        #cv2.waitKey(0)
        raw_h, raw_w, _ = img.shape
        info = {'raw_height': raw_h, 'raw_width': raw_w}

        real_point = np.array(self.data_packs[idx].gt_data['quad'])
        point = np.array(self.data_packs[idx].quadrangle)

        point_flatten = point.flatten()
        real_point_flatten = real_point.flatten()

        boxes = np.array([point_flatten])
        real_boxes = np.array([real_point_flatten])

        label = [[0]]

        img, boxes, real_boxes = self.preprocess_img_boxes(img, self.resize_size, boxes, real_boxes)
        info['resize_height'], info['resize_width'] = img.shape[:2]
        classes = label

        img = transforms.ToTensor()(img)
        boxes = torch.from_numpy(boxes).float()
        classes = torch.LongTensor(classes)

        boxes_w, boxes_h = real_boxes[:, 2] - real_boxes[:, 0], real_boxes[:, 5] - real_boxes[:, 3]
        ct = np.array([(real_boxes[:, 0] + real_boxes[:, 2] + real_boxes[:, 4] + real_boxes[:, 6]) / 4,
                       (real_boxes[:, 1] + real_boxes[:, 3] + real_boxes[:, 5] + real_boxes[:, 7]) / 4], dtype=np.float32).T

        output_h, output_w = info['resize_height'] // self.down_stride, info['resize_width'] // self.down_stride
        boxes_h, boxes_w, ct = boxes_h / self.down_stride, boxes_w / self.down_stride, ct / self.down_stride
        hm = np.zeros((1, output_h, output_w), dtype=np.float32)

        info['gt_hm_height'], info['gt_hm_witdh'] = output_h, output_w
        obj_mask = torch.ones(len(classes))

        for i, cls_id in enumerate(classes):
            radius = gaussian_radius((np.ceil(boxes_h[i]), np.ceil(boxes_w[i])))
            radius = max(0, int(radius))
            ct_int = ct[i].astype(np.int32)
            draw_umich_gaussian(hm[cls_id], ct_int, radius)

        hm = torch.from_numpy(hm)
        obj_mask = obj_mask.eq(1)
        boxes = boxes[obj_mask]
        classes = classes[obj_mask]
        info['ct'] = torch.tensor(ct)[obj_mask]
        return img, boxes, classes, hm, info

    def preprocess_img_boxes(self, image, input_ksize, boxes=None, real_boxes=None):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side = input_ksize
        h, w, _ = image.shape
        _pad = 32  # 32

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w = _pad - nw % _pad
        pad_h = _pad - nh % _pad

        image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2, 4, 6]] = boxes[:, [0, 2, 4, 6]] * scale
            boxes[:, [1, 3, 5, 7]] = boxes[:, [1, 3, 5, 7]] * scale

            real_boxes[:, [0, 2, 4, 6]] = real_boxes[:, [0, 2, 4, 6]] * scale
            real_boxes[:, [1, 3, 5, 7]] = real_boxes[:, [1, 3, 5, 7]] * scale
            return image_paded, boxes, real_boxes

    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list, hm_list, infos = zip(*data)
        assert len(imgs_list) == len(boxes_list) == len(classes_list)
        batch_size = len(boxes_list)
        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []
        pad_hm_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()

        for i in range(batch_size):
            img = imgs_list[i]
            hm = hm_list[i]

            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(
                torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))

            pad_hm_list.append(
                torch.nn.functional.pad(hm, (0, int(max_w // 4 - hm.shape[2]), 0, int(max_h // 4 - hm.shape[1])),
                                        value=0.))

        max_num = 0
        for i in range(batch_size):
            n = boxes_list[i].shape[0]
            if n > max_num:
                max_num = n
        for i in range(batch_size):
            pad_boxes_list.append(
                torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
            pad_classes_list.append(
                torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)
        batch_hms = torch.stack(pad_hm_list)
        return batch_imgs, batch_boxes, batch_classes, batch_hms, infos