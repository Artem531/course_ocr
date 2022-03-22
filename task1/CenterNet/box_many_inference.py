from tqdm import tqdm
import numpy as np
import torch
from model.centernet import CenterNet
from pathlib import Path
from data import MidvPackage
from torchvision import transforms


def preprocess_img(img, input_ksize):
    min_side, max_side = input_ksize
    h, w = img.height, img.width
    _pad = 32
    smallest_side = min(w, h)
    largest_side = max(w, h)
    scale = min_side / smallest_side
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    nw, nh = int(scale * w), int(scale * h)
    img_resized = np.array(img.resize((nw, nh)))

    pad_w = _pad - nw % _pad
    pad_h = _pad - nh % _pad

    img_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
    img_paded[:nh, :nw, :] = img_resized

    return img_paded, {'raw_height': h, 'raw_width': w}

ckp = torch.load('/home/artem/PycharmProjects/course_ocr/course_ocr/task1/CenterNet/ckp/checkpoint.pth')
cfg = ckp['config']

model = CenterNet(cfg).cuda()
model.load_state_dict(ckp['model'])
model = model.eval()

DATASET_PATH = Path("/home/artem/PycharmProjects/course_ocr/course_ocr/task1/midv500_compressed")
data_packs = MidvPackage.read_midv500_dataset(DATASET_PATH)
results_dict = dict()

for dp in tqdm(data_packs):
    for i in range(len(dp)):
        if dp[i].is_test_split():

            img = dp[i].image
            img_paded, info = preprocess_img(img, cfg.resize_size)

            input = transforms.ToTensor()(img_paded)
            input = transforms.Normalize(std=cfg.std, mean=cfg.mean)(input)
            inputs = input.unsqueeze(0).cuda()

            detects = model.inference(inputs, topK=1, return_hm=False, th=0)
            box = detects[0][0][0]

            pred_points = [[box[0].item(), box[1].item()], [box[2].item(), box[3].item()],
                           [box[4].item(), box[5].item()], [box[6].item(), box[7].item()]]

            results_dict[dp[i].unique_key] = pred_points

from course_ocr.task1.course_ocr_t1.metrics import dump_results_dict, measure_crop_accuracy

path = Path("/home/artem/PycharmProjects/course_ocr/course_ocr/task1")
dump_results_dict(results_dict, path / 'pred.json')

acc = measure_crop_accuracy(
    path / 'pred.json',
    path / 'gt.json'
)

print(acc)