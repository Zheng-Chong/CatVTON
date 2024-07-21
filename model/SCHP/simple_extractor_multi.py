#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""
import threading
from queue import Queue
from tqdm import tqdm
import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset
from utils.transforms import get_affine_transform

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}

def _box2cs(box,input_size):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h,input_size)

def _xywh2cs(x, y, w, h,input_size):
    aspect_ratio = input_size[1] * 1.0 / input_size[0]
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w, h], dtype=np.float32)
    return center, scale

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    parser.add_argument("--dataset", type=str, default='atr', choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--model-restore", type=str, 
                        default='/data1/chongzheng/zhangwq/Self-Correction-Human-Parsing-master/exp-schp-201908301523-atr.pth',
                        help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default='/home/chongzheng_p23/data/Datasets/UniFashion/YOOX/YOOX-Images', help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='/home/chongzheng_p23/data/Datasets/UniFashion/YOOX/YOOX-SCHP', help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")

    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def schp_process(image_queue,model,progress_bar,input_size,transform):
    while True:
        img_path = image_queue.get()
        image_queue.task_done()

        if img_path is None: # 收到结束信号
            break

        save_path = img_path.replace("YOOX-Images","YOOX-SCHP").replace(".jpg",".png")
        if os.path.exists(save_path):
            progress_bar.update(1)
            continue

        root = os.path.dirname(img_path)
        img_name = img_path.split("/")[-1].split(".")[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            progress_bar.update(1)
            continue
        if img is not None:
            h, w, _ = img.shape
            # Get person center and scale
            person_center, s = _box2cs([0, 0, w - 1, h - 1],input_size)
            r = 0
            trans = get_affine_transform(person_center, s, r, input_size)
            input = cv2.warpAffine(
                img,
                trans,
                (int(input_size[1]), int(input_size[0])),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0))

            image = transform(input)
            meta = {
                'img_path': img_path,
                'name': img_name,
                'root': root,
                'center': person_center,
                'height': h,
                'width': w,
                'scale': s,
                'rotation': r
            }


        if not os.path.exists(save_path):
            img_name = meta['name'][0]
            c = meta['center'][0]
            # s = meta['scale'][0]
            # w = meta['width'][0]
            # h = meta['height'][0]
            root = meta['root'][0]
            save_root = root.replace("YOOX-Images","YOOX-SCHP")

            if not os.path.exists(save_root):
                os.makedirs(save_root)

            output = model(image.cuda())
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)
            parsing_result_path = save_path
            output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            output_img.putpalette(palette)
            output_img.save(parsing_result_path)
            progress_bar.update(1)


            
def main():
    args = get_arguments()

    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']
    print("Evaluating total class number {} with {}".format(num_classes, label))

    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(args.model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    # dataset = SimpleFolderDataset(root=args.input_dir, input_size=input_size, transform=transform)
    # dataloader = DataLoader(dataset)
    image_queue = Queue()
    for root, dirs, files in os.walk("/home/chongzheng_p23/data/Datasets/UniFashion/YOOX/YOOX-Images"):
        for file in files:
            if file.endswith(".jpg"):
                source_file_path = os.path.join(root, file)
                image_queue.put(source_file_path)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    palette = get_palette(num_classes)
    
    progress_bar = tqdm(total=image_queue.qsize(), desc="Processing SCHP")

    with torch.no_grad():
        devices = [1]*2
        consumer_threads = []
        for i in devices:
            device = f'cuda:{i}'
            consumer_threads.append(threading.Thread(target=schp_process, 
                                    args=(image_queue,model,progress_bar,input_size,transform)))
            consumer_threads[-1].start()

        # for idx, batch in enumerate(tqdm(dataloader)):

    return


if __name__ == '__main__':
    main()
