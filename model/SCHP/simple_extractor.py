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

import os
import threading
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset

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


def process(str):
    data_root = str
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
    dataset = SimpleFolderDataset(root=data_root, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    name = data_root.split("/")[-1]

    palette = get_palette(num_classes)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader,desc=name)):
            image, meta = batch
            
            img_path = meta['img_path'][0]
            save_path = img_path.replace("YOOX-Images","YOOX-SCHP").replace(".jpg",".png")

            if not os.path.exists(save_path):
                img_name = meta['name'][0]
                c = meta['center'].numpy()[0]
                s = meta['scale'].numpy()[0]
                w = meta['width'].numpy()[0]
                h = meta['height'].numpy()[0]
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
                if args.logits:
                    logits_result_path = os.path.join(args.output_dir, img_name[:-4] + '.npy')
                    np.save(logits_result_path, logits_result)
    return


if __name__ == '__main__':
    devices = [2]*11
    # devices = [1]*13
    consumer_threads = []
    data_list=["/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/men/Underwear",
                "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/men/T-Shirts and Tops",
                "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/men/Swimwear",
                "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/men/Sweaters and Sweatshirts",
                "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/men/Suits and Blazers",
                "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/men/Shirts",
                "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/men/Pants",
                "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/men/Jumpsuits and Overalls",
                "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/men/Jeans and Denim",
                "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/men/Coats & Jackets",
                "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/men/Activewear"]

    # data_list=[ "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/women/Underwear",
    #             "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/women/T-Shirts and Tops",
    #             "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/women/Swimwear",
    #             "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/women/Sweaters and Sweatshirts",
    #             "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/women/Suits and Blazers",
    #             "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/women/Skirts",
    #             "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/women/Shirts",
    #             "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/women/Pants",
    #             "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/women/Jumpsuits and Overalls",
    #             "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/women/Jeans and Denim",
    #             "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/women/Dresses",
    #             "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/women/Coats & Jackets",
    #             "/data1/chongzheng/Datasets/UniFashion/YOOX/YOOX-Images/women/Activewear"]

    for i, dataroot in zip(devices,data_list):
        device = f'cuda:{i}'
        consumer_threads.append(threading.Thread(target=process,args=(dataroot,)))
        consumer_threads[-1].start()
    # main()
