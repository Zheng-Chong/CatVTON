from model.SCHP import networks
from model.SCHP.utils.transforms import get_affine_transform, transform_logits

from collections import OrderedDict
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

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

class SCHP:
    def __init__(self, ckpt_path, device):
        dataset_type = None
        if 'lip' in ckpt_path:
            dataset_type = 'lip'
        elif 'atr' in ckpt_path:
            dataset_type = 'atr'
        elif 'pascal' in ckpt_path:
            dataset_type = 'pascal'
        assert dataset_type is not None, 'Dataset type not found in checkpoint path'
        self.device = device
        self.num_classes = dataset_settings[dataset_type]['num_classes']
        self.input_size = dataset_settings[dataset_type]['input_size']
        self.aspect_ratio = self.input_size[1] * 1.0 / self.input_size[0]
        self.palette = get_palette(self.num_classes)

        self.label = dataset_settings[dataset_type]['label']
        self.model = networks.init_model('resnet101', num_classes=self.num_classes, pretrained=None).to(device)
        self.load_ckpt(ckpt_path)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])
        self.upsample = torch.nn.Upsample(size=self.input_size, mode='bilinear', align_corners=True)


    def load_ckpt(self, ckpt_path):
        rename_map = {
            "decoder.conv3.2.weight": "decoder.conv3.3.weight",
            "decoder.conv3.3.weight": "decoder.conv3.4.weight",
            "decoder.conv3.3.bias": "decoder.conv3.4.bias",
            "decoder.conv3.3.running_mean": "decoder.conv3.4.running_mean",
            "decoder.conv3.3.running_var": "decoder.conv3.4.running_var",
            "fushion.3.weight": "fushion.4.weight",
            "fushion.3.bias": "fushion.4.bias",
        }
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        new_state_dict_ = OrderedDict()
        for k, v in list(new_state_dict.items()):
            if k in rename_map:
                new_state_dict_[rename_map[k]] = v
            else:
                new_state_dict_[k] = v
        self.model.load_state_dict(new_state_dict_, strict=False)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def preprocess(self, image):
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
        elif isinstance(image, Image.Image):
            # to cv2 format
            img = np.array(image)
    
        h, w, _ = img.shape
        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size)
        input = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        input = self.transform(input).to(self.device).unsqueeze(0)
        meta = {
                'center': person_center,
                'height': h,
                'width': w,
                'scale': s,
                'rotation': r
        }
        return input, meta


    def __call__(self, image_or_path):
        if isinstance(image_or_path, list):
            image_list = []
            meta_list = []
            for image in image_or_path:
                image, meta = self.preprocess(image)
                image_list.append(image)
                meta_list.append(meta)
            image = torch.cat(image_list, dim=0)
        else:
            image, meta = self.preprocess(image_or_path)
            meta_list = [meta]
                
        output = self.model(image)
        # upsample_outputs = self.upsample(output[0][-1])
        upsample_outputs = self.upsample(output)
        upsample_outputs = upsample_outputs.permute(0, 2, 3, 1)  # BCHW -> BHWC

        output_img_list = []
        for upsample_output, meta in zip(upsample_outputs, meta_list):
            c, s, w, h = meta['center'], meta['scale'], meta['width'], meta['height']
            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=self.input_size)
            parsing_result = np.argmax(logits_result, axis=2)
            output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            output_img.putpalette(self.palette)
            output_img_list.append(output_img)

        return output_img_list[0] if len(output_img_list) == 1 else output_img_list