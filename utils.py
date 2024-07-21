import os

import math
import PIL
import numpy as np
import torch
from PIL import Image
from accelerate.state import AcceleratorState
from packaging import version
import accelerate
from typing import List, Optional, Tuple
from torch.nn import functional as F
from diffusers import UNet2DConditionModel, SchedulerMixin

# Compute DREAM and update latents for diffusion sampling
def compute_dream_and_update_latents_for_inpaint(
    unet: UNet2DConditionModel,
    noise_scheduler: SchedulerMixin,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    target: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    dream_detail_preservation: float = 1.0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Implements "DREAM (Diffusion Rectification and Estimation-Adaptive Models)" from http://arxiv.org/abs/2312.00210.
    DREAM helps align training with sampling to help training be more efficient and accurate at the cost of an extra
    forward step without gradients.

    Args:
        `unet`: The state unet to use to make a prediction.
        `noise_scheduler`: The noise scheduler used to add noise for the given timestep.
        `timesteps`: The timesteps for the noise_scheduler to user.
        `noise`: A tensor of noise in the shape of noisy_latents.
        `noisy_latents`: Previously noise latents from the training loop.
        `target`: The ground-truth tensor to predict after eps is removed.
        `encoder_hidden_states`: Text embeddings from the text model.
        `dream_detail_preservation`: A float value that indicates detail preservation level.
          See reference.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: Adjusted noisy_latents and target.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)[timesteps, None, None, None]
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # The paper uses lambda = sqrt(1 - alpha) ** p, with p = 1 in their experiments.
    dream_lambda = sqrt_one_minus_alphas_cumprod**dream_detail_preservation

    pred = None  # b, 4, h, w
    with torch.no_grad():
        pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    noisy_latents_no_condition = noisy_latents[:, :4]
    _noisy_latents, _target = (None, None)
    if noise_scheduler.config.prediction_type == "epsilon":
        predicted_noise = pred
        delta_noise = (noise - predicted_noise).detach()
        delta_noise.mul_(dream_lambda)
        _noisy_latents = noisy_latents_no_condition.add(sqrt_one_minus_alphas_cumprod * delta_noise)
        _target = target.add(delta_noise)
    elif noise_scheduler.config.prediction_type == "v_prediction":
        raise NotImplementedError("DREAM has not been implemented for v-prediction")
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    
    _noisy_latents = torch.cat([_noisy_latents, noisy_latents[:, 4:]], dim=1)
    return _noisy_latents, _target

# Prepare the input for inpainting model.
def prepare_inpainting_input(
    noisy_latents: torch.Tensor, 
    mask_latents: torch.Tensor,
    condition_latents: torch.Tensor,
    enable_condition_noise: bool = True,
    condition_concat_dim: int = -1,
) -> torch.Tensor:
    """
    Prepare the input for inpainting model.
    
    Args:
        noisy_latents (torch.Tensor): Noisy latents.
        mask_latents (torch.Tensor): Mask latents.
        condition_latents (torch.Tensor): Condition latents.
        enable_condition_noise (bool): Enable condition noise.
    
    Returns:
        torch.Tensor: Inpainting input.
    """
    if not enable_condition_noise:
        condition_latents_ = condition_latents.chunk(2, dim=condition_concat_dim)[-1]
        noisy_latents = torch.cat([noisy_latents, condition_latents_], dim=condition_concat_dim)
    noisy_latents = torch.cat([noisy_latents, mask_latents, condition_latents], dim=1)
    return noisy_latents

# Compute VAE encodings
def compute_vae_encodings(image: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    """
    Args:
        images (torch.Tensor): image to be encoded
        vae (torch.nn.Module): vae model

    Returns:
        torch.Tensor: latent encoding of the image
    """
    pixel_values = image.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return model_input


# Init Accelerator
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration

def init_accelerator(config):
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.project_name,
        logging_dir=os.path.join(config.project_name, "logs"),
    )
    accelerator_ddp_config = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[accelerator_ddp_config],
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
        
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.project_name,
            config={
                "learning_rate": config.learning_rate,
                "train_batch_size": config.train_batch_size,
                "image_size": f"{config.width}x{config.height}",
            },
        )
        
    return accelerator


def init_weight_dtype(wight_dtype):
    return {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[wight_dtype]


def init_add_item_id(config):
    return torch.tensor(
        [
            config.height,
            config.width * 2,
            0,
            0,
            config.height,
            config.width * 2,
        ]
    ).repeat(config.train_batch_size, 1)


def prepare_eval_data(dataset_root, dataset_name, is_pair=True):
    assert dataset_name in ["vitonhd", "dresscode", "farfetch"], "Unknown dataset name {}.".format(dataset_name)
    if dataset_name == "vitonhd":
        data_root = os.path.join(dataset_root, "VITONHD-1024", "test")
        if is_pair:
            keys = os.listdir(os.path.join(data_root, "Images"))
            cloth_image_paths = [
                os.path.join(data_root, "Images", key, key + "-0.jpg") for key in keys
            ]
            person_image_paths = [
                os.path.join(data_root, "Images", key, key + "-1.jpg") for key in keys
            ]
        else:
            # read ../test_pairs.txt
            cloth_image_paths = []
            person_image_paths = []
            with open(
                os.path.join(dataset_root, "VITONHD-1024", "test_pairs.txt"), "r"
            ) as f:
                lines = f.readlines()
                for line in lines:
                    cloth_image, person_image = (
                        line.replace(".jpg", "").strip().split(" ")
                    )
                    cloth_image_paths.append(
                        os.path.join(
                            data_root, "Images", cloth_image, cloth_image + "-0.jpg"
                        )
                    )
                    person_image_paths.append(
                        os.path.join(
                            data_root, "Images", person_image, person_image + "-1.jpg"
                        )
                    )
    elif dataset_name == "dresscode":
        data_root = os.path.join(dataset_root, "DressCode-1024")
        if is_pair:
            part = ["lower", "lower", "upper", "upper", "dresses", "dresses"]
            ids = ["013581", "051685", "000190", "050072", "020829", "053742"]
            cloth_image_paths = [
                os.path.join(data_root, "Images", part[i], ids[i], ids[i] + "_1.jpg")
                for i in range(len(part))
            ]
            person_image_paths = [
                os.path.join(data_root, "Images", part[i], ids[i], ids[i] + "_0.jpg")
                for i in range(len(part))
            ]
        else:
            raise ValueError("DressCode dataset does not support non-pair evaluation.")
    elif dataset_name == "farfetch":
        data_root = os.path.join(dataset_root, "FARFETCH-1024")
        cloth_image_paths = [
            # TryOn
            "/home/chongzheng/Projects/hivton/Datasets/FARFETCH-1024/Images/women/Tops/Blouses/13732751/13732751-2.jpg",
            "/home/chongzheng/Projects/hivton/Datasets/FARFETCH-1024/Images/women/Tops/Hoodies/14661627/14661627-4.jpg",
            "/home/chongzheng/Projects/hivton/Datasets/FARFETCH-1024/Images/women/Tops/Vests & Tank Tops/16532697/16532697-4.jpg",
            "Images/men/Pants/Loose Fit Pants/14750720/14750720-6.jpg",
            # Garment Transfer
            "/home/chongzheng/Projects/hivton/Datasets/FARFETCH-1024/Images/women/Tops/Shirts/10889688/10889688-3.jpg",
            "/home/chongzheng/Projects/hivton/Datasets/FARFETCH-1024/Images/women/Shorts/Leather & Faux Leather Shorts/20143338/20143338-1.jpg",
            "/home/chongzheng/Projects/hivton/Datasets/FARFETCH-1024/Images/women/Jackets/Blazers/15541224/15541224-2.jpg",
            "/home/chongzheng/Projects/hivton/Datasets/FARFETCH-1024/Images/men/Polo Shirts/Polo Shirts/17652415/17652415-0.jpg"
            
            # "Images/men/Jackets/Hooded Jackets/12550261/12550261-1.jpg",
            # "Images/men/Shirts/Shirts/15614589/15614589-4.jpg",
            # "Images/women/Dresses/Day Dresses/10372515/10372515-3.jpg",
            # "Images/women/Dresses/Sundresses/18520992/18520992-4.jpg",
            # "Images/women/Skirts/Asymmetric & Draped Skirts/12404908/12404908-2.jpg",
        ]
        person_image_paths = [
            # TryOn
            "/home/chongzheng/Projects/hivton/Datasets/FARFETCH-1024/Images/women/Tops/Blouses/13732751/13732751-0.jpg",
            "/home/chongzheng/Projects/hivton/Datasets/FARFETCH-1024/Images/women/Tops/Hoodies/14661627/14661627-2.jpg",
            "/home/chongzheng/Projects/hivton/Datasets/FARFETCH-1024/Images/women/Tops/Vests & Tank Tops/16532697/16532697-1.jpg",
            "Images/men/Pants/Loose Fit Pants/14750720/14750720-5.jpg",
            # Garment Transfer
            "/home/chongzheng/Projects/hivton/Datasets/FARFETCH-1024/Images/women/Tops/Shirts/10889688/10889688-1.jpg",
            "/home/chongzheng/Projects/hivton/Datasets/FARFETCH-1024/Images/women/Shorts/Leather & Faux Leather Shorts/20143338/20143338-2.jpg",
            "/home/chongzheng/Projects/hivton/Datasets/FARFETCH-1024/Images/women/Jackets/Blazers/15541224/15541224-0.jpg",
            "/home/chongzheng/Projects/hivton/Datasets/FARFETCH-1024/Images/men/Polo Shirts/Polo Shirts/17652415/17652415-4.jpg",
            
            # "Images/men/Jackets/Hooded Jackets/12550261/12550261-3.jpg",
            # "Images/men/Shirts/Shirts/15614589/15614589-3.jpg",
            # "Images/women/Dresses/Day Dresses/10372515/10372515-0.jpg",
            # "Images/women/Dresses/Sundresses/18520992/18520992-1.jpg",
            # "Images/women/Skirts/Asymmetric & Draped Skirts/12404908/12404908-1.jpg",
        ]
        cloth_image_paths = [
            os.path.join(data_root, path) for path in cloth_image_paths
        ]
        person_image_paths = [
            os.path.join(data_root, path) for path in person_image_paths
        ]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    samples = [
        {
            "folder": os.path.basename(os.path.dirname(cloth_image)),
            "cloth": cloth_image,
            "person": person_image,
        }
        for cloth_image, person_image in zip(
            cloth_image_paths, person_image_paths
        )
    ]
    return samples


def repaint_result(result, person_image, mask_image):
    result, person, mask = np.array(result), np.array(person_image), np.array(mask_image)
    # expand the mask to 3 channels & to 0~1
    mask = np.expand_dims(mask, axis=2)
    mask = mask / 255.0
    # mask for result, ~mask for person
    result_ = result * mask + person * (1 - mask)
    return Image.fromarray(result_.astype(np.uint8))
    
    
# 多通道 Sobel 算子处理 (用于获取模特图像的损失注意力图)
def sobel(batch_image, mask=None, scale=4.0):
    """
    计算输入批量图像的Sobel梯度.

    batch_image: 输入的批量图像张量，大小为 [batch, channels, height, width]
    """
    w, h = batch_image.size(3), batch_image.size(2)
    pool_kernel = (max(w, h) // 16) * 2 + 1
    # 定义Sobel核
    kernel_x = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        .view(1, 1, 3, 3)
        .to(batch_image.device)
        .repeat(1, batch_image.size(1), 1, 1)
    )
    kernel_y = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        .view(1, 1, 3, 3)
        .to(batch_image.device)
        .repeat(1, batch_image.size(1), 1, 1)
    )
    # 初始化梯度张量
    grad_x = torch.zeros_like(batch_image)
    grad_y = torch.zeros_like(batch_image)
    # 边缘填充
    batch_image = F.pad(batch_image, (1, 1, 1, 1), mode="reflect")
    # 应用Sobel算子
    grad_x = F.conv2d(batch_image, kernel_x, padding=0)
    grad_y = F.conv2d(batch_image, kernel_y, padding=0)
    # 计算梯度的幅度
    grad_magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
    # Mask 处理
    if mask is not None:
        grad_magnitude = grad_magnitude * mask
    # 剃度裁剪
    grad_magnitude = torch.clamp(grad_magnitude, 0.2, 2.5)
    # 平均池化
    grad_magnitude = F.avg_pool2d(
        grad_magnitude, kernel_size=pool_kernel, stride=1, padding=pool_kernel // 2
    )
    # 归一化
    grad_magnitude = (grad_magnitude / grad_magnitude.max()) * scale
    return grad_magnitude


# Sobel 加权平方误差, 增强边缘区域的损失（直接用于损失计算）
def sobel_aug_squared_error(x, y, reference, mask=None, reduction="mean"):
    """
    计算x,y的逐元素平方误差，其中x和y是图像张量.
    然后利用 x 的 sobel 结果作为权重，计算加权平方误差.
    x: Tensor, shape [batch, channels, height, width]
    y: Tensor, shape [batch, channels, height, width]
    """
    ref_sobel = sobel(reference, mask=mask)  # 计算 sobel 梯度作为损失权重
    if ref_sobel.isnan().any():
        print("Error: NaN Sobel Gradient")
        loss = F.mse_loss(x, y, reduction="mean")  # 如果梯度为nan，则直接退化为MSE损失
    else:
        squared_error = (x - y).pow(2)
        weighted_squared_error = squared_error * ref_sobel
        if reduction == "mean":
            loss = weighted_squared_error.mean()
        elif reduction == "sum":
            loss = weighted_squared_error.sum()
        elif reduction == "none":
            loss = weighted_squared_error
    # print("WSE Loss:", loss.mean(), loss.dtype)
    return loss


# 准备图像（转换为 Batch 张量）
def prepare_image(image):
    if isinstance(image, torch.Tensor):
        # Batch single image
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(dtype=torch.float32)
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image


def prepare_mask_image(mask_image):
    if isinstance(mask_image, torch.Tensor):
        if mask_image.ndim == 2:
            # Batch and add channel dim for single mask
            mask_image = mask_image.unsqueeze(0).unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] == 1:
            # Single mask, the 0'th dimension is considered to be
            # the existing batch size of 1
            mask_image = mask_image.unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] != 1:
            # Batch of mask, the 0'th dimension is considered to be
            # the batching dimension
            mask_image = mask_image.unsqueeze(1)

        # Binarize mask
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
    else:
        # preprocess mask
        if isinstance(mask_image, (PIL.Image.Image, np.ndarray)):
            mask_image = [mask_image]

        if isinstance(mask_image, list) and isinstance(mask_image[0], PIL.Image.Image):
            mask_image = np.concatenate(
                [np.array(m.convert("L"))[None, None, :] for m in mask_image], axis=0
            )
            mask_image = mask_image.astype(np.float32) / 255.0
        elif isinstance(mask_image, list) and isinstance(mask_image[0], np.ndarray):
            mask_image = np.concatenate([m[None, None, :] for m in mask_image], axis=0)

        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
        mask_image = torch.from_numpy(mask_image)

    return mask_image


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_eval_image_pairs(root, mode="logo"):
    # TODO 加载测试图像对，包括配对和非配对的图像对
    test_name = "test"
    person_image_paths = [
        os.path.join(root, test_name, "image", _)
        for _ in os.listdir(os.path.join(root, test_name, "image"))
        if _.endswith(".jpg")
    ]
    cloth_image_paths = [
        person_image_path.replace("image", "cloth")
        for person_image_path in person_image_paths
    ]
    # 包含图案和文字的部分图像
    if mode == "logo":
        filter_pairs = [
            6648,
            6744,
            6967,
            6985,
            14031,
            12358,
            4963,
            4680,
            499,
            396,
            345,
            6648,
            6744,
            6967,
            6985,
            7510,
            8205,
            8254,
            10545,
            11485,
            11632,
            12354,
            13144,
            14112,
            12570,
            11766,
        ]
        filter_pairs.sort()
        filter_pairs = [f"{_:05d}_00.jpg" for _ in filter_pairs]
        cloth_image_paths = [
            cloth_image_paths[i]
            for i in range(len(cloth_image_paths))
            if os.path.basename(cloth_image_paths[i]) in filter_pairs
        ]
        person_image_paths = [
            person_image_paths[i]
            for i in range(len(person_image_paths))
            if os.path.basename(person_image_paths[i]) in filter_pairs
        ]
    return cloth_image_paths, person_image_paths


def tensor_to_image(tensor: torch.Tensor):
    """
    Converts a torch tensor to PIL Image.
    """
    assert tensor.dim() == 3, "Input tensor should be 3-dimensional."
    assert tensor.dtype == torch.float32, "Input tensor should be float32."
    assert (
        tensor.min() >= 0 and tensor.max() <= 1
    ), "Input tensor should be in range [0, 1]."
    tensor = tensor.cpu()
    tensor = tensor * 255
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.numpy().astype(np.uint8)
    image = Image.fromarray(tensor)
    return image


def concat_images(images: List[Image.Image], divider: int = 4, cols: int = 4):
    """
    Concatenates images horizontally and with
    """
    widths = [image.size[0] for image in images]
    heights = [image.size[1] for image in images]
    total_width = cols * max(widths)
    total_width += divider * (cols - 1)
    # `col` images each row
    rows = math.ceil(len(images) / cols)
    total_height = max(heights) * rows
    # add divider between rows
    total_height += divider * (len(heights) // cols - 1)

    # all black image
    concat_image = Image.new("RGB", (total_width, total_height), (0, 0, 0))

    x_offset = 0
    y_offset = 0
    for i, image in enumerate(images):
        concat_image.paste(image, (x_offset, y_offset))
        x_offset += image.size[0] + divider
        if (i + 1) % cols == 0:
            x_offset = 0
            y_offset += image.size[1] + divider

    return concat_image


def read_prompt_file(prompt_file: str):
    if prompt_file is not None and os.path.isfile(prompt_file):
        with open(prompt_file, "r") as sample_prompt_file:
            sample_prompts = sample_prompt_file.readlines()
            sample_prompts = [sample_prompt.strip() for sample_prompt in sample_prompts]
    else:
        sample_prompts = []
    return sample_prompts


def save_tensors_to_npz(tensors: torch.Tensor, paths: List[str]):
    assert len(tensors) == len(paths), "Length of tensors and paths should be the same!"
    for tensor, path in zip(tensors, paths):
        np.savez_compressed(path, latent=tensor.cpu().numpy())


def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = (
        AcceleratorState().deepspeed_plugin
        if accelerate.state.is_initialized()
        else None
    )
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


def is_xformers_available():
    try:
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            print(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                "please update xFormers to at least 0.0.17. "
                "See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        return True
    except ImportError:
        raise ValueError(
            "xformers is not available. Make sure it is installed correctly"
        )



def resize_and_crop(image, size):
    # Crop to size ratio
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_w = w
        new_h = w * target_h // target_w
    else:
        new_h = h
        new_w = h * target_w // target_h
    image = image.crop(
        ((w - new_w) // 2, (h - new_h) // 2, (w + new_w) // 2, (h + new_h) // 2)
    )
    # resize
    image = image.resize(size, Image.LANCZOS)
    return image


def resize_and_padding(image, size):
    # Padding to size ratio
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_h = target_h
        new_w = w * target_h // h
    else:
        new_w = target_w
        new_h = h * target_w // w
    image = image.resize((new_w, new_h), Image.LANCZOS)
    # padding
    padding = Image.new("RGB", size, (255, 255, 255))
    padding.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return padding



if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image, ImageFilter
    import numpy as np

    def vis_sobel_weight(image_path, mask_path) -> PIL.Image.Image:

        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        l_w, l_h = w // 8, h // 8
        image = image.resize((l_w, l_h))
        mask = Image.open(mask_path).convert("L").resize((l_w, l_h))
        image_pt = transforms.ToTensor()(image).unsqueeze(0).to("cuda")
        mask_pt = transforms.ToTensor()(mask).unsqueeze(0).to("cuda")
        sobel_pt = sobel(image_pt, mask_pt, scale=1.0)
        sobel_image = sobel_pt.squeeze().cpu().numpy()
        sobel_image = Image.fromarray((sobel_image * 255).astype(np.uint8))
        sobel_image = sobel_image.resize((w, h), resample=Image.NEAREST)
        # 图像平滑
        sobel_image = sobel_image.filter(ImageFilter.SMOOTH)
        from data.utils import grayscale_to_heatmap

        sobel_image = grayscale_to_heatmap(sobel_image)
        image = Image.open(image_path).convert("RGB").resize((w, h))
        sobel_image = Image.blend(image, sobel_image, alpha=0.5)
        return sobel_image

    save_folder = "./sobel_vis-2.0"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    from data.utils import scan_files_in_dir

    for i in scan_files_in_dir(
        "/home/chongzheng/Projects/try-on-project/Datasets/VITONHD-1024/test/Images"
    ):
        image_path = i.path

        if i.path.endswith("-1.jpg"):
            result_path = os.path.join(save_folder, os.path.basename(image_path))

            mask_path = image_path.replace("Images", "AgnosticMask").replace(
                "-1.jpg", "_mask-1.png"
            )
            vis_sobel_weight(image_path, mask_path).save(result_path)
    pass
