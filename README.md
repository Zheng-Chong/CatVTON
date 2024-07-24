

# 🐈 CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models

<div style="display: flex; justify-content: center; align-items: center;">
  <a href="http://arxiv.org/abs/2407.15886" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2407.15886-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href='https://huggingface.co/zhengchong/CatVTON' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Hugging Face-ckpts-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface'>
  </a>
  <a href="https://github.com/Zheng-Chong/CatVTON" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>
  <a href="http://120.76.142.206:8888" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
  <a href='https://zheng-chong.github.io/CatVTON/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
  <a href="https://github.com/Zheng-Chong/CatVTON/LICENCE" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/License-CC BY--NC--SA--4.0-lightgreen?style=flat&logo=Lisence' alt='License'>
  </a>
</div>

<div align="center">
  <img src="resource/img/teaser.jpg" width="100%" height="100%"/>
</div>

<!-- This repository is the official implementation of ***CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffsuion Models***. -->

**CatVTON** is a simple and efficient virtual try-on diffusion model with ***1) Lightweight Network (899.06M parameters totally)***, ***2) Parameter-Efficient Training (49.57M parameters trainable)*** and ***3) Simplified Inference (< 8G VRAM for 1024X768 resolution)***.


## Updates
- **`2024/7/24`**: Our [**Paper on ArXiv**](http://arxiv.org/abs/2407.15886) is available now 🥳!
- **`2024/7/22`**: Our [**App Code**](https://github.com/Zheng-Chong/CatVTON/blob/main/app.py) is released, deploy and enjoy CatVTON on your own mechine 🎉!
- **`2024/7/21`**: Our [**Inference Code**](https://github.com/Zheng-Chong/CatVTON/blob/main/inference.py) and [**Weights** 🤗](https://huggingface.co/zhengchong/CatVTON) are released.
- **`2024/7/11`**: Our [**Online Demo**](http://120.76.142.206:8888) is released 😁.

## Installation
An [Installation Guide](https://github.com/Zheng-Chong/CatVTON/INSTALL.md) is provided to help build the conda environment for CatVTON. When deploying the app, you will need Detectron2 & DensePose, but these are not required for inference on datasets. Install the packages according to your needs.

## Deployment (Gradio App)

To deploy the Gradio App for CatVTON on your own mechine, just run the following command, and checkpoints will be automaticly download from HuggingFace.

```PowerShell
CUDA_VISIBLE_DEVICES=0 python app.py \
--output_dir="resource/demo/output" \
--mixed_precision="bf16" \
--allow_tf32 
```
When using `bf16` precision, generating results with a resolution of `1024x768` only requires about `8G` VRAM.

## Inference
### Data Preparation
Before inference, you need to download the [VITON-HD](https://github.com/shadow2496/VITON-HD) or [DressCode](https://github.com/aimagelab/dress-code) dataset.
Once the datasets are downloaded, the folder structures should look like these:
```
├── VITON-HD
|   ├── test_pairs_unpaired.txt
│   ├── test
|   |   ├── image
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── agnostic-mask
│   │   │   ├── [000006_00_mask.png | 000008_00.png | ...]
...
```
For DressCode dataset, we provide [our preprocessed agnostic masks](https://drive.google.com/drive/folders/1uT88nYQl0n5qHz6zngb9WxGlX4ArAbVX?usp=share_link), download and place in `agnostic_masks` folders under each category.
```
├── DressCode
|   ├── test_pairs_paired.txt
|   ├── test_pairs_unpaired.txt
│   ├── [dresses | lower_body | upper_body]
|   |   ├── test_pairs_paired.txt
|   |   ├── test_pairs_unpaired.txt
│   │   ├── images
│   │   │   ├── [013563_0.jpg | 013563_1.jpg | 013564_0.jpg | 013564_1.jpg | ...]
│   │   ├── agnostic_masks
│   │   │   ├── [013563_0.png| 013564_0.png | ...]
...
```

### Inference on VTIONHD/DressCode
To run the inference on the DressCode or VITON-HD dataset, run the following command, checkpoints will be automaticly download from HuggingFace.

```PowerShell
CUDA_VISIBLE_DEVICES=0 python inference.py \
--dataset [dresscode | vitonhd] \
--data_root_path <path> \
--output_dir <path> 
--dataloader_num_workers 8 \
--batch_size 8 \
--seed 555 \
--mixed_precision [no | fp16 | bf16] \
--allow_tf32 \
--repaint \
--eval_pair  
```


## Acknowledgement
Our code is modified based on [Diffusers](https://github.com/huggingface/diffusers). We adopt [Stable Diffusion v1.5 inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) as base model. We use [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing/tree/master) and [DensePose](https://github.com/facebookresearch/DensePose) to automatically generate masks in our [Gradio](https://github.com/gradio-app/gradio) App.  
Thanks to all the contributors!

## License
All the materials, including code, checkpoints, and demo, are made available under the [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. You are free to copy, redistribute, remix, transform, and build upon the project for non-commercial purposes, as long as you give appropriate credit and distribute your contributions under the same license.


## Citation

```
@misc{chong2024catvtonconcatenationneedvirtual,
      title={CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models}, 
      author={Zheng Chong and Xiao Dong and Haoxiang Li and Shiyue Zhang and Wenqing Zhang and Xujie Zhang and Hanqing Zhao and Xiaodan Liang},
      year={2024},
      eprint={2407.15886},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.15886}, 
}
```

