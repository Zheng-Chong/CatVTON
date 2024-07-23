# Installation

## 1. Create A Conda Environment
```shell
conda create -n catvton python==3.9.0
conda activate catvton
```

## 2. Install Requirments
```shell
cd CatVTON-main  # or your path to CatVTON project dir
pip install -r requirements.txt
```
**For inference only, the above packages are enough**. 

## 3. Detectron2 & DensePose
If you want to deploy the gradio app with automatic mask generation, you need to install **Detectron2 & DensePose** by following commands:
```shell
# Detectron2
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
# DensePose
pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
```
When installing, ensure that the runtime CUDA version for PyTorch aligns with the CUDA version of your system; otherwise, errors may occur.

If there is a misalignment, the solution is to reinstall PyTorch with a CUDA version that matches your system's version. For more information on installing PyTorch with the correct CUDA version, you can visit the [PyTorch Get Started page for previous versions](https://pytorch.org/get-started/previous-versions/). 