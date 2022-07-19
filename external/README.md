# Panoptic-DeepLab Code for training on NYUv2

## Installation
```bash
conda create -n multitask-d2 python=3.8 anaconda
conda activate multitask-d2

# conda dependencies
conda install pytorch=1.10.1 torchvision=0.11.2 cudatoolkit=11.3 -c pytorch

# pip dependencies
pip install 'opencv-python>=4.2.0.34'
pip install git+https://github.com/cocodataset/panopticapi.git
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# own package
pip install ../lib/nicr-scene-analysis-datasets

# Not sure if it's still required
pip install setuptools==59.5.0
```

## Training
For training the approach run:
```bash
python train_net.py --config-file configs/NYUv2/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml
```
