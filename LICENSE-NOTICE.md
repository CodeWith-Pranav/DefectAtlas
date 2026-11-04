# Attribution and terms

## MVTec AD data and derived artifacts

MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger. CVPR 2019.

Source: https://www.mvtec.com/research-teaching/datasets/mvtec-ad . Dataset license: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International, https://creativecommons.org/licenses/by-nc-sa/4.0/ . Dataset images, adapted gallery images and trained normality artifacts in this repository are provided for non-commercial research/educational use with this attribution. Adaptations include RGB nearest-neighbor resizing, model training and derived statistics. This project is not endorsed by MVTec. Preserve the attribution and applicable share-alike/non-commercial conditions when redistributing adapted data/artifacts. The full raw dataset is not included; download-data.py acquires the bottle category from the documented public mirror.

## Pretrained feature extractor

The exported feature extractor uses the first four feature blocks of torchvision MobileNetV3 Small with IMAGENET1K_V1 weights. Original weight source: https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth . Torchvision: https://github.com/pytorch/vision . See THIRD-PARTY-TORCHVISION-LICENSE.txt for its BSD 3-Clause notice. These ImageNet weights were not trained by the author of this project.

## Runtime dependencies

ONNX Runtime and sharp retain their respective package licenses. Runtime dependencies are installed from the pinned package-lock.json and not copied into the repository. No commercial or industrial deployment rights beyond the applicable licenses are asserted.
