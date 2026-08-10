# SFDG: Dynamic Graph Learning with Content-guided Spatial-Frequency Relation Reasoning for Deepfake Detection

## 🏗️ Framework

<p align="center">
  <img src="../assets/sfdg_framework.png" alt="SFDG Framework" width="800"/>
</p>

## 🔧 Installation

### 1. Create environment

```bash
conda create -n sfdg python=3.10
conda activate sfdg
```

### 2. Install dependencies

```bash
pip install torch torchvision
pip install dlib kornia albumentations scikit-image opencv-python pillow matplotlib numpy
```

## 📦 Pretrained Model

```text
SFDG_Project/
|-- checkpoints/
|   |-- Freq2_LBP_Graph_23/
|       |-- ckpt_29.pth
|-- runs/
    |-- Freq2_LBP_Graph_23/
        |-- config.pkl
```

## 🚀 Quick Start

```bash
cd SFDG_Project
python SFDG_test_serve.py
```

### Python API

```python
from SFDG_test_serve import SFDG_process

code, probability, label, output_path = SFDG_process("path/to/image.jpg")
```

| Return | Description |
|--------|-------------|
| `code` | 0 for success, 1 for error |
| `probability` | Fake probability (0 = real, 1 = fake) |
| `label` | `"real"` or `"fake"` |
| `output_path` | Path to saved face crop `../result/SFDG/<image_name>.jpg` |


## 📝 Citation

```bibtex
@INPROCEEDINGS{10203558,
  author={Wang, Yuan and Yu, Kun and Chen, Chen and Hu, Xiyuan and Peng, Silong},
  booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Dynamic Graph Learning with Content-guided Spatial-Frequency Relation Reasoning for Deepfake Detection}, 
  year={2023},
  pages={7278-7287},
  doi={10.1109/CVPR52729.2023.00703}
}
```
