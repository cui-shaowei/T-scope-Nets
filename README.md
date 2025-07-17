#  Force Prediction & Video Inpainting Toolkit
## 📁 Project Overview
This repository contains two independent yet complementary pipelines:

1. **force_predict/** – Predict 3-D force vectors from binary images.  
2. **ProPainter/** – Perform video-level object removal / inpainting using masks derived from the force-prediction stage.
## System Requirements
| Component   | Exact Specification                                          |
| ----------- | ------------------------------------------------------------ |
| **OS**      | Ubuntu 20.04.6 LTS 64-bit (GNOME 3.36.8, X11)                |
| **CPU**     | Intel® Core i9-14900KF (32 threads)                          |
| **GPU**     | NVIDIA GPU (CUDA 10.1 **driver/runtime** visible via `nvcc`) |
| **RAM**     | 62.6 GB                                                      |
| **Storage** | 2.0 TB SSD                                                   |
| **CUDA**    | 10.1 (V10.1.243)                                             |
| **Python**  | 3.8.20 (Anaconda)                                            |
## Tested Package Stack
| Environment                      | Key Versions                                                                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **ProPainter** (ProPainter)  | `torch==2.4.1+cu121`, `torchvision==0.19.1+cu121`, `opencv==4.10.0.84`, `numpy==1.24.4`                                         |
| **Force-Predict** (force_predict) | `torch==2.7.1+cu121`, `torchvision==0.22.1+cu121`, `pandas==1.5.3`, `scikit-learn==1.3.0`, `opencv==4.10.0.84`, `numpy==1.24.3` |



---
## 🗂️ Directory Layout

nr/
├── force_predict/
│   ├── main.py
│   ├── model.pth
│   └── utils/
│       ├── init.py
│       ├── CustomDataset.py
│       └── model.py
├── ProPainter/
│   ├── inference_propainter.py


---

## 🚀 Quick Start

### 1. Environment-force_predict
ubuntu：
cd force_predict
conda env create -f environment.yml

### 2. Environment-ProPainter
ubuntu：
cd ProPainter
conda create -n ProPainter python=3.8
pip install -r requirements.txt
### 3. Force Prediction
python main.py    # Outputs per-image predictions & ground-truth
### 4. Video Inpainting
python inference_propainter.py --video inputs/prodata --mask inputs/mask  # Results appear in ProPainter/results/.
##📊 Data Format
| File                           | Description                               |
| ------------------------------ | ----------------------------------------- |
| `data/data.csv`                | `image_path,x,y,z`                        |
| `data/binarydata/*.jpg`        | Binary images with 9 black dots           |
| `data/prodata/*.jpg`           | Identical images used as ProPainter input |
| `ProPainter/inputs/mask/*.jpg` | Masks generated from binary images        |
## Typical Install Time
| Step                   | Time        |
| ---------------------- | ----------- |
| Environment creation   | 34 s        |
| Force_predict environment | 160 s        |
| ProPainter requirements | 52 s        |
| **Total**              | **≈ 4 min** |

## Demo Run Time
| Task                                  | 400*400 100-frame clip |
| ------------------------------------- | -------------------- |
| Force prediction                      | 2.5 s                  |
| Inpainting **GPU **           | 6 s               |



