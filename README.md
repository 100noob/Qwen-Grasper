# 1.YOLO-Edge-Perception

This project is a high-performance color block detection system specifically designed for the **Raspberry Pi 5**. It features a complete pipeline from **YOLOv11** model training (PC side) to optimized inference deployment using the **ncnn** framework (Raspberry Pi side).

> **Note**: It is highly recommended to use **VS Code** with the **CMake Tools**, **C/C++**, and **Python** extensions installed.

## 🚀 Key Features
* **End-to-End Pipeline**: From dataset training on Roboflow to real-world deployment on Raspberry Pi.
* **Automated Build System**: The `color_cube_train` module uses CMake to automatically manage Python virtual environments (venv) and dependency installation.
* **High-Performance Inference**: The `inference` module utilizes the **ncnn** framework to achieve smooth, real-time detection on the Raspberry Pi 5.

## 📁 Project Structure
```text
RPi-ColorBlock-Detection/
├── color_cube_train/          # Model training module (Execute on PC)
│   ├── src/                   # Core training scripts
│   ├── source_data/           # Raw dataset (70% Train, 20% Val, 10% Test)
│   ├── CMakeLists.txt         # Auto-build script (Configures venv & dependencies)
│   ├── data.yaml              # YOLO dataset configuration
│   ├── paths.py               # Path management utility
│   ├── requirements.txt       # Python dependency list for training
│   ├── yolo11s.pt             # Pre-trained weights
│   └── runs/                  # Training outputs and logs
└── inference/                 # Raspberry Pi deployment module (Python + ncnn)
    ├── models/                # Converted ncnn models (.param / .bin)
    ├── requirements.txt       # Inference dependencies (excluding global libs)
    └── main.py
```

## 📊 Dataset
Dataset URL：https://universe.roboflow.com/brian114-xv3lk/color-cube-gvk4q

Classes:
0: purple-cube, 1: green-cube, 2: orange-cube, 3: pink-cube, 4: yellow-cube (Supports extension up to 7 classes)

## 🛠️ Build Instructions
1.Model Training (color-cube-train)
This module is primarily developed in a **Windows (MSYS2)** environment.
* **CMake Automation**:  Automatically creates a Python virtual environment and installs dependencies from `requirements.txt` (e.g., ultralytics).
* **Manual Prerequisite**: Please install a CUDA-enabled PyTorch in advance on Windows.

* **Build Commands**:
```
cd color_cube_train
mkdir build && cd build
cmake -G "Ninja" ..
ninja
```

## 📘 Environment Setup Guide
To ensure CMake, GCC, and Ninja function correctly, follow the instructions based on your system:
 * **Windows: Using MSYS2**:
MSYS2 provides a Linux-like package management experience (pacman), making it ideal for C++/Python mixed development.
 * 1.Download and Install MSYS2：https://www.msys2.org/
 * 2.Update Core Packages (run in MSYS2 UCRT64 terminal):
```pacman -Syu```
* 3.Install Development Toolchain:
```
pacman -S mingw-w64-ucrt-x86_64-gcc \
          mingw-w64-ucrt-x86_64-gdb \
          mingw-w64-ucrt-x86_64-cmake \
          mingw-w64-ucrt-x86_64-ninja \
          mingw-w64-ucrt-x86_64-make
```
* Or install the full toolchain::```pacman -S --needed base-devel mingw-w64-ucrt-x86_64-toolchain```
* This includes gcc, g++, make, gdb, etc., and works well with VS Code or CLion.
* 4.Configure Environment Variables
* Add the following path to your system PATH:```C:\msys64\ucrt64\bin```
* After this, you can directly use these tools in CMD or PowerShell within VS Code.

## 🐧 Linux: Using Ubuntu (Raspberry Pi / PC)
* Ubuntu setup is straightforward using apt.
* 1.Update Package Lists:```sudo apt update && sudo apt upgrade -y```
* 2.Install Development Toolchain:
```
sudo apt install -y build-essential \
                    cmake \
                    gcc \
                    gdb \
                    ninja-build \
                    git
```
* 3.Raspberry Pi Specific (OpenCV)
* To avoid virtual environment conflicts and compilation issues on Raspberry Pi, install OpenCV globally:```sudo apt install -y python3-opencv```
# 2.Interactive-Teaching-to-VLM-Dataset
```
finetune_inference/
├── 📂 click/                                # Stage 1: Annotation Tool
│   ├── 📂 build/                            # Compiled files
│   ├── 📂 destset/                          # Output directory for annotated JSONL
│   ├── 📂 main/                             # Source code for click interaction
│   ├── 📂 model/                            # YOLO model weights (.pt)
│   ├── 📂 my_dir/                           # [User-Created] Target folder for processed images
│   ├── 📂 src_images/                       # [User-Created] Raw source images for annotation
│   ├── CMakeLists.txt
│   ├── paths.py                             # Configuration for local paths
│   ├── readme.txt
│   └── requirements.txt
│
├── 📂 Detection-to-VLM_Conversation_Format_Conversion/  # Stage 2: Data Formatting
│   ├── 📂 JSONL/                            # Converted conversational JSONL output
│   ├── 📂 main/                             # Normalization & formatting scripts
│   ├── 📂 src_images/                       # [User-Created] Symbolic link or copy of images
│   ├── 📂 venv/                             # Virtual environment
│   ├── CMakeLists.txt
│   ├── paths.py
│   └── requirements.txt
│
└── 📂 lora_finetune_unsloth/                 # Stage 3: Training
    ├── 📂 export_weight/                    # Saved LoRA adapters/weights
    ├── 📂 imgs_and_json/
    │   └── 📂 src_imgs/                     # [User-Created] Final dataset images
    ├── 📂 main/                             # Unsloth training scripts
    ├── 📂 venv/                             # Virtual environment
    ├── paths.py
    └── pip.txt                              # Dependency list
```

##  Acknowledgements
* [Ultralytics (YOLOv11)](https://github.com/ultralytics/ultralytics)
* [Tencent ncnn](https://github.com/Tencent/ncnn)
* [Roboflow Universe](https://universe.roboflow.com/)
* https://github.com/opencv/opencv
