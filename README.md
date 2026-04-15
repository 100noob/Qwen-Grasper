# RPi-ColorBlock-Detection

This project is a high-performance color block recognition system specifically designed for the **Raspberry Pi 5**. It features a complete pipeline from **YOLOv11** model training (PC side) to optimized inference deployment using the **ncnn** framework (Raspberry Pi side).

> **Note**: It is highly recommended to use **VS Code** with the **CMake Tools**, **C/C++**, and **Python** extensions installed.

## 🚀 Key Features
* **End-to-End Pipeline**: From dataset training on Roboflow to real-world deployment on Raspberry Pi.
* **Automated Construction**: The `color_cube_train` module uses CMake to automatically manage Python virtual environments (venv) and dependency installation.
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

## 📊 数据集
数据集地址：https://universe.roboflow.com/brian114-xv3lk/color-cube-gvk4q

识别类别：
0: purple-cube, 1: green-cube, 2: orange-cube, 3: pink-cube, 4: yellow-cube (支持扩展至 7 类)

## 🛠️ 项目构建说明
1.模型训练 (color-cube-train)
该模块主要在 Windows (MSYS2) 环境下开发。
* **CMake 自动化**: 通过 CMake 自动创建 Python 虚拟环境，并安装 requirements.txt 中的依赖（如 ultralytics）。
* **手动预装**: 请先在 Windows 环境下安装好支持 CUDA 的 PyTorch。
* **构建指令**:
```
cd color_cube_train
mkdir build && cd build
cmake -G "Ninja" ..
ninja
```

## 📘 环境搭建工具链教程
为了确保 CMake、GCC 和 Ninja 能够正常工作，请根据你的系统参考以下步骤：
 * **Windows 篇：使用 MSYS2 搭建**:
MSYS2 提供了类似 Linux 的软件包管理体验（pacman），非常适合 C++/Python 混合开发。
 * 1.下载并安装 MSYS2：从 msys2.org 下载安装包。
 * 2.更新核心库（在 MSYS2 UCRT64 终端执行）：
```pacman -Syu```
* 3.安装开发工具链：执行以下命令一次性安装 CMake, GCC, GDB, Ninja 以及基础编译组：
```
pacman -S mingw-w64-ucrt-x86_64-gcc \
          mingw-w64-ucrt-x86_64-gdb \
          mingw-w64-ucrt-x86_64-cmake \
          mingw-w64-ucrt-x86_64-ninja \
          mingw-w64-ucrt-x86_64-make
```
* 或者:```pacman -S --needed base-devel mingw-w64-ucrt-x86_64-toolchain```
* 包含了 gcc、g++、make、gdb 等完整编译调试工具，适合配合 VS Code 或 CLion 使用
* 4.配置环境变量将 C:\msys64\ucrt64\bin 添加到 Windows 的 系统环境变量 PATH 中，这样你就可以在 VS Code 的 CMD 或 PowerShell 里直接使用这些命令了。

## Linux 篇：使用 Ubuntu (Raspberry Pi/PC) 搭建
* Ubuntu的安装非常直观，使用 apt 即可
* 1.更新软件源:```sudo apt update && sudo apt upgrade -y```
* 2.安装开发工具链
* 执行以下命令安装 CMake, GCC, GDB, Ninja：
```
sudo apt install -y build-essential \
                    cmake \
                    gcc \
                    gdb \
                    ninja-build \
                    git
```
* 3.针对树莓派的特殊处理 (OpenCV)
* 在树莓派上，为了避免虚拟环境冲突和编译失败，请务必执行```sudo apt install -y python3-opencv```(把opencv直接全局安装)


##  鸣谢
* [Ultralytics (YOLOv11)](https://github.com/ultralytics/ultralytics)
* [Tencent ncnn](https://github.com/Tencent/ncnn)
* [Roboflow Universe](https://universe.roboflow.com/)
