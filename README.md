# RPi-ColorBlock-Detection
这是一个针对你的项目需求优化的 README.md。它明确了 Python 版推理、CMake 自动环境构建以及树莓派环境的特殊说明。

RPi-ColorBlock-Detection
本项目是一个专为 Raspberry Pi 5 设计的颜色方块识别系统。包含基于 YOLOv11 的模型训练流程（PC端）以及基于 ncnn 框架的 Python 高性能推理部署（树莓派端）。

## 🚀 项目特性
* **端到端流程**：从 Roboflow 数据集训练到树莓派实机部署。
* **自动化构建**：`color_cube_train` 使用 CMake 自动管理 Python 虚拟环境及依赖安装。
* **高性能推理**：`inference` 部分采用 ncnn 框架，在树莓派 5 上实现流畅的实时检测。

## 📁 项目结构
```
RPi-ColorBlock-Detection/
├── color_cube_train/          # 模型训练模块 (PC端执行)
│   ├── src/                   # 训练核心代码
│   ├── source_data/           # 原始数据集 (70% Train, 20% Val, 10% Test)
│   ├── CMakeLists.txt         # 自动化构建脚本 (自动配置 venv 与依赖)
│   ├── data.yaml              # YOLO 数据集配置文件
│   ├── paths.py               # 路径管理工具
│   ├── requirements.txt       # 训练端 Python 依赖清单
│   ├── yolo11s.pt             # 预训练权重
│   └── runs/                  # 训练结果与日志输出
└── inference/                 # 树莓派部署模块 (Python + ncnn)             
    ├── models/                # 存放转换后的 ncnn 模型 (.param / .bin)
    ├── requirements.txt       # 推理端依赖 (不含全局库)
    └── main.py                # 推理启动程序
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


## 🤝 鸣谢
* [Ultralytics (YOLOv11)](https://github.com/ultralytics/ultralytics)
* [Tencent ncnn](https://github.com/Tencent/ncnn)
* [Roboflow Universe](https://universe.roboflow.com/)
