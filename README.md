# 1.YOLO-Edge-Perception

This project is a high-performance color block detection system specifically designed for the **Raspberry Pi 5**. It features a complete pipeline from **YOLOv11** model training (PC side) to optimized inference deployment using the **ncnn** framework (Raspberry Pi side).

> **Note**: It is highly recommended to use **VS Code** with the **Python** extensions installed.

## 🚀 Key Features
* **End-to-End Pipeline**: From dataset training on Roboflow to real-world deployment on Raspberry Pi.
* **Automated Build System**: The project uses `uv` to manage Python virtual environments and dependency installation (using `uv` can avoid pip version conflicts).
* **High-Performance Inference**: The `inference` module utilizes the **ncnn** framework to achieve smooth, real-time detection on the Raspberry Pi 5.

## 📁 Project Structure
```text
RPi-ColorBlock-Detection/
├── color_cube_train/          # Model training module (Execute on PC)
│   ├── src/                   # Core training scripts
│   ├── source_data/           # Raw dataset (70% Train, 20% Val, 10% Test)
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

## 📊 Datasets & Resources
All datasets used in this project were entirely collected and manually annotated by the author. They are completely open-source and free to use!

* **YOLO Training & Raspberry Pi 5 Deployment Dataset**:
  [Color Cube Dataset (Roboflow)](https://app.roboflow.com/brian114-xv3lk/color-cube-gvk4q/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
  *Classes: 0: purple-cube, 1: green-cube, 2: orange-cube, 3: pink-cube, 4: yellow-cube (Supports extension up to 7 classes)*

* **LLM Fine-Tuning & Testing Dataset**:
  [LLM Fine-Tuning Test Set (Roboflow)](https://app.roboflow.com/brian114-xv3lk/llm-fine-tuning-test-set-smglh/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

## 🛠️ Build Instructions
1.Model Training (color-cube-train)

* **Environment Setup**: We use `uv` to quickly create a Python virtual environment and install dependencies from `requirements.txt`. Using `uv` is highly recommended because it is extremely fast and can effectively avoid pip version conflicts.
* **Manual Prerequisite**: Please install a CUDA-enabled PyTorch in advance on Windows.

* **Build Commands**:
```bash
cd color_cube_train
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
uv pip install -r requirements.txt
```


## 🐧 Linux: Using Ubuntu (Raspberry Pi / PC)
* Ubuntu setup is straightforward using apt.
* 1.Update Package Lists:```sudo apt update && sudo apt upgrade -y```
* 2.Install Development Toolchain:
```bash
sudo apt install -y build-essential git
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
│   ├── paths.py                             # Configuration for local paths
│   ├── readme.txt
│   └── requirements.txt
│
├── 📂 Detection-to-VLM_Conversation_Format_Conversion/  # Stage 2: Data Formatting
│   ├── 📂 JSONL/                            # Converted conversational JSONL output
│   ├── 📂 main/                             # Normalization & formatting scripts
│   ├── 📂 src_images/                       # [User-Created] Symbolic link or copy of images
│   ├── 📂 venv/                             # Virtual environment
│   ├── paths.py
│   └── requirements.txt
│
├── 📂 lora_finetune_unsloth/                 # Stage 3: Training
│   ├── 📂 export_weight/                    # Saved LoRA adapters/weights
│   ├── 📂 imgs_and_json/
│   │   └── 📂 src_imgs/                     # [User-Created] Final dataset images
│   ├── 📂 main/                             # Unsloth training scripts
│   ├── 📂 venv/                             # Virtual environment
│   ├── paths.py
│   ├── pip.txt                              # Dependency list
│   └── requirements.txt                     # Added
│
├── 📂 test_lora_weight/                      # Stage 5: LoRA Weight Validation
│   ├── 📂 main/                             # Inference & mAP evaluation scripts
│   ├── 📂 src/                              # Evaluation utilities (IoU, mAP calc)
│   ├── 📂 test_set/                         # [User-Created] Testing dataset
│   │   ├── 📂 test_img/                     # [User-Created] Test images
│   │   └── 📂 test_lable/                   # [User-Created] YOLO format GT labels
│   ├── 📂 output_set/                       # Output images with bounding boxes
│   ├── 📂 pred_set/                         # Model predicted labels
│   ├── paths.py                             # Path configuration
│   └── requirements.txt                     # Dependencies
│
└── 📂 model_Finetune_test/                   # Stage 4: Testing
    ├── 📂 main/                             # Inference testing scripts
    └── requirements.txt                     # Added
```
## Project Overview
1. click
Description: A data annotation tool used for interactive object detection. It captures the absolute coordinates of bounding boxes and records the specific click sequences (order) of identified color blocks within an image. This establishes the ground truth for both location and grasping priority.

2. Detection-to-VLM_Conversation_Format_Conversion
Description: A data processing pipeline that converts the raw JSONL output from the click tool into a VLM-specific conversational format. It formats the data to include structured system prompts, user instructions, and assistant responses, specifically tailored for the Qwen (Qwen-VL) model’s fine-tuning requirements.

3. lora_finetune_unsloth
Description: The core training module utilized for fine-tuning the Qwen Large Multimodal Model. It leverages the Unsloth library to implement highly memory-efficient LoRA (Low-Rank Adaptation) training. This program enables the model to learn specific tasks—such as counting color blocks and following a specific grasping order—based on the converted dataset.
    * **`lora_finetune_unsloth/main/main.py`**: This script acts as the main entry point for the fine-tuning process. It loads the formatted conversational JSON data and raw images, configures the Unsloth FastVisionModel with LoRA parameters (for parameter-efficient training), initializes the SFTTrainer, runs the training loop, and finally exports the fine-tuned adapter weights and necessary configuration files.

4. model_Finetune_test
Description: The inference and validation module used to test the fine-tuned Qwen-VL model.
    * **`model_Finetune_test/main/main.py`**: This script sends base64-encoded local test images and instructions to a deployed inference server (via an OpenAI-compatible API). It requests structured JSON output (including object counts, color descriptions, and bounding boxes) and subsequently parses the server's response to visually draw the predicted bounding boxes directly onto the original images, saving the results locally for validation.

5. test_lora_weight
Description: Evaluates the fine-tuned LoRA weights on a designated test set, validating prediction coordinates and grasping priorities. It features mAP computation capabilities and saves visual annotated outputs.
    * **`test_lora_weight/main/main.py`**: Runs prediction over the test dataset, querying the VLM model and saving inference outputs. Afterwards, it invokes evaluation functions.
    * **Note**: The testing directory (`test_set`) and its subdirectories (`test_img` for images and `test_lable` for YOLO format ground-truth labels) need to be manually created and populated by the user before running the evaluation.

### Creating a Virtual Environment using `uv`
For running `model_Finetune_test/main/main.py`, it is recommended to use `uv` (an extremely fast Python package and project manager) to create a virtual environment and install dependencies.

**Step 1: Install `uv`**
If you haven't installed `uv` yet, you can do so using the official standalone installer:
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Step 2: Create a Virtual Environment**
Navigate to the `model_Finetune_test` directory and create the virtual environment:
```bash
cd finetune_inference/model_Finetune_test
uv venv
```

**Step 3: Activate the Virtual Environment**
```bash
# On Linux / macOS
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

**Step 4: Install Dependencies using `uv`**
Use `uv pip` to install the required libraries listed in the `requirements.txt` file quickly:
```bash
uv pip install -r requirements.txt
```

Once installed, you can configure your API settings in `main.py` and run the script.

## 🚧 Project Status: Under Development
The features outlined in the project description—specifically the full robotic arm integration and the end-to-end inference loop—are currently in progress.


##  Acknowledgements
* [Ultralytics (YOLOv11)](https://github.com/ultralytics/ultralytics)
* [Tencent ncnn](https://github.com/Tencent/ncnn)
* [Roboflow Universe](https://universe.roboflow.com/)
* https://github.com/opencv/opencv
