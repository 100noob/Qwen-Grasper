from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import json, os
from PIL import Image
from datasets import Dataset

from unsloth_zoo.tokenizer_utils import fix_untrained_tokens

import unsloth_zoo.tokenizer_utils
unsloth_zoo.tokenizer_utils.fix_untrained_tokens = lambda *args, **kwargs: None

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import MODEL_NAME, JSONL_PATH as JSON_PATH, IMAGE_DIR, EXPORT_WEIGHT

MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True  # 4bit quantization to save VRAM

# ===================== 1. Load and format your JSON data =====================
def load_vision_data(json_path):
    """Strictly build data according to the Unsloth visual fine-tuning format"""
    data_list = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()       # Remove empty lines, spaces
            if not line:
                continue
            # Convert line by line to JSON
            data = json.loads(line)
            data_list.append(data)
    return data_list

def convert_to_conversation(sample):
    instructions = sample["conversations"][0]["value"]
    instructions += ", The ordering rules are as follows: the block furthest from other blocks has the highest priority, blocks closer to the right side of the image have priority over those on the left, for blocks arranged vertically, the upper block has priority over the lower block."
    caption = str(sample["conversations"][1]["value"])
    image_PTL = Image.open(os.path.join(IMAGE_DIR,sample["image"])).convert("RGB")
    conversation = [
        {
            "role":"user",
            "content":[
                {"type":"text","text":instructions},
                {"type":"image","image":image_PTL}
            ]
         },
         {
            "role":"assistant",
            "content":[
                {"type":"text","text":caption},
         ]
         },
    ]
    return {"messages":conversation}


# Load data
dataset_raw = load_vision_data(JSON_PATH)
converted_dataset = Dataset.from_list([convert_to_conversation(sample) for sample in dataset_raw])
print("Data conversion completed")
xxa = 3

# ===================== 2. Load Unsloth visual model =====================
model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_16bit=False,
    load_in_fp8=True,
    dtype=None,  # Auto-select
    trust_remote_code=True,
    full_finetuning=False,
    # device_map=None
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 32,                           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 32,                  # Recommended alpha == r at least
    lora_dropout = 0.05,
    bias = "none",
    random_state = 3407,
    use_rslora = False,               # We support rank stabilized LoRA
    loftq_config = None,               # And LoftQ
    target_modules = "all-linear",    # Optional now! Can specify a list if needed
    # modules_to_save=[
    #     "lm_head",
    #     "embed_tokens",
    # ],
    modules_to_save=None,
)
print("Model loading completed!")

# ===================== 3. Configure Trainer =====================
training_args = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    num_train_epochs=4,
    learning_rate=5e-5,
    fp16=not is_bf16_supported(),
    bf16=is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.001,
    lr_scheduler_type="cosine",
    seed=3407,
    output_dir="outputs_qwen_vl_lora",
    max_seq_length=MAX_SEQ_LENGTH,  # 🔑 Must be explicitly specified for visual models to avoid dynamic alignment OOM
    packing=False,        # 🔑 Packing is not recommended for visual data, easily leads to image tensor misalignment
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = converted_dataset,
    args = training_args,
)
print("Trainer configuration completed!")

# ===================== 5. Start Training =====================
print("Starting Unsloth Qwen-VL LoRA fine-tuning...")
trainer.train()

# # ===================== 6. Save LoRA Model =====================
# model.save_pretrained("/home/xxa/zbr_code/Qwen3.5-9B_FP8_finetuned_int4_gptq1_adapter")  # Save LoRA weights only (small)
# model_path = "/home/xxa/zbr_code/Qwen3.5-9B_FP8_finetuned_int4_gptq1"
# model.save_pretrained_merged(model_path,tokenizer,save_method="merged_gptq") # Save full model
# print(f"Training completed! LoRA model saved to {model_path}")

# ===================== 6. Save Model and Configurations =====================
model_path = EXPORT_WEIGHT

# 1. Save merged full model (weights)
print("Merging and saving model weights...")
model.save_pretrained_merged(model_path, tokenizer, save_method="merged_gptq")

# 2. [Core Modification] Manually force save config.json
print("Forcing config.json export...")
model.config.save_pretrained(model_path)

# 3. [Core Modification] Manually force save tokenizer and processor configurations
print("Exporting tokenizer and processor configurations...")
tokenizer.save_pretrained(model_path)

print(f"All files (including config.json) successfully saved to: {model_path}")
