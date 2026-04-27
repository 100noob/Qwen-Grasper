from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import json, os
from PIL import Image
from datasets import Dataset
from paths import ROOT_DIR, MODEL_NAME, JSONL_PATH, IMAGE_DIR, EXPORT_WEIGHT


MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True  # 4-bit quantization to save VRAM

# 1. Load and format your JSONL data
def load_vision_data(json_path):
    """Strictly construct data according to the Unsloth vision fine-tuning format"""
    data_list = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()       # Remove empty lines and spaces
            if not line:
                continue
            # Convert each line to JSON
            data = json.loads(line)
            data_list.append(data)
    return data_list


# Changing Hugging Face / TRL / Unsloth usually used conversational format:
def convert_to_conversation(sample):
    instructions = sample["conversations"][0]["value"]
    caption = str(sample["conversations"][1]["value"])
    image_PTL = Image.open(os.path.join(IMAGE_DIR, sample["image"])).convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instructions},
                {"type": "image", "image": image_PTL}
            ]
         },
         {
            "role": "assistant",
            "content": [
                {"type": "text", "text": caption},
            ]
         },
    ]
    return {"messages": conversation}


def main():
    # Load data
    dataset_raw = load_vision_data(JSONL_PATH)
    converted_dataset = Dataset.from_list([convert_to_conversation(sample) for sample in dataset_raw])
    print("Data conversion complete")
    xxa = 3

    # 2. Load Unsloth Vision Model
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_16bit=True,
        dtype=None,  # Automatically selected
        trust_remote_code=True,
        full_finetuning=False
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers

        r = 16,                            # The larger, the higher the accuracy, but might overfit
        lora_alpha = 16,                   # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,                # We support rank stabilized LoRA
        loftq_config = None,               # And LoftQ
        target_modules = "all-linear",     # Optional now! Can specify a list if needed
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )
    print("Model loading complete!")

    # 3. Configure Trainer
    training_args = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=EXPORT_WEIGHT,
        max_seq_length=4096,  # 🔑 Vision models must specify this explicitly to avoid dynamic alignment OOM
        packing=False,        # 🔑 Packing is not recommended for vision data, as it can cause image tensor misalignment
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset = converted_dataset,
        args = training_args,
    )
    print("Trainer configuration complete!")

    # 5. Start Training
    print("Starting Unsloth Qwen-VL LoRA fine-tuning...")
    trainer.train()

    # 6. Save LoRA Model
    model.save_pretrained("qwen_vl_lora")  # Only saves LoRA weights (very small)
    print("Training complete! LoRA model saved to qwen_vl_lora")

if __name__=='__main__':
    main()
