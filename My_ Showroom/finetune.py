import logging, warnings, sys, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    LoraConfig,
    TaskType
)

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

bnb_Config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16
)
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Configuration ===
MODEL_NAME = 'microsoft/phi-2'
DATA_PATH = "train_data.jsonl"  # Your training data
OUTPUT_DIR = "fine_tuned_phi_2"
MAX_LENGTH = 512
USE_LORA = True  # Set False if you want full fine-tuning
LOAD_IN_4BIT = True  # Set False for full precision

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config = bnb_Config,
            torch_dtype = torch.float16
            )
    model.to(device)
    model.eval()
    logging.info(f"MOdel {MODEL_NAME} is loaded successfully!")
except Exception as e:
        logging.error(f"{MODEL_NAME} not loaded!")
        raise SystemError("Exit due to model load error!")

if USE_LORA:
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "fc1", "fc2"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)


# === Load Dataset ===
def tokenize(example):
    prompt = f"### Instruction:\n{example['prompt']}\n### Response:\n{example['completion']}"
    inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs


raw_dataset = load_dataset("json", data_files=DATA_PATH)
train_dataset = raw_dataset["train"]
print("Number of samples in dataset:", len(train_dataset))

tokenized_dataset = raw_dataset.map(tokenize, batched=False) 

from transformers import TrainerCallback
import torch

class GPUStatsCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Log only every 500 steps
        if torch.cuda.is_available() and state.global_step % 500 == 0:
            allocated = torch.cuda.memory_allocated() / 1e6  # in MB
            reserved = torch.cuda.memory_reserved() / 1e6    # optional: total memory reserved
            max_allocated = torch.cuda.max_memory_allocated() / 1e6
            print(f"[GPU] Step {state.global_step} | Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Max Allocated: {max_allocated:.2f} MB")

            
# === Training Arguments ===
training_args = TrainingArguments(
    output_dir= OUTPUT_DIR,           
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    max_steps=500,
    learning_rate=3e-5,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    fp16=True,
    logging_dir="./logs",
    report_to="none",
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1
)

model.gradient_checkpointing_enable()
model.config.use_cache = False
# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    callbacks=[GPUStatsCallback()] 
)

# ✅ Calculate steps
effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
steps_per_epoch = len(train_dataset) // effective_batch_size
total_training_steps = steps_per_epoch * training_args.num_train_epochs

print("Effective batch size:", effective_batch_size)
print("Steps per epoch:", steps_per_epoch)
print("Total training steps:", total_training_steps)

# === Train ===
trainer.train(resume_from_checkpoint=True)

# === Save Model ===
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Fine-tuning complete. Model saved at:", OUTPUT_DIR)
 