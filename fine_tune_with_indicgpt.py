
print("starting....")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import tempfile
from .env import token_repo

print("hugging face....")
from huggingface_hub import login
login(token=token_repo)

CONFIG = {
    #"base_model": "ai4bharat/IndicGPT",   # IndicGPT base
    "base_model": "ai4bharat/IndicBART",
    #"base_model": "gyanai/paramanu-telugu-207M-hf",
    #"base_model": "Telugu-LLM-Labs/Telugu-gemma-7b-finetuned-sft",
    "custom_model_path": "llm_from_scratch.pt",  
    "dataset_path": "telugu_corpus.txt",
    "tokenizer_dir": "telugu_bpe_tokenizer",
    "output_dir": "./indicgpt_finetuned",
    "epochs": 3,
    "batch_size": 2,
    "lr": 5e-5,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")


# Load IndicGPT + Tokenizer
print("🔹 Loading IndicGPT and Telugu tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
model = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"]).to(device)


# Load trained weights (if shapes match)

try:
    print("🧩 Loading your custom model weights...")
    custom_state = torch.load(CONFIG["custom_model_path"], map_location=device)
    model.load_state_dict(custom_state, strict=False)
    print("✅ Custom weights loaded successfully.")
except Exception as e:
    print(f"⚠️ Could not load your custom weights: {e}")

# ----------------------------
# Prepare dataset
# ----------------------------
# def load_dataset(file_path, tokenizer, block_size=128):
#     dataset = TextDataset(
#         tokenizer=tokenizer,
#         file_path=file_path,
#         block_size=block_size,
#         overwrite_cache=True,
#     )
#     return dataset
# print("Dataset loading....")
# train_dataset = load_dataset(CONFIG["dataset_path"], tokenizer, max_mb=20)
# print("Dataset loaded....")

from transformers import TextDataset
import tempfile

def load_dataset(file_path, tokenizer, block_size=128, max_kb=None):
    """
    Load a text dataset but only read up to `max_mb` megabytes if specified.
    """
    if max_kb is not None:
        max_bytes = max_kb * 1024 
        # Read only first N MB and save it to a temporary file
        with open(file_path, "rb") as f:
            data = f.read(max_bytes)
        text_data = data.decode("utf-8", errors="ignore")

        # Write partial data to a temporary file so TextDataset can use it
        tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
        tmp.write(text_data)
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        print(f"✅ Loaded first {max_kb} KB of dataset from '{file_path}'")
        file_path = tmp_path  # use the temporary partial file

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
        overwrite_cache=True,
    )
    return dataset


# Usage stays exactly the same:
print("Dataset loading....")
train_dataset = load_dataset(CONFIG["dataset_path"], tokenizer, block_size=128, max_kb=100)
print("Dataset loaded....")


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ----------------------------
# Training setup
# ----------------------------

training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    overwrite_output_dir=True,
    num_train_epochs=CONFIG["epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    save_steps=100,
    save_total_limit=2,
    prediction_loss_only=True,
    learning_rate=CONFIG["lr"],
    logging_dir="./logs",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# ----------------------------
# Fine-tuning
# ----------------------------
print("🚀 Starting fine-tuning...")
trainer.train()
print("Finie tuning completed and saving the model....")
# ----------------------------
# Save model
# ----------------------------
trainer.save_model(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])
print("✅ Fine-tuning complete! Model saved to:", CONFIG["output_dir"])
