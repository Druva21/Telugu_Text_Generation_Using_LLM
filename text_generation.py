"""
text_generation_finetuned.py — Generate Telugu text (~500 tokens) using your fine-tuned IndicGPT
Author: Druva Kumar
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from calculating_perplexity import compute_perplexity

# -------------------------------
# 1️⃣ Device
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {DEVICE}")

# -------------------------------
# 2️⃣ Load fine-tuned model & tokenizer
# -------------------------------
FINETUNED_DIR = "./indicgpt_finetuned"  # path where your fine-tuned model is saved

tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)
print("<te>" in tokenizer.get_vocab())  # True if token exists

model = AutoModelForCausalLM.from_pretrained(FINETUNED_DIR).to(DEVICE)
model.eval()
print("✅ Fine-tuned model & tokenizer loaded successfully!")

# -------------------------------
# 3️⃣ Prepare prompt
# -------------------------------
prompt = "Write the story only in Telugu language. Do not use any other languages.ఒక తెలుగు కథ రాయండి: ఒక రోజు ఒక గ్రామంలో ఒక చిన్న పిల్లవాడు ఉన్నాడు."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
print(f"🧾 Prompt: {prompt}")

# -------------------------------
# 4️⃣ Generate text
# -------------------------------
max_new_tokens = 500

# Basic sampling without repetition penalty (for now)
outputs = model.generate(
    input_ids,
    max_new_tokens=max_new_tokens,
    do_sample=True,         # sampling to get variability
    top_k=50,               # top-k sampling
    top_p=0.9,              # nucleus sampling
    temperature=1.0,
    repetition_penalty=1.2,
    no_repeat_ngram_size=2,  # <--- prevents repeating bigrams
    eos_token_id=tokenizer.eos_token_id
)

# -------------------------------
# 5️⃣ Decode
# -------------------------------
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------------
# 6️⃣ Print
# -------------------------------
#print("\n📝 Generated Telugu Text :\n")
#print(generated_text)

# -------------------------------
# 7️⃣ Optional: Keep only Telugu characters
# -------------------------------
def keep_telugu_only(text):
    # Keep only Telugu characters, spaces, and punctuation
    telugu_chars = "".join([c for c in text if '\u0C00' <= c <= '\u0C7F' or c.isspace() or c in '.,!?'])
    # Replace multiple consecutive whitespace (spaces, tabs, newlines) with a single space
    telugu_chars = " ".join(telugu_chars.split())
    return telugu_chars

print("Only telugu text is: ")
print(keep_telugu_only(generated_text))

print("Perplexity of the model is : ",compute_perplexity(model,tokenizer,generated_text))