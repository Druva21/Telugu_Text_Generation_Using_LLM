"""
text_gen.py — Generate Telugu text using trained GPTSmall (ByteLevel BPE Decoding)
Author: Druva Kumar
"""

import torch
import re
import json
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from train_llm import GPTSmall, CONFIG
from calculating_perplexity import compute_perplexity

# =====================================================
# 1️⃣ Device
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {DEVICE}")

# =====================================================
# 2️⃣ Load the trained ByteLevel BPE tokenizer
# =====================================================
try:
    tokenizer = Tokenizer.from_file("telugu_bpe_tokenizer/tokenizer.json")
    print("✅ Loaded tokenizer.json successfully!")
except Exception as e:
    print(f"⚠️ tokenizer.json not found ({e}), loading from vocab & merges instead...")
    tokenizer = Tokenizer(models.BPE.from_file(
        "telugu_bpe_tokenizer/vocab.json",
        "telugu_bpe_tokenizer/merges.txt"
    ))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

print("✅ Tokenizer ready!")

# =====================================================
# 3️⃣ Initialize Model
# =====================================================
model = GPTSmall(
    vocab_size=CONFIG["vocab_size"],
    embedding_dim=CONFIG["embedding_dim"],
    num_heads=CONFIG["num_heads"],
    hidden_dim=CONFIG["hidden_dim"],
    num_layers=CONFIG["num_layers"],
    seq_len=CONFIG["sequence_length"],
    dropout=CONFIG["dropout"]
)

# Load trained model weights
state_dict = torch.load(CONFIG["model_save_path"], map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
print("✅ Model loaded successfully!")

# =====================================================
# 4️⃣ Encode the prompt
# =====================================================
prompt = CONFIG.get("prompt", "ఒక తెలుగు కథ రాయండి: ఒక రోజు ఒక గ్రామంలో ఒక చిన్న పిల్లవాడు ఉన్నాడు.")
encoded_prompt = tokenizer.encode(prompt)
prompt_ids = encoded_prompt.ids

seq_len = CONFIG["sequence_length"]
prompt_ids = prompt_ids[-seq_len:]  # truncate if too long
generated = torch.tensor([prompt_ids], dtype=torch.long).to(DEVICE)

print(f"🧾 Prompt: {prompt}")
print(f"🧩 Encoded Prompt IDs: {prompt_ids[:20]} ...")

# =====================================================
# 5️⃣ Generate new text
# =====================================================
max_new_tokens = 10000  # You can increase this if needed
# Initialize two tensors:
generated = torch.tensor([prompt_ids], dtype=torch.long).to(DEVICE)
full_sequence = generated.clone()  # 👈 store everything

for _ in range(max_new_tokens):
    logits = model(generated)
    logits = logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)

    next_token = torch.multinomial(probs, num_samples=1)

    # Update generated (context)
    generated = torch.cat([generated, next_token], dim=1)

    # Append to full sequence
    full_sequence = torch.cat([full_sequence, next_token], dim=1)

    # Keep only last seq_len tokens for model input
    if generated.size(1) > seq_len:
        generated = generated[:, -seq_len:]


# =====================================================
# 6️⃣ Decode the generated text using the tokenizer
# =====================================================
generated_ids = full_sequence[0].tolist()
#print("🧩 Generated token IDs:", generated_ids[:50], "...")

decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
#print(decoded_text)

# =====================================================
# 7️⃣ Post-process to clean Telugu
# =====================================================
def clean_telugu(text):
    """Remove non-Telugu or junk Unicode artifacts."""
    return "".join([c for c in text if '\u0C00' <= c <= '\u0C7F' or c.isspace() or c in '.,!?'])

def make_readable(text):
    """Make Telugu words spaced properly."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

telugu_text = clean_telugu(decoded_text)
telugu_text = make_readable(telugu_text)

# =====================================================
# 8️⃣ Final Output
# =====================================================
print("\n📝 Decoded Telugu Text:\n")
print(telugu_text if telugu_text else decoded_text)

print("Perplexity of the model is : ",compute_perplexity(model,tokenizer,decoded_text))
