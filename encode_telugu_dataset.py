from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
import pandas as pd
import json

# Load trained BPE tokenizer from vocab & merges 
tokenizer = Tokenizer(models.BPE.from_file(
    "telugu_bpe_tokenizer/vocab.json",
    "telugu_bpe_tokenizer/merges.txt"
))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

# Load your cleaned Telugu dataset 
print("📥 Loading cleaned dataset...")
df = pd.read_csv("Final_Dataset.csv", encoding='utf-8')

# Tokenize (encode) each sentence into subword token IDs 
encoded_data = []
skipped = 0
for i, text in enumerate(df['cleaned_tweet']):
    if pd.isna(text) or not str(text).strip():
        skipped +=1
        continue
    text = str(text)    
    encoded = tokenizer.encode(text).ids
    encoded_data.append(encoded)

    if i % 100000 == 0 and i > 0:
        print(f"Processed {i} rows...")

# Save encoded token IDs to a JSON file 
with open("Telugu_Tokenized_Data.json", "w", encoding="utf-8") as f:
    json.dump(encoded_data, f, ensure_ascii=False)

print(f"✅ Saved {len(encoded_data)} tokenized rows to 'Telugu_Tokenized_Data.json'")
