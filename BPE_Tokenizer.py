import pandas as pd
from tokenizers import ByteLevelBPETokenizer
import os

#File Paths
INPUT_FILE = "Final_Dataset.csv"            
TEXT_CORPUS_FILE = "telugu_corpus.txt"      
TOKENIZER_DIR = "telugu_bpe_tokenizer"      
OUTPUT_FILE = "Tokenized_Telugu_BPE.csv"    
VOCAB_SIZE = 25000                         

#Load cleaned Telugu text
print("📥 Loading cleaned dataset...")
df = pd.read_csv(INPUT_FILE, encoding="utf-8")
if 'cleaned_tweet' not in df.columns:
    raise ValueError("❌ 'cleaned_tweet' column not found in Final_Dataset.csv")

#Prepare plain text corpus
print("🧾 Preparing plain text corpus for tokenizer training...")
with open(TEXT_CORPUS_FILE, "w", encoding="utf-8") as f:
    for line in df["cleaned_tweet"].astype(str):
        line = line.strip()
        if line:
            f.write(line + "\n")

#Train the BPE tokenizer
print("🧠 Training Byte-Pair Encoding (BPE) tokenizer...")
tokenizer = ByteLevelBPETokenizer()
os.makedirs(TOKENIZER_DIR, exist_ok=True)
tokenizer.train(
    files=TEXT_CORPUS_FILE,
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=["<pad>", "<unk>", "<s>", "</s>"]
)

tokenizer.save_model(TOKENIZER_DIR)
print(f"✅ Tokenizer saved in '{TOKENIZER_DIR}/' with {VOCAB_SIZE} vocab size.")

#Tokenize the dataset
print("🔤 Tokenizing the dataset using trained tokenizer...")
def tokenize_text(text):
    if not isinstance(text, str):
        return ""
    tokens = tokenizer.encode(text).tokens
    return " ".join(tokens)

df_tokenized = pd.DataFrame()
df_tokenized["cleaned_tweet"] = df["cleaned_tweet"]
df_tokenized["subword_tokens"] = df["cleaned_tweet"].astype(str).apply(tokenize_text)

#Save the tokenized output
df_tokenized.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
print(f"✅ Tokenized dataset saved as '{OUTPUT_FILE}'")
print(f"📄 Vocabulary & merges stored in: '{TOKENIZER_DIR}/'")


print("\n🔍 Sample preview:")
print(df_tokenized.head(5).to_markdown(index=False))
