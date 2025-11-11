import pandas as pd
import re
import unicodedata
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory


# File paths and parameters
INPUT_FILE = "telugu_tweets.csv"   
OUTPUT_FILE = "Final_Cleaned_Telugu_Dataset_300MB.csv"
TARGET_MB = 20
TARGET_BYTES = TARGET_MB * 1024 * 1024


# Telugu Normalizer (Indic NLP)
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("te")

# Pattern to remove URLs, mentions, hashtags, emojis, and non-Telugu symbols
URL_PATTERN = re.compile(r"http\S+|www\S+|@\S+|#\S+")
NON_TELUGU_PATTERN = re.compile(r"[^\u0C00-\u0C7F\s]")

def clean_telugu_text(text):
    """Clean and normalize Telugu tweet text."""
    if not isinstance(text, str):
        return ""

    # Remove URLs, mentions, hashtags, emojis, etc.
    text = URL_PATTERN.sub(" ", text)

    # Normalize Unicode text
    text = unicodedata.normalize("NFC", text)

    # Keep only Telugu characters and spaces
    text = NON_TELUGU_PATTERN.sub(" ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Apply IndicNLP normalization
    text = normalizer.normalize(text)

    return text


# Process the file chunk by chunk (~50k rows per chunk)
chunks = []
approx_bytes = 0
total_rows = 0

for chunk in pd.read_csv(INPUT_FILE, chunksize=50000, encoding="utf-8", on_bad_lines="skip"):
    if "tweet" not in chunk.columns:
        print("❌ Error: 'tweet' column not found in CSV.")
        break

    # Clean the Telugu tweets
    chunk["cleaned_tweet"] = chunk["tweet"].astype(str).apply(clean_telugu_text)

    # Keep only the cleaned_tweet column
    chunk = chunk[["cleaned_tweet"]]

    chunks.append(chunk)
    approx_bytes += chunk.memory_usage(deep=True).sum()
    total_rows += len(chunk)

    print(f"Processed chunk, approx {approx_bytes / 1024 / 1024:.2f} MB so far")

    if approx_bytes >= TARGET_BYTES:
        break

# Combine all chunks and save to CSV
final_df = pd.concat(chunks, ignore_index=True)
final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"\n✅ Saved {len(final_df)} rows (~{approx_bytes / 1024 / 1024:.2f} MB) to '{OUTPUT_FILE}'")
print("\n🔍 Sample cleaned tweets:")
print(final_df.head(10).to_string(index=False))
