# 🌐 Telugu Text Generation using LLM

**Project Title:** LLM-Based Text Generation for Creative Writing  
**Repository:** [Telugu_Text_Generation_Using_LLM](https://github.com/Druva21/Telugu_Text_Generation_Using_LLM.git)

---

## 🧠 Project Overview
This project focuses on building a **Telugu language text generation system** using **Large Language Model (LLM)** techniques.  
While large-scale models like GPT perform well for high-resource languages (e.g., English), **Telugu**, a low-resource language, faces data scarcity.  

Our objective is to develop a model capable of generating **coherent, natural, and contextually relevant Telugu text** using a combination of:
1. A **custom SmallGPT model** trained from scratch.  
2. A **fine-tuned Indic language model** (IndicGPT / IndicBART).

---

## 🚩 Problem Statement
- Natural Language Generation (NLG) is a key challenge in NLP, especially for low-resource languages.  
- Lack of high-quality Telugu datasets and pretrained generative models limits progress.  

**Goal:**  
- Build a Telugu text generation model.  
- Evaluate its quality using **Perplexity (PPL)**.

---

## 📊 Dataset Details

**Source:** [LOIT Dataset](https://github.com/bedapudi6788/LOIT)  
**Type:** Telugu tweets containing millions of samples.

### Preprocessing Steps:
- Removed URLs, mentions, hashtags, emojis, and special characters using regex.  
- Retained only Telugu Unicode characters (`\u0C00–\u0C7F`).  
- Applied Unicode normalization (NFC) and IndicNLP normalization.  
- Removed mixed-language words and extra spaces.  
- Implemented **Byte Pair Encoding (BPE)** for subword-level tokenization.  
- Created vocabulary files: `vocab.json` and `merges.txt`.  

---

## ⚙️ Methodology

### **Stage 1 – Training SmallGPT**
Custom **GPT-style Transformer** implemented in PyTorch (`train_llm.py`).

| Parameter | Value |
|------------|--------|
| Architecture | Decoder-only Transformer |
| Transformer Blocks | 2 |
| Attention Heads | 4 |
| Embedding Dimension | 128 |
| Hidden Dimension | 256 |
| Sequence Length | 50 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Learning Rate | 0.002 |
| Epochs | 5 |
| Batch Size | 32 |
| Device | CPU |

**Output:** `llm_from_scratch.pt`

---

### **Stage 2 – Fine-Tuning Indic Model**
Fine-tuned pretrained Indic model (`ai4bharat/IndicGPT` or `IndicBART`) using weights from SmallGPT.

| Parameter | Value |
|------------|--------|
| Learning Rate | 5e-5 |
| Batch Size | 2 |
| Epochs | 3 |
| Framework | Hugging Face Transformers (Trainer API) |
| Dataset | ~100 KB Telugu Corpus |

**Output Directory:** `indicgpt_finetuned/`

---

## 📈 Evaluation

**Metric:** Perplexity (PPL)  
- Lower PPL → Better fluency and confidence.  
- PPL = e^(Cross-Entropy Loss)

### **Results**

| Tokens | SmallGPT PPL | Fine-tuned Model PPL |
|---------|---------------|----------------------|
| 20 | 5,957,041.42 | **16.97** |
| 50 | 1,890,206.71 | **38.62** |
| 100 | 3,771,991.83 | **45.55** |
| 200 | 5,039,043.35 | **10.69** |
| 500 | 3,534,705.64 | **39.22** |

**Observation:** Fine-tuning dramatically improves coherence and reduces perplexity.

---

## 📂 Folder Structure

Telugu_Text_Generation_Using_LLM/
│
├── requirements.txt

├── telugu_tweets.csv

├── Final_Dataset.csv

├── BPE_Tokenizer.py

├── encode_telugu_dataset.py

├── train_llm.py

├── fine_tune_with_indicgpt.py

├── Calculating_Perplexity.py

├── telugu_bpe_tokenizer/

│ ├── merges.txt

│ └── vocab.json

├── Telugu_corpus.txt

└── Tokenized_Telugu_BPE.json


---

## 📈 Results

| Tokens | SmallGPT PPL | Fine-tuned Model PPL |
|---------|---------------|----------------------|
| 20 | 5,957,041.42 | **16.97** |
| 50 | 1,890,206.71 | **38.62** |
| 100 | 3,771,991.83 | **45.55** |
| 200 | 5,039,043.35 | **10.69** |
| 500 | 3,534,705.64 | **39.22** |
