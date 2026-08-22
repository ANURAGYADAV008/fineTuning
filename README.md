# GPT-2 Fine-Tuning for Spam Classification

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.3%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project contains Python scripts and a Jupyter Notebook for fine-tuning a pre-trained **GPT-2 (124M parameter version)** to perform binary email/SMS spam classification. It is based on **Chapter 6: Fine-Tuning for Classification** from the book *Build a Large Language Model (From Scratch)* by Sebastian Raschka.

---

## 🌟 Features

- **Pretrained Weights Loader**: Dynamically fetches GPT-2 model parameters (124M) from OpenAI's backup links or directly and converts checkpoint tensors to PyTorch-compatible formats.
- **LLM Architecture Modification**: Modifies the pretrained language model to act as a classifier. The language modeling head (50,257 output dimensions) is replaced with a classification head (2 output dimensions, corresponding to `ham` / `spam`).
- **Parameter-Efficient Finetuning**: Freezes the early transformer blocks and trains only the final transformer block, final LayerNorm, and the newly added classification head.
- **Custom Tokenization & Padding**: Pre-tokenizes raw dataset texts with the `tiktoken` vocabulary, utilizing `<|endoftext|>` token IDs (`50256`) for custom batch padding.
- **Extensive Evaluation & Plots**: Automatically generates training metrics (loss and classification accuracy) across training, validation, and test datasets. Plots are saved as PDFs (`loss-plot.pdf`, `accuracy-plot.pdf`).

---

## 📂 Project Structure

```
fineTuning/
├── gptfinetune.ipynb      # Main execution notebook implementing the end-to-end pipeline
├── gpt_download3.py       # Helper script to download & load GPT-2 checkpoint Weights
├── previous_chapters.py   # Core PyTorch implementation of the GPT-2 model architecture
├── pyproject.toml         # Packaging configuration, metadata, and dependencies
├── review_classifier.pth  # Serialized state dictionary of the finetuned model
├── train.csv              # Balanced training subset (70%)
├── validation.csv         # Balanced validation subset (10%)
├── test.csv               # Balanced testing subset (20%)
├── loss-plot.pdf          # Post-training loss curve visualization
├── accuracy-plot.pdf      # Post-training classification accuracy curve visualization
├── sms_spam_collection/   # Extracted dataset directory
└── gpt2/                  # Local directory for cached GPT-2 weights
```

---

## ⚙️ Installation & Setup

We recommend managing the project dependencies using `uv` or standard virtual environments (`venv`).

### Prerequisites
- Python >= 3.12
- PyTorch (compiled with CUDA or MPS for hardware acceleration)

### Using UV (Recommended)
Add or run with the project environment using:
```bash
uv sync --all-extras
# To execute the notebook via the UV environment:
uv run jupyter notebook gptfinetune.ipynb
```

### Standard PIP installation
If you prefer standard pip inside a python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r pyproject.toml
```

---

## 🚀 Pipeline & Implementation Details

### 1. Data Preparation
The SMS Spam Collection raw data is downloaded from the UCI Machine Learning Repository, cleaned, and balanced between `ham` (non-spam) and `spam` to resolve label imbalance. It is split as follows:
- **Train (70%)** - 1,045 samples
- **Validation (10%)** - 149 samples
- **Test (20%)** - 300 samples

### 2. Custom Padding (`SpamDataset`)
Prior to loader batching, each sequence is tokenized using `tiktoken`. All sequences are padded via **Option 2** (preserving full length by padding shorter sequences to the size of the longest sequence in the batch using the `<|endoftext|>` token ID `50256`):
```python
# Truncate sequences exceeding max_length (e.g. 120 or 1024 tokens)
self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
# Pad sequences
self.encoded_texts = [
    encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
    for encoded_text in self.encoded_texts
]
```

### 3. Model Architecture Modification
The out-head weight matrix is detached and replaced with a linear mapping head:
```python
# Replace 50,257 vocab logits head with binary classes (HAM vs SPAM)
num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
```
Only the final transformer block and final layer norm weight parameters are set to track gradients (`param.requires_grad = True`), yielding high-speed parameter-efficient training.

### 4. Loss & Training
The output logits correspond to the last sequence token (which aggregates the sequential model context):
```python
logits = model(input_batch)[:, -1, :]  # Logits of last output token
loss = torch.nn.functional.cross_entropy(logits, target_batch)
```
The model is optimized using `AdamW` with a learning rate of `5e-5` and weight decay of `0.1` over 5 epochs.

---

## 📈 Results

After 5 epochs of training, the model achieves the following classification metrics (approximate):
- **Training Accuracy**: ~97.21%
- **Validation Accuracy**: ~97.32%
- **Test Accuracy**: ~95.67%

The training history is visualised in:
1. `loss-plot.pdf`
2. `accuracy-plot.pdf`

---

## 🔮 Usage as a Custom Classifier
To run the classifier on your own inputs, you can call the `classify_review` helper function defined in the notebook:
```python
from previous_chapters import GPTModel
import tiktoken
import torch

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model structure and finetuned weights
model = GPTModel(BASE_CONFIG)
model.out_head = torch.nn.Linear(in_features=768, out_features=2)
model.load_state_dict(torch.load("review_classifier.pth", map_location=device))
model.to(device)

def classify_review(text, model, tokenizer, device, max_length=120):
    model.eval()
    input_ids = tokenizer.encode(text)[:max_length]
    input_ids += [50256] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"

print(classify_review("Free entry in 2 a wkly comp to win FA Cup final tickets!", model, tokenizer, device))
# Output: spam
```

---

## 📄 License
This project is licensed under the MIT License. Feel free to use and modify it for your research and applications.
