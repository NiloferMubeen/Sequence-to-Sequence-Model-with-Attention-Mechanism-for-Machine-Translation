# Seq2Seq Neural Machine Translation with Bahdanau Attention

This project implements a **Sequence-to-Sequence (Seq2Seq) Neural Machine Translation (NMT)** model using **Bahdanau (Additive) Attention** from scratch in **PyTorch**.  
The model is trained on a small **English–French parallel dataset** and demonstrates how attention overcomes the limitations of a fixed-length context vector.

---

## 📌 Project Highlights

- End-to-end **Encoder–Decoder architecture**
- **Bahdanau Attention** for dynamic word alignment
- Comparison between **baseline Seq2Seq** and **attention-based Seq2Seq**
- **Greedy decoding** for inference
- **BLEU score evaluation**
- **Attention heatmap visualization** for interpretability

---

## 🧠 Why Attention?

Traditional Seq2Seq models compress the entire source sentence into a single context vector, which causes information loss for long sentences.

Bahdanau Attention:
- Computes **soft alignments** between source and target words
- Allows the decoder to focus on **different parts of the input** at each timestep
- Improves translation quality and convergence

---

## 📂 Project Structure

├── eng-fra.txt

├── encoder.py

├── decoder.py

├── attn_decoder.py

├── attention.py

├── attn_seq2seq.py

├── train.py

├── translate.py

├── bleu.py

├── visualize_attention.py

├── vocab.py

├── dataset.py

└── README.md


---

## 🔧 Technologies Used

- Python 3.x
- PyTorch
- sacreBLEU
- matplotlib
- NumPy

---

## 📊 Dataset

- **English–French parallel corpus**
- Tab-separated sentence pairs
- Preprocessing includes:
  - Lowercasing
  - Punctuation handling
  - Tokenization
  - Vocabulary creation
  - Padding and tensor conversion

---

## 🏗️ Model Architecture

### Encoder
- Embedding layer
- GRU
- Outputs hidden states for all source tokens

### Decoder (Attention-based)
- Embedding layer
- Bahdanau Attention
- GRU with context vector
- Linear output layer

### Attention Mechanism
\[
score(s_t, h_i) = v^T \tanh(W_s s_t + W_h h_i)
\]

\[
c_t = \sum_i \alpha_{t,i} h_i
\]

---

## 🚀 Training

- Optimizer: **Adam**
- Loss: **CrossEntropyLoss** (ignoring padding tokens)
- Teacher Forcing enabled
- Gradient clipping applied

Example training loop:
```python
loss = train_one_epoch_attn(model, loader, optimizer, criterion)



