# Improving Domain-Specific Question Answering using Retrieval-Augmented Generation

## Group Members
| Name | Student ID | 
|------|-----------|
| Nishi Patel | 501356244 | 
| Avikumar Patel | 501376903 | 

## Course
Natural Language Processing — Final Project
Toronto Metropolitan University, April 2025

---

## Paper
Zhang, Z., Fang, M., & Chen, L. (2024). RetrievalQA: Assessing Adaptive
Retrieval-Augmented Generation for Short-form Open-Domain Question Answering.
**ACL Findings 2024**.
- arXiv: https://arxiv.org/abs/2402.16457
- GitHub (original): https://github.com/hyintell/RetrievalQA

---

## Project Description

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline
to reduce hallucination in Large Language Models for domain-specific Question
Answering. We compare three retrieval strategies from the RetrievalQA paper
on a 100-question balanced sample from the public benchmark dataset.

**The core result:** Always Retrieval achieved a **14× improvement** over
No Retrieval on knowledge-intensive questions (0.017 → 0.250 Exact Match),
confirming that LLMs cannot reliably answer recent or specialised questions
without external retrieval.

---

## Results

| Strategy | Exact Match | Token F1 |
|----------|-------------|----------|
| No Retrieval (baseline) | 0.110 | 0.133 |
| **Always Retrieval (RAG)** | **0.290** | **0.339** |
| Adaptive RAG | 0.250 | 0.287 |

**On retrieval-needed questions only (n=60):**

| Strategy | Exact Match | Token F1 |
|----------|-------------|----------|
| No Retrieval | 0.017 | 0.022 |
| Always Retrieval | 0.250 | 0.279 |
| Adaptive RAG | 0.250 | 0.279 |

---

## Repository Structure

```
NLP-RAG-Project/
├── README.md                    ← this file
├── RAG_NLP_Project.ipynb        ← main notebook (report)
├── data/
│   └── sample_retrievalqa.jsonl ← 100-question sample (subset)
│                                  Full dataset: huggingface.co/datasets/hyintell/RetrievalQA
└── src/
    ├── data_loader.py           ← dataset loading, exploration, sampling
    ├── retriever.py             ← FAISS index building and retrieve()
    ├── llm.py                   ← flan-t5-base loading and generate_answer()
    ├── strategies.py            ← 3 prompt builder functions
    ├── evaluation.py            ← EM, Token F1, experiment runner, error analysis
    └── visualisation.py         ← pipeline diagram and results chart
```

---

## How to Run

### Option A — Google Colab (recommended)
1. Upload the entire project folder to Google Drive
2. Open `RAG_NLP_Project.ipynb` in Colab
3. Run `Runtime → Restart and run all`
4. All libraries install automatically in the first cell

### Option B — Local (Anaconda / Jupyter)
```bash
# Clone the repository
git clone https://github.com/YourUsername/NLP-RAG-Project.git
cd NLP-RAG-Project

# Install dependencies
pip install datasets sentence-transformers faiss-cpu transformers

# Launch Jupyter
jupyter notebook RAG_NLP_Project.ipynb
```

---

## Dataset

**Full dataset:** `hyintell/RetrievalQA` on HuggingFace
```python
from datasets import load_dataset
ds = load_dataset("hyintell/RetrievalQA")
```

A 100-question sample is included in `data/sample_retrievalqa.jsonl`
for quick reference. The full 2,785-question dataset is downloaded
automatically from HuggingFace when you run the notebook.

---

## Key Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `datasets` | ≥2.18 | Load RetrievalQA from HuggingFace |
| `sentence-transformers` | ≥2.7 | Text → 384-dim embeddings |
| `faiss-cpu` | ≥1.8 | Fast vector similarity search |
| `transformers` | ≥4.39 | flan-t5-base LLM generation |
| `pandas` | any | Data manipulation |
| `matplotlib` | any | Visualisation |

---

## Source Code Overview

| File | Key Functions |
|------|--------------|
| `src/data_loader.py` | `load_retrievalqa_dataset()`, `explore_dataset()`, `create_sample()` |
| `src/retriever.py` | `build_index()`, `retrieve()` |
| `src/llm.py` | `load_model()`, `generate_answer()` |
| `src/strategies.py` | `build_prompt_no_retrieval()`, `build_prompt_always_retrieval()`, `build_prompt_adaptive()` |
| `src/evaluation.py` | `run_experiments()`, `compute_scores()`, `error_analysis()` |
| `src/visualisation.py` | `plot_pipeline()`, `plot_results()` |
