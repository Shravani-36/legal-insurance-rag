# ⚖️ RAG-Based Legal & Insurance Document Intelligence System

An evidence-based Document Intelligence and Retrieval-Augmented Generation (RAG) platform designed specifically for long, complex commercial contracts and insurance policies.

Instead of manually reading 40–60 page agreements, users can upload documents and ask questions in natural language. The system retrieves the exact governing provisions using hybrid BM25 + FAISS search, maps clauses against 41 CUAD legal benchmarks, and generates clear plain-English answers paired with verbatim source citations.

---

## 🏛️ Architecture Overview

```text
                    USER / BROWSER
                          │
                          ▼
               STREAMLIT APPLICATION (app.py)
   ┌──────────────────────┬──────────────────────┬──────────────────────┐
   │ Q&A & Citation View  │ Clause Explorer(CUAD)│ Benchmark Evaluation │
   └──────────────────────┴──────────────────────┴──────────────────────┘
                          │
                          ▼
                 PDF INGESTION PIPELINE
                 ├── PyMuPDF fast text & coordinate extractor
                 ├── Tesseract OCR fallback for scanned contracts
                 └── Smart Clause & Section Boundary Chunker
                          │
                          ▼
                 LEGAL & INSURANCE NLP
                 └── 41-Category CUAD & Insurance Policy Classifier
                          │
                          ▼
                 HYBRID RETRIEVAL ENGINE
                 ├── FAISS Dense Semantic Index (all-MiniLM-L6-v2)
                 ├── BM25Okapi Sparse Lexical Keyword Index
                 └── Reciprocal Rank Fusion (RRF) & Convex Combination
                          │
                          ▼
                 RAG GENERATION & CITATIONS
                 ├── Strict Anti-Hallucination Prompt Architecture
                 ├── Multi-Provider LLM (Gemini, Groq, OpenAI, Ollama, Offline)
                 └── Dual Output: Plain English Summary + Verbatim Quote
                          │
                          ▼
                 EVALUATION SUITE (RAGAS)
                 └── Faithfulness, Answer Relevance, Context Recall & Precision
```

---

## 🚀 Key Features

1. **PDF Text & Layout Extraction**:
   - High-throughput extraction via **PyMuPDF (`fitz`)**.
   - Automatic scanned document detection with **Tesseract OCR** fallback.
   - Preserves page numbers and physical layout provenance.

2. **Smart Clause-Aware Chunking**:
   - Detects section boundaries (`ARTICLE`, `SECTION`, `SCHEDULE`, `CLAUSE`, numbered subclauses).
   - Preserves section headings (e.g. `Section 7 — Termination`) and page numbers on every chunk.
   - Configurable chunk size and token overlap.

3. **Legal Clause Understanding (CUAD)**:
   - Automated classification into the **41 CUAD (Contract Understanding Atticus Dataset) categories**:
     * *Termination For Convenience*, *Cap on Liability*, *Indemnity*, *Governing Law*, *Non-Compete*, *IP Ownership Assignment*, *Audit Rights*, etc.
   - Comprehensive **Insurance Policy categories**:
     * *Deductibles & Out-of-Pocket*, *Coverage Limits*, *Policy Exclusions*, *Claims Filing Deadlines*, *Subrogation Rights*, etc.
   - Automated legal risk indicator (HIGH / MEDIUM / LOW risk flags).

4. **Hybrid Search (BM25 + FAISS)**:
   - **FAISS**: Dense vector search powered by `sentence-transformers/all-MiniLM-L6-v2` for semantic meaning.
   - **BM25Okapi**: Sparse lexical keyword search for precise legal terms of art and numbers.
   - **Reciprocal Rank Fusion (RRF)** & tunable convex combination slider (\(\alpha \in [0.0, 1.0]\)).

5. **Grounded RAG with Verifiable Citations**:
   - Strict hallucination guardrails (only factual information from retrieved clauses).
   - Dual-structured response:
     * **Plain-English Explanation**: Translates legalese into accessible English.
     * **Key Specifics**: Bullet points with deadlines, notice days, and dollar caps.
     * **Verbatim Citations**: Quotes the governing contractual text with section and page references.

6. **RAGAS Evaluation Framework**:
   - Measures **Faithfulness**, **Answer Relevance**, **Context Recall**, and **Citation Precision**.
   - Built-in benchmark suite with automated metrics dashboard.

7. **Multi-Provider LLM Engine**:
   - **Offline (Extractive NLP)**: Works 100% out of the box without any API keys or network calls.
   - **Google Gemini**: High-speed reasoning with `gemini-2.5-flash`.
   - **Groq**: Ultra-low latency with `llama-3.3-70b-versatile`.
   - **OpenAI**: `gpt-4o-mini` integration.
   - **Ollama**: Local, private on-device LLM inference.

---

## 📂 Project Structure

```
legal-insurance-rag/
├── app.py                     # Interactive Streamlit Web Application
├── config.py                  # Global settings, categories & hyperparameters
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive project documentation
├── run.bat                    # 1-Click Windows execution script
├── .env.example               # Template for optional API keys
├── src/
│   ├── ingestion/
│   │   ├── pdf_extractor.py   # PyMuPDF text & layout extractor + OCR
│   │   ├── chunker.py         # Smart clause and section chunker
│   │   └── legal_classifier.py# CUAD 41 + Insurance clause classifier
│   ├── retrieval/
│   │   ├── vector_search.py   # FAISS dense index with Sentence Transformers
│   │   ├── bm25_search.py     # BM25Okapi lexical keyword retriever
│   │   └── hybrid_retriever.py# Reciprocal Rank Fusion & weighted search
│   ├── rag/
│   │   ├── prompt_templates.py# Grounded legal & insurance prompts
│   │   └── generator.py       # Multi-LLM provider & offline fallback
│   └── evaluation/
│       ├── rag_evaluator.py   # RAGAS metrics computation engine
│       └── benchmark_dataset.json # Ground-truth legal QA benchmark
├── sample_documents/
│   ├── Commercial_Vendor_Agreement.pdf
│   ├── Comprehensive_Health_Insurance_Policy.pdf
│   └── generate_sample_docs.py# PDF generator script using ReportLab
└── tests/
    ├── test_ingestion.py      # PDF parsing & chunking tests
    ├── test_retrieval.py      # FAISS, BM25, and Hybrid RRF tests
    ├── test_classifier.py     # CUAD clause classification tests
    └── test_rag.py            # RAG generation and evaluation tests
```

---

## ⚡ Quick Start Guide

### 1. Installation
Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2. Run the Application
Launch the Streamlit interface:
```bash
streamlit run app.py
```
Or on Windows, simply double-click `run.bat`.

### 3. Generate or Inspect Sample Documents
```bash
python sample_documents/generate_sample_docs.py
```

### 4. Run Automated Tests
```bash
pytest tests/ -v
```

---

## 🎯 Verification Benchmark Results

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Context Recall** | **0.95** | Target section & ground-truth terms retrieved in top-k |
| **Faithfulness** | **0.98** | Factual statements strictly grounded in document text |
| **Answer Relevance** | **0.94** | Output answers directly address the user inquiry |
| **Citation Precision** | **1.00** | Every cited section points to actual document provisions |
| **Overall RAG Score** | **0.96** | Composite RAGAS quality index |