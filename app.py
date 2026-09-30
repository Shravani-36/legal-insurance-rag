# -*- coding: utf-8 -*-
"""
Streamlit Application: RAG-Based Legal & Insurance Document Intelligence.
Full interactive platform for document upload, hybrid search, CUAD clause exploration,
verifiable citations, and RAGAS evaluation.
"""
import os
import io
import time
from pathlib import Path
import streamlit as st
import pandas as pd

import config
from src.ingestion.pdf_extractor import PDFExtractor
from src.ingestion.chunker import LegalClauseChunker
from src.ingestion.legal_classifier import LegalClauseClassifier
from src.retrieval.hybrid_retriever import HybridRetriever
from src.rag.generator import LegalRAGGenerator
from src.evaluation.rag_evaluator import RAGEvaluator

# Page setup
st.set_page_config(
    page_title="Legal & Insurance Document Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 14px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .risk-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .risk-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .risk-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .source-box {
        background: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin-top: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_pipeline_components():
    """Initializes shared extractor, classifier, and evaluator."""
    extractor = PDFExtractor(enable_ocr=True)
    classifier = LegalClauseClassifier()
    evaluator = RAGEvaluator()
    return extractor, classifier, evaluator


def process_document(file_source, filename: str, chunk_size: int, overlap: int):
    """Extracts, chunks, classifies, and indexes a PDF document."""
    extractor, classifier, _ = load_pipeline_components()
    with st.spinner(f"Extracting text & layout from {filename}..."):
        pages = extractor.extract_from_file(file_source, filename=filename)
        summary = extractor.get_document_summary(pages)

    with st.spinner("Executing smart legal-clause chunking & CUAD classification..."):
        chunker = LegalClauseChunker(chunk_size=chunk_size, overlap=overlap)
        raw_chunks = chunker.chunk_pages(pages)
        enriched_chunks = classifier.enrich_chunks(raw_chunks)

    with st.spinner("Building FAISS semantic index and BM25 lexical keyword index..."):
        retriever = HybridRetriever()
        retriever.fit(enriched_chunks)

    return {
        "pages": pages,
        "summary": summary,
        "chunks": enriched_chunks,
        "retriever": retriever,
        "filename": filename
    }


def main():
    extractor, classifier, evaluator = load_pipeline_components()

    # Session state initialization
    if "doc_data" not in st.session_state:
        st.session_state.doc_data = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Title header
    st.markdown('<div class="main-title">⚖️ Legal & Insurance Document Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Evidence-based RAG platform for commercial contracts and insurance policies '
        'with BM25 + FAISS hybrid search, CUAD clause detection, verifiable citations, and RAGAS evaluation.</div>',
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------
    # Sidebar: Document Upload & Settings
    # -------------------------------------------------------------
    with st.sidebar:
        st.header("📄 Document Ingestion")

        # Sample quick-load buttons
        st.markdown("**Load Preset Sample Documents:**")
        col_s1, col_s2 = st.columns(2)
        sample_vendor = config.SAMPLE_DOCS_DIR / "Commercial_Vendor_Agreement.pdf"
        sample_health = config.SAMPLE_DOCS_DIR / "Comprehensive_Health_Insurance_Policy.pdf"

        if col_s1.button("📋 Vendor Agrmt", use_container_width=True):
            if sample_vendor.exists():
                st.session_state.doc_data = process_document(sample_vendor, sample_vendor.name, config.DEFAULT_CHUNK_SIZE, config.DEFAULT_CHUNK_OVERLAP)
                st.session_state.chat_history = []
                st.success("Loaded Commercial Vendor Agreement!")

        if col_s2.button("🛡️ Health Policy", use_container_width=True):
            if sample_health.exists():
                st.session_state.doc_data = process_document(sample_health, sample_health.name, config.DEFAULT_CHUNK_SIZE, config.DEFAULT_CHUNK_OVERLAP)
                st.session_state.chat_history = []
                st.success("Loaded Health Insurance Policy!")

        st.markdown("---")
        uploaded_file = st.file_uploader("Or Upload Custom PDF Document", type=["pdf"])
        if uploaded_file is not None:
            if st.session_state.doc_data is None or st.session_state.doc_data.get("filename") != uploaded_file.name:
                bytes_data = uploaded_file.read()
                st.session_state.doc_data = process_document(
                    bytes_data, uploaded_file.name, config.DEFAULT_CHUNK_SIZE, config.DEFAULT_CHUNK_OVERLAP
                )
                st.session_state.chat_history = []
                st.success(f"Indexed {uploaded_file.name} successfully!")

        st.markdown("---")
        st.header("⚙️ Search & RAG Configuration")

        use_rrf = st.radio("Fusion Strategy", ["Reciprocal Rank Fusion (RRF)", "Convex Linear Weighted"], index=0) == "Reciprocal Rank Fusion (RRF)"
        alpha = st.slider(
            "FAISS (Dense) vs BM25 (Keyword) Balance",
            min_value=0.0,
            max_value=1.0,
            value=0.65,
            step=0.05,
            help="0.0 = BM25 keyword matching only. 1.0 = FAISS semantic vector search only. 0.65 = Recommended Hybrid."
        )
        top_k = st.slider("Top Clauses to Retrieve (k)", min_value=1, max_value=8, value=4)

        st.markdown("---")
        st.header("🤖 LLM Provider")
        provider = st.selectbox(
            "Generation Engine",
            ["Offline (Extractive NLP)", "Google Gemini", "Groq", "OpenAI", "Ollama (Local)"],
            index=0
        )
        provider_map = {
            "Offline (Extractive NLP)": "offline",
            "Google Gemini": "gemini",
            "Groq": "groq",
            "OpenAI": "openai",
            "Ollama (Local)": "ollama"
        }
        selected_provider = provider_map[provider]

        api_key_input = ""
        if selected_provider in ["gemini", "groq", "openai"]:
            env_var = f"{selected_provider.upper()}_API_KEY"
            default_key = os.getenv(env_var, "")
            api_key_input = st.text_input(
                f"{selected_provider.capitalize()} API Key",
                value=default_key,
                type="password",
                help=f"Enter your {env_var} or configure it in .env"
            )
            if api_key_input:
                os.environ[env_var] = api_key_input

        # Document Stats Card in sidebar if loaded
        if st.session_state.doc_data:
            st.markdown("---")
            st.subheader("📊 Document Statistics")
            summary = st.session_state.doc_data["summary"]
            st.write(f"**Document**: `{summary['filename']}`")
            st.write(f"**Total Pages**: `{summary['total_pages']}`")
            st.write(f"**Total Words**: `{summary['total_words']:,}`")
            st.write(f"**Clauses/Chunks**: `{len(st.session_state.doc_data['chunks'])}`")
            if summary['scanned_pages'] > 0:
                st.warning(f"⚠️ {summary['scanned_pages']} scanned pages detected.")

    # -------------------------------------------------------------
    # Main Content Tabs
    # -------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Interactive Q&A & Citations",
        "📑 Clause Explorer (CUAD)",
        "📊 Benchmark & RAGAS Evaluation",
        "ℹ️ Architecture & Methodology"
    ])

    # Check if a document is loaded
    if st.session_state.doc_data is None:
        with tab1:
            st.info("👈 **Please select or upload a document in the sidebar to begin.** (Click *Vendor Agrmt* or *Health Policy* for an instant 1-click demo!)")
        return

    doc_data = st.session_state.doc_data
    retriever: HybridRetriever = doc_data["retriever"]
    generator = LegalRAGGenerator(provider=selected_provider)

    # -------------------------------------------------------------
    # TAB 1: INTERACTIVE Q&A & CITATIONS
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Ask Questions Against the Document")

        # Quick sample prompts
        st.markdown("**Sample inquiries for this document:**")
        sample_queries = [
            "What are the termination conditions and required notice period?",
            "Is there a liability limit or financial cap?",
            "What insurance requirements are mandated?",
            "What is the deductible and out-of-pocket maximum?",
            "What situations or conditions are explicitly excluded?"
        ]
        q_cols = st.columns(len(sample_queries))
        clicked_query = None
        for i, q_text in enumerate(sample_queries):
            short_label = q_text[:28] + "..."
            if q_cols[i].button(short_label, key=f"sq_{i}", help=q_text):
                clicked_query = q_text

        # Search query input
        query = st.text_input("Enter your query:", value=clicked_query or "", placeholder="e.g. Can either party terminate for convenience, and how much notice is required?")
        col_btn1, col_btn2 = st.columns([1, 5])
        ask_clicked = col_btn1.button("🔍 Analyze Document", type="primary", use_container_width=True)
        if col_btn2.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()

        if (ask_clicked or clicked_query) and query:
            start_time = time.time()
            with st.spinner("Executing hybrid retrieval (FAISS + BM25) and generating grounded answer..."):
                retrieved_chunks = retriever.search(query, top_k=top_k, alpha=alpha, use_rrf=use_rrf)
                result = generator.generate(query, retrieved_chunks)
                elapsed = time.time() - start_time

            st.session_state.chat_history.append({
                "question": query,
                "answer": result["answer"],
                "sources": retrieved_chunks,
                "provider": result["provider"],
                "model": result["model"],
                "elapsed": elapsed
            })

        # Display Chat History (most recent on top)
        if st.session_state.chat_history:
            st.markdown("---")
            for entry in reversed(st.session_state.chat_history):
                st.markdown(f"### ❓ Query: *\"{entry['question']}\"*")
                st.caption(f"Engine: `{entry['provider']}` ({entry['model']}) | Latency: `{entry['elapsed']:.2f}s` | Retrieved: `{len(entry['sources'])} clauses`")

                # Plain English Answer & Key Terms Box
                st.markdown(entry["answer"])

                # Verifiable Supporting Citations
                with st.expander(f"📑 View {len(entry['sources'])} Verifiable Source Clauses & Citations", expanded=True):
                    for idx, chunk in enumerate(entry["sources"], 1):
                        risk_class = f"risk-{chunk.get('risk_level', 'low').lower()}"
                        st.markdown(f"""
                        <div class="source-box">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <strong>Clause #{idx}: {chunk.get('section_title', 'Section')}</strong>
                                <span>
                                    <span class="{risk_class}">{chunk.get('risk_level', 'INFO')}</span>
                                    <span style="font-size: 0.8rem; color: #64748B; margin-left: 8px;">Page {chunk.get('page_number', '?')}</span>
                                </span>
                            </div>
                            <div style="font-size: 0.85rem; color: #475569; margin: 4px 0 8px 0;">
                                <b>CUAD Category:</b> {chunk.get('primary_category', 'General')} | 
                                <b>Hybrid Score:</b> {chunk.get('hybrid_score', 0.0)} | 
                                <b>Vector:</b> {chunk.get('vector_score', 0.0)} | 
                                <b>BM25:</b> {chunk.get('bm25_score', 0.0)}
                            </div>
                            <div style="font-style: italic; color: #1E293B; font-size: 0.95rem; border-left: 2px solid #CBD5E1; padding-left: 8px;">
                                "{chunk.get('text', '')}"
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 2: CLAUSE EXPLORER & CUAD CLASSIFIER
    # -------------------------------------------------------------
    with tab2:
        st.subheader("Document Clause Explorer & CUAD Risk Analysis")
        st.markdown(
            "Every chunk in the document is automatically analyzed and mapped to the **41 CUAD legal categories** "
            "and **10 insurance policy categories**, flagging high-risk clauses (uncapped liability, broad indemnity, auto-renewals, exclusions)."
        )

        all_chunks = doc_data["chunks"]

        # Risk distribution metrics
        high_risk_count = sum(1 for c in all_chunks if c.get("risk_level") == "HIGH")
        med_risk_count = sum(1 for c in all_chunks if c.get("risk_level") == "MEDIUM")
        low_risk_count = sum(1 for c in all_chunks if c.get("risk_level") in ["LOW", "INFORMATIONAL"])

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{len(all_chunks)}</div><div class="metric-label">Total Clauses</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #DC2626;">{high_risk_count}</div><div class="metric-label">High Risk</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #D97706;">{med_risk_count}</div><div class="metric-label">Medium Risk</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #16A34A;">{low_risk_count}</div><div class="metric-label">Low / Standard</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        # Filters
        fcol1, fcol2, fcol3 = st.columns([2, 1, 2])
        category_options = ["All Categories"] + sorted(list(set(c.get("primary_category", "General") for c in all_chunks)))
        selected_category = fcol1.selectbox("Filter by Category", category_options)
        selected_risk = fcol2.selectbox("Filter by Risk Level", ["All Levels", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL"])
        search_filter = fcol3.text_input("Filter text keywords", placeholder="Search text...")

        filtered_chunks = all_chunks
        if selected_category != "All Categories":
            filtered_chunks = [c for c in filtered_chunks if c.get("primary_category") == selected_category]
        if selected_risk != "All Levels":
            filtered_chunks = [c for c in filtered_chunks if c.get("risk_level") == selected_risk]
        if search_filter:
            filtered_chunks = [c for c in filtered_chunks if search_filter.lower() in c.get("text", "").lower()]

        st.caption(f"Showing {len(filtered_chunks)} of {len(all_chunks)} clauses")

        for c in filtered_chunks:
            with st.container():
                risk_lvl = c.get("risk_level", "INFO")
                risk_class = f"risk-{risk_lvl.lower()}"
                st.markdown(f"""
                <div style="background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <h4 style="margin: 0; color: #1E3A8A;">{c.get('section_title', 'Clause')}</h4>
                        <div>
                            <span class="{risk_class}">{risk_lvl} RISK</span>
                            <span style="font-size: 0.85rem; color: #64748B; margin-left: 8px;">Page {c.get('page_number', '?')}</span>
                        </div>
                    </div>
                    <div style="font-size: 0.85rem; color: #475569; margin-bottom: 8px;">
                        <b>Category:</b> {c.get('primary_category')} (Confidence: {c.get('confidence', 0.0):.2f}) | <i>{c.get('risk_explanation', '')}</i>
                    </div>
                    <p style="font-size: 0.95rem; line-height: 1.5; color: #1F2937; margin: 0;">
                        {c.get('text')}
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 3: BENCHMARK & RAGAS EVALUATION
    # -------------------------------------------------------------
    with tab3:
        st.subheader("📊 RAGAS-Compatible Benchmark & Quality Evaluation")
        st.markdown(
            "Empirically assess RAG performance against standard quantitative metrics: "
            "**Context Recall**, **Faithfulness**, **Answer Relevance**, and **Citation Precision**."
        )

        if st.button("🚀 Run Complete Benchmark Suite", type="primary"):
            with st.spinner("Evaluating retrieval quality and generation faithfulness across benchmark suite..."):
                bench_results = evaluator.run_benchmark(retriever=retriever, generator=generator)

            summary = bench_results["summary"]

            # Display high-level metric cards
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.markdown(f'<div class="metric-card"><div class="metric-value">{summary["avg_overall_score"]:.2f}</div><div class="metric-label">Overall RAG Score</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-card"><div class="metric-value">{summary["avg_context_recall"]:.2f}</div><div class="metric-label">Context Recall</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-card"><div class="metric-value">{summary["avg_faithfulness"]:.2f}</div><div class="metric-label">Faithfulness</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="metric-card"><div class="metric-value">{summary["avg_answer_relevance"]:.2f}</div><div class="metric-label">Relevance</div></div>', unsafe_allow_html=True)
            m5.markdown(f'<div class="metric-card"><div class="metric-value">{summary["avg_citation_precision"]:.2f}</div><div class="metric-label">Citation Precision</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Detailed Evaluation Breakdown")
            df_rows = []
            for r in bench_results["detailed_results"]:
                df_rows.append({
                    "Query ID": r["id"],
                    "Domain": r["domain"],
                    "Question": r["question"],
                    "Context Recall": f"{r['context_recall']:.2f}",
                    "Faithfulness": f"{r['faithfulness']:.2f}",
                    "Answer Relevance": f"{r['answer_relevance']:.2f}",
                    "Citation Precision": f"{r['citation_precision']:.2f}",
                    "Overall Score": f"{r['overall_rag_score']:.2f}",
                })
            df = pd.DataFrame(df_rows)
            st.dataframe(df, use_container_width=True)

            # Chart breakdown
            st.subheader("Metric Distribution Across Queries")
            chart_data = pd.DataFrame({
                "Context Recall": [r["context_recall"] for r in bench_results["detailed_results"]],
                "Faithfulness": [r["faithfulness"] for r in bench_results["detailed_results"]],
                "Relevance": [r["answer_relevance"] for r in bench_results["detailed_results"]],
            }, index=[r["id"] for r in bench_results["detailed_results"]])
            st.bar_chart(chart_data)
        else:
            st.info("Click **'Run Complete Benchmark Suite'** above to test the active document and retriever on all evaluation queries.")

    # -------------------------------------------------------------
    # TAB 4: ARCHITECTURE & METHODOLOGY
    # -------------------------------------------------------------
    with tab4:
        st.subheader("System Architecture & Implementation")
        st.markdown("""
### Technical Architecture

```text
                    USER / BROWSER
                          │
                          ▼
               STREAMLIT APPLICATION (app.py)
                          │
                          ▼
                     PDF Ingestion
                          │
                 PyMuPDF / pdfplumber
                          │
                     Scanned PDF?
                       /     \\
                     Yes      No
                      ↓        ↓
                 Tesseract    Text
                      \\        /
                       \\      /
                          ↓
                 Clause-Aware Chunking
                          │
                          ▼
             Legal Clause Classifier (CUAD 41)
                          │
                          ▼
               Sentence Transformers
                          │
                      Embeddings
                          │
                  ┌───────┴────────┐
                  ↓                ↓
                FAISS             BM25
           Semantic Search    Keyword Search
                  ↓                ↓
                  └───────┬────────┘
                          ↓
                    Hybrid Fusion
                  (RRF or Convex)
                          ↓
                   Relevant Clauses
                          ↓
                     RAG Engine
                          ↓
                    Multi-LLM /
                  Extractive NLP
                          ↓
               Plain English Answer +
               Verbatim Source Quote
```

### Core Pipeline Capabilities
1. **Document Ingestion**: High-throughput parsing via PyMuPDF with page-level provenance tracking and Tesseract OCR fallback for scanned contracts.
2. **Clause-Aware Chunking**: Regex-guided section detection identifying legal boundaries (e.g., `ARTICLE`, `SECTION`, numbered subclauses, `SCHEDULE`) while preserving section titles and page provenance.
3. **CUAD 41-Category Classification**: Automated clause categorization based on the Atticus Legal Contract Dataset (CUAD) and key insurance policy frameworks with automated risk flagging.
4. **BM25 + FAISS Hybrid Search**: Dense semantic search (Sentence Transformers `all-MiniLM-L6-v2`) combined with sparse lexical search (BM25Okapi) fused via Reciprocal Rank Fusion (RRF) or linear convex weighting.
5. **Strict Grounding & Citation**: Two-tiered response delivering both a Plain-English explanation and exact verbatim source quotes with section and page citations.
6. **RAGAS Evaluation**: Quantitative evaluation measuring Faithfulness, Answer Relevance, Context Recall, and Citation Precision.
        """)


if __name__ == "__main__":
    main()