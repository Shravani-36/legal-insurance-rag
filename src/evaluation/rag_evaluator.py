# -*- coding: utf-8 -*-
"""
RAG Evaluation Suite for Legal & Insurance Document Intelligence.
Computes Faithfulness, Answer Relevance, Context Recall, and Citation Precision.
"""
from typing import List, Dict, Any, Optional
import json
import re
from pathlib import Path
import config


class RAGEvaluator:
    """
    Evaluator providing RAGAS-compatible metrics for legal document intelligence.
    """

    def __init__(self, benchmark_file: Optional[Path] = None):
        self.benchmark_file = benchmark_file or config.BENCHMARK_PATH
        self.benchmark_data: List[Dict[str, Any]] = []
        if self.benchmark_file.exists():
            with open(self.benchmark_file, "r", encoding="utf-8-sig") as f:
                self.benchmark_data = json.load(f)

    @staticmethod
    def evaluate_context_recall(retrieved_chunks: List[Dict[str, Any]], target_section: str, expected_keywords: List[str]) -> float:
        """
        Measures if the ground-truth section or expected keywords were successfully retrieved.
        """
        if not retrieved_chunks:
            return 0.0

        all_retrieved_text = " ".join([
            f"{c.get('section_title', '')} {c.get('text', '')}" for c in retrieved_chunks
        ]).lower()

        section_hit = 1.0 if target_section.lower() in all_retrieved_text else 0.0

        keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in all_retrieved_text)
        keyword_score = keyword_hits / len(expected_keywords) if expected_keywords else 1.0

        return round(0.4 * section_hit + 0.6 * keyword_score, 3)

    @staticmethod
    def evaluate_faithfulness(answer: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
        """
        Measures if factual statements (numbers, dollar amounts, durations, entities) in the answer
        are present in the retrieved context clauses.
        """
        if not retrieved_chunks or not answer:
            return 0.0

        context_combined = " ".join([c.get("text", "") for c in retrieved_chunks]).lower()
        answer_clean = answer.lower()

        facts = re.findall(r"\b(?:\$\d+(?:,\d{3})*(?:\.\d+)?|\d+\s*(?:days|months|years|percent|%))\b", answer_clean)
        if not facts:
            words = [w for w in answer_clean.split() if len(w) > 5 and w.isalnum()]
            if not words:
                return 1.0
            hits = sum(1 for w in words if w in context_combined)
            return round(min(1.0, hits / len(words)), 3)

        grounded_facts = sum(1 for f in facts if f in context_combined)
        return round(grounded_facts / len(facts), 3)

    @staticmethod
    def evaluate_answer_relevance(question: str, answer: str) -> float:
        """
        Measures lexical and semantic relevance of the answer to the question query.
        """
        if not answer:
            return 0.0

        q_terms = [t for t in re.findall(r"\w+", question.lower()) if len(t) > 3]
        if not q_terms:
            return 1.0

        ans_lower = answer.lower()
        hits = sum(1 for t in q_terms if t in ans_lower)
        relevance = hits / len(q_terms)
        return round(min(1.0, max(0.2, relevance)), 3)

    @staticmethod
    def evaluate_citation_precision(answer: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
        """
        Verifies that cited section names in the answer match the retrieved chunks.
        """
        retrieved_secs = [c.get("section_title", "").lower() for c in retrieved_chunks]
        if not retrieved_secs:
            return 0.0

        candidates = re.findall(r"\*\*([^\*\n]+)\*\*", answer)
        sec_candidates = re.findall(r"(?:Section|Article|Clause|Schedule)\s+[0-9A-Za-z]+", answer, re.IGNORECASE)
        all_cites = set([c.strip().lower() for c in candidates + sec_candidates if len(c.strip()) > 3])

        if not all_cites:
            return 0.8

        valid = sum(1 for cite in all_cites if any(cite in sec or sec in cite or any(w in sec for w in cite.split() if len(w) > 4) for sec in retrieved_secs))
        return round(min(1.0, max(0.5, valid / len(all_cites))), 3)

    def evaluate_single_turn(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        target_section: str = "",
        expected_keywords: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Computes the complete suite of RAG metrics for a single QA turn.
        """
        expected_keywords = expected_keywords or []
        recall = self.evaluate_context_recall(retrieved_chunks, target_section, expected_keywords)
        faithfulness = self.evaluate_faithfulness(answer, retrieved_chunks)
        relevance = self.evaluate_answer_relevance(question, answer)
        citation = self.evaluate_citation_precision(answer, retrieved_chunks)
        rag_score = round((recall * 0.3) + (faithfulness * 0.3) + (relevance * 0.25) + (citation * 0.15), 3)

        return {
            "context_recall": recall,
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "citation_precision": citation,
            "overall_rag_score": rag_score
        }

    def run_benchmark(self, retriever: Any, generator: Any) -> Dict[str, Any]:
        """
        Runs the benchmark dataset across the retriever and generator.
        """
        results: List[Dict[str, Any]] = []

        for item in self.benchmark_data:
            q = item["question"]
            target_sec = item["target_section"]
            expected_kw = item["expected_keywords"]

            retrieved = retriever.search(q, top_k=3)
            gen_res = generator.generate(q, retrieved)
            ans = gen_res["answer"]

            metrics = self.evaluate_single_turn(
                question=q,
                answer=ans,
                retrieved_chunks=retrieved,
                target_section=target_sec,
                expected_keywords=expected_kw
            )

            result_entry = {
                "id": item["id"],
                "domain": item["domain"],
                "question": q,
                "answer_snippet": ans[:140] + "...",
                **metrics
            }
            results.append(result_entry)

        avg_recall = round(sum(r["context_recall"] for r in results) / len(results), 3) if results else 0.0
        avg_faithfulness = round(sum(r["faithfulness"] for r in results) / len(results), 3) if results else 0.0
        avg_relevance = round(sum(r["answer_relevance"] for r in results) / len(results), 3) if results else 0.0
        avg_citation = round(sum(r["citation_precision"] for r in results) / len(results), 3) if results else 0.0
        avg_overall = round(sum(r["overall_rag_score"] for r in results) / len(results), 3) if results else 0.0

        return {
            "summary": {
                "total_queries": len(results),
                "avg_context_recall": avg_recall,
                "avg_faithfulness": avg_faithfulness,
                "avg_answer_relevance": avg_relevance,
                "avg_citation_precision": avg_citation,
                "avg_overall_score": avg_overall
            },
            "detailed_results": results
        }