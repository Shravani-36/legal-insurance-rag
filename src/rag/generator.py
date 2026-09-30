"""
Multi-provider LLM generation module supporting Gemini, Groq, OpenAI, Ollama,
and an intelligent offline extractive fallback engine.
"""
from typing import List, Dict, Any, Optional
import os
import re
import json
import requests
from src.rag.prompt_templates import LEGAL_RAG_SYSTEM_PROMPT, USER_QUERY_TEMPLATE
import config


class LegalRAGGenerator:
    """
    RAG Generator handling prompt formatting, context assembly, and LLM inference.
    """

    def __init__(self, provider: str = "offline", model_name: Optional[str] = None):
        self.provider = provider.lower()
        self.model_name = model_name

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved chunks into numbered context blocks with metadata.
        """
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            sec = chunk.get("section_title", "Unknown Section")
            page = chunk.get("page_number", "?")
            category = chunk.get("primary_category", "General")
            text = chunk.get("text", "").strip()

            block = (
                f"[Clause Source #{i}]\n"
                f"Section: {sec} | Page: {page} | Classification: {category}\n"
                f"Content:\n{text}\n"
            )
            context_blocks.append(block)

        return "\n----------------------------------------\n".join(context_blocks)

    def generate(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates grounded legal intelligence answer given question and retrieved clauses.
        """
        if not retrieved_chunks:
            return {
                "answer": "No relevant clauses were found in the uploaded document matching your inquiry.",
                "sources": [],
                "provider": self.provider,
                "model": "none"
            }

        context_text = self.format_context(retrieved_chunks)
        user_prompt = USER_QUERY_TEMPLATE.format(context_text=context_text, question=question)

        # Detect available API keys if currently set to offline
        active_provider = self.provider
        if active_provider == "offline":
            if os.getenv("GEMINI_API_KEY"):
                active_provider = "gemini"
            elif os.getenv("GROQ_API_KEY"):
                active_provider = "groq"
            elif os.getenv("OPENAI_API_KEY"):
                active_provider = "openai"

        try:
            if active_provider == "gemini":
                answer = self._call_gemini(user_prompt)
                model_used = self.model_name or "gemini-2.5-flash"
            elif active_provider == "groq":
                answer = self._call_groq(user_prompt)
                model_used = self.model_name or "llama-3.3-70b-versatile"
            elif active_provider == "openai":
                answer = self._call_openai(user_prompt)
                model_used = self.model_name or "gpt-4o-mini"
            elif active_provider == "ollama":
                answer = self._call_ollama(user_prompt)
                model_used = self.model_name or "llama3"
            else:
                answer = self._generate_extractive_fallback(question, retrieved_chunks)
                model_used = "extractive-nlp-engine"
                active_provider = "offline"
        except Exception as e:
            # Graceful fallback to offline engine upon network or quota error
            answer = (
                f"*[API Notice: {str(e)[:120]}... Falling back to local extractive engine]*\n\n"
                + self._generate_extractive_fallback(question, retrieved_chunks)
            )
            model_used = "extractive-nlp-engine"

        return {
            "answer": answer,
            "sources": retrieved_chunks,
            "provider": active_provider,
            "model": model_used
        }

    def _call_gemini(self, prompt: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        # Try google-genai library if installed
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            model_id = self.model_name or "gemini-2.5-flash"
            response = client.models.generate_content(
                model=model_id,
                contents=f"{LEGAL_RAG_SYSTEM_PROMPT}\n\n{prompt}"
            )
            return response.text
        except Exception:
            pass

        # Direct REST API call
        model_id = self.model_name or "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": LEGAL_RAG_SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024}
        }
        res = requests.post(url, json=payload, timeout=30)
        res.raise_for_status()
        data = res.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _call_groq(self, prompt: str) -> str:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name or "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": LEGAL_RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

    def _call_openai(self, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name or "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": LEGAL_RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

    def _call_ollama(self, prompt: str) -> str:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        url = f"{base_url}/api/generate"
        payload = {
            "model": self.model_name or "llama3",
            "prompt": f"{LEGAL_RAG_SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False
        }
        res = requests.post(url, json=payload, timeout=45)
        res.raise_for_status()
        return res.json().get("response", "")

    def _generate_extractive_fallback(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Deterministic, offline legal intelligence generator.
        Extracts key sentences, identifies numerical figures, and formats verbatim quotes.
        """
        top_chunk = retrieved_chunks[0]
        top_text = top_chunk["text"]
        top_sec = top_chunk.get("section_title", "Document Section")
        top_page = top_chunk.get("page_number", 1)
        top_cat = top_chunk.get("primary_category", "Contract Provision")
        risk_lvl = top_chunk.get("risk_level", "INFORMATIONAL")

        # Extract sentences from top chunk
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", top_text) if len(s.strip()) > 15]

        # Find key metrics (numbers, dollars, days)
        metrics = re.findall(r"\b(?:\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+\s*(?:days|months|years|percent|%))\b", top_text, re.IGNORECASE)
        metric_str = ", ".join(set(metrics)) if metrics else "Standard contractual terms"

        explanation = (
            f"Based on **{top_sec}** (Page {top_page}), this provision addresses **{top_cat}** "
            f"classified with **{risk_lvl}** impact. "
        )
        if sentences:
            explanation += f"The agreement stipulates that: {sentences[0]}"

        citations_md = []
        for i, chunk in enumerate(retrieved_chunks[:3], 1):
            sec = chunk.get("section_title", "Section")
            pg = chunk.get("page_number", "?")
            txt = chunk.get("text", "")
            # Grab first 2 sentences for citation
            sub_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt) if len(s.strip()) > 10][:2]
            quote = " ".join(sub_sentences) if sub_sentences else txt[:200]
            citations_md.append(f"- **{sec}** (Page {pg})\n  > \"{quote}\"")

        citations_text = "\n".join(citations_md)

        return f"""### Plain English Explanation
{explanation}

### Key Terms & Deadlines
- **Primary Classification**: {top_cat} ({risk_lvl} Risk)
- **Identified Quantities / Timelines**: {metric_str}
- **Governing Section**: {top_sec} on Page {top_page}

### Supporting Clause Citations
{citations_text}
"""
