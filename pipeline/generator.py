"""
Sends the retrieved, timestamped evidence to a local Ollama model for final
answer generation. Nothing leaves the Pi -- Ollama serves the model locally
at config.OLLAMA_HOST.
"""
import requests

import config

_PROMPT_TEMPLATE = """You are answering questions about a video using only the evidence below. \
Each piece of evidence is tagged with a timestamp (HH:MM:SS). Cite timestamps \
when relevant. If the evidence doesn't answer the question, say so plainly.

Evidence:
{context}

Question: {question}

Answer:"""


def _build_context(hits):
    lines = []
    for h in hits:
        ts = h["meta"]["timestamp_str"]
        lines.append(f"[{ts}] {h['text']}")
    return "\n".join(lines)


def generate_answer(question, hits, model=config.OLLAMA_MODEL, host=config.OLLAMA_HOST):
    context = _build_context(hits)
    prompt = _PROMPT_TEMPLATE.format(context=context, question=question)

    resp = requests.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()
