"""
Minimal single-example RLM completion runner (verbose=True).

Edit the constants below (DATASET_FILE / DATASET_INDEX / MARKDOWN_DIR / MODEL)
and run:

  uv run python qa_evaluation/one_rlm_completion_debug.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from rlm import RLM

DATASET_FILE = Path("qa_evaluation/nvdla_qa_dataset.jsonl")
DATASET_INDEX = 0  # 0-based
MARKDOWN_DIR = Path("nvdla-markdowns/markdown")

BACKEND = "openai"
MODEL = "gpt-5"


def load_all_documentation(markdown_dir: Path) -> str:
    markdown_files = sorted(markdown_dir.glob("*.md"))
    if not markdown_files:
        raise ValueError(f"No markdown files found in {str(markdown_dir)!r}")

    chunks: list[str] = []
    for md_file in markdown_files:
        content = md_file.read_text(encoding="utf-8")
        chunks.append(f"# File: {md_file.name}\n\n{content}\n\n{'=' * 80}\n\n")
    return "\n".join(chunks)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if BACKEND == "openai" and not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    dataset = load_jsonl(DATASET_FILE)
    example = dataset[DATASET_INDEX]
    question = example["question"]

    print("=" * 80)
    print(f"id: {example.get('id')}")
    print(f"source_file: {example.get('source_file')}")
    print(f"question: {question}")
    print("=" * 80)

    print("Loading documentation context...")
    documentation = load_all_documentation(MARKDOWN_DIR)
    print(f"Loaded {len(documentation)} characters")

    rlm_prompt = f"""You are answering questions about NVDLA (NVIDIA Deep Learning Accelerator) documentation.

Use the provided documentation context to answer the following question accurately and concisely.

Question: {question}

Context available: NVDLA technical documentation

Please provide a clear, accurate answer based on the documentation."""

    backend_kwargs: dict = {"model_name": MODEL}
    if BACKEND == "openai":
        backend_kwargs["api_key"] = api_key

    rlm = RLM(backend=BACKEND, backend_kwargs=backend_kwargs, verbose=True)

    start = time.perf_counter()
    result = rlm.completion(prompt=documentation, root_prompt=rlm_prompt)
    elapsed = time.perf_counter() - start

    print("\n" + "=" * 80)
    print(f"elapsed_sec: {elapsed:.3f}")
    print(f"root_model: {result.root_model}")
    print(f"usage_summary: {result.usage_summary.to_dict()}")
    print("\nAnswer:\n")
    print(result.response)
    print("=" * 80)


if __name__ == "__main__":
    main()


