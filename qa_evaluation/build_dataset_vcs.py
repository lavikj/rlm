"""
Build a question-answer dataset from VCS documentation for RLM evaluation.

This script implements an improved QA dataset generation pipeline:
1. Reads markdown documentation from VCS User Guide
2. Extracts meaningful text snippets (paragraphs with technical content)
3. Uses an LLM to generate questions for each snippet
4. Uses an LLM to generate golden answers given the question and snippet
5. Uses an LLM to validate whether the QA pair makes sense
6. Saves validated QA pairs as JSON dataset

This approach produces higher quality QA pairs than simply using snippets as answers.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from rlm.clients.openai import OpenAIClient


@dataclass
class QAPair:
    """A question-answer pair with metadata."""

    id: str
    source_file: str
    snippet: str
    question: str
    answer: str
    is_valid: bool
    validation_reason: str
    snippet_length: int


def extract_paragraphs(text: str, min_length: int = 150, max_length: int = 2000) -> list[str]:
    """
    Extract meaningful paragraphs from markdown text.

    Args:
        text: Raw markdown text
        min_length: Minimum character length for a paragraph to be included
        max_length: Maximum character length for a paragraph

    Returns:
        List of paragraph strings
    """
    # Remove page markers and feedback sections
    text = re.sub(r"<!-- CHUNK \d+ START -->", "", text)
    text = re.sub(r"<!-- pagebreak -->", "", text)
    text = re.sub(r"\*Complete document merged from \d+ chunks\*", "", text)
    text = re.sub(r"^Feedback\s*$", "", text, flags=re.MULTILINE)

    # Remove table of contents patterns (lines that are mostly | and -)
    lines = text.split("\n")
    filtered_lines = []
    for line in lines:
        # Skip lines that look like table of contents entries
        if re.match(r"^\s*\|.*\|\s*[\d\-]+\s*\|?\s*$", line):
            continue
        # Skip separator lines
        if re.match(r"^\s*\|[-\s|]+\|\s*$", line):
            continue
        # Skip lines that are just page numbers or section references
        if re.match(r"^\s*[\d\-]+\s*$", line):
            continue
        filtered_lines.append(line)

    text = "\n".join(filtered_lines)

    # Split by double newlines to get paragraphs
    paragraphs = re.split(r"\n\n+", text)

    cleaned_paragraphs = []
    for para in paragraphs:
        # Remove extra whitespace
        para = " ".join(para.split())

        # Skip if too short or too long
        if len(para) < min_length or len(para) > max_length:
            continue

        # Skip if it's just a header (starts with #)
        if para.strip().startswith("#"):
            continue

        # Skip if it's mostly code (has too many backticks or special chars)
        if para.count("`") > 10 or para.count("```") > 0:
            continue

        # Skip if it's mostly markdown table syntax
        if para.count("|") > 5:
            continue

        # Skip if it looks like a list of links
        if para.count("](#") > 3:
            continue

        # Skip copyright, legal notices, and boilerplate
        skip_terms = [
            "copyright",
            "proprietary",
            "trademark",
            "license agreement",
            "third-party",
            "export control",
            "disclaimer",
            "synopsys, inc",
            "all rights reserved",
            "warranty of any kind",
            "open-source",
            "free and open-source",
            "www.synopsys.com",
            "third_party_notices",
        ]
        if any(term in para.lower() for term in skip_terms):
            continue

        # Skip if it's just navigation or feedback text
        if para.strip() in ["Feedback", "Note:", "Note"]:
            continue

        # Skip short generic statements
        if len(para) < 200 and any(
            phrase in para.lower()
            for phrase in ["see the following", "following sections", "this chapter"]
        ):
            continue

        # Skip paragraphs that are mostly bullet point lists (- or *)
        bullet_count = para.count(" - ") + para.count("- \"") + para.count("- You")
        if bullet_count > 2:
            continue

        # Skip if paragraph starts with a bullet point
        if para.strip().startswith("- ") or para.strip().startswith("* "):
            continue

        # Prefer paragraphs with technical content indicators
        technical_indicators = [
            "vcs",
            "simulation",
            "compile",
            "verilog",
            "vhdl",
            "systemverilog",
            "module",
            "signal",
            "register",
            "option",
            "command",
            "execute",
            "debug",
            "timing",
            "clock",
            "testbench",
            "assertion",
            "coverage",
        ]

        # Give slight preference to paragraphs with technical terms
        has_technical_content = any(term in para.lower() for term in technical_indicators)

        # Still include paragraphs without explicit technical terms if they're substantial
        if has_technical_content or len(para) > 250:
            cleaned_paragraphs.append(para)

    # Remove duplicates while preserving order
    seen = set()
    unique_paragraphs = []
    for para in cleaned_paragraphs:
        # Use first 100 chars as dedup key to catch near-duplicates
        key = para[:100].lower()
        if key not in seen:
            seen.add(key)
            unique_paragraphs.append(para)

    return unique_paragraphs


def generate_question(client: OpenAIClient, snippet: str, source_file: str) -> str:
    """
    Use an LLM to generate a question for which the snippet provides relevant context.

    Args:
        client: LLM client
        snippet: Text snippet that provides context for answering
        source_file: Source file name for context

    Returns:
        Generated question string
    """
    prompt = f"""You are helping build a dataset to evaluate a question-answering system for technical EDA (Electronic Design Automation) documentation.

Given the following snippet from VCS (Synopsys Verilog Compiler Simulator) documentation, generate a natural, specific question that someone working with VCS would ask, where this snippet contains information relevant to answering that question.

Requirements:
- The question should be specific and technical
- The question should be answerable with information from this snippet
- The question should be something an engineer working with VCS simulation would actually ask
- Use natural language (avoid overly formal or templated questions)
- Don't reference "the document" or "this section" - ask as if you're genuinely seeking this information
- Focus on practical usage, configuration, or understanding of VCS features

Documentation snippet:
{snippet}

Source: {source_file}

Generate ONE question only. Output just the question text, nothing else."""

    question = client.completion(prompt)
    return question.strip()


def generate_golden_answer(
    client: OpenAIClient, question: str, snippet: str, source_file: str
) -> str:
    """
    Use an LLM to generate a high-quality answer given the question and relevant snippet.

    Args:
        client: LLM client
        question: The question to answer
        snippet: Text snippet containing relevant information
        source_file: Source file name for context

    Returns:
        Generated answer string
    """
    prompt = f"""You are an expert technical writer creating golden answers for a QA evaluation dataset about VCS (Synopsys Verilog Compiler Simulator).

Given the following question and a documentation snippet that contains relevant information, write a clear, accurate, and complete answer.

Requirements:
- Using the information from the provided snippet, answer the question directly and concisely
- Write in a clear, professional tone
- Include specific technical details when relevant
- The answer should be self-contained (reader shouldn't need to see the original snippet)

Question: {question}

Documentation snippet (source: {source_file}):
{snippet}

Write the answer now. Output only the answer text, nothing else."""

    answer = client.completion(prompt)
    return answer.strip()


def validate_qa_pair(
    client: OpenAIClient, question: str, answer: str, snippet: str
) -> tuple[bool, str]:
    """
    Use an LLM to validate whether a QA pair makes sense and is high quality.

    Args:
        client: LLM client
        question: The question
        answer: The generated answer
        snippet: Original snippet for reference

    Returns:
        Tuple of (is_valid, reason)
    """
    prompt = f"""You are a quality assurance expert evaluating question-answer pairs for a technical question-answering dataset about VCS (Verilog Compiler Simulator).

Evaluate whether the following question-answer pair is valid and high-quality.

Question: {question}

Answer: {answer}

Original documentation snippet (for reference):
{snippet}

Evaluation criteria:
1. The question is clear and well-formed
2. The answer directly addresses the question
3. The answer is factually consistent with the snippet
4. The answer provides useful, actionable information
5. The question-answer pair would be valuable for evaluating a question answering system

Output your evaluation in the following JSON format only:
{{
  "is_valid": true or false,
  "reason": "Brief explanation of your decision"
}}

Output only the JSON, nothing else."""

    response = client.completion(prompt)

    # Parse JSON response
    try:
        # Try to extract JSON if wrapped in markdown
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        result = json.loads(response)
        return bool(result.get("is_valid", False)), result.get("reason", "No reason provided")
    except Exception as e:
        print(f"  Warning: Could not parse validation response: {e}")
        return False, f"Parse error: {str(e)}"


def build_dataset(
    markdown_file: str,
    output_file: str,
    llm_client: OpenAIClient,
    max_snippets: int | None = None,
    min_paragraph_length: int = 150,
    max_paragraph_length: int = 2000,
) -> None:
    """
    Build the complete QA dataset with validation.

    Args:
        markdown_file: Path to the markdown documentation file
        output_file: Path to save the output JSON file
        llm_client: LLM client for generation and validation
        max_snippets: Maximum snippets to process (None = all)
        min_paragraph_length: Minimum length for paragraphs to include
        max_paragraph_length: Maximum length for paragraphs to include
    """
    markdown_path = Path(markdown_file)
    print(f"Reading documentation from: {markdown_path}")

    # Read the file
    with open(markdown_path, encoding="utf-8") as f:
        content = f.read()

    # Extract paragraphs
    paragraphs = extract_paragraphs(
        content, min_length=min_paragraph_length, max_length=max_paragraph_length
    )
    print(f"Extracted {len(paragraphs)} candidate paragraphs")

    # Limit snippets if specified
    if max_snippets:
        paragraphs = paragraphs[:max_snippets]
        print(f"Limited to {len(paragraphs)} paragraphs")

    dataset: list[dict] = []
    valid_count = 0
    invalid_count = 0

    print("\nStarting QA pair generation...")
    print("=" * 60)

    for i, snippet in enumerate(paragraphs):
        print(f"\n[{i + 1}/{len(paragraphs)}] Processing snippet ({len(snippet)} chars)...")

        try:
            # Step 1: Generate question
            print("  Step 1: Generating question...")
            question = generate_question(llm_client, snippet, markdown_path.name)
            print(f"  Question: {question[:80]}...")

            # Step 2: Generate golden answer
            print("  Step 2: Generating golden answer...")
            answer = generate_golden_answer(llm_client, question, snippet, markdown_path.name)
            print(f"  Answer: {answer[:80]}...")

            # Step 3: Validate QA pair
            print("  Step 3: Validating QA pair...")
            is_valid, validation_reason = validate_qa_pair(llm_client, question, answer, snippet)
            print(f"  Valid: {is_valid} - {validation_reason[:60]}...")

            qa_pair = {
                "id": f"vcs_{i}",
                "source_file": markdown_path.name,
                "snippet": snippet,
                "question": question,
                "answer": answer,
                "is_valid": is_valid,
                "validation_reason": validation_reason,
                "snippet_length": len(snippet),
            }

            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1

            dataset.append(qa_pair)

        except Exception as e:
            print(f"  Error processing snippet {i}: {e}")
            continue

        # Save intermediate results every 10 items
        if (i + 1) % 10 == 0:
            save_dataset(dataset, output_file, valid_count, invalid_count)
            print(f"  [Checkpoint] Saved {len(dataset)} pairs to {output_file}")

    # Final save
    save_dataset(dataset, output_file, valid_count, invalid_count)

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET CREATION COMPLETE")
    print("=" * 60)
    print(f"Total QA pairs generated: {len(dataset)}")
    print(f"Valid pairs: {valid_count}")
    print(f"Invalid pairs: {invalid_count}")
    print(f"Validation rate: {valid_count / len(dataset) * 100:.1f}%")
    print(f"Saved to: {output_file}")
    print("=" * 60)


def save_dataset(dataset: list[dict], output_file: str, valid_count: int, invalid_count: int):
    """Save dataset to JSON and JSONL files."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save full dataset (including invalid pairs for analysis)
    full_output = {
        "metadata": {
            "total_pairs": len(dataset),
            "valid_pairs": valid_count,
            "invalid_pairs": invalid_count,
            "source": "VCS User Guide 2019.06-SP1",
        },
        "data": dataset,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    # Save valid pairs only in a format compatible with evaluate_rlm_qa.py
    valid_pairs = [
        {
            "id": item["id"],
            "source_file": item["source_file"],
            "question": item["question"],
            "answer": item["answer"],
            "snippet_length": item["snippet_length"],
        }
        for item in dataset
        if item.get("is_valid", False)
    ]

    valid_output_file = output_file.replace(".json", "_valid.json")
    with open(valid_output_file, "w", encoding="utf-8") as f:
        json.dump(valid_pairs, f, indent=2, ensure_ascii=False)

    # Also save as JSONL for easier streaming
    jsonl_file = valid_output_file.replace(".json", ".jsonl")
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in valid_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    """Main execution function."""
    load_dotenv()

    # Paths (anchored to this file)
    qa_dir = Path(__file__).resolve().parent
    repo_root = qa_dir.parent

    MARKDOWN_FILE = str(
        repo_root / "example-docs/Synopsys/VCS/VCS User Guide 2019.06-SP1/markdown.md"
    )
    OUTPUT_FILE = str(qa_dir / "vcs_dataset.json")
    MAX_SNIPPETS = None  # Set to a number to limit, None = all

    # Check for API key (supports OpenRouter or OpenAI)
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY or OPENAI_API_KEY environment variable not set. "
            "Please set one before running this script."
        )

    # Determine base URL
    if os.getenv("OPENROUTER_API_KEY"):
        base_url = "https://openrouter.ai/api/v1"
        model_name = os.getenv("QA_MODEL", "anthropic/claude-sonnet-4.5")
    else:
        base_url = None  # Use OpenAI default
        model_name = os.getenv("QA_MODEL", "gpt-4o")

    client_kwargs = {"model_name": model_name}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAIClient(**client_kwargs)

    print("VCS QA Dataset Builder (Improved)")
    print("=" * 60)
    print(f"Source file: {MARKDOWN_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"LLM model: {client.model_name}")
    print(f"Max snippets: {MAX_SNIPPETS or 'all'}")
    print("=" * 60)

    # Build the dataset
    build_dataset(
        markdown_file=MARKDOWN_FILE,
        output_file=OUTPUT_FILE,
        llm_client=client,
        max_snippets=MAX_SNIPPETS,
        min_paragraph_length=150,
        max_paragraph_length=2000,
    )


if __name__ == "__main__":
    main()
