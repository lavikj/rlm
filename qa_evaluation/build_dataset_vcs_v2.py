"""
Build an intelligent QA dataset from VCS documentation for RLM evaluation.

This improved script generates higher-quality questions through:
1. Selective paragraph filtering - only paragraphs with specific, actionable content
2. Topic extraction - understanding what each paragraph is about
3. Multi-hop question generation - questions requiring multiple source paragraphs
4. Question type diversity - factual, procedural, comparison, and synthesis questions

Question Types Generated:
- SPECIFIC: Detailed questions from high-specificity paragraphs
- COMPARISON: Questions comparing related concepts/features
- SYNTHESIS: Questions requiring information from multiple paragraphs
"""

import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from rlm.clients.openai import OpenAIClient


@dataclass
class AnnotatedParagraph:
    """A paragraph with extracted metadata."""

    id: int
    text: str
    section: str
    specificity_score: float  # 0-1, how specific/detailed is this
    topics: list[str]  # key concepts covered
    paragraph_type: str  # procedure, configuration, explanation, example, overview
    is_question_worthy: bool


@dataclass
class QAPair:
    """A question-answer pair with metadata."""

    id: str
    question_type: str  # SPECIFIC, COMPARISON, SYNTHESIS
    question: str
    answer: str
    source_paragraphs: list[int]  # IDs of source paragraphs
    is_valid: bool = True
    validation_reason: str = ""


def extract_sections(text: str) -> list[tuple[str, str]]:
    """
    Extract sections from markdown, returning (section_name, content) tuples.
    """
    # Split on headers
    sections = []
    current_section = "Introduction"
    current_content = []

    for line in text.split("\n"):
        if line.startswith("## "):
            if current_content:
                sections.append((current_section, "\n".join(current_content)))
            current_section = line[3:].strip()
            current_content = []
        elif line.startswith("# "):
            if current_content:
                sections.append((current_section, "\n".join(current_content)))
            current_section = line[2:].strip()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections.append((current_section, "\n".join(current_content)))

    return sections


def extract_paragraphs_with_sections(text: str, min_length: int = 100) -> list[dict]:
    """
    Extract paragraphs while preserving section context.
    """
    sections = extract_sections(text)
    paragraphs = []
    para_id = 0

    for section_name, section_content in sections:
        # Split section into paragraphs
        raw_paras = re.split(r"\n\n+", section_content)

        for para in raw_paras:
            para = " ".join(para.split())

            # Basic filtering
            if len(para) < min_length:
                continue
            if para.startswith("#"):
                continue
            if para.count("|") > 5:  # Skip tables
                continue
            if para.count("```") > 0:  # Skip code blocks
                continue

            # Skip boilerplate
            skip_terms = [
                "copyright",
                "proprietary",
                "trademark",
                "synopsys, inc",
                "all rights reserved",
                "www.synopsys.com",
            ]
            if any(term in para.lower() for term in skip_terms):
                continue

            paragraphs.append(
                {"id": para_id, "text": para, "section": section_name}
            )
            para_id += 1

    return paragraphs


def analyze_paragraph(client: OpenAIClient, paragraph: dict) -> AnnotatedParagraph:
    """
    Use LLM to analyze a paragraph for question-worthiness and extract metadata.
    """
    prompt = f"""Analyze this technical documentation paragraph for its suitability for generating QA test questions.

Paragraph (from section "{paragraph['section']}"):
{paragraph['text']}

Evaluate and respond in JSON format:
{{
  "specificity_score": <0.0-1.0, where 1.0 means highly specific with concrete details like commands, parameters, values, procedures>,
  "topics": [<list of 1-3 key technical concepts/features mentioned>],
  "paragraph_type": "<one of: procedure, configuration, explanation, example, overview, reference>",
  "is_question_worthy": <true if contains specific, testable information; false if generic/introductory>
}}

Criteria for question-worthiness:
- Contains specific commands, options, or parameters
- Describes a concrete procedure or configuration
- Explains specific behavior or requirements
- NOT just an overview or introduction
- NOT just a list of section references

Output only the JSON, nothing else."""

    response = client.completion(prompt)

    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        data = json.loads(response)
        return AnnotatedParagraph(
            id=paragraph["id"],
            text=paragraph["text"],
            section=paragraph["section"],
            specificity_score=float(data.get("specificity_score", 0)),
            topics=data.get("topics", []),
            paragraph_type=data.get("paragraph_type", "unknown"),
            is_question_worthy=data.get("is_question_worthy", False),
        )
    except Exception as e:
        print(f"  Warning: Could not parse analysis for paragraph {paragraph['id']}: {e}")
        return AnnotatedParagraph(
            id=paragraph["id"],
            text=paragraph["text"],
            section=paragraph["section"],
            specificity_score=0,
            topics=[],
            paragraph_type="unknown",
            is_question_worthy=False,
        )


def generate_specific_question(
    client: OpenAIClient, paragraph: AnnotatedParagraph
) -> tuple[str, str]:
    """
    Generate a specific, detailed question from a high-quality paragraph.
    Focus on specifics, not generalities.
    """
    prompt = f"""You are creating a challenging technical question for a QA evaluation dataset about VCS (Synopsys Verilog Compiler Simulator).

Source paragraph (topics: {', '.join(paragraph.topics)}):
{paragraph.text}

Generate a SPECIFIC, DETAILED question that:
1. Tests understanding of concrete details (commands, parameters, values, procedures)
2. Cannot be answered with just general knowledge
3. Requires the specific information in this paragraph
4. Would be asked by an engineer who needs to USE this feature

BAD question examples (too generic):
- "What is VCS?"
- "How do I use simulation?"
- "What are the features of VCS?"

GOOD question examples (specific):
- "What option do I use to enable transport delays instead of inertial delays?"
- "How do I specify the maximum latency for DBBIF in cycles?"
- "What happens if I set the SNPSLMD_LICENSE_FILE environment variable?"

Output format (JSON):
{{
  "question": "<specific technical question>",
  "answer": "<clear, complete answer using information from the paragraph>"
}}

Output only the JSON."""

    response = client.completion(prompt)

    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        data = json.loads(response)
        return data["question"], data["answer"]
    except Exception as e:
        raise ValueError(f"Could not parse response: {e}")


def find_related_paragraphs(
    paragraphs: list[AnnotatedParagraph],
) -> list[tuple[AnnotatedParagraph, AnnotatedParagraph]]:
    """
    Find pairs of paragraphs that share topics or are in related sections.
    These are candidates for comparison/synthesis questions.
    """
    pairs = []

    # Group by topics
    topic_to_paras: dict[str, list[AnnotatedParagraph]] = {}
    for para in paragraphs:
        for topic in para.topics:
            topic_lower = topic.lower()
            if topic_lower not in topic_to_paras:
                topic_to_paras[topic_lower] = []
            topic_to_paras[topic_lower].append(para)

    # Find pairs that share topics but are from different sections
    seen_pairs = set()
    for topic, paras in topic_to_paras.items():
        if len(paras) >= 2:
            for i, p1 in enumerate(paras):
                for p2 in paras[i + 1 :]:
                    # Prefer pairs from different sections
                    if p1.section != p2.section:
                        pair_key = tuple(sorted([p1.id, p2.id]))
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            pairs.append((p1, p2))

    # Also find pairs in same section (for comparison within a feature)
    section_to_paras: dict[str, list[AnnotatedParagraph]] = {}
    for para in paragraphs:
        if para.section not in section_to_paras:
            section_to_paras[para.section] = []
        section_to_paras[para.section].append(para)

    for section, paras in section_to_paras.items():
        if len(paras) >= 2:
            # Take some pairs from same section
            for i in range(min(3, len(paras) - 1)):
                p1, p2 = paras[i], paras[i + 1]
                pair_key = tuple(sorted([p1.id, p2.id]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pairs.append((p1, p2))

    return pairs


def generate_comparison_question(
    client: OpenAIClient, para1: AnnotatedParagraph, para2: AnnotatedParagraph
) -> tuple[str, str] | None:
    """
    Generate a comparison question that requires understanding both paragraphs.
    """
    prompt = f"""You are creating a COMPARISON question for a QA evaluation dataset about VCS (Synopsys Verilog Compiler Simulator).

The question should require understanding BOTH of these paragraphs to answer properly.

Paragraph 1 (section: {para1.section}, topics: {', '.join(para1.topics)}):
{para1.text}

Paragraph 2 (section: {para2.section}, topics: {', '.join(para2.topics)}):
{para2.text}

Generate a comparison or contrast question such as:
- "What is the difference between X and Y?"
- "When should I use X instead of Y?"
- "How does X compare to Y in terms of Z?"
- "What are the trade-offs between X and Y?"

If these paragraphs don't have a meaningful comparison, respond with: {{"skip": true}}

Otherwise, output format (JSON):
{{
  "question": "<comparison question requiring both paragraphs>",
  "answer": "<answer that synthesizes information from both paragraphs>"
}}

Output only the JSON."""

    response = client.completion(prompt)

    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        data = json.loads(response)
        if data.get("skip"):
            return None
        return data["question"], data["answer"]
    except Exception:
        return None


def generate_synthesis_question(
    client: OpenAIClient, paragraphs: list[AnnotatedParagraph]
) -> tuple[str, str] | None:
    """
    Generate a synthesis question that requires combining multiple paragraphs.
    """
    if len(paragraphs) < 2:
        return None

    para_texts = "\n\n---\n\n".join(
        [
            f"Paragraph {i+1} (section: {p.section}, topics: {', '.join(p.topics)}):\n{p.text}"
            for i, p in enumerate(paragraphs[:3])  # Max 3 paragraphs
        ]
    )

    prompt = f"""You are creating a SYNTHESIS question for a QA evaluation dataset about VCS (Synopsys Verilog Compiler Simulator).

The question should require combining information from MULTIPLE paragraphs to answer completely.

{para_texts}

Generate a synthesis question such as:
- "How would I configure VCS to achieve X while also ensuring Y?"
- "What steps do I need to take to set up X, and what are the prerequisites?"
- "If I want to do X, what related configurations should I also consider?"

The answer MUST require information from at least 2 of the paragraphs.

If these paragraphs can't form a meaningful synthesis question, respond with: {{"skip": true}}

Otherwise, output format (JSON):
{{
  "question": "<synthesis question requiring multiple sources>",
  "answer": "<comprehensive answer combining information from multiple paragraphs>"
}}

Output only the JSON."""

    response = client.completion(prompt)

    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        data = json.loads(response)
        if data.get("skip"):
            return None
        return data["question"], data["answer"]
    except Exception:
        return None


def validate_qa_pair(
    client: OpenAIClient, question: str, answer: str, question_type: str
) -> tuple[bool, str]:
    """Validate a QA pair for quality."""
    prompt = f"""Evaluate this {question_type} question-answer pair for a VCS documentation QA dataset.

Question: {question}

Answer: {answer}

Criteria:
1. Question is clear, specific, and well-formed
2. Answer directly and completely addresses the question
3. Answer contains concrete, useful information
4. The QA pair tests real understanding, not trivia

Output JSON:
{{
  "is_valid": true/false,
  "reason": "<brief explanation>"
}}

Output only the JSON."""

    response = client.completion(prompt)

    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        data = json.loads(response)
        return data.get("is_valid", False), data.get("reason", "")
    except Exception:
        return False, "Parse error"


def build_intelligent_dataset(
    markdown_file: str,
    output_file: str,
    client: OpenAIClient,
    max_paragraphs_to_analyze: int | None = None,
    target_specific_questions: int = 50,
    target_comparison_questions: int = 20,
    target_synthesis_questions: int = 15,
) -> None:
    """
    Build an intelligent QA dataset with diverse question types.
    """
    print(f"Reading documentation from: {markdown_file}")
    with open(markdown_file, encoding="utf-8") as f:
        content = f.read()

    # Extract paragraphs with section context
    print("Extracting paragraphs...")
    raw_paragraphs = extract_paragraphs_with_sections(content, min_length=100)
    print(f"Found {len(raw_paragraphs)} candidate paragraphs")

    if max_paragraphs_to_analyze:
        raw_paragraphs = raw_paragraphs[:max_paragraphs_to_analyze]
        print(f"Limited to {len(raw_paragraphs)} for analysis")

    # Phase 1: Analyze paragraphs for question-worthiness
    print("\n" + "=" * 60)
    print("PHASE 1: Analyzing paragraphs for question-worthiness")
    print("=" * 60)

    annotated_paragraphs: list[AnnotatedParagraph] = []
    for i, para in enumerate(raw_paragraphs):
        if (i + 1) % 20 == 0:
            print(f"  Analyzed {i + 1}/{len(raw_paragraphs)} paragraphs...")

        annotated = analyze_paragraph(client, para)
        annotated_paragraphs.append(annotated)

    # Filter to question-worthy paragraphs
    worthy_paragraphs = [p for p in annotated_paragraphs if p.is_question_worthy]
    high_specificity = [p for p in worthy_paragraphs if p.specificity_score >= 0.7]

    print(f"\nAnalysis complete:")
    print(f"  Total paragraphs: {len(annotated_paragraphs)}")
    print(f"  Question-worthy: {len(worthy_paragraphs)}")
    print(f"  High specificity (>=0.7): {len(high_specificity)}")

    # Phase 2: Generate SPECIFIC questions from high-quality paragraphs
    print("\n" + "=" * 60)
    print("PHASE 2: Generating SPECIFIC questions")
    print("=" * 60)

    qa_pairs: list[dict] = []

    # Prioritize high-specificity paragraphs
    specific_candidates = sorted(
        high_specificity, key=lambda p: p.specificity_score, reverse=True
    )[:target_specific_questions * 2]  # Get more candidates than needed

    specific_count = 0
    for para in specific_candidates:
        if specific_count >= target_specific_questions:
            break

        try:
            question, answer = generate_specific_question(client, para)
            is_valid, reason = validate_qa_pair(client, question, answer, "SPECIFIC")

            if is_valid:
                qa_pairs.append({
                    "id": f"specific_{specific_count}",
                    "question_type": "SPECIFIC",
                    "question": question,
                    "answer": answer,
                    "source_paragraphs": [para.id],
                    "source_sections": [para.section],
                    "topics": para.topics,
                    "is_valid": True,
                })
                specific_count += 1
                print(f"  [{specific_count}/{target_specific_questions}] Generated: {question[:60]}...")

        except Exception as e:
            print(f"  Error generating question: {e}")

    print(f"\nGenerated {specific_count} SPECIFIC questions")

    # Phase 3: Generate COMPARISON questions
    print("\n" + "=" * 60)
    print("PHASE 3: Generating COMPARISON questions")
    print("=" * 60)

    related_pairs = find_related_paragraphs(worthy_paragraphs)
    random.shuffle(related_pairs)  # Randomize to get variety
    print(f"Found {len(related_pairs)} related paragraph pairs")

    comparison_count = 0
    for para1, para2 in related_pairs:
        if comparison_count >= target_comparison_questions:
            break

        result = generate_comparison_question(client, para1, para2)
        if result:
            question, answer = result
            is_valid, reason = validate_qa_pair(client, question, answer, "COMPARISON")

            if is_valid:
                qa_pairs.append({
                    "id": f"comparison_{comparison_count}",
                    "question_type": "COMPARISON",
                    "question": question,
                    "answer": answer,
                    "source_paragraphs": [para1.id, para2.id],
                    "source_sections": [para1.section, para2.section],
                    "topics": list(set(para1.topics + para2.topics)),
                    "is_valid": True,
                })
                comparison_count += 1
                print(f"  [{comparison_count}/{target_comparison_questions}] Generated: {question[:60]}...")

    print(f"\nGenerated {comparison_count} COMPARISON questions")

    # Phase 4: Generate SYNTHESIS questions
    print("\n" + "=" * 60)
    print("PHASE 4: Generating SYNTHESIS questions")
    print("=" * 60)

    # Group paragraphs by section for synthesis
    section_groups: dict[str, list[AnnotatedParagraph]] = {}
    for para in worthy_paragraphs:
        if para.section not in section_groups:
            section_groups[para.section] = []
        section_groups[para.section].append(para)

    synthesis_candidates = [
        paras for paras in section_groups.values() if len(paras) >= 2
    ]
    random.shuffle(synthesis_candidates)

    synthesis_count = 0
    for paras in synthesis_candidates:
        if synthesis_count >= target_synthesis_questions:
            break

        # Take 2-3 paragraphs from the group
        selected = paras[: min(3, len(paras))]
        result = generate_synthesis_question(client, selected)

        if result:
            question, answer = result
            is_valid, reason = validate_qa_pair(client, question, answer, "SYNTHESIS")

            if is_valid:
                qa_pairs.append({
                    "id": f"synthesis_{synthesis_count}",
                    "question_type": "SYNTHESIS",
                    "question": question,
                    "answer": answer,
                    "source_paragraphs": [p.id for p in selected],
                    "source_sections": [selected[0].section],
                    "topics": list(set(t for p in selected for t in p.topics)),
                    "is_valid": True,
                })
                synthesis_count += 1
                print(f"  [{synthesis_count}/{target_synthesis_questions}] Generated: {question[:60]}...")

    print(f"\nGenerated {synthesis_count} SYNTHESIS questions")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Full output with metadata
    full_output = {
        "metadata": {
            "total_questions": len(qa_pairs),
            "specific_questions": specific_count,
            "comparison_questions": comparison_count,
            "synthesis_questions": synthesis_count,
            "paragraphs_analyzed": len(annotated_paragraphs),
            "question_worthy_paragraphs": len(worthy_paragraphs),
            "source": "VCS User Guide 2019.06-SP1",
        },
        "data": qa_pairs,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    # Save evaluation-compatible format (just the valid QA pairs)
    eval_format = [
        {
            "id": item["id"],
            "source_file": "markdown.md",
            "question": item["question"],
            "answer": item["answer"],
            "question_type": item["question_type"],
            "snippet_length": 0,  # Not applicable for multi-source
        }
        for item in qa_pairs
    ]

    valid_file = output_file.replace(".json", "_valid.json")
    with open(valid_file, "w", encoding="utf-8") as f:
        json.dump(eval_format, f, indent=2, ensure_ascii=False)

    jsonl_file = valid_file.replace(".json", ".jsonl")
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in eval_format:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nDataset saved to: {output_file}")
    print(f"Evaluation format: {valid_file}")
    print(f"JSONL format: {jsonl_file}")

    # Summary
    print("\n" + "=" * 60)
    print("DATASET CREATION COMPLETE")
    print("=" * 60)
    print(f"Total questions: {len(qa_pairs)}")
    print(f"  - SPECIFIC: {specific_count}")
    print(f"  - COMPARISON: {comparison_count}")
    print(f"  - SYNTHESIS: {synthesis_count}")
    print("=" * 60)


def main():
    """Main execution function."""
    load_dotenv()

    qa_dir = Path(__file__).resolve().parent
    repo_root = qa_dir.parent

    MARKDOWN_FILE = str(
        repo_root / "example-docs/Synopsys/VCS/VCS User Guide 2019.06-SP1/markdown.md"
    )
    OUTPUT_FILE = str(qa_dir / "vcs_qa_dataset_v2.json")

    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY or OPENAI_API_KEY required.")

    # Configure client
    if os.getenv("OPENROUTER_API_KEY"):
        base_url = "https://openrouter.ai/api/v1"
        model_name = os.getenv("QA_MODEL", "openai/gpt-4o")
    else:
        base_url = None
        model_name = os.getenv("QA_MODEL", "gpt-4o")

    client_kwargs = {"model_name": model_name}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAIClient(**client_kwargs)

    print("VCS Intelligent QA Dataset Builder (v2)")
    print("=" * 60)
    print(f"Source: {MARKDOWN_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Model: {client.model_name}")
    print("=" * 60)

    build_intelligent_dataset(
        markdown_file=MARKDOWN_FILE,
        output_file=OUTPUT_FILE,
        client=client,
        max_paragraphs_to_analyze=200,  # Analyze first 200 paragraphs (for testing, remove limit for full run)
        target_specific_questions=50,
        target_comparison_questions=20,
        target_synthesis_questions=15,
    )


if __name__ == "__main__":
    main()
