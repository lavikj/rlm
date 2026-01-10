"""
Build a question-answer dataset from NVDLA documentation for RLM evaluation.

This script:
1. Reads markdown files from nvdla-markdowns/markdown/
2. Extracts meaningful text snippets (paragraphs)
3. Uses an LLM to generate questions for each snippet
4. Saves as JSON dataset with question-answer pairs
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
from rlm.clients.openai import OpenAIClient


def extract_paragraphs(text: str, min_length: int = 100) -> List[str]:
    """
    Extract meaningful paragraphs from markdown text.
    
    Args:
        text: Raw markdown text
        min_length: Minimum character length for a paragraph to be included
        
    Returns:
        List of paragraph strings
    """
    # Remove markdown artifacts
    text = re.sub(r'<!-- CHUNK \d+ START -->', '', text)
    text = re.sub(r'<!-- pagebreak -->', '', text)
    text = re.sub(r'\*Complete document merged from \d+ chunks\*', '', text)
    text = re.sub(r'---+', '', text)
    
    # Remove image references but keep surrounding context
    text = re.sub(r'!\[Image\]\(images/[^)]+\)', '', text)
    text = re.sub(r'Fig\. \d+[^\n]*\n', '', text)
    
    # Split by double newlines to get paragraphs
    paragraphs = text.split('\n\n')
    
    # Filter and clean paragraphs
    cleaned_paragraphs = []
    for para in paragraphs:
        # Remove extra whitespace
        para = ' '.join(para.split())
        
        # Skip if too short
        if len(para) < min_length:
            continue
            
        # Skip if it's just a header (starts with #)
        if para.strip().startswith('#'):
            continue
            
        # Skip if it's just a bullet point without context
        if para.strip().startswith('-') and len(para) < 200:
            continue
            
        # Skip if it's mostly markdown formatting
        if para.count('[') > 3 or para.count('](') > 3:
            continue
            
        cleaned_paragraphs.append(para)
    
    return cleaned_paragraphs


def generate_question_for_snippet(client: OpenAIClient, snippet: str, source_file: str) -> str:
    """
    Use an LLM to generate a question for which the snippet is the answer.
    
    Args:
        client: LLM client
        snippet: Text snippet that serves as the answer
        source_file: Source file name for context
        
    Returns:
        Generated question string
    """
    prompt = f"""You are helping build a dataset to evaluate a system for that answers questions on technical documentation.

Given the following snippet from NVDLA (NVIDIA Deep Learning Accelerator) documentation, generate a natural, specific question that this snippet would answer.

Requirements:
- The question should be specific enough that this snippet provides a complete answer
- The question should be something a developer/engineer working with NVDLA would actually ask
- Use natural language (avoid overly formal or templated questions)
- Don't reference "the document" or "this section" - ask as if you're genuinely seeking this information
- The question should be answerable with the information in the snippet alone

Documentation snippet:
{snippet}

Source: {source_file}

Generate ONE question only. Output just the question text, nothing else."""

    question = client.completion(prompt)
    return question.strip()


def build_dataset(
    markdown_dir: str,
    output_file: str,
    llm_client: OpenAIClient,
    max_snippets_per_file: int = 10,
    min_paragraph_length: int = 150
) -> None:
    """
    Build the complete QA dataset.
    
    Args:
        markdown_dir: Directory containing markdown files
        output_file: Path to save the output JSON file
        llm_client: LLM client for question generation
        max_snippets_per_file: Maximum snippets to extract per file
        min_paragraph_length: Minimum length for paragraphs to include
    """
    markdown_path = Path(markdown_dir)
    markdown_files = sorted(markdown_path.glob("*.md"))
    
    print(f"Found {len(markdown_files)} markdown files")
    
    dataset = []
    total_snippets = 0
    
    for md_file in markdown_files:
        print(f"\nProcessing: {md_file.name}")
        
        # Read the file
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract paragraphs
        paragraphs = extract_paragraphs(content, min_length=min_paragraph_length)
        
        # Limit snippets per file
        selected_paragraphs = paragraphs[:max_snippets_per_file]
        print(f"  Found {len(paragraphs)} paragraphs, using {len(selected_paragraphs)}")
        
        # Generate questions for each snippet
        for i, snippet in enumerate(selected_paragraphs):
            try:
                print(f"  Generating question {i+1}/{len(selected_paragraphs)}...")
                question = generate_question_for_snippet(llm_client, snippet, md_file.name)
                
                qa_pair = {
                    "id": f"{md_file.stem}_{i}",
                    "source_file": md_file.name,
                    "question": question,
                    "answer": snippet,
                    "snippet_length": len(snippet)
                }
                
                dataset.append(qa_pair)
                total_snippets += 1
                
            except Exception as e:
                print(f"  Error generating question for snippet {i}: {e}")
                continue
    
    # Save dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Dataset creation complete!")
    print(f"Total QA pairs: {total_snippets}")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}")
    
    # Also save as JSONL for easier streaming
    jsonl_file = output_file.replace('.json', '.jsonl')
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Also saved as JSONL: {jsonl_file}")


def main():
    """Main execution function."""
    # Configuration
    MARKDOWN_DIR = "../nvdla-markdowns/markdown"
    OUTPUT_FILE = "nvdla_qa_dataset.json"
    
    # Make sure OPENAI_API_KEY is set in environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it before running this script."
        )
    
    client = OpenAIClient(
        api_key=api_key,
        model_name="gpt-5",
    )
    
    print("Building NVDLA QA Dataset")
    print("="*60)
    print(f"Source directory: {MARKDOWN_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"LLM model: {client.model_name}")
    print("="*60)
    
    # Build the dataset
    build_dataset(
        markdown_dir=MARKDOWN_DIR,
        output_file=OUTPUT_FILE,
        llm_client=client,
        max_snippets_per_file=10,  # Adjust this to control dataset size
        min_paragraph_length=150
    )


if __name__ == "__main__":
    main()

