"""
Quick test script to verify the QA dataset pipeline works correctly.

This creates a mini dataset with just 2-3 snippets and runs evaluation,
useful for verifying setup before running the full pipeline.
"""

import os
import json
from build_qa_dataset import build_dataset
from evaluate_rlm_qa import run_evaluation
from rlm.clients.openai import OpenAIClient


def main():
    """Run a quick smoke test of the QA pipeline."""
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please run: export OPENAI_API_KEY='your-key-here'")
        return
    
    print("="*60)
    print("QA Pipeline Smoke Test")
    print("="*60)
    print("\nThis will create a mini dataset and run a quick evaluation")
    print("to verify everything works correctly.\n")
    
    # Step 1: Build mini dataset
    print("Step 1: Building mini dataset (3 snippets per file max)...")
    print("-"*60)
    
    client = OpenAIClient(api_key=api_key, model_name="gpt-5")
    
    try:
        build_dataset(
            markdown_dir="../nvdla-markdowns/markdown",
            output_file="test_qa_dataset.json",
            llm_client=client,
            max_snippets_per_file=2,  # Only 2 per file for quick test
            min_paragraph_length=150
        )
    except Exception as e:
        print(f"\n❌ Dataset creation failed: {e}")
        return
    
    # Check dataset was created
    if not os.path.exists("test_qa_dataset.json"):
        print("\n❌ Dataset file not created")
        return
    
    with open("test_qa_dataset.json", 'r') as f:
        dataset = json.load(f)
    
    print(f"\n✓ Successfully created dataset with {len(dataset)} QA pairs")
    
    # Step 2: Run mini evaluation (just first 3 questions)
    print("\n\nStep 2: Running mini evaluation (first 3 questions)...")
    print("-"*60)
    
    backend = "openai"
    backend_kwargs = {
        "api_key": api_key,
        "model_name": "gpt-5",
    }
    evaluator_client = OpenAIClient(api_key=api_key, model_name="gpt-4o")
    
    try:
        metrics = run_evaluation(
            dataset_file="test_qa_dataset.json",
            markdown_dir="../nvdla-markdowns/markdown",
            backend=backend,
            backend_kwargs=backend_kwargs,
            evaluator_client=evaluator_client,
            output_file="test_evaluation_results.json",
            max_questions=3  # Only evaluate 3 questions
        )
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Success
    print("\n\n" + "="*60)
    print("✓ SMOKE TEST PASSED")
    print("="*60)
    print("\nThe QA pipeline is working correctly!")
    print("\nNext steps:")
    print("1. Review test files:")
    print("   - test_qa_dataset.json")
    print("   - test_evaluation_results.json")
    print("\n2. Run full pipeline:")
    print("   - python build_qa_dataset.py")
    print("   - python evaluate_rlm_qa.py")
    print("\n3. Clean up test files:")
    print("   - rm test_qa_dataset.json*")
    print("   - rm test_evaluation_results.json")
    print("="*60)


if __name__ == "__main__":
    main()

