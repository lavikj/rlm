#!/bin/bash

# Run the complete QA dataset creation and evaluation pipeline for RLM

set -e  # Exit on error

echo "=========================================="
echo "RLM QA Evaluation Pipeline"
echo "=========================================="
echo ""

# Load .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
elif [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
fi

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable not set"
    echo ""
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    exit 1
fi

echo "✓ API key found"
echo ""

# Parse arguments
MODE=${1:-full}

case $MODE in
    test)
        echo "Running SMOKE TEST mode..."
        echo "=========================================="
        echo ""
        python test_qa_pipeline.py
        ;;
        
    build)
        echo "Running DATASET BUILD only..."
        echo "=========================================="
        echo ""
        python build_qa_dataset.py
        ;;
        
    evaluate)
        echo "Running EVALUATION only..."
        echo "=========================================="
        echo ""
        if [ ! -f "nvdla_qa_dataset.json" ]; then
            echo "❌ Error: Dataset file 'nvdla_qa_dataset.json' not found"
            echo "Please run: ./run_qa_evaluation.sh build"
            exit 1
        fi
        python evaluate_rlm_qa.py
        ;;
        
    analyze)
        echo "Running ANALYSIS only..."
        echo "=========================================="
        echo ""
        if [ ! -f "rlm_evaluation_results.json" ]; then
            echo "❌ Error: Results file 'rlm_evaluation_results.json' not found"
            echo "Please run: ./run_qa_evaluation.sh evaluate"
            exit 1
        fi
        python analyze_results.py
        ;;
        
    full)
        echo "Running FULL PIPELINE..."
        echo "=========================================="
        echo ""
        
        echo "Step 1/3: Building dataset..."
        echo "------------------------------------------"
        python build_qa_dataset.py
        echo ""
        
        echo "Step 2/3: Running evaluation..."
        echo "------------------------------------------"
        python evaluate_rlm_qa.py
        echo ""
        
        echo "Step 3/3: Analyzing results..."
        echo "------------------------------------------"
        python analyze_results.py
        echo ""
        ;;
        
    clean)
        echo "Cleaning up generated files..."
        echo "=========================================="
        echo ""
        rm -f nvdla_qa_dataset.json
        rm -f nvdla_qa_dataset.jsonl
        rm -f rlm_evaluation_results.json
        rm -f test_qa_dataset.json
        rm -f test_qa_dataset.jsonl
        rm -f test_evaluation_results.json
        echo "✓ Cleanup complete"
        ;;
        
    *)
        echo "Usage: ./run_qa_evaluation.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  test      - Run smoke test with mini dataset (default for first run)"
        echo "  build     - Build the QA dataset only"
        echo "  evaluate  - Run evaluation only (requires dataset)"
        echo "  analyze   - Analyze results only (requires evaluation results)"
        echo "  full      - Run complete pipeline: build → evaluate → analyze (default)"
        echo "  clean     - Remove all generated files"
        echo ""
        echo "Examples:"
        echo "  ./run_qa_evaluation.sh test      # Quick test"
        echo "  ./run_qa_evaluation.sh full      # Complete pipeline"
        echo "  ./run_qa_evaluation.sh analyze   # Just analyze existing results"
        echo ""
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✓ Pipeline complete!"
echo "=========================================="

