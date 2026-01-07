"""
Interactive example: Query your own documents with RLM

A simple script to test RLM with your documents.
Just modify the DOCS_PATH and YOUR_QUERY variables.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

# ============ CONFIGURATION - MODIFY THESE ============
DOCS_PATH = "./nvdla-markdowns/markdown"  # Path to your documents
YOUR_QUERY = "What is NVDLA and what are its main components?"  # Your question

# Choose your model and environment
MODEL = "gpt-5-nano"  # Options: "gpt-4o", "gpt-4-turbo-2024-04-09", etc.
ENVIRONMENT = "local"  # Options: "local", "modal", "docker"
MAX_DEPTH = 2  # How deep RLM can recurse
# ======================================================


def load_documents(path: str) -> dict[str, str]:
    """Load all text files from a directory."""
    docs = {}
    docs_path = Path(path)
    
    if not docs_path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    
    # Support markdown, txt, and other text files
    for pattern in ["*.md", "*.txt", "*.rst"]:
        for file in docs_path.glob(pattern):
            print(f"  Loading: {file.name}")
            docs[file.name] = file.read_text(encoding='utf-8')
    
    return docs


def main():
    print("="*80)
    print("RLM Long Context Query System")
    print("="*80)
    
    # Load documents
    print(f"\nLoading documents from: {DOCS_PATH}")
    docs = load_documents(DOCS_PATH)
    
    if not docs:
        print("No documents found! Check your DOCS_PATH.")
        return
    
    total_chars = sum(len(content) for content in docs.values())
    print(f"Loaded {len(docs)} files ({total_chars:,} characters total)")
    print(f"Files: {', '.join(list(docs.keys())[:3])}{'...' if len(docs) > 3 else ''}\n")
    
    # Set up RLM with logging
    logger = RLMLogger(log_dir="./logs")
    
    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": MODEL,
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        environment=ENVIRONMENT,
        max_depth=MAX_DEPTH,
        logger=logger,
        verbose=True,
    )
    
    # Create the prompt with your documents as context
    # RLM will have access to the documents via the 'context' variable
    print(f"Query: {YOUR_QUERY}\n")
    print("Processing with RLM...\n")
    
    result = rlm.completion(
        prompt=docs,  # Documents become the 'context' variable in the REPL
        root_prompt=YOUR_QUERY  # Your actual question
    )
    
    # Display results
    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(result.response)
    print("\n" + "="*80)
    print(f"\nLog file: {logger.log_file_path}")
    print("To visualize the RLM trajectory:")
    print("  cd visualizer && npm run dev")
    print("  Then open http://localhost:3001 and load the log file")


if __name__ == "__main__":
    main()

