#!/usr/bin/env python3
"""
RLM Timing Benchmark Script

Runs multiple queries through RLM and measures the distribution of response times.
Features:
- Progress tracking with live updates
- Checkpointing (results saved after each query)
- Graceful interruption (Ctrl+C saves progress)
- Distribution plotting

Usage:
    python benchmark_rlm_timing.py [--max-queries N] [--checkpoint-file FILE] [--resume]
"""

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path so we can import rlm
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    if shutdown_requested:
        print("\n\nForced exit. Progress has been saved.")
        sys.exit(1)
    print("\n\nShutdown requested. Finishing current query and saving progress...")
    print("(Press Ctrl+C again to force exit)")
    shutdown_requested = True


def load_documents(doc_path: str) -> dict[str, str]:
    """Load the VCS user guide document."""
    with open(doc_path, encoding="utf-8") as f:
        content = f.read()
    return {"vcs_user_guide_markdown.md": content}


def load_queries(dataset_path: str) -> list[dict]:
    """Load queries from the dataset file."""
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load existing checkpoint if it exists."""
    path = Path(checkpoint_path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "started_at": datetime.now().isoformat(),
        "completed_queries": 0,
        "results": [],
        "errors": [],
    }


def save_checkpoint(checkpoint_path: str, data: dict):
    """Save checkpoint to file."""
    data["last_updated"] = datetime.now().isoformat()
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def print_progress(current: int, total: int, last_time: float, avg_time: float):
    """Print progress bar and statistics."""
    pct = (current / total) * 100
    bar_len = 40
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    
    # Estimate remaining time
    remaining = (total - current) * avg_time if avg_time > 0 else 0
    remaining_str = f"{remaining/60:.1f}m" if remaining > 60 else f"{remaining:.0f}s"
    
    print(f"\r[{bar}] {current}/{total} ({pct:.1f}%) | Last: {last_time:.1f}s | Avg: {avg_time:.1f}s | ETA: {remaining_str}    ", end="", flush=True)


def run_single_query(documents: dict[str, str], query: str, max_iterations: int) -> dict:
    """Run a single query and return timing information."""
    from rlm.api import query_documents_detailed
    
    start_time = time.time()
    
    try:
        result = query_documents_detailed(
            documents=documents,
            query=query,
            max_iterations=max_iterations,
            verbose=False,
        )
        
        end_time = time.time()
        wall_time = end_time - start_time
        
        return {
            "success": True,
            "wall_time": wall_time,
            "execution_time": result.get("execution_time", wall_time),
            "usage": result.get("usage", {}),
            "answer_preview": result.get("answer", "")[:200] + "..." if len(result.get("answer", "")) > 200 else result.get("answer", ""),
        }
        
    except Exception as e:
        end_time = time.time()
        return {
            "success": False,
            "wall_time": end_time - start_time,
            "error": str(e),
        }


def plot_distribution(results: list[dict], output_path: str = None):
    """Plot the distribution of query times."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\nMatplotlib not installed. Skipping plot generation.")
        print("Install with: pip install matplotlib")
        return
    
    # Extract successful query times
    times = [r["wall_time"] for r in results if r.get("success", False)]
    
    if not times:
        print("\nNo successful queries to plot.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    bins = np.linspace(0, max(times) + 10, 20)
    ax1.hist(times, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.axvline(np.mean(times), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(times):.1f}s")
    ax1.axvline(np.median(times), color="orange", linestyle="--", linewidth=2, label=f"Median: {np.median(times):.1f}s")
    ax1.set_xlabel("Query Time (seconds)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of RLM Query Times")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot and individual points
    ax2 = axes[1]
    bp = ax2.boxplot(times, vert=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    
    # Scatter individual points
    x_jitter = np.random.normal(1, 0.04, len(times))
    ax2.scatter(x_jitter, times, alpha=0.5, color="steelblue", s=30)
    
    ax2.set_ylabel("Query Time (seconds)")
    ax2.set_title("Query Time Distribution")
    ax2.set_xticklabels(["RLM Queries"])
    ax2.grid(True, alpha=0.3, axis="y")
    
    # Add statistics text
    stats_text = (
        f"n = {len(times)}\n"
        f"Mean: {np.mean(times):.1f}s\n"
        f"Median: {np.median(times):.1f}s\n"
        f"Std: {np.std(times):.1f}s\n"
        f"Min: {min(times):.1f}s\n"
        f"Max: {max(times):.1f}s\n"
        f"P90: {np.percentile(times, 90):.1f}s\n"
        f"P95: {np.percentile(times, 95):.1f}s"
    )
    ax2.text(1.35, np.mean(times), stats_text, fontsize=10, verticalalignment="center",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path}")
    
    plt.show()


def print_summary(results: list[dict]):
    """Print summary statistics."""
    import numpy as np
    
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    times = [r["wall_time"] for r in successful]
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total queries:     {len(results)}")
    print(f"Successful:        {len(successful)}")
    print(f"Failed:            {len(failed)}")
    
    if times:
        print(f"\nTiming Statistics (successful queries):")
        print(f"  Mean:            {np.mean(times):.1f}s")
        print(f"  Median:          {np.median(times):.1f}s")
        print(f"  Std Dev:         {np.std(times):.1f}s")
        print(f"  Min:             {min(times):.1f}s")
        print(f"  Max:             {max(times):.1f}s")
        print(f"  P25:             {np.percentile(times, 25):.1f}s")
        print(f"  P75:             {np.percentile(times, 75):.1f}s")
        print(f"  P90:             {np.percentile(times, 90):.1f}s")
        print(f"  P95:             {np.percentile(times, 95):.1f}s")
        print(f"  Total time:      {sum(times):.1f}s ({sum(times)/60:.1f}m)")
    
    if failed:
        print(f"\nFailed queries:")
        for r in failed[:5]:
            print(f"  - {r.get('query_id', 'unknown')}: {r.get('error', 'unknown error')[:50]}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark RLM query timing distribution")
    parser.add_argument("--max-queries", type=int, default=None, 
                        help="Maximum number of queries to run (default: all)")
    parser.add_argument("--checkpoint-file", type=str, default="rlm_benchmark_checkpoint.json",
                        help="Path to checkpoint file (default: rlm_benchmark_checkpoint.json)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint")
    parser.add_argument("--max-iterations", type=int, default=30,
                        help="Max RLM iterations per query (default: 30)")
    parser.add_argument("--doc-path", type=str, 
                        default="../example-docs/Synopsys/VCS/VCS User Guide 2019.06-SP1/markdown.md",
                        help="Path to VCS user guide markdown")
    parser.add_argument("--dataset-path", type=str,
                        default="../qa_evaluation/vcs_dataset_valid.json",
                        help="Path to query dataset")
    parser.add_argument("--output-plot", type=str, default="rlm_timing_distribution.png",
                        help="Output path for the distribution plot")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting")
    
    args = parser.parse_args()
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 60)
    print("RLM Timing Benchmark")
    print("=" * 60)
    
    # Load documents
    print(f"\nLoading documents from: {args.doc_path}")
    try:
        documents = load_documents(args.doc_path)
        doc_size = sum(len(v) for v in documents.values())
        print(f"  Loaded {len(documents)} document(s), {doc_size:,} total characters")
    except FileNotFoundError:
        print(f"Error: Document not found at {args.doc_path}")
        sys.exit(1)
    
    # Load queries
    print(f"\nLoading queries from: {args.dataset_path}")
    try:
        queries = load_queries(args.dataset_path)
        print(f"  Loaded {len(queries)} queries")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {args.dataset_path}")
        sys.exit(1)
    
    # Limit queries if specified
    if args.max_queries:
        queries = queries[:args.max_queries]
        print(f"  Limited to {len(queries)} queries")
    
    # Fixed model for all queries
    fixed_model = "qwen/qwen3-235b-a22b-2507"
    
    # Load or initialize checkpoint
    checkpoint = load_checkpoint(args.checkpoint_file) if args.resume else {
        "started_at": datetime.now().isoformat(),
        "model": fixed_model,
        "max_iterations": args.max_iterations,
        "completed_queries": 0,
        "results": [],
        "errors": [],
    }
    
    start_idx = checkpoint["completed_queries"] if args.resume else 0
    
    if args.resume and start_idx > 0:
        print(f"\nResuming from query {start_idx + 1}/{len(queries)}")
        print(f"  Previous results: {len(checkpoint['results'])} successful, {len(checkpoint['errors'])} errors")
    
    print(f"\nModel: {fixed_model} (fixed, via Cerebras)")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Checkpoint file: {args.checkpoint_file}")
    print("\n" + "-" * 60)
    print("Starting benchmark... (Press Ctrl+C to stop and save progress)")
    print("-" * 60 + "\n")
    
    # Track timing for progress estimate
    all_times = [r["wall_time"] for r in checkpoint["results"]] if args.resume else []
    
    # Run queries
    for i, query_data in enumerate(queries[start_idx:], start=start_idx):
        if shutdown_requested:
            print("\n\nStopping benchmark due to user request.")
            break
        
        query_id = query_data.get("id", f"query_{i}")
        query_text = query_data["question"]
        
        # Run the query
        result = run_single_query(
            documents=documents,
            query=query_text,
            max_iterations=args.max_iterations,
        )
        
        # Add metadata
        result["query_id"] = query_id
        result["query_text"] = query_text[:100] + "..." if len(query_text) > 100 else query_text
        result["timestamp"] = datetime.now().isoformat()
        
        # Update checkpoint
        if result["success"]:
            checkpoint["results"].append(result)
            all_times.append(result["wall_time"])
        else:
            checkpoint["errors"].append(result)
        
        checkpoint["completed_queries"] = i + 1
        
        # Save checkpoint after each query
        save_checkpoint(args.checkpoint_file, checkpoint)
        
        # Print progress
        avg_time = sum(all_times) / len(all_times) if all_times else 0
        print_progress(i + 1, len(queries), result["wall_time"], avg_time)
    
    print("\n\n" + "-" * 60)
    print("Benchmark complete!")
    print("-" * 60)
    
    # Print summary
    all_results = checkpoint["results"] + checkpoint["errors"]
    print_summary(all_results)
    
    # Save final checkpoint
    checkpoint["completed_at"] = datetime.now().isoformat()
    save_checkpoint(args.checkpoint_file, checkpoint)
    print(f"\nResults saved to: {args.checkpoint_file}")
    
    # Plot distribution
    if not args.no_plot and checkpoint["results"]:
        plot_distribution(checkpoint["results"], args.output_plot)


if __name__ == "__main__":
    main()
