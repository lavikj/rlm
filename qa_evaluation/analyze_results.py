"""
Analyze RLM evaluation results and provide detailed insights.

This script loads evaluation results and provides:
- Performance breakdown by source file
- Common failure patterns
- Best and worst performing questions
- Detailed statistics
"""

import json
from collections import defaultdict
from typing import Dict, List, Any


def load_results(results_file: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_by_source(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Analyze performance breakdown by source file."""
    by_source = defaultdict(list)
    
    for result in results:
        source = result['source_file']
        by_source[source].append(result['score'])
    
    analysis = {}
    for source, scores in by_source.items():
        analysis[source] = {
            'count': len(scores),
            'avg_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'high_quality': sum(1 for s in scores if s >= 0.8) / len(scores)
        }
    
    return analysis


def find_outliers(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Find best and worst performing questions."""
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    return {
        'best': sorted_results[:5],
        'worst': sorted_results[-5:]
    }


def analyze_generation_time(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze generation time statistics."""
    times = [r['generation_time'] for r in results if r.get('generation_time', 0) > 0]
    
    if not times:
        return {}
    
    times.sort()
    return {
        'mean': sum(times) / len(times),
        'median': times[len(times) // 2],
        'min': min(times),
        'max': max(times),
        'p95': times[int(len(times) * 0.95)] if len(times) > 20 else max(times)
    }


def print_analysis(results_file: str):
    """Print comprehensive analysis of evaluation results."""
    
    # Load results
    data = load_results(results_file)
    metrics = data['metrics']
    results = data['results']
    
    print("="*80)
    print("RLM EVALUATION ANALYSIS")
    print("="*80)
    
    # Overall metrics
    print("\n📊 OVERALL METRICS")
    print("-"*80)
    print(f"Total Questions:          {metrics['total_questions']}")
    print(f"Average Score:            {metrics['average_score']:.3f}")
    print(f"High Quality (≥ 0.8):     {metrics['high_quality_percentage']:.1f}%")
    print(f"Acceptable (≥ 0.6):       {metrics['acceptable_percentage']:.1f}%")
    
    print("\n📈 Score Distribution:")
    for category, count in metrics['score_distribution'].items():
        percentage = (count / metrics['total_questions']) * 100
        print(f"  {category:.<25} {count:>3} ({percentage:>5.1f}%)")
    
    # Performance by source file
    print("\n\n📁 PERFORMANCE BY SOURCE FILE")
    print("-"*80)
    source_analysis = analyze_by_source(results)
    
    # Sort by average score
    sorted_sources = sorted(
        source_analysis.items(),
        key=lambda x: x[1]['avg_score'],
        reverse=True
    )
    
    print(f"{'Source File':<50} {'Count':>6} {'Avg':>6} {'High%':>7}")
    print("-"*80)
    for source, stats in sorted_sources:
        source_short = source[:47] + "..." if len(source) > 50 else source
        print(f"{source_short:<50} {stats['count']:>6} "
              f"{stats['avg_score']:>6.3f} {stats['high_quality']*100:>6.1f}%")
    
    # Generation time analysis
    print("\n\n⏱️  GENERATION TIME STATISTICS")
    print("-"*80)
    time_stats = analyze_generation_time(results)
    
    if time_stats:
        print(f"Mean:              {time_stats['mean']:.2f}s")
        print(f"Median:            {time_stats['median']:.2f}s")
        print(f"95th Percentile:   {time_stats['p95']:.2f}s")
        print(f"Min:               {time_stats['min']:.2f}s")
        print(f"Max:               {time_stats['max']:.2f}s")
    else:
        print("No generation time data available")
    
    # Best performing questions
    print("\n\n🏆 TOP 5 BEST PERFORMING QUESTIONS")
    print("-"*80)
    outliers = find_outliers(results)
    
    for i, result in enumerate(outliers['best'], 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Q: {result['question'][:100]}...")
        print(f"   Source: {result['source_file']}")
        print(f"   Evaluation: {result['explanation']}")
    
    # Worst performing questions
    print("\n\n⚠️  BOTTOM 5 WORST PERFORMING QUESTIONS")
    print("-"*80)
    
    for i, result in enumerate(outliers['worst'], 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Q: {result['question'][:100]}...")
        print(f"   Source: {result['source_file']}")
        print(f"   Evaluation: {result['explanation']}")
        print(f"   Generated: {result['generated_answer'][:150]}...")
    
    # Score threshold analysis
    print("\n\n🎯 DETAILED SCORE ANALYSIS")
    print("-"*80)
    
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    scores = [r['score'] for r in results]
    
    print(f"{'Threshold':>10} {'Count ≥':>10} {'Percentage':>12}")
    print("-"*40)
    for threshold in thresholds:
        count = sum(1 for s in scores if s >= threshold)
        percentage = (count / len(scores)) * 100
        print(f"{threshold:>10.1f} {count:>10} {percentage:>11.1f}%")
    
    # Questions with errors
    error_results = [r for r in results if 'ERROR' in r['generated_answer']]
    if error_results:
        print("\n\n❌ QUESTIONS WITH ERRORS")
        print("-"*80)
        print(f"Total errors: {len(error_results)}")
        for result in error_results[:5]:
            print(f"\n  Q: {result['question'][:80]}...")
            print(f"  Error: {result['generated_answer'][:100]}")
    
    print("\n" + "="*80)


def main():
    """Main execution function."""
    import sys
    
    results_file = "rlm_evaluation_results.json"
    
    # Allow custom results file as command line argument
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    
    try:
        print_analysis(results_file)
    except FileNotFoundError:
        print(f"Error: Results file '{results_file}' not found.")
        print("\nUsage: python analyze_results.py [results_file.json]")
        print("Default: rlm_evaluation_results.json")
        return
    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

