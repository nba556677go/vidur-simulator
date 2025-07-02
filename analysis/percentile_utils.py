#!/usr/bin/env python3
"""
Utility functions for calculating percentiles from benchmark JSON files.
Can be imported as a module or run standalone.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional


def calculate_percentiles(data: List[float], percentiles: List[int] = [50, 95, 99]) -> Dict[str, float]:
    """
    Calculate percentiles for given data.
    
    Args:
        data: List of numeric values
        percentiles: List of percentile values to calculate (default: [50, 95, 99])
    
    Returns:
        Dictionary with percentile keys (e.g., 'p50', 'p95') and their values
    """
    if not data:
        return {f"p{p}": 0.0 for p in percentiles}
    
    result = {}
    for p in percentiles:
        result[f"p{p}"] = float(np.percentile(data, p))
    return result


def extract_latency_data(json_file_path: str) -> Tuple[List[float], List[float]]:
    """
    Extract total_latency and time_to_first_token data from JSON file.
    
    Args:
        json_file_path: Path to the benchmark results JSON file
    
    Returns:
        Tuple of (total_latencies, time_to_first_tokens) lists
    
    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is invalid
        KeyError: If required keys are missing from the JSON
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    individual_requests = data.get('individual_requests', [])
    if not individual_requests:
        raise KeyError("No 'individual_requests' found in the JSON file")
    
    total_latencies = []
    time_to_first_tokens = []
    
    for request in individual_requests:
        if 'total_latency' in request:
            total_latencies.append(float(request['total_latency']))
        if 'time_to_first_token' in request:
            time_to_first_tokens.append(float(request['time_to_first_token']))
    
    return total_latencies, time_to_first_tokens


def analyze_json_file(json_file_path: str, percentiles: List[int] = [50, 95, 99]) -> Dict:
    """
    Analyze benchmark JSON file and return percentile statistics.
    
    Args:
        json_file_path: Path to the benchmark results JSON file
        percentiles: List of percentiles to calculate (default: [50, 95, 99])
    
    Returns:
        Dictionary containing analysis results
    """
    total_latencies, time_to_first_tokens = extract_latency_data(json_file_path)
    
    return {
        'total_requests': len(total_latencies) if total_latencies else len(time_to_first_tokens),
        'total_latency_percentiles': calculate_percentiles(total_latencies, percentiles),
        'time_to_first_token_percentiles': calculate_percentiles(time_to_first_tokens, percentiles),
        'raw_data': {
            'total_latencies': total_latencies,
            'time_to_first_tokens': time_to_first_tokens
        }
    }


def format_results(results: Dict, show_raw_data: bool = False) -> str:
    """
    Format analysis results as a readable string.
    
    Args:
        results: Results dictionary from analyze_json_file()
        show_raw_data: Whether to include raw data in output
    
    Returns:
        Formatted string representation of results
    """
    output = []
    output.append("=" * 60)
    output.append("BENCHMARK PERCENTILE ANALYSIS")
    output.append("=" * 60)
    output.append(f"Total Requests Analyzed: {results['total_requests']}")
    
    output.append("\n" + "-" * 40)
    output.append("TOTAL LATENCY PERCENTILES")
    output.append("-" * 40)
    latency_percentiles = results['total_latency_percentiles']
    for metric, value in latency_percentiles.items():
        output.append(f"{metric.upper()}: {value:.6f} seconds ({value*1000:.3f} ms)")
    
    output.append("\n" + "-" * 40)
    output.append("TIME TO FIRST TOKEN PERCENTILES")
    output.append("-" * 40)
    ttft_percentiles = results['time_to_first_token_percentiles']
    for metric, value in ttft_percentiles.items():
        output.append(f"{metric.upper()}: {value:.6f} seconds ({value*1000:.3f} ms)")
    
    if show_raw_data:
        output.append("\n" + "-" * 40)
        output.append("RAW DATA SUMMARY")
        output.append("-" * 40)
        raw_data = results['raw_data']
        output.append(f"Total Latency Values: {len(raw_data['total_latencies'])}")
        output.append(f"TTFT Values: {len(raw_data['time_to_first_tokens'])}")
        if raw_data['total_latencies']:
            output.append(f"Total Latency Range: {min(raw_data['total_latencies']):.6f}s - {max(raw_data['total_latencies']):.6f}s")
        if raw_data['time_to_first_tokens']:
            output.append(f"TTFT Range: {min(raw_data['time_to_first_tokens']):.6f}s - {max(raw_data['time_to_first_tokens']):.6f}s")
    
    output.append("\n" + "=" * 60)
    return "\n".join(output)


def main():
    """Example usage when run as a script."""
    import sys
    
    if len(sys.argv) < 2:
        json_file_path = "benchmarks/llm/vllm/latency/vidur_test/meta-llama_Meta-Llama-3-8B/run_20250702_030923/benchmark_results.json"
        print(f"No file specified, using default: {json_file_path}")
    else:
        json_file_path = sys.argv[1]
    
    try:
        results = analyze_json_file(json_file_path)
        print(format_results(results, show_raw_data=True))
        
        # Example of accessing specific metrics programmatically
        print("\nPROGRAMMATIC ACCESS EXAMPLE:")
        print(f"P50 Total Latency: {results['total_latency_percentiles']['p50']:.3f}s")
        print(f"P99 TTFT: {results['time_to_first_token_percentiles']['p99']:.3f}s")
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
