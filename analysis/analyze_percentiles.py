#!/usr/bin/env python3
"""
Script to analyze percentiles (p50, p95, p99) for total_latency and time_to_first_token
from benchmark results JSON file.
"""

import json
import numpy as np
import argparse
import sys
from pathlib import Path


def calculate_percentiles(data, percentiles=[50, 95, 99]):
    """Calculate percentiles for given data."""
    if not data:
        return {f"p{p}": 0 for p in percentiles}
    
    result = {}
    for p in percentiles:
        result[f"p{p}"] = np.percentile(data, p)
    return result


def analyze_benchmark_results(json_file_path):
    """Analyze benchmark results and calculate percentiles."""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{json_file_path}'.")
        return None
    
    # Extract individual requests
    individual_requests = data.get('individual_requests', [])
    if not individual_requests:
        print("Error: No 'individual_requests' found in the JSON file.")
        return None
    
    # Extract metrics
    total_latencies = []
    time_to_first_tokens = []
    
    for request in individual_requests:
        if 'total_latency' in request:
            total_latencies.append(request['total_latency'])
        if 'time_to_first_token' in request:
            time_to_first_tokens.append(request['time_to_first_token'])
    
    if not total_latencies:
        print("Warning: No 'total_latency' values found in individual requests.")
    if not time_to_first_tokens:
        print("Warning: No 'time_to_first_token' values found in individual requests.")
    
    # Calculate percentiles
    results = {
        'total_requests': len(individual_requests),
        'total_latency_percentiles': calculate_percentiles(total_latencies),
        'time_to_first_token_percentiles': calculate_percentiles(time_to_first_tokens)
    }
    
    return results


def print_results(results):
    """Print the calculated percentiles in a formatted way."""
    if not results:
        return
    
    print("\n" + "="*60)
    print("BENCHMARK PERCENTILE ANALYSIS")
    print("="*60)
    print(f"Total Requests Analyzed: {results['total_requests']}")
    
    print("\n" + "-"*40)
    print("TOTAL LATENCY PERCENTILES (seconds)")
    print("-"*40)
    latency_percentiles = results['total_latency_percentiles']
    for metric, value in latency_percentiles.items():
        print(f"{metric.upper():<4}: {value:.6f} seconds ({value*1000:.3f} ms)")
    
    print("\n" + "-"*40)
    print("TIME TO FIRST TOKEN PERCENTILES (seconds)")
    print("-"*40)
    ttft_percentiles = results['time_to_first_token_percentiles']
    for metric, value in ttft_percentiles.items():
        print(f"{metric.upper():<4}: {value:.6f} seconds ({value*1000:.3f} ms)")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze percentiles for total_latency and time_to_first_token from benchmark JSON file"
    )
    parser.add_argument(
        'json_file', 
        help='Path to the benchmark results JSON file'
    )
    parser.add_argument(
        '--output-json', 
        help='Output results to JSON file'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.json_file).exists():
        print(f"Error: File '{args.json_file}' does not exist.")
        sys.exit(1)
    
    # Analyze the results
    results = analyze_benchmark_results(args.json_file)
    
    if results:
        # Print to console
        print_results(results)
        
        # Optionally save to JSON file
        if args.output_json:
            try:
                with open(args.output_json, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to: {args.output_json}")
            except Exception as e:
                print(f"Error saving to JSON file: {e}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
