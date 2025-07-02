#!/usr/bin/env python3
"""
Simple script to analyze percentiles for total_latency and time_to_first_token
from benchmark results JSON file (hardcoded path for easy testing).
"""

import json
import numpy as np


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
    
    # Calculate percentiles
    results = {
        'total_requests': len(individual_requests),
        'total_latency_percentiles': calculate_percentiles(total_latencies),
        'time_to_first_token_percentiles': calculate_percentiles(time_to_first_tokens)
    }
    
    return results


def main():
    # Update this path to your JSON file
    json_file_path = "benchmarks/llm/vllm/latency/vidur_test/meta-llama_Meta-Llama-3-8B/run_20250702_030923/benchmark_results.json"
    
    print(f"Analyzing file: {json_file_path}")
    results = analyze_benchmark_results(json_file_path)
    
    if results:
        print("\n" + "="*60)
        print("BENCHMARK PERCENTILE ANALYSIS")
        print("="*60)
        print(f"Total Requests Analyzed: {results['total_requests']}")
        
        print("\n" + "-"*40)
        print("TOTAL LATENCY PERCENTILES")
        print("-"*40)
        latency_percentiles = results['total_latency_percentiles']
        for metric, value in latency_percentiles.items():
            print(f"{metric.upper()}: {value:.6f} seconds ({value*1000:.3f} ms)")
        
        print("\n" + "-"*40)
        print("TIME TO FIRST TOKEN PERCENTILES")
        print("-"*40)
        ttft_percentiles = results['time_to_first_token_percentiles']
        for metric, value in ttft_percentiles.items():
            print(f"{metric.upper()}: {value:.6f} seconds ({value*1000:.3f} ms)")
        
        print("\n" + "="*60)
        
        # Also show comparison with existing benchmark stats
        print("\nCOMPARISON WITH EXISTING BENCHMARK STATS:")
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            existing_stats = data.get('benchmark_stats', {}).get('latency_percentiles_ms', {})
            
            if 'total_generation_latency' in existing_stats:
                print("\nExisting Total Latency Stats (ms):")
                tgl = existing_stats['total_generation_latency']
                print(f"  P50: {tgl.get('total_p50', 'N/A')}")
                print(f"  P95: {tgl.get('total_p95', 'N/A')}")
                print(f"  P99: {tgl.get('total_p99', 'N/A')}")
            
            if 'time_to_first_token' in existing_stats:
                print("\nExisting TTFT Stats (ms):")
                ttft = existing_stats['time_to_first_token']
                print(f"  P50: {ttft.get('first_token_p50', 'N/A')}")
                print(f"  P95: {ttft.get('first_token_p95', 'N/A')}")
                print(f"  P99: {ttft.get('first_token_p99', 'N/A')}")
        except:
            pass


if __name__ == "__main__":
    main()
