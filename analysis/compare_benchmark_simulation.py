#!/usr/bin/env python3
"""
Script to compare percentiles between benchmark results (JSON) and simulation results (CSV).
- JSON: total_latency and time_to_first_token from individual_requests
- CSV: request_e2e_time and prefill_e2e_time from vidur simulation
Includes CDF plotting functionality for visual comparison.
"""

import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Set matplotlib backend for headless environments
matplotlib.use('Agg')


def calculate_percentiles(data: List[float], percentiles: List[int] = [50, 95, 99]) -> Dict[str, float]:
    """Calculate percentiles for given data."""
    if not data:
        return {f"p{p}": 0.0 for p in percentiles}
    
    result = {}
    for p in percentiles:
        result[f"p{p}"] = float(np.percentile(data, p))
    return result


def read_benchmark_json(json_file_path: str) -> Tuple[List[float], List[float]]:
    """Read benchmark JSON file and extract latency data."""
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


def read_simulation_csv(csv_file_path: str) -> Tuple[List[float], List[float]]:
    """Read simulation CSV file and extract latency data."""
    df = pd.read_csv(csv_file_path)
    
    # Extract request_e2e_time (corresponds to total_latency)
    request_e2e_times = df['request_e2e_time'].dropna().tolist()
    
    # Extract prefill_e2e_time (corresponds to time_to_first_token)
    prefill_e2e_times = df['prefill_e2e_time'].dropna().tolist()
    
    return request_e2e_times, prefill_e2e_times


def plot_cdf(data1: List[float], data2: List[float], labels: List[str], 
             title: str, xlabel: str, output_path: Optional[str] = None):
    """Plot CDF comparison between two datasets."""
    
    plt.figure(figsize=(12, 7))
    
    # Calculate percentiles for both datasets
    p50_1, p90_1, p99_1 = np.percentile(data1, [50, 90, 99])
    p50_2, p90_2, p99_2 = np.percentile(data2, [50, 90, 99])
    
    # Calculate percentage differences
    p50_diff = ((p50_2 - p50_1) / p50_1 * 100) if p50_1 != 0 else 0
    p90_diff = ((p90_2 - p90_1) / p90_1 * 100) if p90_1 != 0 else 0
    p99_diff = ((p99_2 - p99_1) / p99_1 * 100) if p99_1 != 0 else 0
    
    # Create enhanced labels with percentage differences
    enhanced_labels = [
        f"{labels[0]} (Baseline)",
        f"{labels[1]} (P50: {p50_diff:+.1f}%, P90: {p90_diff:+.1f}%, P99: {p99_diff:+.1f}%)"
    ]
    
    # Calculate CDFs
    for i, (data, label) in enumerate(zip([data1, data2], enhanced_labels)):
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, cdf, label=label, linewidth=2)
    
    plt.xlabel(xlabel)
    plt.ylabel('Cumulative Probability')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add percentile lines
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    percentiles = [50, 90, 99]
    line_styles = ['--', ':', '-.']
    
    for i, (data, original_label) in enumerate(zip([data1, data2], labels)):
        for p, style in zip(percentiles, line_styles):
            percentile_val = np.percentile(data, p)
            plt.axvline(percentile_val, color=colors[i], linestyle=style, alpha=0.6, linewidth=1.5)
    
    # Add legend for percentile lines (only once to avoid clutter)
    for p, style in zip(percentiles, line_styles):
        plt.axvline(-1, color='gray', linestyle=style, alpha=0.7, label=f'P{p}', linewidth=1.5)
    
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"CDF plot saved to: {output_path}")
    else:
        plt.show()


def plot_all_cdfs(results: Dict, output_dir: Optional[str] = None):
    """Plot CDFs for both total latency and TTFT comparisons."""
    
    # Total Latency CDF
    bench_total = results['benchmark']['total_latency']['data']
    sim_total = results['simulation']['total_latency']['data']
    
    total_lat_path = None
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        total_lat_path = Path(output_dir) / "total_latency_cdf.png"
    
    plot_cdf(
        bench_total, sim_total,
        ['Benchmark', 'Simulation'],
        'Total Latency CDF Comparison',
        'Total Latency (seconds)',
        total_lat_path
    )
    
    # TTFT CDF
    bench_ttft = results['benchmark']['ttft']['data']
    sim_ttft = results['simulation']['ttft']['data']
    
    ttft_path = None
    if output_dir:
        ttft_path = Path(output_dir) / "ttft_cdf.png"
    
    plot_cdf(
        bench_ttft, sim_ttft,
        ['Benchmark', 'Simulation'],
        'Time to First Token CDF Comparison',
        'Time to First Token (seconds)',
        ttft_path
    )


def compare_datasets(benchmark_file: str, simulation_file: str, percentiles: List[int] = [50, 95, 99]):
    """Compare benchmark and simulation datasets."""
    
    # Read data
    print("Reading benchmark data...")
    bench_total_lat, bench_ttft = read_benchmark_json(benchmark_file)
    
    print("Reading simulation data...")
    sim_total_lat, sim_ttft = read_simulation_csv(simulation_file)
    
    # Calculate percentiles
    bench_total_percentiles = calculate_percentiles(bench_total_lat, percentiles)
    bench_ttft_percentiles = calculate_percentiles(bench_ttft, percentiles)
    sim_total_percentiles = calculate_percentiles(sim_total_lat, percentiles)
    sim_ttft_percentiles = calculate_percentiles(sim_ttft, percentiles)
    
    # Print comparison
    print("\n" + "="*80)
    print("BENCHMARK vs SIMULATION COMPARISON")
    print("="*80)
    print(f"Benchmark Requests: {len(bench_total_lat)}")
    print(f"Simulation Requests: {len(sim_total_lat)}")
    
    print("\n" + "-"*50)
    print("TOTAL LATENCY COMPARISON")
    print("-"*50)
    print(f"{'Percentile':<10} {'Benchmark (s)':<15} {'Simulation (s)':<15} {'Diff (s)':<12} {'Diff (%)':<10}")
    print("-"*50)
    
    for p in percentiles:
        p_key = f"p{p}"
        bench_val = bench_total_percentiles[p_key]
        sim_val = sim_total_percentiles[p_key]
        diff = sim_val - bench_val
        diff_pct = (diff / bench_val * 100) if bench_val != 0 else 0
        
        print(f"P{p:<9} {bench_val:<15.6f} {sim_val:<15.6f} {diff:<12.6f} {diff_pct:<10.2f}")
    
    print("\n" + "-"*50)
    print("TIME TO FIRST TOKEN COMPARISON")
    print("-"*50)
    print(f"{'Percentile':<10} {'Benchmark (s)':<15} {'Simulation (s)':<15} {'Diff (s)':<12} {'Diff (%)':<10}")
    print("-"*50)
    
    for p in percentiles:
        p_key = f"p{p}"
        bench_val = bench_ttft_percentiles[p_key]
        sim_val = sim_ttft_percentiles[p_key]
        diff = sim_val - bench_val
        diff_pct = (diff / bench_val * 100) if bench_val != 0 else 0
        
        print(f"P{p:<9} {bench_val:<15.6f} {sim_val:<15.6f} {diff:<12.6f} {diff_pct:<10.2f}")
    
    # Summary statistics
    print("\n" + "-"*50)
    print("SUMMARY STATISTICS")
    print("-"*50)
    
    print("\nTotal Latency:")
    print(f"  Benchmark - Mean: {np.mean(bench_total_lat):.6f}s, Std: {np.std(bench_total_lat):.6f}s")
    print(f"  Simulation - Mean: {np.mean(sim_total_lat):.6f}s, Std: {np.std(sim_total_lat):.6f}s")
    print(f"  Range - Benchmark: [{min(bench_total_lat):.6f}, {max(bench_total_lat):.6f}]")
    print(f"  Range - Simulation: [{min(sim_total_lat):.6f}, {max(sim_total_lat):.6f}]")
    
    print("\nTime to First Token:")
    print(f"  Benchmark - Mean: {np.mean(bench_ttft):.6f}s, Std: {np.std(bench_ttft):.6f}s")
    print(f"  Simulation - Mean: {np.mean(sim_ttft):.6f}s, Std: {np.std(sim_ttft):.6f}s")
    print(f"  Range - Benchmark: [{min(bench_ttft):.6f}, {max(bench_ttft):.6f}]")
    print(f"  Range - Simulation: [{min(sim_ttft):.6f}, {max(sim_ttft):.6f}]")
    
    print("\n" + "="*80)
    
    # Return structured results
    return {
        'benchmark': {
            'total_latency': {
                'data': bench_total_lat,
                'percentiles': bench_total_percentiles,
                'mean': np.mean(bench_total_lat),
                'std': np.std(bench_total_lat)
            },
            'ttft': {
                'data': bench_ttft,
                'percentiles': bench_ttft_percentiles,
                'mean': np.mean(bench_ttft),
                'std': np.std(bench_ttft)
            }
        },
        'simulation': {
            'total_latency': {
                'data': sim_total_lat,
                'percentiles': sim_total_percentiles,
                'mean': np.mean(sim_total_lat),
                'std': np.std(sim_total_lat)
            },
            'ttft': {
                'data': sim_ttft,
                'percentiles': sim_ttft_percentiles,
                'mean': np.mean(sim_ttft),
                'std': np.std(sim_ttft)
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare latency percentiles between benchmark JSON and simulation CSV"
    )
    parser.add_argument(
        '--benchmark', 
        default="benchmarks/llm/vllm/latency/vidur_test/meta-llama_Meta-Llama-3-8B/run_20250702_030923/benchmark_results.json",
        help='Path to benchmark JSON file'
    )
    parser.add_argument(
        '--simulation', 
        default="benchmarks/llm/vllm/latency/vidur_output/request_metrics.csv",
        help='Path to simulation CSV file'
    )
    parser.add_argument(
        '--percentiles',
        type=int,
        nargs='+',
        default=[50, 95, 99],
        help='Percentiles to calculate (default: 50 95 99)'
    )
    parser.add_argument(
        '--output-json',
        help='Output detailed results to JSON file'
    )
    parser.add_argument(
        '--plot-cdf',
        action='store_true',
        help='Generate CDF plots for visual comparison'
    )
    parser.add_argument(
        '--plot-output-dir',
        default='./plots',
        help='Directory to save CDF plots (default: ./plots)'
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    if not Path(args.benchmark).exists():
        print(f"Error: Benchmark file '{args.benchmark}' does not exist.")
        sys.exit(1)
    
    if not Path(args.simulation).exists():
        print(f"Error: Simulation file '{args.simulation}' does not exist.")
        sys.exit(1)
    
    try:
        results = compare_datasets(args.benchmark, args.simulation, args.percentiles)
        
        # Save to JSON if requested
        if args.output_json:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for dataset_name, dataset in results.items():
                json_results[dataset_name] = {}
                for metric_name, metric in dataset.items():
                    json_results[dataset_name][metric_name] = {
                        'percentiles': metric['percentiles'],
                        'mean': float(metric['mean']),
                        'std': float(metric['std']),
                        'count': len(metric['data']),
                        'min': float(min(metric['data'])),
                        'max': float(max(metric['data']))
                    }
            
            with open(args.output_json, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"\nDetailed results saved to: {args.output_json}")
        
        # Generate CDF plots if requested
        if args.plot_cdf:
            print("\nGenerating CDF plots...")
            plot_all_cdfs(results, args.plot_output_dir)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
