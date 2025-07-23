#!/usr/bin/env python3
"""
Script to compare VLLM benchmark and Vidur simulation results by finding
the configuration with minimum P50 execution latency for each QPS.
Includes P50, P90, and P99 latencies in the comparison.
"""

import os
import pandas as pd
import numpy as np

def find_min_latency_config(df, qps, metric='P50'):
    """Find the configuration with minimum execution latency for the given metric."""
    try:
        # For VLLM, use Total_Gen_P50/P90/P99 as the execution latency
        if f'Total_Gen_{metric}' in df.columns:
            min_idx = df[f'Total_Gen_{metric}'].idxmin()
            return df.loc[min_idx]
        # For Vidur, use Exec_P50/P90/P99 as the execution latency
        elif f'Exec_{metric.lower()}' in df.columns:
            min_idx = df[f'Exec_{metric.lower()}'].idxmin()
            return df.loc[min_idx]
        else:
            print(f"Could not find {metric} column in DataFrame")
            return None
    except Exception as e:
        print(f"Error finding min latency config: {e}")
        return None

def main():
    # QPS values to analyze
    qps_values = [0.25, 0.5, 2, 8, 15, 25]
    
    # Map QPS values to directory names
    vllm_qps_dir_map = {
        0.25: "0.25",
        0.5: "0.5",
        2: "2",
        8: "8",
        15: "15",
        25: "25"
    }
    
    vidur_qps_dir_map = {
        0.25: "0.25",
        0.5: "0.5",
        2: "2.0",
        8: "8.0",
        15: "15.0",
        25: "25.0"
    }
    
    # Get the current directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Results for P50, P90, P99
    results_p50 = []
    results_p90 = []
    results_p99 = []
    
    for qps in qps_values:
        # Get directory names for this QPS
        vllm_qps_dir = vllm_qps_dir_map.get(qps, str(qps))
        vidur_qps_dir = vidur_qps_dir_map.get(qps, str(qps))
        
        # Load VLLM benchmark results
        vllm_dir = os.path.join(base_dir, f"vllm_bench_results/a100_p4d/qps{vllm_qps_dir}")
        if not os.path.exists(vllm_dir):
            print(f"Directory not found: {vllm_dir}")
            continue
            
        vllm_files = [f for f in os.listdir(vllm_dir) if f.startswith("summary_")]
        if not vllm_files:
            print(f"No VLLM summary files found in {vllm_dir}")
            continue
        
        vllm_path = os.path.join(vllm_dir, vllm_files[0])
        vllm_df = pd.read_csv(vllm_path)
        
        # Load Vidur simulation results - use only chunk8192
        vidur_dir = os.path.join(base_dir, f"vidur_results/a100_default/chunk8192/qps{vidur_qps_dir}")
        if not os.path.exists(vidur_dir):
            print(f"Directory not found: {vidur_dir}")
            continue
            
        vidur_files = [f for f in os.listdir(vidur_dir) if f.startswith("summary_")]
        if not vidur_files:
            print(f"No Vidur summary files found in {vidur_dir}")
            continue
        
        vidur_path = os.path.join(vidur_dir, vidur_files[0])
        vidur_df = pd.read_csv(vidur_path)
        
        # Find configurations with minimum P50 execution latency
        vllm_min_p50 = find_min_latency_config(vllm_df, qps, 'P50')
        vidur_min_p50 = find_min_latency_config(vidur_df, qps, 'P50')
        
        # Find configurations with minimum P90 execution latency
        vllm_min_p90 = find_min_latency_config(vllm_df, qps, 'P90')
        vidur_min_p90 = find_min_latency_config(vidur_df, qps, 'P90')
        
        # Find configurations with minimum P99 execution latency
        vllm_min_p99 = find_min_latency_config(vllm_df, qps, 'P99')
        vidur_min_p99 = find_min_latency_config(vidur_df, qps, 'P99')
        
        if vllm_min_p50 is None or vidur_min_p50 is None:
            print(f"Could not find min P50 latency config for QPS {qps}")
            continue
        
        if vllm_min_p90 is None or vidur_min_p90 is None:
            print(f"Could not find min P90 latency config for QPS {qps}")
            continue
        
        if vllm_min_p99 is None or vidur_min_p99 is None:
            print(f"Could not find min P99 latency config for QPS {qps}")
            continue
        
        # Extract relevant information for P50
        vllm_config_p50 = (
            int(vllm_min_p50['Num_Replicas']), 
            int(vllm_min_p50['Tensor_Parallel']), 
            int(vllm_min_p50['Pipeline_Parallel'])
        )
        vidur_config_p50 = (
            int(vidur_min_p50['Num_Replicas']), 
            int(vidur_min_p50['Tensor_Parallel']), 
            int(vidur_min_p50['Pipeline_Parallel'])
        )
        
        # Extract relevant information for P90
        vllm_config_p90 = (
            int(vllm_min_p90['Num_Replicas']), 
            int(vllm_min_p90['Tensor_Parallel']), 
            int(vllm_min_p90['Pipeline_Parallel'])
        )
        vidur_config_p90 = (
            int(vidur_min_p90['Num_Replicas']), 
            int(vidur_min_p90['Tensor_Parallel']), 
            int(vidur_min_p90['Pipeline_Parallel'])
        )
        
        # Extract relevant information for P99
        vllm_config_p99 = (
            int(vllm_min_p99['Num_Replicas']), 
            int(vllm_min_p99['Tensor_Parallel']), 
            int(vllm_min_p99['Pipeline_Parallel'])
        )
        vidur_config_p99 = (
            int(vidur_min_p99['Num_Replicas']), 
            int(vidur_min_p99['Tensor_Parallel']), 
            int(vidur_min_p99['Pipeline_Parallel'])
        )
        
        # Get latency values (convert from ms to s for VLLM)
        vllm_latency_p50 = vllm_min_p50['Total_Gen_P50'] / 1000.0  # ms to s
        vidur_latency_p50 = vidur_min_p50['Exec_P50']
        
        vllm_latency_p90 = vllm_min_p90['Total_Gen_P90'] / 1000.0  # ms to s
        vidur_latency_p90 = vidur_min_p90['Exec_P90']
        
        vllm_latency_p99 = vllm_min_p99['Total_Gen_P99'] / 1000.0  # ms to s
        vidur_latency_p99 = vidur_min_p99['Exec_P99']
        
        # Calculate ratios
        ratio_p50 = vllm_latency_p50 / vidur_latency_p50 if vidur_latency_p50 > 0 else float('inf')
        ratio_p90 = vllm_latency_p90 / vidur_latency_p90 if vidur_latency_p90 > 0 else float('inf')
        ratio_p99 = vllm_latency_p99 / vidur_latency_p99 if vidur_latency_p99 > 0 else float('inf')
        
        # Add to results
        results_p50.append({
            'QPS': qps,
            'VLLM_Config': f"R{vllm_config_p50[0]}-TP{vllm_config_p50[1]}-PP{vllm_config_p50[2]}",
            'VLLM_Latency': vllm_latency_p50,
            'Vidur_Config': f"R{vidur_config_p50[0]}-TP{vidur_config_p50[1]}-PP{vidur_config_p50[2]}",
            'Vidur_Latency': vidur_latency_p50,
            'Ratio': ratio_p50
        })
        
        results_p90.append({
            'QPS': qps,
            'VLLM_Config': f"R{vllm_config_p90[0]}-TP{vllm_config_p90[1]}-PP{vllm_config_p90[2]}",
            'VLLM_Latency': vllm_latency_p90,
            'Vidur_Config': f"R{vidur_config_p90[0]}-TP{vidur_config_p90[1]}-PP{vidur_config_p90[2]}",
            'Vidur_Latency': vidur_latency_p90,
            'Ratio': ratio_p90
        })
        
        results_p99.append({
            'QPS': qps,
            'VLLM_Config': f"R{vllm_config_p99[0]}-TP{vllm_config_p99[1]}-PP{vllm_config_p99[2]}",
            'VLLM_Latency': vllm_latency_p99,
            'Vidur_Config': f"R{vidur_config_p99[0]}-TP{vidur_config_p99[1]}-PP{vidur_config_p99[2]}",
            'Vidur_Latency': vidur_latency_p99,
            'Ratio': ratio_p99
        })
    
    # Create DataFrames
    df_p50 = pd.DataFrame(results_p50)
    df_p90 = pd.DataFrame(results_p90)
    df_p99 = pd.DataFrame(results_p99)
    
    # Print results
    print("\nP50 Results:")
    print(df_p50.to_string(index=False, float_format='%.6f'))
    
    print("\nP90 Results:")
    print(df_p90.to_string(index=False, float_format='%.6f'))
    
    print("\nP99 Results:")
    print(df_p99.to_string(index=False, float_format='%.6f'))
    
    # Save results to CSV
    df_p50.to_csv("min_latency_p50_comparison.csv", index=False)
    df_p90.to_csv("min_latency_p90_comparison.csv", index=False)
    df_p99.to_csv("min_latency_p99_comparison.csv", index=False)
    
    # Create a combined results DataFrame
    combined_results = []
    for qps in qps_values:
        p50_row = df_p50[df_p50['QPS'] == qps]
        p90_row = df_p90[df_p90['QPS'] == qps]
        p99_row = df_p99[df_p99['QPS'] == qps]
        
        if len(p50_row) > 0 and len(p90_row) > 0 and len(p99_row) > 0:
            combined_results.append({
                'QPS': qps,
                'VLLM_Config_P50': p50_row.iloc[0]['VLLM_Config'],
                'VLLM_Latency_P50': p50_row.iloc[0]['VLLM_Latency'],
                'Vidur_Config_P50': p50_row.iloc[0]['Vidur_Config'],
                'Vidur_Latency_P50': p50_row.iloc[0]['Vidur_Latency'],
                'Ratio_P50': p50_row.iloc[0]['Ratio'],
                'VLLM_Config_P90': p90_row.iloc[0]['VLLM_Config'],
                'VLLM_Latency_P90': p90_row.iloc[0]['VLLM_Latency'],
                'Vidur_Config_P90': p90_row.iloc[0]['Vidur_Config'],
                'Vidur_Latency_P90': p90_row.iloc[0]['Vidur_Latency'],
                'Ratio_P90': p90_row.iloc[0]['Ratio'],
                'VLLM_Config_P99': p99_row.iloc[0]['VLLM_Config'],
                'VLLM_Latency_P99': p99_row.iloc[0]['VLLM_Latency'],
                'Vidur_Config_P99': p99_row.iloc[0]['Vidur_Config'],
                'Vidur_Latency_P99': p99_row.iloc[0]['Vidur_Latency'],
                'Ratio_P99': p99_row.iloc[0]['Ratio']
            })
    
    df_combined = pd.DataFrame(combined_results)
    df_combined.to_csv("min_latency_combined_comparison.csv", index=False)
    
    print("\nResults saved to CSV files.")

if __name__ == "__main__":
    main()