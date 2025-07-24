#!/usr/bin/env python3
"""
Script to compare VLLM benchmark and Vidur simulation results for P50, P90, and P99 latencies.
It generates a CSV with the results and then creates plots.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

def plot_results(df, output_dir):
    """
    Generates and saves plots based on the comparison results.
    This function is robust to single QPS entries and logs warnings for non-finite values.

    Args:
        df (pd.DataFrame): The DataFrame containing the comparison results.
        output_dir (str): The directory where the plots will be saved.
    """
    print("\nGenerating plots...")
    
    plot_df = df.copy()
    x_labels = plot_df['QPS'].astype(str)
    x = np.arange(len(x_labels))
    num_qps = len(x_labels)

    # --- Plot 1: Latency Ratio Comparison ---
    fig, axes = plt.subplots(3, 1, figsize=(15, 22))
    fig.suptitle('Comparison of Latency Ratios for different QPS', fontsize=20, y=0.995)
    percentiles = ['P50', 'P90', 'P99']
    bar_width_latency = 0.8 / 2 if num_qps > 1 else 0.4
    bar_label_fontsize = 12 # Fontsize for the new labels

    for i, p in enumerate(percentiles):
        ax = axes[i]
        
        for col_name_template in ['VLLM_Latency_Ratio_{}', 'R1TP1PP1_vs_VLLM_Min_{}']:
            col_name = col_name_template.format(p)
            if col_name in plot_df.columns:
                plot_df[col_name] = pd.to_numeric(plot_df[col_name], errors='coerce')
                if not np.isfinite(plot_df[col_name]).all():
                    invalid_qps = plot_df['QPS'][~np.isfinite(plot_df[col_name])]
                    warnings.warn(f"\n[!] Warning: Non-finite values in '{col_name}' for QPS: {list(invalid_qps)}. Skipped in plot.")
            else:
                warnings.warn(f"\n[!] Warning: Column '{col_name}' not found. Skipping.")
                continue

        vllm_latency_ratio = plot_df.get(f'VLLM_Latency_Ratio_{p}', pd.Series(np.nan, index=plot_df.index))
        r1tp1pp1_vs_vllm_min = plot_df.get(f'R1TP1PP1_vs_VLLM_Min_{p}', pd.Series(np.nan, index=plot_df.index))

        bar1 = ax.bar(x - bar_width_latency/2, vllm_latency_ratio, bar_width_latency, label='Vidur_vs_VLLM_Min')
        bar2 = ax.bar(x + bar_width_latency/2, r1tp1pp1_vs_vllm_min, bar_width_latency, label='R1TP1PP1_vs_VLLM_Min')

        # --- ADDED THIS SECTION FOR LABELS ---
        ax.bar_label(bar1, fmt='%.2f', padding=3, fontsize=bar_label_fontsize)
        ax.bar_label(bar2, fmt='%.2f', padding=3, fontsize=bar_label_fontsize)
        # --- END OF ADDITION ---

        ax.set_title(f'{p} Latency Comparison', fontsize=16)
        ax.set_ylabel('Latency Ratio', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='both', which='major', labelsize=14)
        if num_qps == 1:
            ax.set_xlim(-1, 1)
        if ax.get_ylim()[1] > 0:
            ax.set_ylim(top=ax.get_ylim()[1] * 1.25) # Increased padding for labels
        ax.legend(fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.xlabel('QPS (Queries Per Second)', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    latency_plot_path = os.path.join(output_dir, 'latency_comparison_by_qps_detailed.png')
    plt.savefig(latency_plot_path)
    plt.close(fig)
    print(f"Latency comparison plot saved to: {latency_plot_path}")

    # --- Plot 2: Prediction Error Comparison ---
    fig, ax = plt.subplots(figsize=(14, 9))
    bar_width_error = 0.25

    error_cols = ['Avg_Pred_Error_P50', 'Avg_Pred_Error_P90', 'Avg_Pred_Error_P99']
    for col_name in error_cols:
        if col_name in plot_df.columns:
            plot_df[col_name] = pd.to_numeric(plot_df[col_name], errors='coerce')
            if not np.isfinite(plot_df[col_name]).all():
                invalid_qps = plot_df['QPS'][~np.isfinite(plot_df[col_name])]
                warnings.warn(f"\n[!] Warning: Non-finite values in '{col_name}' for QPS: {list(invalid_qps)}. Skipped in plot.")
        else:
             warnings.warn(f"\n[!] Warning: Column '{col_name}' not found. Skipping.")

    error_p50 = plot_df.get('Avg_Pred_Error_P50', pd.Series(np.nan, index=plot_df.index))
    error_p90 = plot_df.get('Avg_Pred_Error_P90', pd.Series(np.nan, index=plot_df.index))
    error_p99 = plot_df.get('Avg_Pred_Error_P99', pd.Series(np.nan, index=plot_df.index))
    
    bar1 = ax.bar(x - bar_width_error, error_p50, bar_width_error, label='P50 Error')
    bar2 = ax.bar(x, error_p90, bar_width_error, label='P90 Error')
    bar3 = ax.bar(x + bar_width_error, error_p99, bar_width_error, label='P99 Error')

    ax.bar_label(bar1, fmt='%.2f%%', padding=3, fontsize=10)
    ax.bar_label(bar2, fmt='%.2f%%', padding=3, fontsize=10)
    ax.bar_label(bar3, fmt='%.2f%%', padding=3, fontsize=10)

    ax.set_title('Average Prediction Error by QPS', fontsize=18, pad=20)
    ax.set_xlabel('QPS (Queries Per Second)', fontsize=14)
    ax.set_ylabel('Average Prediction Error (%)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.axhline(0, color='grey', linewidth=0.8)
    
    valid_errors = plot_df[error_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if not valid_errors.empty:
        min_val = valid_errors.min().min()
        max_val = valid_errors.max().max()
        bottom_limit = min(0, min_val * 1.1)
        top_limit = max(0, max_val * 1.1)
        ax.set_ylim(bottom_limit, top_limit)
    
    ax.legend(fontsize=12)
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    error_plot_path = os.path.join(output_dir, 'prediction_error_by_qps.png')
    plt.savefig(error_plot_path)
    plt.close(fig)
    print(f"Prediction error plot saved to: {error_plot_path}")


def main():
    # This main function is IDENTICAL to your original script
    vidur_profile = "a100_default_model_reprofiled"
    vidur_compute_profile = "a100_p4d"
    vidur_network_device = "a100_dgx"
    vllm_device = "a100_p4d"
    vidur_base_dir = f"vidur_results/{vidur_profile}/compute_{vidur_compute_profile}/network_{vidur_network_device}/chunk8192"
    vllm_bench_base_dir = f"vllm_bench_results/{vllm_device}/nprompt150"

    qps_values = [2, 5, 8]
    #qps_values.extend([15,20,40])
    #map qps with float
    qps_dir_map = {qps: str(qps) for qps in qps_values}
    #append qps_values with 2,5,8,..., 2.0,5.0,8.0... based on qps_values. d
    
    vllm_qps_dir_map = { 0.25: "0.25", 0.5: "0.5", 2: "2", 5:"5",  8: "8", 15: "15", 25: "25" }
    vidur_qps_dir_map = { 0.25: "0.25", 0.5: "0.5", 2: "2.0", 5:"5.0", 8: "8.0", 15: "15.0", 25: "25.0" }
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results = []
    
    for qps in qps_values:
        vllm_qps_dir = qps_dir_map.get(qps, str(qps))
        vidur_qps_dir = qps_dir_map.get(qps, str(qps))
        #vllm_qps_dir = vllm_qps_dir_map.get(qps, str(qps))
        #vidur_qps_dir = vidur_qps_dir_map.get(qps, str(qps))
        
        vllm_dir = os.path.join(base_dir, f"{vllm_bench_base_dir}/qps{vllm_qps_dir}")
        if not os.path.exists(vllm_dir):
            print(f"Directory not found: {vllm_dir}")
            continue
            
        vllm_files = [f for f in os.listdir(vllm_dir) if f.startswith("summary_")]
        if not vllm_files:
            print(f"No VLLM summary files found in {vllm_dir}")
            continue
        vllm_path = os.path.join(vllm_dir, vllm_files[0])
        vllm_df = pd.read_csv(vllm_path)
        
        vidur_dir = os.path.join(base_dir, f"{vidur_base_dir}/qps{vidur_qps_dir}")
        if not os.path.exists(vidur_dir):
            print(f"Directory not found: {vidur_dir}")
            continue
            
        vidur_files = [f for f in os.listdir(vidur_dir) if f.startswith("summary_")]
        if not vidur_files:
            print(f"No Vidur summary files found in {vidur_dir}")
            continue
        
        vidur_path = os.path.join(vidur_dir, vidur_files[0])
        vidur_df = pd.read_csv(vidur_path)
        
        vllm_min_p50_config = vllm_df.loc[vllm_df['Total_Gen_P50'].idxmin()]
        vllm_min_p90_config = vllm_df.loc[vllm_df['Total_Gen_P90'].idxmin()]
        vllm_min_p99_config = vllm_df.loc[vllm_df['Total_Gen_P99'].idxmin()]
        
        vidur_min_p50_config = vidur_df.loc[vidur_df['Exec_P50'].idxmin()]
        vidur_min_p90_config = vidur_df.loc[vidur_df['Exec_P90'].idxmin()]
        vidur_min_p99_config = vidur_df.loc[vidur_df['Exec_P99'].idxmin()]
        
        vllm_with_vidur_p50_config = vllm_df[(vllm_df['Num_Replicas'] == vidur_min_p50_config['Num_Replicas']) & (vllm_df['Tensor_Parallel'] == vidur_min_p50_config['Tensor_Parallel']) & (vllm_df['Pipeline_Parallel'] == vidur_min_p50_config['Pipeline_Parallel'])]
        vllm_with_vidur_p90_config = vllm_df[(vllm_df['Num_Replicas'] == vidur_min_p90_config['Num_Replicas']) & (vllm_df['Tensor_Parallel'] == vidur_min_p90_config['Tensor_Parallel']) & (vllm_df['Pipeline_Parallel'] == vidur_min_p90_config['Pipeline_Parallel'])]
        vllm_with_vidur_p99_config = vllm_df[(vllm_df['Num_Replicas'] == vidur_min_p99_config['Num_Replicas']) & (vllm_df['Tensor_Parallel'] == vidur_min_p99_config['Tensor_Parallel']) & (vllm_df['Pipeline_Parallel'] == vidur_min_p99_config['Pipeline_Parallel'])]
        
        vllm_using_vidur_p50_config_p50 = np.inf
        vllm_using_vidur_p90_config_p90 = np.inf
        vllm_using_vidur_p99_config_p99 = np.inf
        
        if not vllm_with_vidur_p50_config.empty:
            vllm_using_vidur_p50_config_p50 = vllm_with_vidur_p50_config['Total_Gen_P50'].values[0] / 1000.0
        if not vllm_with_vidur_p90_config.empty:
            vllm_using_vidur_p90_config_p90 = vllm_with_vidur_p90_config['Total_Gen_P90'].values[0] / 1000.0
        if not vllm_with_vidur_p99_config.empty:
            vllm_using_vidur_p99_config_p99 = vllm_with_vidur_p99_config['Total_Gen_P99'].values[0] / 1000.0
            
        vllm_r1_tp1_pp1 = vllm_df[(vllm_df['Num_Replicas'] == 1) & (vllm_df['Tensor_Parallel'] == 1) & (vllm_df['Pipeline_Parallel'] == 1)]
        
        vllm_r1_tp1_pp1_p50, vllm_r1_tp1_pp1_p90, vllm_r1_tp1_pp1_p99 = np.inf, np.inf, np.inf
        if not vllm_r1_tp1_pp1.empty:
            vllm_r1_tp1_pp1_p50 = vllm_r1_tp1_pp1['Total_Gen_P50'].values[0] / 1000.0
            vllm_r1_tp1_pp1_p90 = vllm_r1_tp1_pp1['Total_Gen_P90'].values[0] / 1000.0
            vllm_r1_tp1_pp1_p99 = vllm_r1_tp1_pp1['Total_Gen_P99'].values[0] / 1000.0
        
        vllm_p50 = vllm_min_p50_config['Total_Gen_P50'] / 1000.0
        vllm_p90 = vllm_min_p90_config['Total_Gen_P90'] / 1000.0
        vllm_p99 = vllm_min_p99_config['Total_Gen_P99'] / 1000.0
        
        vidur_p50 = vidur_min_p50_config['Exec_P50']
        vidur_p90 = vidur_min_p90_config['Exec_P90']
        vidur_p99 = vidur_min_p99_config['Exec_P99']
        
        vllm_latency_ratio_p50 = vllm_using_vidur_p50_config_p50 / vllm_p50 if vllm_p50 > 0 else np.inf
        vllm_latency_ratio_p90 = vllm_using_vidur_p90_config_p90 / vllm_p90 if vllm_p90 > 0 else np.inf
        vllm_latency_ratio_p99 = vllm_using_vidur_p99_config_p99 / vllm_p99 if vllm_p99 > 0 else np.inf
        
        r1_tp1_pp1_vs_vllm_min_p50 = vllm_r1_tp1_pp1_p50 / vllm_p50 if vllm_p50 > 0 else np.inf
        r1_tp1_pp1_vs_vllm_min_p90 = vllm_r1_tp1_pp1_p90 / vllm_p90 if vllm_p90 > 0 else np.inf
        r1_tp1_pp1_vs_vllm_min_p99 = vllm_r1_tp1_pp1_p99 / vllm_p99 if vllm_p99 > 0 else np.inf
        
        merged_df = pd.merge(vllm_df, vidur_df, on=['Num_Replicas', 'Tensor_Parallel', 'Pipeline_Parallel'], suffixes=('_vllm', '_vidur'))
        avg_pred_error_p50, avg_pred_error_p90, avg_pred_error_p99 = np.inf, np.inf, np.inf
        if not merged_df.empty:
            p50_error = ((merged_df['Exec_P50'] - (merged_df['Total_Gen_P50'] / 1000)) / (merged_df['Total_Gen_P50'] / 1000)) * 100
            p90_error = ((merged_df['Exec_P90'] - (merged_df['Total_Gen_P90'] / 1000)) / (merged_df['Total_Gen_P90'] / 1000)) * 100
            p99_error = ((merged_df['Exec_P99'] - (merged_df['Total_Gen_P99'] / 1000)) / (merged_df['Total_Gen_P99'] / 1000)) * 100
            avg_pred_error_p50 = p50_error.mean()
            avg_pred_error_p90 = p90_error.mean()
            avg_pred_error_p99 = p99_error.mean()
        
        results.append({
            'QPS': qps,
            'VLLM_P50_Config': f"R{int(vllm_min_p50_config['Num_Replicas'])}-TP{int(vllm_min_p50_config['Tensor_Parallel'])}-PP{int(vllm_min_p50_config['Pipeline_Parallel'])}",
            'VLLM_P90_Config': f"R{int(vllm_min_p90_config['Num_Replicas'])}-TP{int(vllm_min_p90_config['Tensor_Parallel'])}-PP{int(vllm_min_p90_config['Pipeline_Parallel'])}",
            'VLLM_P99_Config': f"R{int(vllm_min_p99_config['Num_Replicas'])}-TP{int(vllm_min_p99_config['Tensor_Parallel'])}-PP{int(vllm_min_p99_config['Pipeline_Parallel'])}",
            'VLLM_P50': vllm_p50, 'VLLM_P90': vllm_p90, 'VLLM_P99': vllm_p99,
            'Vidur_P50_Config': f"R{int(vidur_min_p50_config['Num_Replicas'])}-TP{int(vidur_min_p50_config['Tensor_Parallel'])}-PP{int(vidur_min_p50_config['Pipeline_Parallel'])}",
            'Vidur_P90_Config': f"R{int(vidur_min_p90_config['Num_Replicas'])}-TP{int(vidur_min_p90_config['Tensor_Parallel'])}-PP{int(vidur_min_p90_config['Pipeline_Parallel'])}",
            'Vidur_P99_Config': f"R{int(vidur_min_p99_config['Num_Replicas'])}-TP{int(vidur_min_p99_config['Tensor_Parallel'])}-PP{int(vidur_min_p99_config['Pipeline_Parallel'])}",
            'Vidur_P50': vidur_p50, 'Vidur_P90': vidur_p90, 'Vidur_P99': vidur_p99,
            'Ratio_P50': vllm_p50 / vidur_p50 if vidur_p50 > 0 else np.inf,
            'Ratio_P90': vllm_p90 / vidur_p90 if vidur_p90 > 0 else np.inf,
            'Ratio_P99': vllm_p99 / vidur_p99 if vidur_p99 > 0 else np.inf,
            'VLLM_Latency_Ratio_P50': vllm_latency_ratio_p50,
            'VLLM_Latency_Ratio_P90': vllm_latency_ratio_p90,
            'VLLM_Latency_Ratio_P99': vllm_latency_ratio_p99,
            'R1TP1PP1_vs_VLLM_Min_P50': r1_tp1_pp1_vs_vllm_min_p50,
            'R1TP1PP1_vs_VLLM_Min_P90': r1_tp1_pp1_vs_vllm_min_p90,
            'R1TP1PP1_vs_VLLM_Min_P99': r1_tp1_pp1_vs_vllm_min_p99,
            'Avg_Pred_Error_P50': avg_pred_error_p50,
            'Avg_Pred_Error_P90': avg_pred_error_p90,
            'Avg_Pred_Error_P99': avg_pred_error_p99
        })
    
    if not results:
        print("No results were generated. Exiting.")
        return

    results_df = pd.DataFrame(results)
    print("\nResults:")
    print(results_df.to_string(index=False, float_format='%.6f'))
    
    os.makedirs(vidur_base_dir, exist_ok=True)
    output_path = f"{vidur_base_dir}/all_percentiles_comparison.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    if not results_df.empty:
        plot_results(results_df, vidur_base_dir)

if __name__ == "__main__":
    main()
