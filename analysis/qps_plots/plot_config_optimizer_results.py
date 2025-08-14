
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import csv
import re

def get_device_costs():
    """
    Parse the Plannable_Public EC2_US East.csv to get hourly costs for different device types
    and include GPUs per node information
    mostly network_device cost
    """
    device_costs = {
        'p4d_a100_40g_nvlink': {'cost': None, 'gpus_per_node': 8},  # p4d.24xlarge has 8 A100 GPUs
        'h100': {'cost': None, 'gpus_per_node': 8},  # p5.48xlarge has 8 H100 GPUs
        'l40s_g6e48': {'cost': None, 'gpus_per_node': 8},  # g6e.48xlarge has 8 L40S GPUs
        'a10g_g5': {'cost': None, 'gpus_per_node': 8},  # g5.48xlarge has 8 A10G GPUs
        'l4_g6': {'cost': None, 'gpus_per_node': 8},  # g6.48xlarge has 8 L4 GPUs
    }
    
    # Instance type to device mapping
    instance_to_device = {
        'p4d.24xlarge': 'p4d_a100_40g_nvlink',
        'p5.48xlarge': 'h100',
        'g6e.48xlarge': 'l40s_g6e48',
        'g5.48xlarge': 'a10g_g5',
        'g6.48xlarge': 'l4_g6'
    }
    
    csv_path = os.path.join(os.path.dirname(__file__), "Plannable_Public_EC2_US_East.csv")
    
    with open(csv_path, 'r') as f:
        # Skip the first line which contains 'sep=;'
        f.readline()
        
        # The file uses semicolons as separators
        reader = csv.DictReader(f, delimiter=';')
        
        for row in reader:
            instance_type = row.get('Instance Type', '').strip()
            #print(f'Checking instance type: {instance_type}')
            
            if instance_type in instance_to_device:
                device = instance_to_device[instance_type]
                
                # Get the 2025 IMR cost (hourly rate)
                try:
                    imr_cost_str = row.get('2025 IMR', '0').strip()
                    if imr_cost_str:  # Make sure it's not empty
                        cost = float(imr_cost_str)
                        device_costs[device]['cost'] = cost
                        print(f'Found cost for {device} ({instance_type}): ${cost:.2f}')
                except (ValueError, TypeError) as e:
                    print(f'Error parsing cost for {instance_type}: {e}')
    
    # Print the costs and set fallback values if needed
    print("Device information:")
    fallback_costs = {
        'a100': 4.73,  # p4d.24xlarge approximate cost
        'h100': 11.85,  # p5.48xlarge approximate cost
        'l40s_g6e': 4.68,  # g6e.48xlarge approximate cost
        #'a10g_g5': 3.89,  # g5.48xlarge approximate cost
        #'l4_g6': 2.95  # g6.48xlarge approximate cost
    }
    
    for device, info in device_costs.items():
        # Use fallback values if costs weren't found
        if info['cost'] is None:
            info['cost'] = fallback_costs.get(device, 1.0)
            print(f"  {device}: ${info['cost']:.2f} per hour (FALLBACK VALUE), {info['gpus_per_node']} GPUs per node")
        else:
            print(f"  {device}: ${info['cost']:.2f} per hour, {info['gpus_per_node']} GPUs per node")
        
    return device_costs

# Directory containing the optimizer output
#CONFIG_DIR = "/home/ec2-user/vidur-simulator/config_optimizer_output_r8_r16/runs"
CONFIG_DIRS = [
    "/home/ec2-user/vidur-simulator/config_optimizer_output_qwen1.5_r1_r2_r4_r8_r16_a10g_g5/runs",
    "/home/ec2-user/vidur-simulator/config_optimizer_output_qwen1.5_r1_r2_r4_r8_r16_l4_g6/runs",
    "/home/ec2-user/vidur-simulator/config_optimizer_output_llama_8b_r1_r2_r4_r8_r16_a10g_g5/runs",
    "/home/ec2-user/vidur-simulator/config_optimizer_output_llama_8b_r1_r2_r4_r8_r16_l4_g6/runs",
    "/home/ec2-user/vidur-simulator/config_optimizer_output_llama_8b_r1_r2_r4_r8_r16_a100_p4d/runs",
    "/home/ec2-user/vidur-simulator/config_optimizer_output_llama_8b_r1_r2_r4_r8_r16_h100_p5/runs",

]
# Data structure to hold results
results = []

# Get device costs
device_costs = get_device_costs()

# SLO limit example
slo_limit = 0.25  # 200ms example for TTFT
exec_slo = 7.8  # slo for total execution time
inter_token_slo = 0.015  # 8ms in seconds

# Walk through all config directories
for config_dir in CONFIG_DIRS:
    base_dir = os.path.expanduser(config_dir)

    # Walk through all run directories in each config dir
    for run_dir in os.listdir(base_dir):
        run_path = os.path.join(base_dir, run_dir)
        if not os.path.isdir(run_path):
            continue
        
        # Process each QPS directory
        for qps_dir in os.listdir(run_path):
            if not re.match(r'^r\d+_q', qps_dir):
                continue
                
            # Extract QPS value from directory name
            try:
                qps = float(qps_dir.split('_q')[1])
            except:
                continue
                
            qps_path = os.path.join(run_path, qps_dir)
            
            # Find the timestamped directory
            timestamp_dirs = [d for d in os.listdir(qps_path) if os.path.isdir(os.path.join(qps_path, d))]
            if not timestamp_dirs:
                continue
                
            # Use the first timestamped directory
            timestamp_path = os.path.join(qps_path, timestamp_dirs[0])
            
            # Check if config.json and request_metrics.csv exist
            config_path = os.path.join(timestamp_path, "config.json")
            metrics_path = os.path.join(timestamp_path, "request_metrics.csv")
            
            if not (os.path.exists(config_path) and os.path.exists(metrics_path)):
                continue
                
            # Parse config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract relevant config details
            replica_config = config.get('cluster_config', {}).get('replica_config', {})
            num_replicas = config.get('cluster_config', {}).get('num_replicas', 0)
            tensor_parallel_size = replica_config.get('tensor_parallel_size', 0)
            num_pipeline_stages = replica_config.get('num_pipeline_stages', 0)
            device = replica_config.get('device', 'Unknown')
            network_device = replica_config.get('network_device', 'Unknown')
            
            # Extract token lengths (prefill and decode)
            request_generator_config = config.get('request_generator_config', {})
            length_generator_config = request_generator_config.get('length_generator_config', {})
            prefill_tokens = length_generator_config.get('prefill_tokens', 0)
            decode_tokens = length_generator_config.get('decode_tokens', 0)
            
            # Get scheduler info and related parameters
            scheduler_config = config.get('scheduler_config', {})
            replica_scheduler_config = config.get('replica_scheduler_config', {})
            cluster_config = config.get('cluster_config', {})
            
            # Look for scheduler_type - need to check in several possible locations
            scheduler_type = None
            
            # 1. Check in scheduler_config
            if scheduler_type is None:
                scheduler_type = scheduler_config.get('scheduler_type', None)
                
            # 2. Check in cluster_config -> replica_scheduler_config -> name
            if scheduler_type is None and 'replica_scheduler_config' in cluster_config:
                scheduler_type = cluster_config['replica_scheduler_config'].get('name', None)
                
            # 3. Check in global_scheduler_config -> name
            if scheduler_type is None and 'global_scheduler_config' in cluster_config:
                scheduler_type = cluster_config['global_scheduler_config'].get('name', None)
                
            # Default to Unknown if not found
            if scheduler_type is None:
                scheduler_type = "Unknown"
            
            # Get chunk_size - need to check in several possible locations
            chunk_size = None
            
            # 1. Try cluster_config -> replica_scheduler_config
            if chunk_size is None and 'replica_scheduler_config' in cluster_config:
                chunk_size = cluster_config['replica_scheduler_config'].get('chunk_size', None)
            
            # 2. Try replica_scheduler_config directly (may be at top level)
            if chunk_size is None:
                chunk_size = replica_scheduler_config.get('chunk_size', None)
                
            # 3. Try scheduler_config -> replica_scheduler_config
            if chunk_size is None:
                scheduler_replica_config = scheduler_config.get('replica_scheduler_config', {})
                chunk_size = scheduler_replica_config.get('chunk_size', None)
                
            # 4. Fall back to sarathi_chunk_size if needed
            if chunk_size is None:
                chunk_size = scheduler_config.get('sarathi_chunk_size', None)
                
            # Get batch size in similar way
            sarathi_batch_size = None
            
            # 1. Try directly in scheduler_config
            if sarathi_batch_size is None:
                sarathi_batch_size = scheduler_config.get('batch_size', None)
                
            # 2. Try in replica_scheduler_config
            if sarathi_batch_size is None and 'replica_scheduler_config' in cluster_config:
                sarathi_batch_size = cluster_config['replica_scheduler_config'].get('batch_size_cap', None)
            # Get model name
            model_name = replica_config.get('model_name', 'Unknown')

            # Read metrics
            metrics_df = pd.read_csv(metrics_path)
            
            # Calculate total runtime: max(request_arrived_at + request_e2e_time)
            if 'request_arrived_at' in metrics_df.columns and 'request_e2e_time' in metrics_df.columns:
                total_runtime_seconds = (metrics_df['request_arrived_at'] + metrics_df['request_e2e_time']).max()
            else:
                # Fallback: use max request_e2e_time if columns are missing
                total_runtime_seconds = metrics_df['request_e2e_time'].max() if 'request_e2e_time' in metrics_df.columns else 0
            
            # Convert to hours
            total_runtime_hours = total_runtime_seconds / 3600.0
            
            # Calculate P99 of prefill_e2e_time
            p99_ttft = metrics_df['prefill_e2e_time'].quantile(0.99)
            
            # Calculate P99 and P50 of total request execution time from request_execution_time column
            p99_exec_time = None
            p50_exec_time = None
            if 'request_execution_time' in metrics_df.columns:
                p99_exec_time = metrics_df['request_execution_time'].quantile(0.99)
                p50_exec_time = metrics_df['request_execution_time'].quantile(0.50)
            
            # Calculate P99 of decode_time_execution_plus_preemption_normalized
            p99_inter_token_latency = None
            if 'decode_time_execution_plus_preemption_normalized' in metrics_df.columns:
                p99_inter_token_latency = metrics_df['decode_time_execution_plus_preemption_normalized'].quantile(0.99)
            
            # Store the result
            results.append({
                'run_id': run_dir,
                'qps': qps,
                'p99_ttft': p99_ttft,
                'p99_exec_time': p99_exec_time,
                'p50_exec_time': p50_exec_time,
                'p99_inter_token_latency': p99_inter_token_latency,
                'num_replicas': num_replicas,
                'tensor_parallel_size': tensor_parallel_size,
                'num_pipeline_stages': num_pipeline_stages,
                'device': device,
                'network_device': network_device,
                'model_name': model_name,
                'scheduler_type': scheduler_type,
                'chunk_size': chunk_size,
                'batch_size': sarathi_batch_size,
                'prefill_tokens': prefill_tokens,
                'decode_tokens': decode_tokens,
                'total_runtime_hours': total_runtime_hours,
                'config_path': config_path,
                'metrics_path': metrics_path
            })

# Convert results to DataFrame
df = pd.DataFrame(results)

if len(df) == 0:
    print("No data found!")
else:
    # Save the raw data
    
    
    # Display top 5 results sorted by p99_ttft
    print("Top 5 configs by lowest P99 TTFT:")
    top_configs = df.sort_values('p99_ttft').head(5)
    print(top_configs)
    
    # Create a unique color map for the different configurations
    unique_configs = {}
    
    # Use network_device for coloring
    unique_network_device = df['network_device'].unique()
    
    for i, val in enumerate(unique_network_device):
        unique_configs[val] = i
    
    # Create custom colormap
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(unique_configs), 1)))
    
    # Calculate QPS per dollar using the actual device costs and proper device count
    # Get cost per hour and GPUs per node for each device
    df['device_cost_per_hour'] = df['network_device'].apply(lambda x: device_costs.get(x, {}).get('cost', 0))
    df['gpus_per_node'] = df['network_device'].apply(lambda x: device_costs.get(x, {}).get('gpus_per_node', 8))
    
    # Calculate how many replicas can fit on one node based on device GPU count
    # Add comments to explain the data flow
    # These columns are calculated on-the-fly and not persisted in the DataFrame
    # To save them, we need to store them before saving to CSV
    
    # Calculate replicas that can fit on one node
    df['replica_per_node'] = df.apply(
        lambda x: x['gpus_per_node'] / (x['tensor_parallel_size'] * x['num_pipeline_stages']), 
        axis=1
    )
    assert all(df['replica_per_node'] > 0), "Replica per node must be greater than 0"    
    
    # Calculate number of nodes needed
    df['nodes_needed'] = df.apply(
        lambda x: np.ceil(x['num_replicas'] / x['replica_per_node']),
        axis=1
    )
    
    # Calculate total cost per hour
    df['total_cost_per_hour'] = df['device_cost_per_hour'] * df['nodes_needed']
    
    # Calculate total cost for the entire run
    df['total_cost'] = df['total_cost_per_hour'] * df['total_runtime_hours']
    
    # Calculate QPS per dollar using total cost
    df['qps_per_dollar'] = df.apply(
        lambda x: x['qps'] / x['total_cost'] if x['total_cost'] > 0 else 0, 
        axis=1
    )
    
    # Save all columns including the calculated ones
    df.to_csv("config_optimizer_results.csv", index=False)
    # Calculate best configs for different metrics
    slo_compliant = df[(df['p99_ttft'] <= slo_limit) & (df['p99_exec_time'] <= exec_slo)]
    
    # Best config for QPS (max QPS under SLO)
    if len(slo_compliant) > 0:
        best_config_qps = slo_compliant.loc[slo_compliant['qps'].idxmax()]
    else:
        print("Warning: no config under SLO configured for QPS, falling back to min p99_ttft...")
        best_config_qps = df.sort_values('p99_ttft').iloc[0]
    
    # Best config for QPS per dollar (max QPS per dollar under SLO)
    if len(slo_compliant) > 0:
        best_config_qps_per_dollar = slo_compliant.loc[slo_compliant['qps_per_dollar'].idxmax()]
    else:
        print("Warning: no config under SLO configured for QPS per dollar, falling back to min p99_ttft...")
        best_config_qps_per_dollar = df.sort_values('p99_ttft').iloc[0]
    
    # For the third subplot, use the QPS best config
    best_config_third_plot = best_config_qps
    
    # Check if we have execution time data
    has_exec_time = all(pd.notna(df['p99_exec_time']))
    
    # Create the best config descriptions
    best_desc_qps = (f"Best QPS Config: PP={best_config_qps['num_pipeline_stages']}, "
                    f"TP={best_config_qps['tensor_parallel_size']}, "
                    f"Replicas={best_config_qps['num_replicas']}, "
                    f"Nodes={best_config_qps['nodes_needed']}, "
                    f"Scheduler={best_config_qps['scheduler_type']}, "
                    f"Chunk={best_config_qps['chunk_size']}, "
                    f"Batch={best_config_qps['batch_size']}, "
                    f"SKU={best_config_qps['device']}, "
                    f"QPS = {best_config_qps['qps']:.2f}")
    
    best_desc_qps_per_dollar = (f"Best QPS/Dollar Config: PP={best_config_qps_per_dollar['num_pipeline_stages']}, "
                               f"TP={best_config_qps_per_dollar['tensor_parallel_size']}, "
                               f"Replicas={best_config_qps_per_dollar['num_replicas']}, "
                               f"Nodes={best_config_qps_per_dollar['nodes_needed']}, "
                               f"Scheduler={best_config_qps_per_dollar['scheduler_type']}, "
                               f"Chunk={best_config_qps_per_dollar['chunk_size']}, "
                               f"Batch={best_config_qps_per_dollar['batch_size']}, "
                               f"SKU={best_config_qps_per_dollar['device']}, "
                               f"QPS = {best_config_qps['qps']:.2f}, "
                               f"QPS/$ = {best_config_qps_per_dollar['qps_per_dollar']:.4f}")

    # =========================================
    # Figure 1: QPS Scatter Plot (4 subplots - including p99_inter_token_latency)
    # =========================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(18, 16), gridspec_kw={'width_ratios': [1, 1.2], 'height_ratios': [1, 1]})
    #fig1.suptitle("LLM Performance: QPS Analysis", fontsize=21)
    fig1.text(0.5, 0.97, best_desc_qps, ha='center', fontsize=17)

    # Subplot 1: QPS vs P99 TTFT
    ax = axes1[0, 0]
    ax.set_title("QPS vs P99 Time to First Token", fontsize=19)
    
    for network_device in unique_network_device:
        subset = df[df['network_device'] == network_device]
        ax.scatter(subset['p99_ttft'], subset['qps'], 
                   label=network_device,
                   color=colors[unique_configs[network_device]],
                   s=80, alpha=0.7)
    
    # No star for first subplot
    
    # Add SLO limit line
    ax.axvline(x=slo_limit, color='red', linestyle='--', label='SLO Limit (200ms)')
    ax.axvspan(0, slo_limit, alpha=0.1, color='green', label='SLO Compliant Region')
    
    ax.set_xlabel("Time to First Token - P99 (s)", fontsize=17)
    ax.set_ylabel("QPS", fontsize=17)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Configuration", fontsize=13)
    
    # Subplot 2: QPS vs P99 Request Total Latency
    ax = axes1[0, 1]
    ax.set_title("QPS vs P99 Request Total Latency", fontsize=19)
    
    if has_exec_time:
        for network_device in unique_network_device:
            subset = df[df['network_device'] == network_device]
            ax.scatter(subset['p99_exec_time'], subset['qps'], 
                      label=network_device,
                      color=colors[unique_configs[network_device]],
                      s=80, alpha=0.7)
        
        # No star for second subplot
        
        # Add SLO limit line
        ax.axvline(x=exec_slo, color='red', linestyle='--', label=f'SLO Limit ({exec_slo}s)')
        ax.axvspan(0, exec_slo, alpha=0.1, color='green', label='SLO Compliant Region')
        
        ax.set_xlabel("Request Execution Time - P99 (s)", fontsize=17)
        ax.set_ylabel("QPS", fontsize=17)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Configuration", fontsize=13)
    else:
        ax.text(0.5, 0.5, "P99 Request Total Latency metrics not available", 
                ha='center', va='center', fontsize=19)
    
    # Subplot 3: QPS vs P99 Inter-Token Latency
    ax = axes1[1, 0]
    ax.set_title("QPS vs P99 Inter-Token Latency", fontsize=19)
    
    # Check if we have inter-token latency data
    has_inter_token_latency = pd.notna(df['p99_inter_token_latency']).any()
    
    if has_inter_token_latency:
        for network_device in unique_network_device:
            subset = df[df['network_device'] == network_device]
            ax.scatter(subset['p99_inter_token_latency'], subset['qps'], 
                     label=network_device,
                     color=colors[unique_configs[network_device]],
                     s=80, alpha=0.7)
        
        # No star for third subplot
        
        # Add SLO limit line for inter-token latency (8ms)
        ax.axvline(x=inter_token_slo, color='red', linestyle='--', label=f'Inter-Token SLO ({inter_token_slo*1000}ms)')
        ax.axvspan(0, inter_token_slo, alpha=0.1, color='green', label='SLO Compliant Region')
        
        ax.set_xlabel("Inter-Token Latency - P99 (s)", fontsize=17)
        ax.set_ylabel("QPS", fontsize=17)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Configuration", fontsize=13)
    else:
        ax.text(0.5, 0.5, "P99 Inter-Token Latency metrics not available", 
               ha='center', va='center', fontsize=19)
    
    # Subplot 4: P99 Inter-Token Latency vs P99 TTFT, colored by QPS
    ax = axes1[1, 1]
    ax.set_title("P99 Inter-Token Latency vs P99 TTFT (Colored by QPS)", fontsize=19)
    
    # Check if we have inter-token latency data
    has_inter_token_latency = pd.notna(df['p99_inter_token_latency']).any()
    
    if has_inter_token_latency:
        # Create scatter plot with QPS as color
        scatter = ax.scatter(df['p99_ttft'], df['p99_inter_token_latency'], 
                           c=df['qps'], cmap='viridis', s=100, alpha=0.7)
        
        # Add colorbar with proper spacing
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label('QPS', fontsize=17)
        
        # Highlight the best config (only on fourth subplot)
        ax.scatter(best_config_third_plot['p99_ttft'], best_config_third_plot['p99_inter_token_latency'], 
                  color='gold', s=200, marker='*', 
                  label=f"Best: {best_config_third_plot['device']} TP{best_config_third_plot['tensor_parallel_size']}/PP{best_config_third_plot['num_pipeline_stages']}", 
                  edgecolor='black', zorder=10)
        
        # Add SLO limit lines
        ax.axvline(x=slo_limit, color='red', linestyle='--', label=f'TTFT SLO ({slo_limit*1000}ms)')
        ax.axhline(y=inter_token_slo, color='red', linestyle=':', label=f'Inter-Token SLO ({inter_token_slo*1000}ms)')
        
        # Add SLO compliant region (bottom-left rectangle)
        ax.fill_between([0, slo_limit], 0, inter_token_slo, alpha=0.1, color='green', label='SLO Compliant Region')
        
        ax.set_xlabel("Time to First Token - P99 (s)", fontsize=17)
        ax.set_ylabel("Inter-Token Latency - P99 (s)", fontsize=17)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=13)
    else:
        ax.text(0.5, 0.5, "P99 Inter-Token Latency metrics not available", 
                ha='center', va='center', fontsize=19)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("qps_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # =================================================
    # Figure 2: QPS per Dollar Scatter Plot (4 subplots - including p99_inter_token_latency)
    # =================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 16), gridspec_kw={'width_ratios': [1, 1.2], 'height_ratios': [1, 1]})
    #fig2.suptitle("LLM Cost Efficiency: QPS per Dollar Analysis", fontsize=21)
    fig2.text(0.5, 0.97, best_desc_qps_per_dollar, ha='center', fontsize=17)
    
    # Subplot 1: QPS per Dollar vs P99 TTFT
    ax = axes2[0, 0]
    ax.set_title("QPS per Dollar vs P99 TTFT", fontsize=19)
    
    for network_device in unique_network_device:
        subset = df[df['network_device'] == network_device]
        ax.scatter(subset['p99_ttft'], subset['qps_per_dollar'], 
                   label=network_device,
                   color=colors[unique_configs[network_device]],
                   s=80, alpha=0.7)
    
    # No star for first subplot
    
    # Add SLO limit line
    ax.axvline(x=slo_limit, color='red', linestyle='--', label=f'SLO Limit ({slo_limit*1000}ms)')
    ax.axvspan(0, slo_limit, alpha=0.1, color='green', label='SLO Compliant Region')
    
    ax.set_xlabel("Time to First Token - P99 (s)", fontsize=17)
    ax.set_ylabel("QPS per Dollar", fontsize=17)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Configuration", fontsize=13)
    
    # Subplot 2: QPS per Dollar vs P99 Request Total Latency
    ax = axes2[0, 1]
    ax.set_title("QPS per Dollar vs P99 Request Latency", fontsize=19)
    
    if has_exec_time:
        for network_device in unique_network_device:
            subset = df[df['network_device'] == network_device]
            ax.scatter(subset['p99_exec_time'], subset['qps_per_dollar'], 
                      label=network_device,
                      color=colors[unique_configs[network_device]],
                      s=80, alpha=0.7)
        
        # No star for second subplot
        
        # Add SLO limit line
        ax.axvline(x=exec_slo, color='red', linestyle='--', label='SLO Limit (5s)')
        ax.axvspan(0, exec_slo, alpha=0.1, color='green', label='SLO Compliant Region')
        
        ax.set_xlabel("Request Execution Time - P99 (s)", fontsize=17)
        ax.set_ylabel("QPS per Dollar", fontsize=17)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Configuration", fontsize=13)
    else:
        ax.text(0.5, 0.5, "P99 Request Total Latency metrics not available", 
                ha='center', va='center', fontsize=19)
    
    # Subplot 3: QPS per Dollar vs P99 Inter-Token Latency
    ax = axes2[1, 0]
    ax.set_title("QPS per Dollar vs P99 Inter-Token Latency", fontsize=19)
    
    # Check if we have inter-token latency data
    has_inter_token_latency = pd.notna(df['p99_inter_token_latency']).any()
    
    if has_inter_token_latency:
        for network_device in unique_network_device:
            subset = df[df['network_device'] == network_device]
            ax.scatter(subset['p99_inter_token_latency'], subset['qps_per_dollar'], 
                     label=network_device,
                     color=colors[unique_configs[network_device]],
                     s=80, alpha=0.7)
        
        # No star for third subplot
        
        # Add a reasonable SLO limit line for inter-token latency (e.g., 20ms)
        ax.axvline(x=inter_token_slo, color='red', linestyle='--', label=f'Inter-Token SLO ({inter_token_slo*1000}ms)')
        ax.axvspan(0, inter_token_slo, alpha=0.1, color='green', label='SLO Compliant Region')
        
        ax.set_xlabel("Inter-Token Latency - P99 (s)", fontsize=17)
        ax.set_ylabel("QPS per Dollar", fontsize=17)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Configuration", fontsize=13)
    else:
        ax.text(0.5, 0.5, "P99 Inter-Token Latency metrics not available", 
               ha='center', va='center', fontsize=19)
    
    # Subplot 4: P99 Inter-Token Latency vs P99 TTFT, colored by QPS per Dollar
    ax = axes2[1, 1]
    ax.set_title("P99 Inter-Token Latency vs P99 TTFT (Colored by QPS/$)", fontsize=19)
    
    # Check if we have inter-token latency data
    has_inter_token_latency = pd.notna(df['p99_inter_token_latency']).any()
    
    if has_inter_token_latency:
        # Create scatter plot with QPS per dollar as color
        scatter = ax.scatter(df['p99_ttft'], df['p99_inter_token_latency'], 
                           c=df['qps_per_dollar'], cmap='viridis', s=100, alpha=0.7)
        
        # Add colorbar with proper spacing
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label('QPS per Dollar', fontsize=17)
        
        # Highlight the best config (only on fourth subplot)
        ax.scatter(best_config_third_plot['p99_ttft'], best_config_third_plot['p99_inter_token_latency'], 
                  color='gold', s=250, marker='*', 
                  label=f"Best: {best_config_third_plot['device']} TP{best_config_third_plot['tensor_parallel_size']}/PP{best_config_third_plot['num_pipeline_stages']}", 
                  edgecolor='black', zorder=10)
        
        # Add SLO limit lines
        ax.axvline(x=slo_limit, color='red', linestyle='--', label=f'TTFT SLO ({slo_limit*1000}ms)')
        ax.axhline(y=inter_token_slo, color='red', linestyle=':', label=f'Inter-Token SLO ({inter_token_slo*1000}ms)')
        
        # Add SLO compliant region (bottom-left rectangle)
        ax.fill_between([0, slo_limit], 0, inter_token_slo, alpha=0.1, color='green', label='SLO Compliant Region')
        
        ax.set_xlabel("Time to First Token - P99 (s)", fontsize=17)
        ax.set_ylabel("Inter-Token Latency - P99 (s)", fontsize=17)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=13)
    else:
        ax.text(0.5, 0.5, "P99 Inter-Token Latency metrics not available", 
                ha='center', va='center', fontsize=19)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("qps_per_dollar_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Analysis complete.")
    print(f"Raw data saved: config_optimizer_results.csv")
    print(f"Plots saved:")
    print(f"  - qps_scatter.png - QPS performance metrics")
    print(f"  - qps_per_dollar_scatter.png - Cost efficiency metrics")
