import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Prepare the data
csv_dir = "vidur_results/a100_default_model_reprofiled/chunk8192"
csv_data =  f"{csv_dir}/all_percentiles_comparison.csv"
df = pd.read_csv((csv_data))

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(15, 22))
fig.suptitle('Comparison of Latency Ratios for different QPS', fontsize=20, y=0.995)

# --- Plotting Parameters ---
percentiles = ['P50', 'P90', 'P99']
bar_width = 0.35
x = np.arange(len(df['QPS']))
label_fontsize = 14
tick_fontsize = 14
title_fontsize = 16
bar_label_fontsize = 14

for i, p in enumerate(percentiles):
    ax = axes[i]
    
    # Get data for the current percentile
    vllm_latency_ratio = df[f'VLLM_Latency_Ratio_{p}']
    r1tp1pp1_vs_vllm_min = df[f'R1TP1PP1_vs_VLLM_Min_{p}']

    # Create bars and store the container objects
    bar1 = ax.bar(x - bar_width/2, vllm_latency_ratio, bar_width, label='vidur_vs_VLLM_Min')
    bar2 = ax.bar(x + bar_width/2, r1tp1pp1_vs_vllm_min, bar_width, label='R1TP1PP1_vs_VLLM_Min')

    # Add text labels on top of each bar
    ax.bar_label(bar1, fmt='%.2f', padding=3, fontsize=bar_label_fontsize)
    ax.bar_label(bar2, fmt='%.2f', padding=3, fontsize=bar_label_fontsize)
    
    # Set titles and labels with larger font sizes
    ax.set_title(f'{p} Latency Comparison', fontsize=title_fontsize)
    ax.set_ylabel('Latency Ratio', fontsize=label_fontsize)
    
    # Set tick parameters
    ax.set_xticks(x)
    ax.set_xticklabels(df['QPS'])
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    # Adjust y-axis limit to make space for labels
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    
    ax.legend(fontsize=label_fontsize)
    ax.grid()
# Set common X-axis label
plt.xlabel('QPS (Queries Per Second)', fontsize=label_fontsize)

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(f'{csv_dir}/latency_comparison_by_qps_detailed.png')

plt.close()

## prediction

# --- Plot Setup ---
fig, ax = plt.subplots(figsize=(14, 9))

# Data to plot
error_p50 = df['Avg_Pred_Error_P50']
error_p90 = df['Avg_Pred_Error_P90']
error_p99 = df['Avg_Pred_Error_P99']
#add error parsing if these number error_p50, p90.p99  nan

print(error_p99)
print(error_p99)
# Add error checking
if error_p50.isnull().any() or error_p90.isnull().any() or error_p99.isnull().any():
    print("Warning: NaN values detected in prediction error data")# Bar positions
x = np.arange(len(df['QPS']))  # the label locations
bar_width = 0.25  # the width of the bars

# Create grouped bars
bar1 = ax.bar(x - bar_width, error_p50, bar_width, label='P50 Error')
bar2 = ax.bar(x, error_p90, bar_width, label='P90 Error')
bar3 = ax.bar(x + bar_width, error_p99, bar_width, label='P99 Error')

# --- Formatting ---
# Add text labels on top of each bar
ax.bar_label(bar1, fmt='%.2f', padding=3, fontsize=10)
ax.bar_label(bar2, fmt='%.2f', padding=3, fontsize=10)
ax.bar_label(bar3, fmt='%.2f', padding=3, fontsize=10)

# Add titles and labels with larger font sizes
ax.set_title('Average Prediction Error by QPS', fontsize=18, pad=20)
ax.set_xlabel('QPS (Queries Per Second)', fontsize=14)
ax.set_ylabel('Average Prediction Error (%)', fontsize=14)

# Set X-axis ticks and labels
ax.set_xticks(x)
ax.set_xticklabels(df['QPS'])

# Enlarge tick labels
ax.tick_params(axis='both', which='major', labelsize=12)

# Add a horizontal line at y=0 for reference
ax.axhline(0, color='grey', linewidth=0.8)

# Adjust y-axis limits to make space for labels, accounting for negative values
min_val = min(df[['Avg_Pred_Error_P50', 'Avg_Pred_Error_P90', 'Avg_Pred_Error_P99']].min())
max_val = max(df[['Avg_Pred_Error_P50', 'Avg_Pred_Error_P90', 'Avg_Pred_Error_P99']].max())
ax.set_ylim(min_val * 1.15, max_val * 1.15)

# Add legend
ax.legend(fontsize=12)

# Final layout adjustments
plt.tight_layout()
plt.savefig(f'{csv_dir}/prediction_error_by_qps.png')
plt.close()
