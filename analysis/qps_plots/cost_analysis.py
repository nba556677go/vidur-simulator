#!/usr/bin/env python3
import pandas as pd

# Read the config optimizer results CSV
df = pd.read_csv('config_optimizer_results.csv')

# Filter for nodes_needed == 1
df_filtered = df[df['nodes_needed'] == 1]

# Save to CSV
df_filtered.to_csv('single_node_results.csv', index=False)

print(f"Filtered {len(df_filtered)} rows with nodes_needed == 1")
print(f"Original dataset had {len(df)} rows")
print("Results saved to 'single_node_results.csv'")
print("\nFirst 5 rows:")
print(df_filtered[['network_device', 'model_name', 'qps', 'total_cost', 'nodes_needed']].head().to_string(index=False))