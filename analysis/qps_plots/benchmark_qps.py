#!/usr/bin/env python3
import pandas as pd

# Read the single node results CSV
df = pd.read_csv('single_node_results.csv')

# Group by model, tensor_parallel_size, num_replicas, network_device and show unique QPS values
qps_by_config = df.groupby(['model_name', 'tensor_parallel_size', 'num_replicas', 'network_device'])['qps'].apply(lambda x: [float(v) for v in sorted(x.unique())]).reset_index()

# Save to CSV for easy reference
qps_by_config.to_csv('qps_by_model_parallelism_device.csv', index=False)

print("QPS values by model, parallelism and device configuration:")
for _, row in qps_by_config.iterrows():
    print(f"{row['model_name']} | TP={row['tensor_parallel_size']} | Replicas={row['num_replicas']} | Device={row['network_device']}: {row['qps']}")