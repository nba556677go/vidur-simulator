#!/usr/bin/env python3
import pandas as pd

# Read the single node results CSV
df = pd.read_csv('single_node_results.csv')

# Group by network_device and model_name, show unique QPS values
qps_summary = df.groupby(['network_device', 'model_name'])['qps'].apply(lambda x: sorted(x.unique())).reset_index()

print("QPS values used for each network_device and model_name combination:")
for _, row in qps_summary.iterrows():
    print(f"{row['network_device']} + {row['model_name']}: {row['qps']}")