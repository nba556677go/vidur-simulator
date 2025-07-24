import pandas as pd
import numpy as np
import os
import re
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModelConfig:
    num_replicas: int
    tensor_parallel_size: int
    num_pipeline_stages: int
    model_name: str

class VidurParser:
    def __init__(self, base_dir: str, output_dir: str):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.results = {}
        os.makedirs(output_dir, exist_ok=True)
        
    def parse_config_json(self, config_path: str) -> ModelConfig:
        """Parse configuration from config.json file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            cluster_config = config.get('cluster_config', {})
            replica_config = cluster_config.get('replica_config', {})
            
            return ModelConfig(
                num_replicas=cluster_config.get('num_replicas', 1),
                tensor_parallel_size=replica_config.get('tensor_parallel_size', 1),
                num_pipeline_stages=replica_config.get('num_pipeline_stages', 1),
                model_name=replica_config.get('model_name', 'unknown')
            )
        except Exception as e:
            print(f"Error parsing config.json: {e}")
            return None

    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate latency metrics for a given DataFrame."""
        metrics = {}
        
        # Calculate metrics for e2e time
        metrics['e2e_avg'] = df['request_e2e_time'].mean()
        metrics['e2e_p50'] = df['request_e2e_time'].quantile(0.5)
        metrics['e2e_p90'] = df['request_e2e_time'].quantile(0.9)
        metrics['e2e_p99'] = df['request_e2e_time'].quantile(0.99)

        # Calculate metrics for execution time
        metrics['exec_avg'] = df['request_execution_time'].mean()
        metrics['exec_p50'] = df['request_execution_time'].quantile(0.5)
        metrics['exec_p90'] = df['request_execution_time'].quantile(0.9)
        metrics['exec_p99'] = df['request_execution_time'].quantile(0.99)

        return metrics

    def process_directory(self, timestamp_dir: str) -> Tuple[Dict, ModelConfig]:
        """Process a specific timestamp directory."""
        full_path = os.path.join(self.base_dir, timestamp_dir)
        
        # Check for config.json and request_metrics.csv
        config_path = os.path.join(full_path, 'config.json')
        metrics_path = os.path.join(full_path, 'request_metrics.csv')
        
        if not os.path.exists(config_path) or not os.path.exists(metrics_path):
            return None, None
            
        config = self.parse_config_json(config_path)
        if config is None:
            return None, None
            
        df = pd.read_csv(metrics_path)
        metrics = self.calculate_metrics(df)
        
        return metrics, config

    def parse_all(self) -> pd.DataFrame:
        """Parse all directories and create a summary DataFrame."""
        results_list = []
        
        for timestamp_dir in os.listdir(self.base_dir):
            if os.path.isdir(os.path.join(self.base_dir, timestamp_dir)):
                metrics, config = self.process_directory(timestamp_dir)
                
                if metrics and config:
                    results_list.append({
                        'Timestamp': timestamp_dir,
                        'Model': config.model_name,
                        'Num_Replicas': config.num_replicas,
                        'Tensor_Parallel': config.tensor_parallel_size,
                        'Pipeline_Parallel': config.num_pipeline_stages,
                        'E2E_Avg': metrics['e2e_avg'],
                        'E2E_P50': metrics['e2e_p50'],
                        'E2E_P90': metrics['e2e_p90'],
                        'E2E_P99': metrics['e2e_p99'],
                        'Exec_Avg': metrics['exec_avg'],
                        'Exec_P50': metrics['exec_p50'],
                        'Exec_P90': metrics['exec_p90'],
                        'Exec_P99': metrics['exec_p99']
                    })

        return pd.DataFrame(results_list)

    def save_results(self, df: pd.DataFrame):
        """Save results to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed CSV
        csv_path = os.path.join(self.output_dir, f'latency_metrics_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        
        # Save summary by configuration
        summary_path = os.path.join(self.output_dir, f'summary_{timestamp}.csv')
        
        # Select only numeric columns for mean calculation
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        summary = df.groupby(['Model', 'Num_Replicas', 'Tensor_Parallel', 'Pipeline_Parallel'])[numeric_columns].mean()
        summary.to_csv(summary_path)
        
        return csv_path, summary_path

def main():
    # Example usage
    #base_dir = "/home/ec2-user/s3-local/vidur_outputs/a100_default/qps0.25"
    QPS=[15, 20 , 40]
    profile_name = "h100_p5"
    compute_profile = "h100_p5"
    network_device = "h100_p5"
    for qps in QPS:
        
        base_dir  = f"/home/ec2-user/vidur-simulator/simulator_output/{profile_name}/compute_{compute_profile}/network_{network_device}/qps{qps}"
        output_dir = f"./vidur_results/{profile_name}/compute_{compute_profile}/network_{network_device}/chunk8192/qps{qps}"
        
        parser = VidurParser(base_dir, output_dir)
        results_df = parser.parse_all()
        
        # Save results
        csv_path, summary_path = parser.save_results(results_df)
        
        # Print summary
        print(f"\nResults saved to: {csv_path}")
        print(f"Summary saved to: {summary_path}")
        
        print("\nResults summary:")
        print(results_df.to_string())
        
        # Print configuration-wise summary
        print("\nAverage metrics by configuration:")
        numeric_columns = results_df.select_dtypes(include=[np.number]).columns
        summary = results_df.groupby(['Model', 'Num_Replicas', 'Tensor_Parallel', 'Pipeline_Parallel'])[numeric_columns].mean()
        print(summary)

if __name__ == "__main__":
    main()