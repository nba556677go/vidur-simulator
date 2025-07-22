from vidurparser import VidurParser
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class VLLMConfig:
    model_name: str
    num_replicas: int
    tensor_parallel_size: int
    num_pipeline_stages: int
    max_num_batched_tokens: int
    max_num_seqs: int
    concurrency: int
    qps: float
    max_tokens: int
    temperature: float
    top_p: float

class VLLMBenchParser(VidurParser):
    def __init__(self, base_dir: str, output_dir: str):
        super().__init__(base_dir, output_dir)

    def parse_config_json(self, config_path: str) -> VLLMConfig:
        """Parse configuration from VLLM config.json file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            return VLLMConfig(
                model_name=config.get('model', 'unknown'),
                num_replicas=config.get('data_parallel_size', 1),
                tensor_parallel_size=config.get('tensor_parallel_size', 1),
                num_pipeline_stages=config.get('pipeline_parallel_size', 1),
                max_num_batched_tokens=config.get('max_num_batched_tokens', 0),
                max_num_seqs=config.get('max_num_seqs', 0),
                concurrency=config.get('concurrency', 0),
                qps=config.get('qps', 0.0),
                max_tokens=config.get('max_tokens', 0),
                temperature=config.get('temperature', 0.0),
                top_p=config.get('top_p', 0.0)
            )
        except Exception as e:
            print(f"Error parsing config.json: {e}")
            return None

    def parse_metrics(self, results_path: str) -> Dict:
        """Parse metrics from benchmark_results.json file."""
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)
            
            latency_percentiles = data.get('benchmark_stats', {}).get('latency_percentiles_ms', {})
            summary = data.get('benchmark_stats', {}).get('summary', {})
            
            metrics = {
                'total_gen_avg': latency_percentiles.get('total_generation_latency', {}).get('total_avg'),
                'total_gen_p50': latency_percentiles.get('total_generation_latency', {}).get('total_p50'),
                'total_gen_p90': latency_percentiles.get('total_generation_latency', {}).get('total_p90'),
                'total_gen_p99': latency_percentiles.get('total_generation_latency', {}).get('total_p99'),
                'first_token_avg': latency_percentiles.get('time_to_first_token', {}).get('first_token_avg'),
                'first_token_p50': latency_percentiles.get('time_to_first_token', {}).get('first_token_p50'),
                'first_token_p90': latency_percentiles.get('time_to_first_token', {}).get('first_token_p90'),
                'first_token_p99': latency_percentiles.get('time_to_first_token', {}).get('first_token_p99'),
                'inter_token_avg': latency_percentiles.get('inter_token_latency', {}).get('inter_token_avg'),
                'inter_token_p50': latency_percentiles.get('inter_token_latency', {}).get('inter_token_p50'),
                'inter_token_p90': latency_percentiles.get('inter_token_latency', {}).get('inter_token_p90'),
                'inter_token_p99': latency_percentiles.get('inter_token_latency', {}).get('inter_token_p99'),
                'throughput': summary.get('overall_throughput_tokens_per_sec'),
                'total_requests': summary.get('total_requests'),
                'total_input_tokens': summary.get('total_input_tokens'),
                'total_output_tokens': summary.get('total_output_tokens')
            }
            return metrics
        except Exception as e:
            print(f"Error parsing metrics from benchmark_results.json: {e}")
            return None

    def process_directory(self, path: str) -> Tuple[Dict, VLLMConfig]:
        """Process a specific directory."""
        # Check for both config.json and benchmark_results.json
        config_path = os.path.join(path, 'config.json')
        results_path = os.path.join(path, 'benchmark_results.json')
        
        if not os.path.exists(config_path) or not os.path.exists(results_path):
            return None, None
            
        config = self.parse_config_json(config_path)
        metrics = self.parse_metrics(results_path)
        
        return metrics, config

    def parse_all(self) -> pd.DataFrame:
        """Parse all directories and create a summary DataFrame."""
        results_list = []
        
        for model_dir in os.listdir(self.base_dir):
            model_path = os.path.join(self.base_dir, model_dir)
            if os.path.isdir(model_path):
                for run_dir in os.listdir(model_path):
                    if run_dir.startswith('run_'):
                        full_path = os.path.join(model_path, run_dir)
                        metrics, config = self.process_directory(full_path)
                        
                        if metrics and config:
                            results_list.append({
                                'Timestamp': run_dir,
                                'Model': config.model_name,
                                'Num_Replicas': config.num_replicas,
                                'Tensor_Parallel': config.tensor_parallel_size,
                                'Pipeline_Parallel': config.num_pipeline_stages,
                                'Max_Batch_Tokens': config.max_num_batched_tokens,
                                'Max_Num_Seqs': config.max_num_seqs,
                                'Concurrency': config.concurrency,
                                'QPS': config.qps,
                                'Max_Tokens': config.max_tokens,
                                'Temperature': config.temperature,
                                'Top_P': config.top_p,
                                'Total_Gen_Avg': metrics['total_gen_avg'],
                                'Total_Gen_P50': metrics['total_gen_p50'],
                                'Total_Gen_P90': metrics['total_gen_p90'],
                                'Total_Gen_P99': metrics['total_gen_p99'],
                                'First_Token_Avg': metrics['first_token_avg'],
                                'First_Token_P50': metrics['first_token_p50'],
                                'First_Token_P90': metrics['first_token_p90'],
                                'First_Token_P99': metrics['first_token_p99'],
                                'Inter_Token_Avg': metrics['inter_token_avg'],
                                'Inter_Token_P50': metrics['inter_token_p50'],
                                'Inter_Token_P90': metrics['inter_token_p90'],
                                'Inter_Token_P99': metrics['inter_token_p99'],
                                'Throughput': metrics['throughput'],
                                'Total_Requests': metrics['total_requests'],
                                'Total_Input_Tokens': metrics['total_input_tokens'],
                                'Total_Output_Tokens': metrics['total_output_tokens']
                            })

        return pd.DataFrame(results_list)

def main():
    QPS=2
    base_dir = f"../benchmarks/llm/vllm/latency/vllm_output/a100_p4d/qps{QPS}"
    output_dir = f"./vllm_bench_results/a100_p4d/qps{QPS}"
    
    parser = VLLMBenchParser(base_dir, output_dir)
    results_df = parser.parse_all()
    
    # Save results
    csv_path, summary_path = parser.save_results(results_df)
    
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