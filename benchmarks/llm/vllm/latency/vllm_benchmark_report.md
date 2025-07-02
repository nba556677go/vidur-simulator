# vLLM Benchmark Analysis Report for Qwen/Qwen3-8B

## Executive Summary

This report analyzes the performance of the Qwen/Qwen3-8B model using vLLM with various parallelism and batch size configurations. The analysis is based on benchmark results from 36 different configurations, with 4 configurations missing due to likely memory constraints.

### Key Findings:

1. **Best Overall Throughput**: TP=2, DP=4, max_tokens=4096 (29.90 tokens/sec)
2. **Best Time to First Token (TTFT)**: TP=1, DP=8, max_tokens=4096 (1081.40 ms)
3. **Best Inter-Token Latency**: TP=2, DP=2, max_tokens=40960 (24.85 ms)
4. **Missing Configurations**: All TP=8, DP=1 configurations (likely due to memory constraints)
5. **Batch Size Impact**: For most configurations, smaller batch sizes (4096) yield better throughput

## Detailed Analysis

### Missing Configurations

The following configurations are missing from the benchmark results:
- TP=8, DP=1, max_tokens=4096 (attempted but failed)
- TP=8, DP=1, max_tokens=20480
- TP=8, DP=1, max_tokens=40960
- TP=8, DP=1, max_tokens=73728

The logs for the attempted TP=8, DP=1, max_tokens=4096 runs show that the vLLM engine was initialized but the benchmarks were interrupted before completion. This suggests that using tensor parallelism across all 8 GPUs may cause memory issues, especially with larger batch sizes.

### Performance by Parallelism Strategy

#### Data Parallelism (DP) vs. Tensor Parallelism (TP)

- **Higher DP values** generally yield better throughput than higher TP values
- The best throughput configuration uses TP=2, DP=4 (29.90 tokens/sec)
- The second-best throughput configuration uses TP=4, DP=2 (29.08 tokens/sec)
- For TTFT, higher DP values are significantly better (TP=1, DP=8 gives the best TTFT at 1081.40 ms)

#### Effect of Batch Size (max_num_batched_tokens)

For most configurations, increasing the batch size from 4096 to 73728 tokens actually **decreases throughput**:
- TP=1, DP=1: -0.74% change
- TP=1, DP=2: -6.11% change
- TP=1, DP=4: -3.41% change
- TP=1, DP=8: -0.68% change
- TP=2, DP=1: -4.14% change
- TP=2, DP=2: -11.07% change
- TP=2, DP=4: -8.32% change
- TP=4, DP=2: -14.48% change

Only one configuration showed improved throughput with larger batch size:
- TP=4, DP=1: +1.23% change from 4096 to 73728 tokens

This suggests that for this model, smaller batch sizes are generally more efficient, with the optimal batch size being the smallest tested (4096) for most configurations.

### Best Configurations by Metric

| Metric | Best Value | TP | DP | max_tokens |
|--------|------------|----|----|------------|
| Throughput (tokens/sec) | 29.90 | 2 | 4 | 4096 |
| TTFT p50 (ms) | 1081.40 | 1 | 8 | 4096 |
| Inter-token p50 (ms) | 24.85 | 2 | 2 | 40960 |
| Total Latency p50 (ms) | 30840.69 | 2 | 4 | 4096 |

## Recommendations

Based on the benchmark results, we recommend the following configurations for different use cases:

1. **For Maximum Throughput**: 
   - TP=2, DP=4, max_tokens=4096
   - This configuration provides the highest throughput (29.90 tokens/sec) with reasonable latency

2. **For Lowest Latency (fastest first response)**:
   - TP=1, DP=8, max_tokens=4096
   - This configuration provides the fastest time to first token (1081.40 ms) with good throughput (28.67 tokens/sec)

3. **For Smoothest Generation (lowest inter-token latency)**:
   - TP=2, DP=2, max_tokens=40960
   - This configuration provides the lowest inter-token latency (24.85 ms) but with lower throughput (24.75 tokens/sec)

4. **Balanced Performance**:
   - TP=2, DP=4, max_tokens=4096
   - This configuration provides the best overall balance of throughput and latency

## Conclusion

The benchmark results show that data parallelism (DP) generally yields better performance than tensor parallelism (TP) for the Qwen/Qwen3-8B model. The optimal configuration uses a combination of both (TP=2, DP=4) with a smaller batch size (4096 tokens).

The missing TP=8 configurations suggest that using tensor parallelism across all 8 GPUs may not be feasible due to memory constraints, especially with larger batch sizes. This is an important consideration for deployment planning.

For most configurations, smaller batch sizes yield better throughput, which is counter to the common assumption that larger batch sizes improve throughput. This suggests that for this model and hardware configuration, the overhead of processing larger batches outweighs the benefits of increased parallelism.
