import asyncio
import logging
import time
import uuid
import numpy as np
import torch
import json
import argparse
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from transformers import AutoTokenizer

# Initialize logger
logger = logging.getLogger(__name__)

class LLMBenchmark:
    """
    A class to benchmark the performance of a vLLM engine from either a prompts file or a trace file,
    including detailed latency and throughput metrics.
    """
    def __init__(self,
                 model_name: str,
                 nsys: bool = False,
                 download_dir: str = None,
                 tensor_parallel_size: int = 1,
                 pipeline_parallel_size: int = 1,
                 data_parallel_size: int = 1,
                 max_num_batched_tokens: int = 4096):
        self.model_name = model_name
        self.engine = None
        self.nsys = nsys
        self.results = []
        
        # vLLM Engine Parameters
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = data_parallel_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.download_dir = download_dir
        
        logger.info("Initializing tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Tokenizer initialized.")
        
    async def initialize_engine(self):
        """Initialize the vLLM engine with specified settings."""
        logger.info(f"Initializing vLLM engine with tp={self.tensor_parallel_size}, pp={self.pipeline_parallel_size}, dp={self.data_parallel_size}...")
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            data_parallel_size=self.data_parallel_size,
            gpu_memory_utilization=0.85,
            # NOTE: max_model_len is removed to allow vLLM to auto-detect from the model's config.
            # This resolves the ValidationError.
            max_num_batched_tokens=self.max_num_batched_tokens,
            dtype="auto",
            trust_remote_code=True,
            enforce_eager=True,
            download_dir=self.download_dir
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"Initialized vLLM engine for {self.model_name}")

    async def generate_text(self, prompt: str, sampling_params: SamplingParams, prompt_info: Dict):
        """
        Generate text for a single request and record metrics.
        This method is now unified for both prompt-file and trace-file modes.
        """
        request_id = str(uuid.uuid4())
        
        # Record request arrival time
        arrival_time = prompt_info.get("arrival_time", time.time())
        
        # In trace mode, input_tokens is pre-calculated and passed in prompt_info.
        # In prompt mode, we calculate it on the fly.
        input_tokens = prompt_info.get("input_tokens", len(self.tokenizer.encode(prompt)))
        
        # Record when request processing actually starts (prefill begins)
        processing_start_time = time.time()
        schedule_delay = processing_start_time - arrival_time
        
        start_time = processing_start_time
        first_token_time = None
        token_times = []

        # NVTX range for prefill
        if self.nsys:
            torch.cuda.nvtx.range_push("vllm_prefill")
        
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for output in results_generator:
            current_time = time.time()
            if first_token_time is None:
                first_token_time = current_time
                if self.nsys:
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_push("vllm_decode")
            token_times.append(current_time)
            final_output = output
        
        if self.nsys:
            torch.cuda.nvtx.range_pop()

        if final_output is None:
            logger.error(f"Request {request_id} failed, no output was generated.")
            # We don't raise an error, just skip recording results for this request.
            return
        
        end_time = time.time()
        output_tokens = len(final_output.outputs[0].token_ids)
        
        total_latency = end_time - start_time
        time_to_first_token = first_token_time - start_time if first_token_time else 0
        inter_token_latencies = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
        avg_inter_token_latency = np.mean(inter_token_latencies) if inter_token_latencies else 0
        
        result = {
            "request_id": request_id,
            "prompt": prompt if not prompt_info.get("is_trace") else f"Trace-generated prompt ({input_tokens} tokens)",
            "output": final_output.outputs[0].text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "arrival_time": arrival_time,
            "processing_start_time": processing_start_time,
            "schedule_delay": schedule_delay,
            "total_latency": total_latency,
            "time_to_first_token": time_to_first_token,
            "avg_inter_token_latency": avg_inter_token_latency,
            "tokens_per_second": output_tokens / total_latency if total_latency > 0 else 0,
            "token_latencies": inter_token_latencies,
        }
        self.results.append(result)

    def _parse_trace_file(self, trace_path: str) -> List[Dict[str, Any]]:
        """Parses a CSV trace file and prepares requests."""
        logger.info(f"Parsing trace file from {trace_path}...")
        try:
            df = pd.read_csv(trace_path)
            required_cols = {'request_arrived_at', 'request_num_prefill_tokens', 'request_num_decode_tokens'}
            
            df.columns = df.columns.str.strip()
            if not required_cols.issubset(df.columns):
                raise ValueError(f"Trace file must contain columns: {required_cols}. Found: {list(df.columns)}")

            requests = []
            for _, row in df.iterrows():
                num_prefill = int(row['request_num_prefill_tokens'])
                num_decode = int(row['request_num_decode_tokens'])
                arrival_time = float(row['request_arrived_at'])

                # Generate a dummy prompt that tokenizes to the desired prefill length
                prompt_token_ids = [self.tokenizer.bos_token_id] * num_prefill
                prompt_text = self.tokenizer.decode(prompt_token_ids)

                sampling_params = SamplingParams(max_tokens=num_decode)
                
                requests.append({
                    "prompt": prompt_text,
                    "params": sampling_params,
                    "arrived_at": arrival_time,
                    "arrival_time": arrival_time,  # Add explicit arrival_time for consistency
                    "input_tokens": num_prefill,
                    "is_trace": True,
                })
            logger.info(f"Successfully parsed {len(requests)} requests from trace file.")
            return requests
        except Exception as e:
            logger.error(f"Failed to read or parse trace file: {e}")
            raise

    async def run_benchmark(self, 
                            prompts: List[str] = None, 
                            trace_path: str = None, 
                            gen_params: Dict = None, 
                            concurrency: int = 1):
        """Run benchmark from either a list of prompts or a trace file."""
        if not self.engine:
            await self.initialize_engine()
        
        self._benchmark_start_mono = time.monotonic()
        tasks = []

        if trace_path:
            # --- Trace-driven Workload ---
            logger.info("Starting trace-driven benchmark...")
            requests = self._parse_trace_file(trace_path)
            for req in requests:
                target_arrival_mono = self._benchmark_start_mono + req['arrived_at']
                sleep_duration = max(0, target_arrival_mono - time.monotonic())
                await asyncio.sleep(sleep_duration)
                
                # Set the actual arrival time when the request is processed
                req['arrival_time'] = time.time()
                
                task = asyncio.create_task(self.generate_text(req['prompt'], req['params'], req))
                tasks.append(task)
            
        elif prompts:
            # --- Prompt File Workload ---
            logger.info("Starting prompt file-driven benchmark...")
            sampling_params = SamplingParams(
                temperature=gen_params.get("temperature", 0.7),
                top_p=gen_params.get("top_p", 0.9),
                max_tokens=gen_params.get("max_tokens", 256)
            )
            for prompt in prompts:
                # Record arrival time for prompt-based requests
                prompt_info = {"arrival_time": time.time()}
                task = asyncio.create_task(self.generate_text(prompt, sampling_params, prompt_info))
                tasks.append(task)
                if len(tasks) >= concurrency:
                    await asyncio.gather(*tasks)
                    tasks = []
        else:
            raise ValueError("Either prompts or a trace_path must be provided.")

        if tasks:
            await asyncio.gather(*tasks)

    def _calculate_stats(self) -> Dict:
        """Calculate and return comprehensive benchmark statistics."""
        if not self.results:
            return {}

        total_latencies_ms = [r["total_latency"] * 1000 for r in self.results]
        first_token_latencies_ms = [r["time_to_first_token"] * 1000 for r in self.results]
        inter_token_latencies_ms = [lat * 1000 for r in self.results for lat in r["token_latencies"]]
        schedule_delays_ms = [r["schedule_delay"] * 1000 for r in self.results]

        def calculate_percentiles(values, prefix=""):
            if not values: return {}
            return {
                f"{prefix}p50": float(np.percentile(values, 50)), f"{prefix}p90": float(np.percentile(values, 90)),
                f"{prefix}p95": float(np.percentile(values, 95)), f"{prefix}p99": float(np.percentile(values, 99)),
                f"{prefix}max": float(np.max(values)), f"{prefix}avg": float(np.mean(values))
            }

        total_latency_stats = calculate_percentiles(total_latencies_ms, "total_")
        first_token_stats = calculate_percentiles(first_token_latencies_ms, "first_token_")
        inter_token_stats = calculate_percentiles(inter_token_latencies_ms, "inter_token_")
        schedule_delay_stats = calculate_percentiles(schedule_delays_ms, "schedule_delay_")

        total_input_tokens = sum(r['input_tokens'] for r in self.results)
        total_output_tokens = sum(r['output_tokens'] for r in self.results)
        
        # For overall throughput, use the total wall clock time from the first request to the last.
        benchmark_duration = time.monotonic() - getattr(self, '_benchmark_start_mono', time.monotonic())
        overall_throughput_tps = total_output_tokens / benchmark_duration if benchmark_duration > 0 else 0

        return {
            "summary": {
                "total_requests": len(self.results),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "overall_throughput_tokens_per_sec": overall_throughput_tps
            },
            "latency_percentiles_ms": {
                "total_generation_latency": total_latency_stats,
                "time_to_first_token": first_token_stats,
                "inter_token_latency": inter_token_stats,
                "schedule_delay": schedule_delay_stats,
            }
        }

    def print_results(self):
        """Print comprehensive benchmark results."""
        if not self.results:
            logger.warning("No results to display")
            return
            
        stats = self._calculate_stats()
        summary = stats["summary"]
        latency_stats = stats["latency_percentiles_ms"]
        total_latency_stats = latency_stats["total_generation_latency"]
        first_token_stats = latency_stats["time_to_first_token"]
        inter_token_stats = latency_stats["inter_token_latency"]
        schedule_delay_stats = latency_stats["schedule_delay"]

        print("\n=== Benchmark Results ===")
        print(f"Total requests: {summary['total_requests']}")
        print(f"Total input tokens: {summary['total_input_tokens']}")
        print(f"Total output tokens: {summary['total_output_tokens']}")
        print(f"Overall Throughput: {summary['overall_throughput_tokens_per_sec']:.2f} tokens/sec")
        
        print("\n=== Latency Percentiles (ms) ===")
        if total_latency_stats:
            print("Total Generation Latency:")
            print(f"  Avg: {total_latency_stats['total_avg']:.2f}  P50: {total_latency_stats['total_p50']:.2f}")
            print(f"  P90: {total_latency_stats['total_p90']:.2f}  P95: {total_latency_stats['total_p95']:.2f}")
            print(f"  P99: {total_latency_stats['total_p99']:.2f}  Max: {total_latency_stats['total_max']:.2f}")
        
        if first_token_stats:
            print("\nTime To First Token:")
            print(f"  Avg: {first_token_stats['first_token_avg']:.2f}  P50: {first_token_stats['first_token_p50']:.2f}")
            print(f"  P90: {first_token_stats['first_token_p90']:.2f}  P95: {first_token_stats['first_token_p95']:.2f}")
            print(f"  P99: {first_token_stats['first_token_p99']:.2f}  Max: {first_token_stats['first_token_max']:.2f}")
        
        if inter_token_stats:
            print("\nInter-Token Latency:")
            print(f"  Avg: {inter_token_stats['inter_token_avg']:.2f}  P50: {inter_token_stats['inter_token_p50']:.2f}")
            print(f"  P90: {inter_token_stats['inter_token_p90']:.2f}  P95: {inter_token_stats['inter_token_p95']:.2f}")
            print(f"  P99: {inter_token_stats['inter_token_p99']:.2f}  Max: {inter_token_stats['inter_token_max']:.2f}")
        
        if schedule_delay_stats:
            print("\nSchedule Delay:")
            print(f"  Avg: {schedule_delay_stats['schedule_delay_avg']:.2f}  P50: {schedule_delay_stats['schedule_delay_p50']:.2f}")
            print(f"  P90: {schedule_delay_stats['schedule_delay_p90']:.2f}  P95: {schedule_delay_stats['schedule_delay_p95']:.2f}")
            print(f"  P99: {schedule_delay_stats['schedule_delay_p99']:.2f}  Max: {schedule_delay_stats['schedule_delay_max']:.2f}")
        
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"\nGPU Memory Usage: {((total - free)/1024**3):.2f}GB / {total/1024**3:.2f}GB")

    def save_results_to_json(self, filename: str):
        """Save benchmark statistics and full results to a JSON file."""
        if not self.results:
            logger.warning("No results to save.")
            return

        stats = self._calculate_stats()
        
        full_results = {
            "benchmark_config": {
                "model_name": self.model_name,
                "tensor_parallel_size": self.tensor_parallel_size,
                "pipeline_parallel_size": self.pipeline_parallel_size,
                "data_parallel_size": self.data_parallel_size,
                "max_num_batched_tokens": self.max_num_batched_tokens,
            },
            "benchmark_stats": stats,
            "individual_requests": self.results
        }

        try:
            with open(filename, 'w') as f:
                json.dump(full_results, f, indent=4)
            logger.info(f"Benchmark results saved to {filename}")
        except IOError as e:
            logger.error(f"Failed to write results to {filename}: {e}")


async def main():
    parser = argparse.ArgumentParser(description="vLLM Benchmarking Tool with Trace Support")
    
    # --- Input Source (Mutually Exclusive) ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--prompts-file", type=str, help="Path to a file with prompts, one per line.")
    input_group.add_argument("--trace", type=str, help="Path to a CSV trace file for request simulation.")
    
    # --- Model and vLLM Engine Arguments ---
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-0.5B", help="Name of the Hugging Face model to benchmark.")
    parser.add_argument("--download-dir", type=str, default=None, help="Directory to download and cache model files. Defaults to Hugging Face cache dir.")
    parser.add_argument("--tp", "--tensor-parallel-size", dest="tensor_parallel_size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--pp", "--pipeline-parallel-size", dest="pipeline_parallel_size", type=int, default=1, help="Pipeline parallel size.")
    parser.add_argument("--dp", "--data-parallel-size", dest="data_parallel_size", type=int, default=1, help="Data parallel size.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096, help="Maximum number of batched tokens.")

    # --- Benchmark Run Arguments ---
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent requests (for --prompts-file mode).")
    
    # --- Generation Parameters (for --prompts-file mode) ---
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Sampling top-p.")

    # --- Profiling and Output ---
    parser.add_argument("--nsys", action="store_true", help="Wrap prefill and decode phases in CUDA NVTX ranges for nsys profiling.")
    parser.add_argument("--output-dir", type=str, default="./benchmark_runs", help="Base directory to save timestamped run folders.")
      
    args = parser.parse_args()

    # --- Setup Output Directory and Logging ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir_name = args.model.replace("/", "_")
    run_dir = Path(args.output_dir) / model_dir_name / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "benchmark.log"
    config_path = run_dir / "config.json"
    results_path = run_dir / "benchmark_results.json"

    # Configure logging
    log_formatter = logging.Formatter("%(asctime)s [%(name)s:%(levelname)s] %(message)s")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Benchmark run started. Results will be saved in: {run_dir}")

    # Save arguments to config.json
    try:
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
        logger.info(f"Run configuration saved to {config_path}")
    except IOError as e:
        logger.error(f"Failed to save config: {e}")

    # --- Run Benchmark ---
    try:
        benchmark = LLMBenchmark(
            model_name=args.model,
            nsys=args.nsys,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            data_parallel_size=args.data_parallel_size,
            max_num_batched_tokens=args.max_num_batched_tokens,
            download_dir=args.download_dir
        )
        
        if args.trace:
            await benchmark.run_benchmark(trace_path=args.trace)
        else:
            try:
                with open(args.prompts_file, "r", encoding="utf-8") as f:
                    prompts = [line.strip() for line in f if line.strip()]
                logger.info(f"Read {len(prompts)} prompts from {args.prompts_file}")
            except FileNotFoundError:
                logger.error(f"Error: Prompts file not found at {args.prompts_file}")
                return
            
            gen_params = {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens}
            await benchmark.run_benchmark(prompts=prompts, gen_params=gen_params, concurrency=args.concurrency)
        
        benchmark.print_results()
        benchmark.save_results_to_json(results_path)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
