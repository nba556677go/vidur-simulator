import subprocess
import os
import argparse
from datetime import datetime

# --- Argument Parsing ---
# Set up an argument parser to accept command-line arguments
parser = argparse.ArgumentParser(description="Run Vidur with different configurations.")
parser.add_argument(
    '--total_gpus', 
    type=int, 
    default=8, 
    help='Total number of GPUs available for the experiment. Default is 8.'
)
parser.add_argument(
    '--log_dir',
    type=str,
    default='./simulation_logs',
    help='Base directory to save the log files. A timestamped subdirectory will be created here. Default is ./logs.'
)
args = parser.parse_args()
TOTAL_GPUS = args.total_gpus
BASE_LOG_DIR = args.log_dir
# --- End Argument Parsing ---

# --- Setup ---
# Create a unique, timestamped directory for this run's logs to prevent overwrites.
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR = os.path.join(BASE_LOG_DIR, timestamp)

# Create the log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)
# --- End Setup ---

# Define the variables and their possible values
cluster_config_num_replicas_list = [1, 2, 4, 8]
replica_config_tensor_parallel_size_list = [1, 2, 4, 8]
replica_config_pipeline_parallel_size_list = [1, 2, 4, 8]

# Base command template
base_command = (
    "python -m vidur.main "
    "--time_limit 10000 "
    "--replica_config_model_name meta-llama/Meta-Llama-3-8B "
    "--replica_config_device a100 "
    "--replica_config_network_device a100_dgx "
    "--request_generator_config_type synthetic "
    "--synthetic_request_generator_config_num_requests 128 "
    "--length_generator_config_type trace "
    "--trace_request_length_generator_config_trace_file ./misc/scaled_mooncake.csv "
    "--interval_generator_config_type poisson "
    "--poisson_request_interval_generator_config_qps 8.0 "
    "--global_scheduler_config_type round_robin "
    "--replica_scheduler_config_type vllm_v1 "
    "--vllm_v1_scheduler_config_chunk_size 512 "
    "--vllm_v1_scheduler_config_batch_size_cap 512 "
    "--cache_config_enable_prefix_caching"
)

print(f"Starting experiment with a total of {TOTAL_GPUS} GPUs available.")
print(f"Logs for this run will be saved in the '{LOG_DIR}' directory.")

# Iterate through all combinations of the variables
for num_replicas in cluster_config_num_replicas_list:
    for tensor_parallel_size in replica_config_tensor_parallel_size_list:
        for pipeline_parallel_size in replica_config_pipeline_parallel_size_list:

            # Calculate the total number of GPUs required for this configuration
            gpus_required = num_replicas * tensor_parallel_size * pipeline_parallel_size

            # Check if the configuration is possible with the available GPUs
            if gpus_required > TOTAL_GPUS:
                print("-" * 80)
                print(f"Skipping impossible configuration:")
                print(f"  - Replicas (DP): {num_replicas}, TP: {tensor_parallel_size}, PP: {pipeline_parallel_size}")
                print(f"  - GPUs Required ({gpus_required}) > Total GPUs Available ({TOTAL_GPUS})")
                print("-" * 80)
                continue  # Skip to the next iteration

            # Create a unique log file name for the current configuration
            log_filename = f"dp{num_replicas}_tp{tensor_parallel_size}_pp{pipeline_parallel_size}.log"
            log_filepath = os.path.join(LOG_DIR, log_filename)

            # Construct the python part of the command
            python_command = (
                f"{base_command} "
                f"--cluster_config_num_replicas {num_replicas} "
                f"--replica_config_tensor_parallel_size {tensor_parallel_size} "
                f"--replica_config_num_pipeline_stages {pipeline_parallel_size}"
            )

            # Construct the full command to activate venv, run the python script,
            # and redirect both stdout and stderr to the log file.
            command_to_run = f"source .venv/bin/activate && {python_command} > {log_filepath} 2>&1"

            print("="*80)
            print(f"Running configuration:")
            print(f"  - Replicas (DP): {num_replicas}")
            print(f"  - Tensor Parallel (TP): {tensor_parallel_size}")
            print(f"  - Pipeline Parallel (PP): {pipeline_parallel_size}")
            print(f"  - GPUs Required: {gpus_required} (of {TOTAL_GPUS} available)")
            print(f"  - Logging output to: {log_filepath}")
            print("="*80)

            try:
                # Execute the command using a shell. We don't need to capture output
                # here anymore since it's being redirected to a file.
                process = subprocess.run(
                    command_to_run, 
                    shell=True,
                    check=True, 
                    text=True,
                    executable='/bin/bash'
                )
                print(f"Command executed successfully. See log for details: {log_filepath}")
            except subprocess.CalledProcessError as e:
                print(f"Command failed with exit code {e.returncode}.")
                print(f"Check the log file for error details: {log_filepath}")
            except FileNotFoundError:
                print(f"Error: The shell '/bin/bash' was not found.")
                print("Please ensure you are on a Unix-like system (Linux, macOS).")
                break # Exit the loop if the shell is not found
            except Exception as e:
                print(f"An unexpected error occurred: {e}")


print("\nAll configurations have been processed.")
