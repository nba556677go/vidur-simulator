import pandas as pd
import numpy as np
import sys

# --- 1. Data Preparation ---
# This script reads a CSV file from a command-line argument,
# scales token counts, and saves the result to a new CSV file,
# overwriting the original token columns.
#
# Usage: python your_script_name.py input.csv output.csv
if len(sys.argv) != 3:
    print("Usage: python your_script_name.py <input_csv_path> <output_csv_path>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: Input file not found at {input_file}")
    sys.exit(1)


# --- 2. Scaling Logic Definition ---
# Define the maximum total tokens allowed for prefill and decode combined.
MAX_TOKEN_LENGTH = 8000

# Calculate the original total tokens for scaling purposes.
original_total_tokens = df['num_prefill_tokens'] + df['num_decode_tokens']

# Calculate the scaling ratio.
# The ratio is calculated only for rows where the total tokens exceed the max length.
# For rows already under the limit, the ratio is 1 (no change).
# We use np.where for a vectorized and efficient conditional calculation.
ratio = np.where(
    original_total_tokens > MAX_TOKEN_LENGTH,
    MAX_TOKEN_LENGTH / original_total_tokens,
    1.0  # Keep original values if total is within the limit
)

# --- 3. Apply Scaling ---
# Apply the scaling ratio and create temporary scaled columns.
# The result is cast to an integer, as token counts must be whole numbers.
scaled_prefill = (df['num_prefill_tokens'] * ratio).astype(int)
scaled_decode = (df['num_decode_tokens'] * ratio).astype(int)

# --- 4. Ensure Decode Tokens > 0 and Adjust ---
# Identify rows where scaling resulted in zero decode tokens but was not zero originally.
# This can happen if num_decode_tokens is very small compared to num_prefill_tokens.
zero_decode_mask = (scaled_decode == 0) & (df['num_decode_tokens'] > 0)

# For those rows, set decode tokens to 1 and subtract 1 from prefill tokens
# to maintain the total token count, ensuring prefill doesn't go below zero.
scaled_decode.loc[zero_decode_mask] = 1
scaled_prefill.loc[zero_decode_mask] = (scaled_prefill.loc[zero_decode_mask] - 1).clip(lower=0)

# Overwrite the original columns with the final scaled and adjusted values.
df['num_prefill_tokens'] = scaled_prefill
df['num_decode_tokens'] = scaled_decode

#assert df['num_decode_tokens'] > 0
# --- 5. Display and Save Results ---
# The DataFrame now has the same columns as the input, but with scaled values.
print(f"Scaling token counts to a max total length of {MAX_TOKEN_LENGTH}")
print("Ensuring 'num_decode_tokens' is always greater than 0.")
print("Displaying the first 10 rows of the modified data (original columns are overwritten):")
print(df.head(10).to_string())

# Save the modified DataFrame to a new CSV file.
# The output file will have the exact same columns as the input.
df.to_csv(output_file, index=False)

print(f"\nSuccessfully processed the file.")
print(f"Original data from '{input_file}' has been scaled and saved to '{output_file}'.")
