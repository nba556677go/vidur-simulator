import pandas as pd
import ast
import numpy as np
import math

# Read the CSV file
df = pd.read_csv('misc/scaled_mooncake.csv')

print(f"Total rows in the dataset: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()

# Function to safely parse block_hash_ids
def parse_block_hash_ids(block_hash_str):
    try:
        # Convert string representation of list to actual list
        block_list = ast.literal_eval(block_hash_str)
        return len(block_list)
    except:
        return None

# Add columns for analysis
df['total_tokens'] = df['num_prefill_tokens'] + df['num_decode_tokens']
df['actual_blocks'] = df['block_hash_ids'].apply(parse_block_hash_ids)
df['expected_blocks_16'] = df['total_tokens'].apply(lambda x: math.floor(x / 16))
df['formula_matches'] = df['actual_blocks'] == df['expected_blocks_16']

# Also check with block_size from the data (if it's not 16)
if 'block_size' in df.columns:
    block_size_values = df['block_size'].unique()
    print(f"Block size values in data: {block_size_values}")
    
    # Use the actual block size from the data
    df['expected_blocks_actual'] = df.apply(lambda row: math.floor(row['total_tokens'] / row['block_size']), axis=1)
    df['formula_matches_actual'] = df['actual_blocks'] == df['expected_blocks_actual']

print("\n=== ANALYSIS: Is number of blocks = floor((prefill + decode) / 16)? ===")
print(f"Rows where formula matches (using 16): {df['formula_matches'].sum()}")
print(f"Rows where formula doesn't match (using 16): {(~df['formula_matches']).sum()}")
print(f"Percentage matching with 16: {df['formula_matches'].mean() * 100:.2f}%")

if 'block_size' in df.columns:
    print(f"\nRows where formula matches (using actual block_size): {df['formula_matches_actual'].sum()}")
    print(f"Rows where formula doesn't match (using actual block_size): {(~df['formula_matches_actual']).sum()}")
    print(f"Percentage matching with actual block_size: {df['formula_matches_actual'].mean() * 100:.2f}%")

print("\n=== SAMPLE COMPARISONS ===")
comparison_cols = ['num_prefill_tokens', 'num_decode_tokens', 'total_tokens', 'actual_blocks', 'expected_blocks_16', 'formula_matches']
if 'block_size' in df.columns:
    comparison_cols.extend(['block_size', 'expected_blocks_actual', 'formula_matches_actual'])

print(df[comparison_cols].head(10).to_string(index=False))

# Show mismatched cases
mismatched_16 = df[~df['formula_matches']]
if len(mismatched_16) > 0:
    print(f"\n=== MISMATCHED CASES (using 16) - First 10 ===")
    print(mismatched_16[comparison_cols].head(10).to_string(index=False))
    
    print(f"\n=== MISMATCHED CASES STATISTICS ===")
    print(f"Difference (actual - expected): {(mismatched_16['actual_blocks'] - mismatched_16['expected_blocks_16']).describe()}")
else:
    print("\n=== NO MISMATCHED CASES FOUND (using 16) ===")

if 'block_size' in df.columns:
    mismatched_actual = df[~df['formula_matches_actual']]
    if len(mismatched_actual) > 0:
        print(f"\n=== MISMATCHED CASES (using actual block_size) - First 10 ===")
        print(mismatched_actual[comparison_cols].head(10).to_string(index=False))
    else:
        print("\n=== NO MISMATCHED CASES FOUND (using actual block_size) ===")

print("\n=== DETAILED STATISTICS ===")
print(f"Total tokens - Min: {df['total_tokens'].min()}, Max: {df['total_tokens'].max()}, Mean: {df['total_tokens'].mean():.2f}")
print(f"Actual blocks - Min: {df['actual_blocks'].min()}, Max: {df['actual_blocks'].max()}, Mean: {df['actual_blocks'].mean():.2f}")
print(f"Expected blocks (÷16) - Min: {df['expected_blocks_16'].min()}, Max: {df['expected_blocks_16'].max()}, Mean: {df['expected_blocks_16'].mean():.2f}")

# Check if there are any edge cases
print(f"\n=== EDGE CASES ===")
print(f"Rows with 0 total tokens: {(df['total_tokens'] == 0).sum()}")
print(f"Rows with 0 actual blocks: {(df['actual_blocks'] == 0).sum()}")
print(f"Rows with total_tokens < 16: {(df['total_tokens'] < 16).sum()}")

# Check specific cases where total_tokens < 16
small_token_cases = df[df['total_tokens'] < 16]
if len(small_token_cases) > 0:
    print(f"\n=== CASES WITH TOTAL_TOKENS < 16 ===")
    print(small_token_cases[comparison_cols].to_string(index=False))
