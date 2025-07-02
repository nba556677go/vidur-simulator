import pandas as pd
import ast
import numpy as np

# Read the CSV file
df = pd.read_csv('data/processed_traces/mooncake_conversation_trace.csv')

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
df['block_hash_count'] = df['block_hash_ids'].apply(parse_block_hash_ids)

# Check if lengths match
df['lengths_match'] = df['total_tokens'] == df['block_hash_count']

# Display summary statistics
print("=== SUMMARY ANALYSIS ===")
print(f"Rows with matching lengths: {df['lengths_match'].sum()}")
print(f"Rows with non-matching lengths: {(~df['lengths_match']).sum()}")
print(f"Percentage matching: {df['lengths_match'].mean() * 100:.2f}%")
print()

# Show first few rows for verification
print("=== FIRST 10 ROWS COMPARISON ===")
comparison_df = df[['num_prefill_tokens', 'num_decode_tokens', 'total_tokens', 'block_hash_count', 'lengths_match']].head(10)
print(comparison_df.to_string(index=False))
print()

# Show cases where lengths don't match (if any)
mismatched = df[~df['lengths_match']]
if len(mismatched) > 0:
    print("=== MISMATCHED CASES ===")
    print(mismatched[['num_prefill_tokens', 'num_decode_tokens', 'total_tokens', 'block_hash_count', 'lengths_match']].head(10))
else:
    print("=== NO MISMATCHED CASES FOUND ===")

print()

# Additional statistics
print("=== DETAILED STATISTICS ===")
print(f"Block size (from block_size column): {df['block_size'].unique()}")
print(f"Average total tokens: {df['total_tokens'].mean():.2f}")
print(f"Average block hash count: {df['block_hash_count'].mean():.2f}")
print(f"Min total tokens: {df['total_tokens'].min()}")
print(f"Max total tokens: {df['total_tokens'].max()}")
print(f"Min block hash count: {df['block_hash_count'].min()}")
print(f"Max block hash count: {df['block_hash_count'].max()}")
