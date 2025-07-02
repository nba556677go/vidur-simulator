import pandas as pd
import ast
import math

# Read the CSV file
df = pd.read_csv('data/processed_traces/mooncake_conversation_trace.csv')

# Parse block_hash_ids and calculate metrics
def parse_block_hash_ids(block_hash_str):
    try:
        block_list = ast.literal_eval(block_hash_str)
        return len(block_list)
    except:
        return None

df['total_tokens'] = df['num_prefill_tokens'] + df['num_decode_tokens']
df['block_hash_count'] = df['block_hash_ids'].apply(parse_block_hash_ids)

# Calculate expected blocks based on block_size
df['expected_blocks_exact'] = df['total_tokens'] / df['block_size']
df['expected_blocks_ceil'] = df.apply(lambda row: math.ceil(row['total_tokens'] / row['block_size']), axis=1)

# Check different relationships
df['matches_total_tokens'] = df['block_hash_count'] == df['total_tokens']
df['matches_expected_ceil'] = df['block_hash_count'] == df['expected_blocks_ceil']

print("=== RELATIONSHIP ANALYSIS ===")
print(f"Total rows: {len(df)}")
print()

print("1. Block hash count == Total tokens:")
print(f"   Matches: {df['matches_total_tokens'].sum()}")
print(f"   Percentage: {df['matches_total_tokens'].mean() * 100:.2f}%")
print()

print("2. Block hash count == Ceiling(total_tokens / block_size):")
print(f"   Matches: {df['matches_expected_ceil'].sum()}")
print(f"   Percentage: {df['matches_expected_ceil'].mean() * 100:.2f}%")
print()

# Show detailed comparison for first 10 rows
print("=== DETAILED COMPARISON (First 10 rows) ===")
comparison_cols = ['num_prefill_tokens', 'num_decode_tokens', 'total_tokens', 
                   'block_size', 'expected_blocks_ceil', 'block_hash_count', 
                   'matches_total_tokens', 'matches_expected_ceil']
print(df[comparison_cols].head(10).to_string(index=False))

# Check actual ratio
df['ratio_tokens_to_blocks'] = df['total_tokens'] / df['block_hash_count']
print(f"\n=== TOKEN TO BLOCK RATIO ANALYSIS ===")
print(f"Average ratio (tokens/blocks): {df['ratio_tokens_to_blocks'].mean():.2f}")
print(f"Min ratio: {df['ratio_tokens_to_blocks'].min():.2f}")
print(f"Max ratio: {df['ratio_tokens_to_blocks'].max():.2f}")
print(f"Standard deviation: {df['ratio_tokens_to_blocks'].std():.2f}")

# Summary statistics
print(f"\n=== SUMMARY ===")
print(f"Block size values: {df['block_size'].unique()}")
print(f"All block_size values are: {df['block_size'].iloc[0]}")

# Check if the relationship is exactly ceil(tokens/16)
print(f"\nFirst few manual calculations:")
for i in range(5):
    tokens = df.iloc[i]['total_tokens']
    blocks = df.iloc[i]['block_hash_count']
    expected = math.ceil(tokens / 16)
    print(f"Row {i}: tokens={tokens}, blocks={blocks}, ceil(tokens/16)={expected}, match={blocks==expected}")
