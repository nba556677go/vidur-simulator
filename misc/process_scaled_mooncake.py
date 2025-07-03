#!/usr/bin/env python3

import pandas as pd
import numpy as np

def process_scaled_mooncake():
    """
    Process the scaled_mooncake.csv file to:
    1. Reverse/swap num_prefill_tokens and num_decode_tokens
    2. Create separate CSVs for short prefill and long prefill scenarios
    """
    
    # Read the original CSV file
    print("Reading original CSV file...")
    df = pd.read_csv('misc/scaled_mooncake.csv')
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Display some statistics about prefill tokens
    print(f"\nOriginal prefill token statistics:")
    print(f"Min: {df['num_prefill_tokens'].min()}")
    print(f"Max: {df['num_prefill_tokens'].max()}")
    print(f"Mean: {df['num_prefill_tokens'].mean():.2f}")
    print(f"Median: {df['num_prefill_tokens'].median():.2f}")
    
    print(f"\nOriginal decode token statistics:")
    print(f"Min: {df['num_decode_tokens'].min()}")
    print(f"Max: {df['num_decode_tokens'].max()}")
    print(f"Mean: {df['num_decode_tokens'].mean():.2f}")
    print(f"Median: {df['num_decode_tokens'].median():.2f}")
    
    # Create a copy for the reversed data
    df_reversed = df.copy()
    
    # Swap/reverse the num_prefill_tokens and num_decode_tokens columns
    print("\nSwapping num_prefill_tokens and num_decode_tokens...")
    df_reversed['num_prefill_tokens'], df_reversed['num_decode_tokens'] = (
        df['num_decode_tokens'].copy(), 
        df['num_prefill_tokens'].copy()
    )
    
    # Save the reversed data
    output_file_reversed = 'misc/scaled_mooncake_reversed.csv'
    df_reversed.to_csv(output_file_reversed, index=False)
    print(f"Saved reversed data to: {output_file_reversed}")
    
    # Determine threshold for short vs long prefill
    # Using median as threshold for categorization
    median_prefill = df_reversed['num_prefill_tokens'].median()
    print(f"\nUsing median prefill tokens ({median_prefill}) as threshold for short/long categorization")
    
    # Create short prefill dataset (prefill tokens <= median)
    df_short_prefill = df_reversed[df_reversed['num_prefill_tokens'] <= median_prefill].copy()
    
    # Create long prefill dataset (prefill tokens > median)
    df_long_prefill = df_reversed[df_reversed['num_prefill_tokens'] > median_prefill].copy()
    
    print(f"\nShort prefill dataset shape: {df_short_prefill.shape}")
    print(f"Long prefill dataset shape: {df_long_prefill.shape}")
    
    # Save short prefill dataset
    output_file_short = 'misc/scaled_mooncake_short_prefill.csv'
    df_short_prefill.to_csv(output_file_short, index=False)
    print(f"Saved short prefill data to: {output_file_short}")
    
    # Save long prefill dataset
    output_file_long = 'misc/scaled_mooncake_long_prefill.csv'
    df_long_prefill.to_csv(output_file_long, index=False)
    print(f"Saved long prefill data to: {output_file_long}")
    
    # Display statistics for the new datasets
    print(f"\nShort prefill statistics:")
    print(f"Prefill tokens - Min: {df_short_prefill['num_prefill_tokens'].min()}, Max: {df_short_prefill['num_prefill_tokens'].max()}")
    print(f"Decode tokens - Min: {df_short_prefill['num_decode_tokens'].min()}, Max: {df_short_prefill['num_decode_tokens'].max()}")
    
    print(f"\nLong prefill statistics:")
    print(f"Prefill tokens - Min: {df_long_prefill['num_prefill_tokens'].min()}, Max: {df_long_prefill['num_prefill_tokens'].max()}")
    print(f"Decode tokens - Min: {df_long_prefill['num_decode_tokens'].min()}, Max: {df_long_prefill['num_decode_tokens'].max()}")
    
    return df_reversed, df_short_prefill, df_long_prefill

if __name__ == "__main__":
    process_scaled_mooncake()
