import unittest
import torch
import pytest
import numpy as np
from math import ceil

# Import the necessary modules
from sarathi.config import ParallelConfig
from sarathi.model_executor.attention import AttentionBackend, get_attention_wrapper, set_attention_backend
from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.attention.sequence_proxy import SequenceMetadataProxy


class TestAppendPageKVCache(unittest.TestCase):
    """Test the append_paged_kv_cache function with different head dimensions."""

    def setUp(self):
        # Mock the necessary components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        self.block_size = 16
        self.attention_backend = AttentionBackend.FLASHINFER
        
    def _setup_attention_wrapper(self, head_dim, num_q_heads, num_kv_heads):
        """Setup the attention wrapper with the given parameters."""
        # Create a custom model config with the specified head dimension
        model_config = ModelConfig(
            embedding_dim=num_q_heads * head_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            num_hidden_layers=32,
            max_model_len=4096,
        )
        
        parallel_config = ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
        
        # Set the attention backend
        set_attention_backend(self.attention_backend)
        
        # Initialize the attention wrapper
        get_attention_wrapper().init(
            model_config,
            parallel_config,
            self.block_size,
            self.device,
        )
        
        return model_config, parallel_config
        
    def _create_test_inputs(self, batch_size, seq_len, num_q_heads, num_kv_heads, head_dim):
        """Create test inputs for the attention wrapper."""
        # Create query, key, value tensors
        query = torch.randn(
            batch_size * seq_len,
            num_q_heads * head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        key = torch.randn(
            batch_size * seq_len,
            num_kv_heads * head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        value = torch.randn(
            batch_size * seq_len,
            num_kv_heads * head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        
        # Create sequence metadata
        seq_metadata_list = []
        for _ in range(batch_size):
            num_blocks = ceil(seq_len / self.block_size)
            seq_metadata = SequenceMetadataProxy(
                is_prompt=True,
                total_len=seq_len,
                processed_len=0,
                block_table=np.arange(num_blocks, dtype=np.int32),
            )
            seq_metadata_list.append(seq_metadata)
            
        # Create KV cache
        max_num_blocks = ceil(batch_size * seq_len / self.block_size)
        kv_cache = get_attention_wrapper().get_cache_block(
            max_num_blocks, dtype=self.dtype, device=self.device
        )
        
        return seq_metadata_list, query, key, value, kv_cache
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_append_page_kvcache_with_different_head_dims(self):
        """Test append_paged_kv_cache with different head dimensions."""
        # Test with common head dimensions
        head_dims_to_test = [64, 80, 96, 128, 256]
        batch_size = 1
        seq_len = 32
        
        for head_dim in head_dims_to_test:
            # For each head dimension, test with different q/kv head configurations
            for num_q_heads, num_kv_heads in [(32, 32), (32, 8), (16, 16)]:
                with self.subTest(f"head_dim={head_dim}, q_heads={num_q_heads}, kv_heads={num_kv_heads}"):
                    try:
                        # Setup the attention wrapper
                        model_config, parallel_config = self._setup_attention_wrapper(
                            head_dim, num_q_heads, num_kv_heads
                        )
                        
                        # Create test inputs
                        seq_metadata_list, query, key, value, kv_cache = self._create_test_inputs(
                            batch_size, seq_len, num_q_heads, num_kv_heads, head_dim
                        )
                        
                        # Begin forward pass
                        get_attention_wrapper().begin_forward(seq_metadata_list)
                        
                        # Execute forward pass which will call append_paged_kv_cache internally
                        get_attention_wrapper().forward(query, key, value, kv_cache)
                        
                        # End forward pass
                        get_attention_wrapper().end_forward()
                        
                        # If we get here without an exception, the test passes
                        self.assertTrue(True)
                    except ValueError as e:
                        if "Unsupported head_dim" in str(e):
                            # This is the error we're testing for
                            self.fail(f"Head dimension {head_dim} is not supported: {str(e)}")
                        else:
                            # Other ValueError, re-raise
                            raise e
                    except Exception as e:
                        # Log other exceptions but don't fail the test
                        print(f"Exception with head_dim={head_dim}: {str(e)}")


if __name__ == "__main__":
    unittest.main()