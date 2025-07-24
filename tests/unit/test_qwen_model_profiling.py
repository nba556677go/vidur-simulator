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
from vidur.profiling.attention.attention_input import AttentionInput


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestQwenModelProfiling(unittest.TestCase):
    """Test profiling with Qwen models that have head_dim=80."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        self.block_size = 16
        self.attention_backend = AttentionBackend.FLASHINFER
        
    def _get_qwen_model_config(self, model_name="Qwen/Qwen3-4B"):
        """Get the model config for Qwen models."""
        try:
            return ModelConfig.from_model_name(model_name)
        except Exception as e:
            self.skipTest(f"Could not load model config for {model_name}: {str(e)}")
            
    def test_qwen_model_head_dim(self):
        """Test that Qwen models have the expected head dimension."""
        model_config = self._get_qwen_model_config()
        head_dim = model_config.get_head_size()
        
        # Verify that Qwen models have head_dim=80
        self.assertEqual(head_dim, 80, "Qwen model should have head_dim=80")
        
    def test_qwen_model_profiling_setup(self):
        """Test the setup for profiling Qwen models."""
        model_config = self._get_qwen_model_config()
        parallel_config = ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
        
        # Set the attention backend
        set_attention_backend(self.attention_backend)
        
        try:
            # Initialize the attention wrapper
            get_attention_wrapper().init(
                model_config,
                parallel_config,
                self.block_size,
                self.device,
            )
            
            # If we get here without an exception, the initialization works
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Failed to initialize attention wrapper: {str(e)}")
            
    def test_qwen_model_forward_pass(self):
        """Test the forward pass with Qwen models."""
        model_config = self._get_qwen_model_config()
        parallel_config = ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
        
        # Set the attention backend
        set_attention_backend(self.attention_backend)
        
        try:
            # Initialize the attention wrapper
            get_attention_wrapper().init(
                model_config,
                parallel_config,
                self.block_size,
                self.device,
            )
            
            # Create test inputs
            batch_size = 1
            seq_len = 32
            num_q_heads = model_config.num_q_heads
            num_kv_heads = model_config.num_kv_heads
            head_dim = model_config.get_head_size()
            
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
            
            # Begin forward pass
            get_attention_wrapper().begin_forward(seq_metadata_list)
            
            # Execute forward pass which will call append_paged_kv_cache internally
            try:
                get_attention_wrapper().forward(query, key, value, kv_cache)
                self.fail("Expected ValueError for unsupported head_dim=80")
            except ValueError as e:
                # This is the error we're expecting
                self.assertIn("Unsupported head_dim: 80", str(e))
            except Exception as e:
                # Other exceptions are unexpected
                self.fail(f"Unexpected exception: {str(e)}")
            finally:
                # End forward pass
                get_attention_wrapper().end_forward()
                
    def test_supported_head_dims(self):
        """Test which head dimensions are supported by FlashInfer."""
        try:
            from flashinfer.page import _kernels
            
            # Test various head dimensions
            head_dims_to_test = [32, 64, 80, 96, 128, 256]
            batch_size = 1
            seq_len = 32
            num_kv_heads = 32
            
            supported_dims = []
            unsupported_dims = []
            
            for head_dim in head_dims_to_test:
                # Create tensors
                k = torch.randn(
                    batch_size * seq_len,
                    num_kv_heads,
                    head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
                v = torch.randn(
                    batch_size * seq_len,
                    num_kv_heads,
                    head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
                
                # Create paged KV cache
                num_pages = batch_size * seq_len
                paged_kv = torch.zeros(
                    (2, num_pages, num_kv_heads, head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
                
                # Create page table and indices
                page_table = torch.zeros((batch_size, seq_len), dtype=torch.int32, device=self.device)
                indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device=self.device)
                indptr[1:] = seq_len
                indices = torch.arange(seq_len, dtype=torch.int32, device=self.device).unsqueeze(0).expand(batch_size, -1)
                
                try:
                    # Call append_paged_kv_cache directly
                    _kernels.append_paged_kv_cache(
                        k, v, paged_kv, page_table, indptr, indices
                    )
                    supported_dims.append(head_dim)
                except ValueError as e:
                    if "Unsupported head_dim" in str(e):
                        unsupported_dims.append(head_dim)
                    else:
                        raise e
                    
            print(f"Supported head dimensions: {supported_dims}")
            print(f"Unsupported head dimensions: {unsupported_dims}")
            
            # Verify that 80 is in the unsupported list
            self.assertIn(80, unsupported_dims, "head_dim=80 should be unsupported")
            
        except ImportError:
            self.skipTest("FlashInfer not available")


if __name__ == "__main__":
    unittest.main()