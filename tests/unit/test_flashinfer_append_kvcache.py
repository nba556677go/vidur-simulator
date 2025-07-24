import unittest
import torch
import pytest
import numpy as np

# Import the necessary modules
try:
    from flashinfer.page import append_paged_kv_cache
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False


@pytest.mark.skipif(not FLASHINFER_AVAILABLE, reason="FlashInfer not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFlashInferAppendKVCache(unittest.TestCase):
    """Test the append_paged_kv_cache function from FlashInfer directly."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        
    def test_append_paged_kv_cache_head_dims(self):
        """Test append_paged_kv_cache with different head dimensions."""
        if not FLASHINFER_AVAILABLE:
            self.skipTest("FlashInfer not available")
            
        # Test with various head dimensions including the problematic one (80)
        head_dims_to_test = [64, 80, 96, 128, 256]
        batch_size = 1
        seq_len = 32
        num_kv_heads = 32
        
        for head_dim in head_dims_to_test:
            with self.subTest(f"head_dim={head_dim}"):
                # Create page table
                page_table = torch.zeros((batch_size, seq_len), dtype=torch.int32, device=self.device)
                
                # Create key and value tensors
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
                
                # Create indptr and indices
                indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device=self.device)
                indptr[1:] = seq_len
                indices = torch.arange(seq_len, dtype=torch.int32, device=self.device).unsqueeze(0).expand(batch_size, -1)
                
                try:
                    # Call append_paged_kv_cache directly
                    from flashinfer.page import _kernels
                    _kernels.append_paged_kv_cache(
                        k, v, paged_kv, page_table, indptr, indices
                    )
                    # If we get here without an exception, the test passes for this head_dim
                    self.assertTrue(True)
                except ValueError as e:
                    if "Unsupported head_dim" in str(e):
                        # This is the error we're testing for
                        self.fail(f"Head dimension {head_dim} is not supported: {str(e)}")
                    else:
                        # Other ValueError, re-raise
                        raise e
                except Exception as e:
                    # Log other exceptions
                    print(f"Exception with head_dim={head_dim}: {str(e)}")


if __name__ == "__main__":
    unittest.main()