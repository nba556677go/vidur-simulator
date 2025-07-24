#!/usr/bin/env python3
"""
Test runner script for FlashInfer-related tests.
This script runs the tests that specifically test the append_paged_kv_cache function.
"""

import unittest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the test modules
from tests.unit.test_append_page_kvcache import TestAppendPageKVCache
from tests.unit.test_flashinfer_append_kvcache import TestFlashInferAppendKVCache
from tests.unit.test_qwen_model_profiling import TestQwenModelProfiling

if __name__ == "__main__":
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add the test cases
    test_suite.addTest(unittest.makeSuite(TestAppendPageKVCache))
    test_suite.addTest(unittest.makeSuite(TestFlashInferAppendKVCache))
    test_suite.addTest(unittest.makeSuite(TestQwenModelProfiling))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful())