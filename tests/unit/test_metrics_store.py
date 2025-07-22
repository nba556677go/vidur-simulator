import unittest
import types
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the mock implementation from the attention wrapper
from vidur.profiling.attention.attention_wrapper import MetricsStore

class TestMetricsStore(unittest.TestCase):
    def test_metrics_store_initialization(self):
        """Test that MetricsStore is properly initialized with dummy methods."""
        # Check that the instance exists
        self.assertIsNotNone(MetricsStore._instance)
        
        # Check that the instance is a SimpleNamespace
        self.assertIsInstance(MetricsStore._instance, types.SimpleNamespace)
        
        # Check that the required methods exist and work as expected
        self.assertFalse(MetricsStore._instance.is_op_enabled(op_name="test"))
        self.assertIsNone(MetricsStore._instance.push_operation_metrics(op_name="test"))
        self.assertIsNone(MetricsStore._instance.push_operation_metrics_events(op_name="test"))
        
        # Check that get_instance returns the instance
        self.assertEqual(MetricsStore.get_instance(), MetricsStore._instance)

if __name__ == '__main__':
    unittest.main()