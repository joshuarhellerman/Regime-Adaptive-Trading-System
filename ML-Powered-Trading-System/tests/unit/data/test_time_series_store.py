import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime

import numpy as np
import pytest

from data.storage.time_series_store import TimeSeriesStore, Resolution, Aggregation


class TestTimeSeriesStore(unittest.TestCase):
    """Unit tests for TimeSeriesStore class"""

    def setUp(self):
        """Set up test environment before each test"""
        # Create a temporary directory for the tests
        self.test_dir = tempfile.mkdtemp()
        self.store = TimeSeriesStore(self.test_dir)

    def tearDown(self):
        """Clean up test environment after each test"""
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    def test_init(self):
        """Test initialization of TimeSeriesStore"""
        # Check that the directory was created
        self.assertTrue(os.path.exists(self.test_dir))
        
        # Check that resolution subdirectories were created
        for resolution in Resolution:
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, resolution.value)))

    def test_store(self):
        """Test storing data in a time series"""
        # Store a value
        series_name = "test_series"
        value = 42
        timestamp = time.time()
        
        result = self.store.store(series_name, value, timestamp)
        self.assertTrue(result)
        
        # Check that the value was stored
        last_point = self.store.get_last(series_name)
        self.assertIsNotNone(last_point)
        self.assertEqual(last_point[0], timestamp)
        self.assertEqual(last_point[1], value)

    def test_store_multiple(self):
        """Test storing multiple values in a time series"""
        series_name = "test_series"
        data = [
            (time.time() - 10, 1),
            (time.time() - 5, 2),
            (time.time(), 3)
        ]
        
        # Store multiple values
        for timestamp, value in data:
            self.store.store(series_name, value, timestamp)
        
        # Check metadata
        self.assertEqual(self.store.count(series_name), 3)
        
        # Check first and last values
        first_point = self.store.get_first(series_name)
        last_point = self.store.get_last(series_name)
        
        self.assertEqual(first_point[0], data[0][0])
        self.assertEqual(first_point[1], data[0][1])
        self.assertEqual(last_point[0], data[2][0])
        self.assertEqual(last_point[1], data[2][1])

    def test_store_unordered(self):
        """Test storing values with unordered timestamps"""
        series_name = "test_series"
        
        # Create data with unordered timestamps
        now = time.time()
        data = [
            (now, 3),
            (now - 10, 1),
            (now - 5, 2)
        ]
        
        # Store values
        for timestamp, value in data:
            self.store.store(series_name, value, timestamp)
        
        # Check that data is properly ordered in get
        start_time = now - 20
        end_time = now + 10
        result = self.store.get(series_name, start_time, end_time)
        
        # Should be sorted by timestamp
        expected = sorted(data, key=lambda x: x[0])
        self.assertEqual(len(result), 3)
        for i, (timestamp, value) in enumerate(expected):
            self.assertEqual(result[i][0], timestamp)
            self.assertEqual(result[i][1], value)

    def test_get_range(self):
        """Test getting a range of values"""
        series_name = "test_series"
        now = time.time()
        
        # Store values
        data = [
            (now - 100, 1),
            (now - 50, 2),
            (now, 3)
        ]
        
        for timestamp, value in data:
            self.store.store(series_name, value, timestamp)
        
        # Test get with different ranges
        # 1. Full range
        result = self.store.get(series_name, now - 200, now + 100)
        self.assertEqual(len(result), 3)
        
        # 2. Partial range
        result = self.store.get(series_name, now - 75, now - 25)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 2)
        
        # 3. Empty range
        result = self.store.get(series_name, now + 10, now + 20)
        self.assertEqual(len(result), 0)

    def test_get_with_resolutions(self):
        """Test getting data with different resolutions"""
        series_name = "test_series"
        now = time.time()
        base_time = int(now / 60) * 60  # Round to minute boundary
        
        # Store values every second for a minute
        for i in range(60):
            self.store.store(series_name, i, base_time + i)
        
        # Test different resolutions
        # 1. Raw resolution
        raw_data = self.store.get(series_name, base_time, base_time + 60)
        self.assertEqual(len(raw_data), 60)
        
        # 2. Second resolution (should be the same as raw in this case)
        sec_data = self.store.get(series_name, base_time, base_time + 60, 
                                  resolution=Resolution.SECOND)
        self.assertEqual(len(sec_data), 60)
        
        # 3. Minute resolution (should downsample to 1 point)
        min_data = self.store.get(series_name, base_time, base_time + 60, 
                                 resolution=Resolution.MINUTE)
        self.assertEqual(len(min_data), 1)
        
        # By default, it uses the LAST aggregation
        self.assertEqual(min_data[0][1], 59)  # Last value in the minute

    def test_aggregations(self):
        """Test different aggregation methods"""
        series_name = "test_series"
        now = time.time()
        base_time = int(now / 60) * 60  # Round to minute boundary
        
        # Store values 0-9 for a single minute
        for i in range(10):
            self.store.store(series_name, i, base_time + i)
        
        # Test different aggregations with MINUTE resolution
        # 1. MEAN
        data = self.store.get(series_name, base_time, base_time + 10,
                             resolution=Resolution.MINUTE,
                             aggregation=Aggregation.MEAN)
        self.assertEqual(data[0][1], 4.5)  # Mean of 0-9
        
        # 2. SUM
        data = self.store.get(series_name, base_time, base_time + 10,
                             resolution=Resolution.MINUTE,
                             aggregation=Aggregation.SUM)
        self.assertEqual(data[0][1], 45)  # Sum of 0-9
        
        # 3. MIN
        data = self.store.get(series_name, base_time, base_time + 10,
                             resolution=Resolution.MINUTE,
                             aggregation=Aggregation.MIN)
        self.assertEqual(data[0][1], 0)
        
        # 4. MAX
        data = self.store.get(series_name, base_time, base_time + 10,
                             resolution=Resolution.MINUTE,
                             aggregation=Aggregation.MAX)
        self.assertEqual(data[0][1], 9)
        
        # 5. FIRST
        data = self.store.get(series_name, base_time, base_time + 10,
                             resolution=Resolution.MINUTE,
                             aggregation=Aggregation.FIRST)
        self.assertEqual(data[0][1], 0)
        
        # 6. LAST
        data = self.store.get(series_name, base_time, base_time + 10,
                             resolution=Resolution.MINUTE,
                             aggregation=Aggregation.LAST)
        self.assertEqual(data[0][1], 9)
        
        # 7. COUNT
        data = self.store.get(series_name, base_time, base_time + 10,
                             resolution=Resolution.MINUTE,
                             aggregation=Aggregation.COUNT)
        self.assertEqual(data[0][1], 10)

    def test_non_numeric_values(self):
        """Test storing and retrieving non-numeric values"""
        series_name = "test_series"
        now = time.time()
        
        # Store string values
        self.store.store(series_name, "value1", now - 10)
        self.store.store(series_name, "value2", now - 5)
        self.store.store(series_name, "value3", now)
        
        # Retrieve values
        data = self.store.get(series_name, now - 20, now + 10)
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0][1], "value1")
        self.assertEqual(data[1][1], "value2")
        self.assertEqual(data[2][1], "value3")
        
        # Non-numeric values should use the last value when aggregating with numeric methods
        data = self.store.get(series_name, now - 20, now + 10,
                             resolution=Resolution.MINUTE,
                             aggregation=Aggregation.MEAN)
        self.assertEqual(data[0][1], "value3")  # Last value

    def test_complex_values(self):
        """Test storing and retrieving complex data structures"""
        series_name = "test_series"
        now = time.time()
        
        # Store dictionary
        dict_value = {"key1": "value1", "key2": 42}
        self.store.store(series_name, dict_value, now - 10)
        
        # Store list
        list_value = [1, 2, 3, 4, 5]
        self.store.store(series_name, list_value, now - 5)
        
        # Store numpy array
        array_value = np.array([1, 2, 3])
        self.store.store(series_name, array_value, now)
        
        # Retrieve and check values
        data = self.store.get(series_name, now - 20, now + 10)
        self.assertEqual(len(data), 3)
        
        self.assertEqual(data[0][1], dict_value)
        self.assertEqual(data[1][1], list_value)
        # Numpy arrays don't compare equal directly
        np.testing.assert_array_equal(data[2][1], array_value)

    def test_get_last(self):
        """Test getting the last value from a time series"""
        series_name = "test_series"
        
        # Case 1: Empty series
        self.assertIsNone(self.store.get_last(series_name))
        
        # Case 2: Series with values
        now = time.time()
        self.store.store(series_name, 1, now - 10)
        self.store.store(series_name, 2, now - 5)
        self.store.store(series_name, 3, now)
        
        last_point = self.store.get_last(series_name)
        self.assertIsNotNone(last_point)
        self.assertEqual(last_point[0], now)
        self.assertEqual(last_point[1], 3)

    def test_get_first(self):
        """Test getting the first value from a time series"""
        series_name = "test_series"
        
        # Case 1: Empty series
        self.assertIsNone(self.store.get_first(series_name))
        
        # Case 2: Series with values
        now = time.time()
        self.store.store(series_name, 1, now - 10)
        self.store.store(series_name, 2, now - 5)
        self.store.store(series_name, 3, now)
        
        first_point = self.store.get_first(series_name)
        self.assertIsNotNone(first_point)
        self.assertEqual(first_point[0], now - 10)
        self.assertEqual(first_point[1], 1)

    def test_get_range_info(self):
        """Test getting the time range of a time series"""
        series_name = "test_series"
        
        # Case 1: Empty series
        self.assertIsNone(self.store.get_range(series_name))
        
        # Case 2: Series with values
        now = time.time()
        self.store.store(series_name, 1, now - 10)
        self.store.store(series_name, 2, now - 5)
        self.store.store(series_name, 3, now)
        
        range_info = self.store.get_range(series_name)
        self.assertIsNotNone(range_info)
        self.assertEqual(range_info[0], now - 10)  # First timestamp
        self.assertEqual(range_info[1], now)       # Last timestamp

    def test_get_series_names(self):
        """Test getting names of all available time series"""
        # Case 1: No series
        self.assertEqual(self.store.get_series_names(), [])
        
        # Case 2: Multiple series
        self.store.store("series1", 1)
        self.store.store("series2", 2)
        self.store.store("series3", 3)
        
        series_names = self.store.get_series_names()
        self.assertEqual(len(series_names), 3)
        self.assertIn("series1", series_names)
        self.assertIn("series2", series_names)
        self.assertIn("series3", series_names)

    def test_get_metadata(self):
        """Test getting metadata for a time series"""
        series_name = "test_series"
        
        # Case 1: Series doesn't exist
        self.assertEqual(self.store.get_metadata(series_name), {})
        
        # Case 2: Series with numeric values
        now = time.time()
        self.store.store(series_name, 10, now - 10)
        self.store.store(series_name, 5, now - 5)
        self.store.store(series_name, 20, now)
        
        metadata = self.store.get_metadata(series_name)
        self.assertEqual(metadata["count"], 3)
        self.assertEqual(metadata["first_timestamp"], now - 10)
        self.assertEqual(metadata["last_timestamp"], now)
        self.assertEqual(metadata["min_value"], 5)
        self.assertEqual(metadata["max_value"], 20)

    def test_count(self):
        """Test counting points in a time series"""
        series_name = "test_series"
        
        # Case 1: Series doesn't exist
        self.assertEqual(self.store.count(series_name), 0)
        
        # Case 2: Series with values
        for i in range(5):
            self.store.store(series_name, i)
        
        self.assertEqual(self.store.count(series_name), 5)

    def test_delete(self):
        """Test deleting a time series"""
        # Create some series
        self.store.store("series1", 1)
        self.store.store("series2", 2)
        
        # Verify they exist
        self.assertIn("series1", self.store.get_series_names())
        self.assertIn("series2", self.store.get_series_names())
        
        # Delete one series
        result = self.store.delete("series1")
        self.assertTrue(result)
        
        # Verify it was deleted
        self.assertNotIn("series1", self.store.get_series_names())
        self.assertIn("series2", self.store.get_series_names())
        
        # Try to delete non-existent series
        result = self.store.delete("non_existent")
        self.assertFalse(result)

    def test_clear(self):
        """Test clearing all time series data"""
        # Create some series
        self.store.store("series1", 1)
        self.store.store("series2", 2)
        self.store.store("series3", 3)
        
        # Verify they exist
        self.assertEqual(len(self.store.get_series_names()), 3)
        
        # Clear all series
        count = self.store.clear()
        self.assertEqual(count, 3)
        
        # Verify all were cleared
        self.assertEqual(len(self.store.get_series_names()), 0)

    def test_flush(self):
        """Test flushing cached data to disk"""
        # Store some data
        self.store.store("series1", 1)
        
        # Flush data
        result = self.store.flush()
        self.assertTrue(result)
        
        # Check that data was saved (by checking that metadata file exists)
        metadata_file = os.path.join(self.test_dir, "metadata.json")
        self.assertTrue(os.path.exists(metadata_file))

    def test_large_data_flushing(self):
        """Test automatic flushing when cache limit is reached"""
        series_name = "test_series"
        
        # Override max memory points for test
        original_max = self.store._max_memory_points
        self.store._max_memory_points = 5
        
        try:
            # Store more points than the cache limit
            for i in range(10):
                self.store.store(series_name, i)
            
            # Check that data was flushed
            # (We can verify by checking the raw cache size is less than what we inserted)
            self.assertLessEqual(
                len(self.store._series_cache.get(series_name, {}).get(Resolution.RAW, [])), 
                5
            )
            
            # But all points should still be accessible
            self.assertEqual(self.store.count(series_name), 10)
        finally:
            # Restore original max
            self.store._max_memory_points = original_max

    def test_compact(self):
        """Test compacting time series data"""
        series_name = "test_series"
        
        # Store some data with duplicate timestamps
        now = time.time()
        self.store.store(series_name, 1, now)
        self.store.store(series_name, 2, now)  # Same timestamp, should replace previous value
        
        # Add more points
        self.store.store(series_name, 3, now + 1)
        self.store.store(series_name, 4, now + 2)
        
        # Flush data
        self.store.flush()
        
        # Compact the series
        result = self.store.compact(series_name)
        self.assertTrue(result)
        
        # Check that duplicate was removed
        self.assertEqual(self.store.count(series_name), 3)
        
        # First point should have value 2 (second store with same timestamp)
        data = self.store.get(series_name, now, now + 0.1)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0][1], 2)


if __name__ == "__main__":
    unittest.main()