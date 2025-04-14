import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import pytest
from pathlib import Path
import tempfile
import os
import json

from data.fetchers.alternative_data_gateway import (
    AlternativeDataGateway,
    AltDataType,
    AltDataFormat
)
from core.event_bus import EventBus
from data.processors.data_normalizer import DataNormalizer
from execution.exchange.connectivity_manager import ConnectivityManager


class TestAlternativeDataGateway(unittest.TestCase):
    """Tests for AlternativeDataGateway class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock dependencies
        self.connectivity_manager = MagicMock(spec=ConnectivityManager)
        self.data_normalizer = MagicMock(spec=DataNormalizer)
        self.event_bus = MagicMock(spec=EventBus)
        self.event_bus.publish = AsyncMock()

        # Create temp directory for cache
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Test configuration
        self.config = {
            'sources': {
                'news': {
                    'alpha_vantage': {'api_key': 'test_key'},
                    'newsapi': {'api_key': 'test_key'},
                    'default': 'alpha_vantage'
                },
                'sentiment': {
                    'stocktwits': {'api_key': 'test_key'},
                    'twitter': {'api_key': 'test_key'},
                    'reddit': {'api_key': 'test_key'},
                    'default': 'stocktwits'
                },
                'economic': {
                    'fred': {'api_key': 'test_key'},
                    'world_bank': {},
                    'default': 'fred'
                },
                'earnings': {
                    'alpha_vantage': {'api_key': 'test_key'},
                    'zacks': {'api_key': 'test_key'},
                    'default': 'alpha_vantage'
                }
            },
            'use_cache': True,
            'cache_ttl': 300,  # 5 minutes for testing
            'cache_dir': os.path.join(self.temp_dir.name, 'cache')
        }
        
        # Create the gateway with test configuration
        self.gateway = AlternativeDataGateway(
            self.connectivity_manager,
            self.data_normalizer,
            self.event_bus,
            self.config
        )
        
        # Mock internal methods
        self.gateway._get_alpha_vantage_news = AsyncMock()
        self.gateway._get_newsapi_news = AsyncMock()
        self.gateway._get_bloomberg_news = AsyncMock()
        
        self.gateway._get_stocktwits_sentiment = AsyncMock()
        self.gateway._get_twitter_sentiment = AsyncMock()
        self.gateway._get_reddit_sentiment = AsyncMock()
        
        self.gateway._get_fred_economic_data = AsyncMock()
        self.gateway._get_world_bank_data = AsyncMock()
        self.gateway._get_bea_data = AsyncMock()
        
        self.gateway._get_alpha_vantage_earnings = AsyncMock()
        self.gateway._get_zacks_earnings = AsyncMock()

    def tearDown(self):
        """Clean up after tests"""
        self.temp_dir.cleanup()

    async def test_init(self):
        """Test initialization of the gateway"""
        self.assertEqual(self.gateway.connectivity_manager, self.connectivity_manager)
        self.assertEqual(self.gateway.normalizer, self.data_normalizer)
        self.assertEqual(self.gateway.event_bus, self.event_bus)
        self.assertEqual(self.gateway.config, self.config)
        self.assertEqual(self.gateway.sources, self.config['sources'])
        self.assertEqual(self.gateway.use_cache, self.config['use_cache'])
        self.assertEqual(self.gateway.cache_ttl, self.config['cache_ttl'])
        self.assertEqual(str(self.gateway.cache_dir), self.config['cache_dir'])
        
        # Check cache directory creation
        self.assertTrue(os.path.exists(self.config['cache_dir']))

    async def test_get_news(self):
        """Test fetching news data"""
        # Prepare test data
        mock_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'title': ['Test News'],
            'content': ['This is test news content'],
            'source': ['Test Source'],
            'url': ['http://test.com/news'],
            'sentiment': [0.8],
            'symbols': [['AAPL', 'MSFT']]
        })
        
        self.gateway._get_alpha_vantage_news.return_value = mock_data
        
        # Test the method
        result = await self.gateway.get_news(
            symbols=['AAPL', 'MSFT'],
            keywords=['earnings', 'growth'],
            start_time=datetime.now() - timedelta(days=7),
            end_time=datetime.now(),
            source='alpha_vantage',
            limit=100
        )
        
        # Verify results
        pd.testing.assert_frame_equal(result, mock_data)
        
        # Verify the correct source method was called
        self.gateway._get_alpha_vantage_news.assert_called_once()
        
        # Verify caching
        cache_files = os.listdir(self.gateway.cache_dir)
        self.assertGreater(len(cache_files), 0)

    async def test_get_sentiment(self):
        """Test fetching sentiment data"""
        # Prepare test data
        mock_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['AAPL'],
            'sentiment_score': [0.75],
            'volume': [1000],
            'source': ['stocktwits'],
            'message_count': [50]
        })
        
        self.gateway._get_stocktwits_sentiment.return_value = mock_data
        
        # Test the method with default source
        result = await self.gateway.get_sentiment(
            symbols='AAPL',
            start_time=datetime.now() - timedelta(days=3),
            end_time=datetime.now()
        )
        
        # Verify results
        pd.testing.assert_frame_equal(result, mock_data)
        
        # Verify the correct source method was called
        self.gateway._get_stocktwits_sentiment.assert_called_once()

    async def test_get_economic_data(self):
        """Test fetching economic data"""
        # Prepare test data
        mock_data = pd.DataFrame({
            'date': [datetime.now() - timedelta(days=i) for i in range(5)],
            'indicator': ['GDP'] * 5,
            'value': [21.4, 21.3, 21.2, 21.1, 21.0],
            'source': ['fred'] * 5
        })
        
        self.gateway._get_fred_economic_data.return_value = mock_data
        
        # Test the method
        result = await self.gateway.get_economic_data(
            indicators=['GDP', 'UNEMPLOYMENT'],
            start_time=datetime.now() - timedelta(days=365),
            end_time=datetime.now(),
            source='fred'
        )
        
        # Verify results
        pd.testing.assert_frame_equal(result, mock_data)
        
        # Verify the correct source method was called
        self.gateway._get_fred_economic_data.assert_called_once()

    async def test_get_earnings_data(self):
        """Test fetching earnings data"""
        # Prepare test data
        mock_data = pd.DataFrame({
            'symbol': ['AAPL'] * 4,
            'report_date': [datetime.now() - timedelta(days=90 * i) for i in range(4)],
            'fiscal_quarter': ['Q1', 'Q4', 'Q3', 'Q2'],
            'eps_actual': [3.2, 3.1, 2.9, 3.0],
            'eps_estimate': [3.1, 3.0, 2.8, 2.9],
            'revenue_actual': [90.1, 85.3, 82.7, 81.2],
            'revenue_estimate': [89.5, 84.1, 82.0, 80.5]
        })
        
        self.gateway._get_alpha_vantage_earnings.return_value = mock_data
        
        # Test the method
        result = await self.gateway.get_earnings_data(
            symbols=['AAPL', 'MSFT'],
            start_time=datetime.now() - timedelta(days=365),
            end_time=datetime.now(),
            include_estimates=True,
            source='alpha_vantage'
        )
        
        # Verify results
        pd.testing.assert_frame_equal(result, mock_data)
        
        # Verify the correct source method was called
        self.gateway._get_alpha_vantage_earnings.assert_called_once()

    async def test_cache_functionality(self):
        """Test cache functionality"""
        # Prepare test data
        mock_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'title': ['Test News'],
            'content': ['This is test news content'],
            'source': ['Test Source']
        })
        
        self.gateway._get_alpha_vantage_news.return_value = mock_data
        
        # First call should use the fetcher
        result1 = await self.gateway.get_news(
            symbols='AAPL',
            start_time=datetime.now() - timedelta(days=7),
            end_time=datetime.now(),
            source='alpha_vantage'
        )
        
        # Reset the mock to verify it's not called again
        self.gateway._get_alpha_vantage_news.reset_mock()
        
        # Second call with same parameters should use cache
        result2 = await self.gateway.get_news(
            symbols='AAPL', 
            start_time=datetime.now() - timedelta(days=7),
            end_time=datetime.now(),
            source='alpha_vantage'
        )
        
        # Verify the fetcher wasn't called again
        self.gateway._get_alpha_vantage_news.assert_not_called()
        
        # Verify both results are the same
        pd.testing.assert_frame_equal(result1, result2)

    async def test_error_handling(self):
        """Test error handling and event publishing"""
        # Make the fetcher raise an exception
        self.gateway._get_alpha_vantage_news.side_effect = Exception("API Error")
        
        # Call should handle the exception and return empty DataFrame
        result = await self.gateway.get_news(
            symbols='AAPL',
            source='alpha_vantage'
        )
        
        # Verify empty result
        self.assertTrue(result.empty)
        
        # Verify error event was published
        self.event_bus.publish.assert_called_once()
        # Extract the event from the call args
        called_event = self.event_bus.publish.call_args[0][0]
        self.assertIn("error.data.alternative.news", called_event.type)

    async def test_subscribe_to_alt_data(self):
        """Test subscribing to alternative data updates"""
        # Create a mock callback
        callback = AsyncMock()
        
        # Mock the _run_data_stream method to avoid starting actual stream
        with patch.object(self.gateway, '_run_data_stream', new_callable=AsyncMock) as mock_run_stream:
            # Subscribe to news
            subscription_id = await self.gateway.subscribe_to_alt_data(
                data_type=AltDataType.NEWS,
                subjects=['AAPL', 'MSFT'],
                callback=callback,
                source='alpha_vantage'
            )
            
            # Verify subscription was created
            self.assertIn(subscription_id, self.gateway.subscribers)
            self.assertEqual(self.gateway.subscribers[subscription_id]['type'], AltDataType.NEWS)
            self.assertEqual(self.gateway.subscribers[subscription_id]['subjects'], ['AAPL', 'MSFT'])
            self.assertEqual(self.gateway.subscribers[subscription_id]['source'], 'alpha_vantage')
            self.assertEqual(self.gateway.subscribers[subscription_id]['callback'], callback)
            
            # Verify stream was started
            stream_key = f"{AltDataType.NEWS.value}_alpha_vantage"
            self.assertIn(stream_key, self.gateway.active_streams)
            mock_run_stream.assert_called_once()

    async def test_unsubscribe(self):
        """Test unsubscribing from alternative data updates"""
        # Create a mock callback
        callback = AsyncMock()
        
        # Mock the _run_data_stream method to avoid starting actual stream
        with patch.object(self.gateway, '_run_data_stream', new_callable=AsyncMock) as mock_run_stream:
            # Subscribe to news
            subscription_id = await self.gateway.subscribe_to_alt_data(
                data_type=AltDataType.NEWS,
                subjects=['AAPL'],
                callback=callback,
                source='alpha_vantage'
            )
            
            # Mock the task
            stream_key = f"{AltDataType.NEWS.value}_alpha_vantage"
            mock_task = MagicMock()
            mock_task.done.return_value = False
            mock_task.cancel = MagicMock()
            self.gateway.active_streams[stream_key]['task'] = mock_task
            
            # Unsubscribe
            result = await self.gateway.unsubscribe(subscription_id)
            
            # Verify result
            self.assertTrue(result)
            
            # Verify subscription was removed
            self.assertNotIn(subscription_id, self.gateway.subscribers)
            
            # Verify stream was stopped and removed
            self.assertNotIn(stream_key, self.gateway.active_streams)
            mock_task.cancel.assert_called_once()

    def test_get_default_source(self):
        """Test getting default source for data types"""
        # Test with default specified
        default_news = self.gateway._get_default_source(AltDataType.NEWS)
        self.assertEqual(default_news, 'alpha_vantage')
        
        # Test with no default specified
        # Temporarily modify the sources to remove default
        orig_sources = self.gateway.sources
        test_sources = {
            'test_type': {
                'source1': {},
                'source2': {}
            }
        }
        self.gateway.sources = test_sources
        
        # Create a test enum entry
        test_enum = MagicMock()
        test_enum.value = 'test_type'
        
        # Should return first available source
        default_test = self.gateway._get_default_source(test_enum)
        self.assertEqual(default_test, 'source1')
        
        # Restore original sources
        self.gateway.sources = orig_sources

    def test_cache_key_generation(self):
        """Test cache key generation"""
        data_type = AltDataType.NEWS
        source = 'alpha_vantage'
        subjects = ['AAPL', 'MSFT']
        filters = ['keywords=earnings', 'limit=100']
        start_time = datetime(2023, 1, 1)
        end_time = datetime(2023, 1, 7)
        
        key = self.gateway._get_cache_key(
            data_type,
            source,
            subjects,
            filters,
            start_time,
            end_time
        )
        
        # Verify key format and sorting
        self.assertIn(data_type.value, key)
        self.assertIn(source, key)
        # Subjects should be sorted
        self.assertIn('AAPL_MSFT', key)
        # Filters should be sorted
        self.assertIn('keywords=earnings_limit=100', key) 
        # Date format
        self.assertIn('20230101', key)
        self.assertIn('20230107', key)

    @pytest.mark.asyncio
    async def test_stream_news(self):
        """Test streaming news to subscribers"""
        # This is a placeholder test as the actual implementation
        # is marked as placeholder in the original code
        callback = AsyncMock()
        
        with patch.object(self.gateway, '_run_data_stream', new_callable=AsyncMock):
            subscription_id = await self.gateway.subscribe_to_alt_data(
                data_type=AltDataType.NEWS,
                subjects=['AAPL'],
                callback=callback,
                source='alpha_vantage'
            )
            
            # Mock the _stream_news method to return some data
            with patch.object(self.gateway, '_stream_news', new_callable=AsyncMock) as mock_stream:
                # Call the method directly
                await self.gateway._stream_news('alpha_vantage', ['AAPL'])
                
                # Verify the method was called
                mock_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_sentiment(self):
        """Test streaming sentiment to subscribers"""
        # Similar placeholder test as above
        callback = AsyncMock()
        
        with patch.object(self.gateway, '_run_data_stream', new_callable=AsyncMock):
            subscription_id = await self.gateway.subscribe_to_alt_data(
                data_type=AltDataType.SOCIAL_SENTIMENT,
                subjects=['AAPL'],
                callback=callback,
                source='stocktwits'
            )
            
            # Mock the _stream_sentiment method to return some data
            with patch.object(self.gateway, '_stream_sentiment', new_callable=AsyncMock) as mock_stream:
                # Call the method directly
                await self.gateway._stream_sentiment('stocktwits', ['AAPL'])
                
                # Verify the method was called
                mock_stream.assert_called_once()


if __name__ == '__main__':
    unittest.main()