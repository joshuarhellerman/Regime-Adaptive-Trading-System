import unittest
from datetime import datetime, timedelta, time
import pytz
from unittest.mock import patch, MagicMock
import calendar

# Import the module under test
from utils.time_utils import (
    now, to_timestamp, from_timestamp, parse_datetime, format_datetime,
    convert_timezone, timeframe_to_timedelta, ceil_dt, floor_dt, round_dt,
    is_trading_hours, get_next_trading_time, get_market_open_close,
    get_trading_days, parse_period, get_interval_timestamp, generate_time_windows,
    is_holiday, get_business_days_between, timeframe_to_seconds, seconds_to_timeframe,
    Timer, TimingStatistics, TimeSeriesResampler, PerformanceTimer, TimingContext,
    TimeResolution, TradingCalendar, UTC, EST, JST, GMT, TRADING_HOURS,
    align_timestamps, create_time_buckets, calculate_execution_stats,
    get_quarter_start, get_quarter_end, get_month_end, get_week_start, get_week_end,
    estimate_data_periodicity, generate_calendar_periods, time_segments_overlap,
    get_overlap_duration, describe_time_difference, rolling_window, time_buckets,
    fill_time_gaps, get_execution_latency_budget, is_historical_time, is_future_time,
    time_until, time_since, localize_timestamp, get_trading_timestamps, format_duration,
    parse_trading_time, market_time_to_utc, utc_to_market_time, get_calendar_for_asset,
    get_market_hours_in_period, align_to_trading_schedule, date_range, 
    get_timezones_for_region, calculate_time_efficiency, get_current_market_session
)


class TestTimeUtilsBasics(unittest.TestCase):
    """Test basic time utilities"""

    def test_now(self):
        """Test the now function"""
        # Test with default timezone (UTC)
        dt = now()
        self.assertIsInstance(dt, datetime)
        self.assertEqual(dt.tzinfo, pytz.UTC)

        # Test with specific timezone
        dt = now(EST)
        self.assertIsInstance(dt, datetime)
        self.assertEqual(dt.tzinfo, EST)

    def test_to_timestamp(self):
        """Test converting datetime to timestamp"""
        # Create a fixed datetime
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)
        ts = to_timestamp(dt)
        self.assertIsInstance(ts, float)
        self.assertEqual(ts, dt.timestamp())

        # Test with string input
        dt_str = "2023-01-01T12:00:00+00:00"
        ts = to_timestamp(dt_str)
        self.assertIsInstance(ts, float)
        self.assertEqual(ts, dt.timestamp())

    def test_from_timestamp(self):
        """Test converting timestamp to datetime"""
        # Create a fixed timestamp
        ts = 1672574400.0  # 2023-01-01 12:00:00 UTC
        dt = from_timestamp(ts)
        self.assertIsInstance(dt, datetime)
        self.assertEqual(dt, datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC))

        # Test with specific timezone
        dt = from_timestamp(ts, EST)
        self.assertIsInstance(dt, datetime)
        self.assertEqual(dt.tzinfo, EST)

    def test_parse_datetime(self):
        """Test parsing datetime strings"""
        # Test ISO format
        dt_str = "2023-01-01T12:00:00+00:00"
        dt = parse_datetime(dt_str)
        self.assertEqual(dt, datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC))

        # Test other formats
        formats = [
            "2023-01-01 12:00:00",
            "2023-01-01 12:00",
            "2023-01-01",
            "01/01/2023 12:00:00",
            "01/01/2023 12:00",
            "01/01/2023"
        ]
        for fmt_str in formats:
            dt = parse_datetime(fmt_str)
            self.assertIsInstance(dt, datetime)
            self.assertIsNotNone(dt.tzinfo)

        # Test with invalid format
        with self.assertRaises(ValueError):
            parse_datetime("invalid-datetime")

    def test_format_datetime(self):
        """Test formatting datetime objects"""
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)

        # Test predefined formats
        self.assertEqual(format_datetime(dt, 'iso'), "2023-01-01T12:00:00+00:00")
        self.assertEqual(format_datetime(dt, 'date'), "2023-01-01")
        self.assertEqual(format_datetime(dt, 'time'), "12:00:00")
        self.assertTrue("2023-01-01 12:00:00" in format_datetime(dt, 'full'))

        # Test custom format
        self.assertEqual(format_datetime(dt, '%Y/%m/%d'), "2023/01/01")

    def test_convert_timezone(self):
        """Test converting between timezones"""
        # Create UTC datetime
        dt_utc = datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)

        # Convert to EST
        dt_est = convert_timezone(dt_utc, EST)
        self.assertEqual(dt_est.tzinfo, EST)
        self.assertEqual(dt_est.hour, 7)  # UTC-5

        # Convert to JST
        dt_jst = convert_timezone(dt_utc, JST)
        self.assertEqual(dt_jst.tzinfo, JST)
        self.assertEqual(dt_jst.hour, 21)  # UTC+9

        # Test with naive datetime
        dt_naive = datetime(2023, 1, 1, 12, 0, 0)
        dt_est = convert_timezone(dt_naive, EST)
        self.assertEqual(dt_est.tzinfo, EST)


class TestTimeFrames(unittest.TestCase):
    """Test timeframe and interval related utilities"""

    def test_timeframe_to_timedelta(self):
        """Test converting timeframe strings to timedeltas"""
        test_cases = [
            ('1m', timedelta(minutes=1)),
            ('5m', timedelta(minutes=5)),
            ('1h', timedelta(hours=1)),
            ('4h', timedelta(hours=4)),
            ('1d', timedelta(days=1)),
            ('7d', timedelta(days=7)),
            ('1w', timedelta(weeks=1)),
            ('4w', timedelta(weeks=4)),
            ('1M', timedelta(days=30)),  # Approximate
            ('3M', timedelta(days=90)),  # Approximate
        ]

        for timeframe, expected in test_cases:
            result = timeframe_to_timedelta(timeframe)
            self.assertEqual(result, expected)

        # Test invalid timeframe
        with self.assertRaises(ValueError):
            timeframe_to_timedelta('1x')

    def test_parse_period(self):
        """Test parsing period strings into timedeltas"""
        test_cases = [
            ('1m', timedelta(minutes=1)),
            ('5min', timedelta(minutes=5)),
            ('1h', timedelta(hours=1)),
            ('4hour', timedelta(hours=4)),
            ('1d', timedelta(days=1)),
            ('7day', timedelta(days=7)),
            ('1w', timedelta(weeks=1)),
            ('4week', timedelta(weeks=4)),
            ('1mo', timedelta(days=30)),  # Approximate
            ('3month', timedelta(days=90)),  # Approximate
            ('1y', timedelta(days=365)),  # Approximate
            ('2year', timedelta(days=730)),  # Approximate
        ]

        for period, expected in test_cases:
            result = parse_period(period)
            self.assertEqual(result, expected)

        # Test empty period
        with self.assertRaises(ValueError):
            parse_period('')

        # Test invalid period
        with self.assertRaises(ValueError):
            parse_period('1x')

    def test_get_interval_timestamp(self):
        """Test aligning timestamps to interval boundaries"""
        # Test minute intervals
        dt = datetime(2023, 1, 1, 12, 34, 56, tzinfo=UTC)
        aligned = get_interval_timestamp(dt, '5m')
        self.assertEqual(aligned, datetime(2023, 1, 1, 12, 30, 0, tzinfo=UTC))

        # Test hour intervals
        aligned = get_interval_timestamp(dt, '1h')
        self.assertEqual(aligned, datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC))
        aligned = get_interval_timestamp(dt, '4h')
        self.assertEqual(aligned, datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC))

        # Test day intervals
        aligned = get_interval_timestamp(dt, '1d')
        self.assertEqual(aligned, datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC))

        # Test week intervals
        dt = datetime(2023, 1, 4, 12, 0, 0, tzinfo=UTC)  # Wednesday
        aligned = get_interval_timestamp(dt, '1w')
        self.assertEqual(aligned, datetime(2023, 1, 2, 0, 0, 0, tzinfo=UTC))  # Monday

        # Test month intervals
        dt = datetime(2023, 1, 15, 12, 0, 0, tzinfo=UTC)
        aligned = get_interval_timestamp(dt, '1M')
        self.assertEqual(aligned, datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC))

    def test_timeframe_to_seconds(self):
        """Test converting timeframes to seconds"""
        test_cases = [
            ('1m', 60),
            ('5m', 300),
            ('1h', 3600),
            ('4h', 14400),
            ('1d', 86400),
            ('1w', 604800),
        ]

        for timeframe, expected_seconds in test_cases:
            result = timeframe_to_seconds(timeframe)
            self.assertEqual(result, expected_seconds)

    def test_seconds_to_timeframe(self):
        """Test converting seconds to timeframe strings"""
        test_cases = [
            (30, '30s'),
            (60, '1m'),
            (300, '5m'),
            (3600, '1h'),
            (14400, '4h'),
            (86400, '1d'),
            (604800, '1w'),
            (2592000, '1M'),
        ]

        for seconds, expected_timeframe in test_cases:
            result = seconds_to_timeframe(seconds)
            self.assertEqual(result, expected_timeframe)

    def test_generate_time_windows(self):
        """Test generating time windows"""
        start = datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC)
        end = datetime(2023, 1, 1, 6, 0, 0, tzinfo=UTC)

        # Test with 1-hour windows
        windows = generate_time_windows(start, end, '1h')
        self.assertEqual(len(windows), 6)
        self.assertEqual(windows[0], (start, start + timedelta(hours=1)))
        self.assertEqual(windows[-1], (end - timedelta(hours=1), end))

        # Test with 2-hour windows
        windows = generate_time_windows(start, end, '2h')
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0], (start, start + timedelta(hours=2)))
        self.assertEqual(windows[-1], (start + timedelta(hours=4), end))

        # Test invalid case (start > end)
        with self.assertRaises(ValueError):
            generate_time_windows(end, start, '1h')


class TestTimeResolution(unittest.TestCase):
    """Test time resolution related functions"""

    def test_ceil_dt(self):
        """Test ceiling datetime to various resolutions"""
        dt = datetime(2023, 1, 1, 12, 34, 56, 789000, tzinfo=UTC)

        # Test MILLISECOND resolution
        result = ceil_dt(dt, TimeResolution.MILLISECOND)
        self.assertEqual(result, dt)

        # Test SECOND resolution
        result = ceil_dt(dt, TimeResolution.SECOND)
        self.assertEqual(result, datetime(2023, 1, 1, 12, 34, 57, 0, tzinfo=UTC))

        # Test MINUTE resolution
        result = ceil_dt(dt, TimeResolution.MINUTE)
        self.assertEqual(result, datetime(2023, 1, 1, 12, 35, 0, 0, tzinfo=UTC))

        # Test HOUR resolution
        result = ceil_dt(dt, TimeResolution.HOUR)
        self.assertEqual(result, datetime(2023, 1, 1, 13, 0, 0, 0, tzinfo=UTC))

        # Test DAY resolution
        result = ceil_dt(dt, TimeResolution.DAY)
        self.assertEqual(result, datetime(2023, 1, 2, 0, 0, 0, 0, tzinfo=UTC))

        # Test WEEK resolution (ceil to next Monday)
        dt = datetime(2023, 1, 4, 12, 0, 0, tzinfo=UTC)  # Wednesday
        result = ceil_dt(dt, TimeResolution.WEEK)
        self.assertEqual(result, datetime(2023, 1, 9, 0, 0, 0, 0, tzinfo=UTC))  # Next Monday

        # Test MONTH resolution
        dt = datetime(2023, 1, 15, 12, 0, 0, tzinfo=UTC)
        result = ceil_dt(dt, TimeResolution.MONTH)
        self.assertEqual(result, datetime(2023, 2, 1, 0, 0, 0, 0, tzinfo=UTC))

        # Test exact boundaries
        dt = datetime(2023, 1, 1, 0, 0, 0, 0, tzinfo=UTC)
        result = ceil_dt(dt, TimeResolution.DAY)
        self.assertEqual(result, dt)

    def test_floor_dt(self):
        """Test flooring datetime to various resolutions"""
        dt = datetime(2023, 1, 1, 12, 34, 56, 789000, tzinfo=UTC)

        # Test MILLISECOND resolution
        result = floor_dt(dt, TimeResolution.MILLISECOND)
        self.assertEqual(result, datetime(2023, 1, 1, 12, 34, 56, 789000, tzinfo=UTC))

        # Test SECOND resolution
        result = floor_dt(dt, TimeResolution.SECOND)
        self.assertEqual(result, datetime(2023, 1, 1, 12, 34, 56, 0, tzinfo=UTC))

        # Test MINUTE resolution
        result = floor_dt(dt, TimeResolution.MINUTE)
        self.assertEqual(result, datetime(2023, 1, 1, 12, 34, 0, 0, tzinfo=UTC))

        # Test HOUR resolution
        result = floor_dt(dt, TimeResolution.HOUR)
        self.assertEqual(result, datetime(2023, 1, 1, 12, 0, 0, 0, tzinfo=UTC))

        # Test DAY resolution
        result = floor_dt(dt, TimeResolution.DAY)
        self.assertEqual(result, datetime(2023, 1, 1, 0, 0, 0, 0, tzinfo=UTC))

        # Test WEEK resolution (floor to previous Monday)
        dt = datetime(2023, 1, 4, 12, 0, 0, tzinfo=UTC)  # Wednesday
        result = floor_dt(dt, TimeResolution.WEEK)
        self.assertEqual(result, datetime(2023, 1, 2, 0, 0, 0, 0, tzinfo=UTC))  # Monday

        # Test MONTH resolution
        dt = datetime(2023, 1, 15, 12, 0, 0, tzinfo=UTC)
        result = floor_dt(dt, TimeResolution.MONTH)
        self.assertEqual(result, datetime(2023, 1, 1, 0, 0, 0, 0, tzinfo=UTC))

    def test_round_dt(self):
        """Test rounding datetime to various resolutions"""
        # Test rounding down
        dt = datetime(2023, 1, 1, 12, 29, 30, 0, tzinfo=UTC)
        result = round_dt(dt, TimeResolution.HOUR)
        self.assertEqual(result, datetime(2023, 1, 1, 12, 0, 0, 0, tzinfo=UTC))

        # Test rounding up
        dt = datetime(2023, 1, 1, 12, 30, 0, 0, tzinfo=UTC)
        result = round_dt(dt, TimeResolution.HOUR)
        self.assertEqual(result, datetime(2023, 1, 1, 13, 0, 0, 0, tzinfo=UTC))

        # Test exact midpoint (rounds up)
        dt = datetime(2023, 1, 1, 12, 30, 0, 0, tzinfo=UTC)
        result = round_dt(dt, TimeResolution.HOUR)
        self.assertEqual(result, datetime(2023, 1, 1, 13, 0, 0, 0, tzinfo=UTC))


class TestTradingCalendar(unittest.TestCase):
    """Test trading calendar related functions"""

    def test_is_trading_hours(self):
        """Test checking if a time is during trading hours"""
        # Test US equities during regular hours
        dt = datetime(2023, 1, 3, 14, 30, 0, tzinfo=UTC)  # 9:30 AM EST on Tuesday
        self.assertTrue(is_trading_hours(dt, TradingCalendar.US_EQUITIES))

        # Test US equities outside regular hours
        dt = datetime(2023, 1, 3, 21, 0, 0, tzinfo=UTC)  # 4:00 PM EST on Tuesday
        self.assertFalse(is_trading_hours(dt, TradingCalendar.US_EQUITIES))

        # Test US equities during extended hours
        dt = datetime(2023, 1, 3, 13, 0, 0, tzinfo=UTC)  # 8:00 AM EST on Tuesday
        self.assertFalse(is_trading_hours(dt, TradingCalendar.US_EQUITIES))
        self.assertTrue(is_trading_hours(dt, TradingCalendar.US_EQUITIES, include_extended=True))

        # Test weekend
        dt = datetime(2023, 1, 1, 14, 30, 0, tzinfo=UTC)  # Sunday
        self.assertFalse(is_trading_hours(dt, TradingCalendar.US_EQUITIES))

        # Test crypto (24/7)
        dt = datetime(2023, 1, 1, 14, 30, 0, tzinfo=UTC)  # Sunday
        self.assertTrue(is_trading_hours(dt, TradingCalendar.CRYPTO))

        # Test forex (24/5)
        dt = datetime(2023, 1, 2, 14, 30, 0, tzinfo=UTC)  # Monday
        self.assertTrue(is_trading_hours(dt, TradingCalendar.FOREX))
        dt = datetime(2023, 1, 7, 14, 30, 0, tzinfo=UTC)  # Saturday
        self.assertFalse(is_trading_hours(dt, TradingCalendar.FOREX))

    def test_get_next_trading_time(self):
        """Test getting next trading time"""
        # Test when already in trading hours
        dt = datetime(2023, 1, 3, 14, 30, 0, tzinfo=UTC)  # 9:30 AM EST on Tuesday
        result = get_next_trading_time(dt, TradingCalendar.US_EQUITIES)
        self.assertEqual(result, dt)

        # Test before market open on trading day
        dt = datetime(2023, 1, 3, 13, 0, 0, tzinfo=UTC)  # 8:00 AM EST on Tuesday
        result = get_next_trading_time(dt, TradingCalendar.US_EQUITIES)
        self.assertEqual(result, datetime(2023, 1, 3, 14, 30, 0, tzinfo=EST).astimezone(UTC))

        # Test after market close on trading day
        dt = datetime(2023, 1, 3, 21, 30, 0, tzinfo=UTC)  # 4:30 PM EST on Tuesday
        result = get_next_trading_time(dt, TradingCalendar.US_EQUITIES)
        next_day = datetime(2023, 1, 4, 14, 30, 0, tzinfo=UTC)  # 9:30 AM EST on Wednesday
        self.assertEqual(result, next_day)

        # Test weekend
        dt = datetime(2023, 1, 1, 14, 30, 0, tzinfo=UTC)  # Sunday
        result = get_next_trading_time(dt, TradingCalendar.US_EQUITIES)
        monday = datetime(2023, 1, 2, 14, 30, 0, tzinfo=UTC)  # 9:30 AM EST on Monday
        self.assertEqual(result, monday)

        # Test crypto (always trading)
        dt = datetime(2023, 1, 1, 14, 30, 0, tzinfo=UTC)  # Sunday
        result = get_next_trading_time(dt, TradingCalendar.CRYPTO)
        self.assertEqual(result, dt)

    def test_get_market_open_close(self):
        """Test getting market open and close times"""
        # Test US equities on trading day
        dt = datetime(2023, 1, 3, 0, 0, 0, tzinfo=UTC)  # Tuesday
        open_time, close_time = get_market_open_close(dt, TradingCalendar.US_EQUITIES)
        self.assertEqual(open_time.hour, 9)
        self.assertEqual(open_time.minute, 30)
        self.assertEqual(close_time.hour, 16)
        self.assertEqual(close_time.minute, 0)

        # Test US equities on weekend
        dt = datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC)  # Sunday
        open_time, close_time = get_market_open_close(dt, TradingCalendar.US_EQUITIES)
        self.assertIsNone(open_time)
        self.assertIsNone(close_time)

        # Test extended hours
        dt = datetime(2023, 1, 3, 0, 0, 0, tzinfo=UTC)  # Tuesday
        open_time, close_time = get_market_open_close(dt, TradingCalendar.US_EQUITIES, include_extended=True)
        self.assertEqual(open_time.hour, 4)
        self.assertEqual(open_time.minute, 0)
        self.assertEqual(close_time.hour, 20)
        self.assertEqual(close_time.minute, 0)

        # Test crypto (24/7)
        dt = datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC)  # Sunday
        open_time, close_time = get_market_open_close(dt, TradingCalendar.CRYPTO)
        self.assertEqual(open_time, dt)
        self.assertEqual(close_time, dt + timedelta(days=1))

    def test_get_trading_days(self):
        """Test getting trading days between dates"""
        # Test one week for US equities
        start = datetime(2023, 1, 2, 0, 0, 0, tzinfo=UTC)  # Monday
        end = datetime(2023, 1, 8, 0, 0, 0, tzinfo=UTC)  # Sunday
        days = get_trading_days(start, end, TradingCalendar.US_EQUITIES)
        self.assertEqual(len(days), 5)  # Monday to Friday

        # Test weekend only
        start = datetime(2023, 1, 7, 0, 0, 0, tzinfo=UTC)  # Saturday
        end = datetime(2023, 1, 8, 0, 0, 0, tzinfo=UTC)  # Sunday
        days = get_trading_days(start, end, TradingCalendar.US_EQUITIES)
        self.assertEqual(len(days), 0)

        # Test crypto (all days)
        start = datetime(2023, 1, 2, 0, 0, 0, tzinfo=UTC)  # Monday
        end = datetime(2023, 1, 8, 0, 0, 0, tzinfo=UTC)  # Sunday
        days = get_trading_days(start, end, TradingCalendar.CRYPTO)
        self.assertEqual(len(days), 7)  # All days


class TestUtilityClasses(unittest.TestCase):
    """Test utility classes like Timer, TimingStatistics, etc."""

    def test_timer(self):
        """Test Timer class"""
        # Test context manager usage
        with Timer('test_timer') as timer:
            # Simulate some work
            pass

        self.assertIsNotNone(timer.start_time)
        self.assertIsNotNone(timer.end_time)
        self.assertIsNotNone(timer.elapsed)

        # Test manual start/stop
        timer = Timer('manual_timer')
        timer.start()
        # Simulate some work
        elapsed = timer.stop()
        self.assertIsNotNone(elapsed)
        self.assertEqual(elapsed, timer.elapsed)

        # Test get_elapsed
        timer = Timer('elapsed_timer')
        timer.start()
        # Simulate some work
        elapsed = timer.get_elapsed()
        self.assertIsNotNone(elapsed)

        # Test reset
        timer.reset()
        self.assertIsNone(timer.start_time)
        self.assertIsNone(timer.end_time)
        self.assertIsNone(timer.elapsed)

    def test_timing_statistics(self):
        """Test TimingStatistics class"""
        stats = TimingStatistics('test_stats')
        
        # Test empty stats
        empty_stats = stats.get_stats()
        self.assertEqual(empty_stats['count'], 0)
        self.assertEqual(empty_stats['min'], 0)
        self.assertEqual(empty_stats['max'], 0)
        
        # Add samples
        stats.add_sample(0.1)
        stats.add_sample(0.2)
        stats.add_sample(0.3)
        
        # Test stats with samples
        result_stats = stats.get_stats()
        self.assertEqual(result_stats['count'], 3)
        self.assertEqual(result_stats['min'], 0.1)
        self.assertEqual(result_stats['max'], 0.3)
        self.assertAlmostEqual(result_stats['mean'], 0.2, places=6)
        
        # Test reset
        stats.reset()
        reset_stats = stats.get_stats()
        self.assertEqual(reset_stats['count'], 0)

    def test_time_series_resampler(self):
        """Test TimeSeriesResampler class"""
        # Create test time series data
        time_series = [
            (datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC), 10),
            (datetime(2023, 1, 1, 12, 5, 0, tzinfo=UTC), 15),
            (datetime(2023, 1, 1, 12, 10, 0, tzinfo=UTC), 20),
            (datetime(2023, 1, 1, 12, 15, 0, tzinfo=UTC), 25),
        ]
        
        # Test 'last' aggregation
        resampled = TimeSeriesResampler.resample(time_series, '10m', 'last')
        self.assertEqual(len(resampled), 2)
        self.assertEqual(resampled[0][1], 15)  # Last value in first bucket
        self.assertEqual(resampled[1][1], 25)  # Last value in second bucket
        
        # Test 'first' aggregation
        resampled = TimeSeriesResampler.resample(time_series, '10m', 'first')
        self.assertEqual(len(resampled), 2)
        self.assertEqual(resampled[0][1], 10)  # First value in first bucket
        self.assertEqual(resampled[1][1], 20)  # First value in second bucket
        
        # Test 'mean' aggregation
        resampled = TimeSeriesResampler.resample(time_series, '10m', 'mean')
        self.assertEqual(len(resampled), 2)
        self.assertEqual(resampled[0][1], 12.5)  # Mean of first bucket (10, 15)
        self.assertEqual(resampled[1][1], 22.5)  # Mean of second bucket (20, 25)

    @patch('threading.RLock')
    def test_performance_timer(self, mock_rlock):
        """Test PerformanceTimer class"""
        # Mock the lock
        mock_lock = MagicMock()
        mock_rlock.return_value = mock_lock
        
        # Test singleton pattern
        timer1 = PerformanceTimer.get_instance('test_timer')
        timer2 = PerformanceTimer.get_instance('test_timer')
        self.assertIs(timer1, timer2)
        
        # Test timing operations
        timer = PerformanceTimer('timer_test')
        timer.start('operation1')
        # Simulate some work
        timer.stop('operation1')
        
        # Test get_stats
        stats = timer.get_stats('operation1')
        self.assertIn('count', stats)
        self.assertEqual(stats['count'], 1)
        
        # Test summary
        summary = timer.summary()
        self.assertIn('Performance Timer', summary)
        self.assertIn('operation1', summary)
        
        # Test reset
        timer.reset('operation1')
        stats = timer.get_stats('operation1')
        self.assertEqual(stats['count'], 0)


class TestTimeUtilsAdvanced(unittest.TestCase):
    """Test advanced time utilities"""

    def test_get_quarter_start(self):
        """Test getting quarter start dates"""
        test_cases = [
            (datetime(2023, 1, 15, tzinfo=UTC), datetime(2023, 1, 1, 0, 0, 0, 0, tzinfo=UTC)),   # Q1
            (datetime(2023, 4, 15, tzinfo=UTC), datetime(2023, 4, 1, 0, 0, 0, 0, tzinfo=UTC)),   # Q2
            (datetime(2023, 7, 15, tzinfo=UTC), datetime(2023, 7, 1, 0, 0, 0, 0, tzinfo=UTC)),   # Q3
            (datetime(2023, 10, 15, tzinfo=UTC), datetime(2023, 10, 1, 0, 0, 0, 0, tzinfo=UTC)), # Q4
        ]
        
        for input_date, expected_date in test_cases:
            result = get_quarter_start(input_date)
            self.assertEqual(result, expected_date)

    def test_get_quarter_end(self):
        """Test getting quarter end dates"""
        test_cases = [
            (datetime(2023, 1, 15, tzinfo=UTC), datetime(2023, 3, 31, 0, 0, 0, 0, tzinfo=UTC)),   # Q1
            (datetime(2023, 4, 15, tzinfo=UTC), datetime(2023, 6, 30, 0, 0, 0, 0, tzinfo=UTC)),   # Q2
            (datetime(2023, 7, 15, tzinfo=UTC), datetime(2023, 9, 30, 0, 0, 0, 0, tzinfo=UTC)),   # Q3
            (datetime(2023, 10, 15, tzinfo=UTC), datetime(2023, 12, 31, 0, 0, 0, 0, tzinfo=UTC)), # Q4
        ]
        
        for input_date, expected_date in test_cases:
            result = get_quarter_end(input_date)
            self.assertEqual(result, expected_date)

    def test_get_month_end(self):
        """Test getting month end dates"""
        test_cases = [
            (datetime(2023, 1, 15, tzinfo=UTC), datetime(2023, 1, 31, 23, 59, 59, 999999, tzinfo=UTC)),
            (datetime(2023, 2, 15, tzinfo=UTC), datetime(2023, 2, 28, 23, 59, 59, 999999, tzinfo=UTC)),
            (datetime(2023, 4, 15, tzinfo=UTC), datetime(2023, 4, 30, 23, 59, 59, 999999, tzinfo=UTC)),
            (datetime(2024, 2, 15, tzinfo=UTC), datetime(2024, 2, 29, 23, 59, 59, 999999, tzinfo=UTC)), # Leap year
        ]
        
        for input_date, expected_date in test_cases:
            result = get_month_end(input_date)
            self.assertEqual(result, expected_date)

    def test_get_week_start_end(self):
        """Test getting week start and end dates"""
        # Test starting Monday (default)
        dt = datetime(2023, 1, 18, tzinfo=UTC)  # Wednesday
        start = get_week_start(dt)
        self.assertEqual(start, datetime(2023, 1, 16, 0, 0, 0, 0, tzinfo=UTC))  # Monday
        
        end = get_week_end(dt)
        self.assertEqual(end, datetime(2023, 1, 22, 23, 59, 59, 999999, tzinfo=UTC))  # Sunday
        
        # Test starting Sunday
        start = get_week_start(dt, 6)
        self.assertEqual(start, datetime(2023, 1, 15, 0, 0, 0, 0, tzinfo=UTC))  # Sunday
        
        end = get_week_end(dt, 6)
        self.assertEqual(end, datetime(2023, 1, 21, 23, 59, 59, 999999, tzinfo=UTC))  # Saturday

    def test_estimate_data_periodicity(self):
        """Test estimating data periodicity"""
        # Test minute data
        timestamps = [
            datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC),
            datetime(2023, 1, 1, 12, 1, 0, tzinfo=UTC),
            datetime(2023, 1, 1, 12, 2, 0, tzinfo=UTC),
            datetime(2023, 1, 1, 12, 3, 0, tzinfo=UTC),
        ]
        self.assertEqual(estimate_data_periodicity(timestamps), "1m")
        
        # Test hour data
        timestamps = [
            datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC),
            datetime(2023, 1, 1, 13, 0, 0, tzinfo=UTC),
            datetime(2023, 1, 1, 14, 0, 0, tzinfo=UTC),
        ]
        self.assertEqual(estimate_data_periodicity(timestamps), "1h")
        
        # Test day data
        timestamps = [
            datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC),
            datetime(2023, 1, 2, 0, 0, 0, tzinfo=UTC),
            datetime(2023, 1, 3, 0, 0, 0, tzinfo=UTC),
        ]
        self.assertEqual(estimate_data_periodicity(timestamps), "1d")
        
        # Test empty data
        self.assertEqual(estimate_data_periodicity([]), "unknown")
        
        # Test single timestamp
        timestamps = [datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC)]
        self.assertEqual(estimate_data_periodicity(timestamps), "unknown")

    def test_generate_calendar_periods(self):
        """Test generating calendar periods"""
        start = datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC)
        end = datetime(2023, 1, 10, 0, 0, 0, tzinfo=UTC)
        
        # Test day periods
        periods = generate_calendar_periods(start, end, 'day')
        self.assertEqual(len(periods), 10)
        self.assertEqual(periods[0][0], start)
        self.assertEqual(periods[0][1], datetime(2023, 1, 1, 23, 59, 59, 999999, tzinfo=UTC))
        
        # Test week periods
        periods = generate_calendar_periods(start, end, 'week')
        # Should span parts of two weeks (Jan 1 is Sunday)
        self.assertTrue(1 <= len(periods) <= 2)
        
        # Test month periods
        start = datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC)
        end = datetime(2023, 2, 15, 0, 0, 0, tzinfo=UTC)
        periods = generate_calendar_periods(start, end, 'month')
        self.assertEqual(len(periods), 2)
        
        # Test quarter periods
        start = datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC)
        end = datetime(2023, 6, 30, 0, 0, 0, tzinfo=UTC)
        periods = generate_calendar_periods(start, end, 'quarter')
        self.assertEqual(len(periods), 2)

    def test_time_segments_overlap(self):
        """Test checking if time segments overlap"""
        # Test overlapping segments
        segment1 = (datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC), 
                   datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC))
        segment2 = (datetime(2023, 1, 1, 11, 0, 0, tzinfo=UTC), 
                   datetime(2023, 1, 1, 13, 0, 0, tzinfo=UTC))
        self.assertTrue(time_segments_overlap(segment1, segment2))
        
        # Test non-overlapping segments
        segment1 = (datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC), 
                   datetime(2023, 1, 1, 11, 0, 0, tzinfo=UTC))
        segment2 = (datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC), 
                   datetime(2023, 1, 1, 13, 0, 0, tzinfo=UTC))
        self.assertFalse(time_segments_overlap(segment1, segment2))
        
        # Test adjacent segments (not overlapping)
        segment1 = (datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC), 
                   datetime(2023, 1, 1, 11, 0, 0, tzinfo=UTC))
        segment2 = (datetime(2023, 1, 1, 11, 0, 0, tzinfo=UTC), 
                   datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC))
        self.assertTrue(time_segments_overlap(segment1, segment2))
        
        # Test one segment completely inside another
        segment1 = (datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC), 
                   datetime(2023, 1, 1, 13, 0, 0, tzinfo=UTC))
        segment2 = (datetime(2023, 1, 1, 11, 0, 0, tzinfo=UTC), 
                   datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC))
        self.assertTrue(time_segments_overlap(segment1, segment2))

    def test_get_overlap_duration(self):
        """Test calculating overlap duration between time segments"""
        # Test overlapping segments
        segment1 = (datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC), 
                   datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC))
        segment2 = (datetime(2023, 1, 1, 11, 0, 0, tzinfo=UTC), 
                   datetime(2023, 1, 1, 13, 0, 0, tzinfo=UTC))
        overlap = get_overlap_duration(segment1, segment2)
        self.assertEqual(overlap, timedelta(hours=1))
        
        # Test non-overlapping segments
        segment1 = (datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC), 
                   datetime(2023, 1, 1, 11, 0, 0, tzinfo=UTC))
        segment2 = (datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC), 
                   datetime(2023, 1, 1, 13, 0, 0, tzinfo=UTC))
        overlap = get_overlap_duration(segment1, segment2)
        self.assertEqual(overlap, timedelta(0))
        
        # Test one segment completely inside another
        segment1 = (datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC), 
                   datetime(2023, 1, 1, 13, 0, 0, tzinfo=UTC))
        segment2 = (datetime(2023, 1, 1, 11, 0, 0, tzinfo=UTC), 
                   datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC))
        overlap = get_overlap_duration(segment1, segment2)
        self.assertEqual(overlap, timedelta(hours=1))

    def test_describe_time_difference(self):
        """Test describing time difference in human-readable format"""
        time1 = datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC)
        
        # Test seconds
        time2 = datetime(2023, 1, 1, 10, 0, 30, tzinfo=UTC)
        self.assertEqual(describe_time_difference(time1, time2), "30 seconds")
        
        # Test minutes
        time2 = datetime(2023, 1, 1, 10, 5, 0, tzinfo=UTC)
        self.assertEqual(describe_time_difference(time1, time2), "5 minutes")
        
        # Test hours
        time2 = datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)
        self.assertEqual(describe_time_difference(time1, time2), "2 hours")
        
        # Test days
        time2 = datetime(2023, 1, 3, 10, 0, 0, tzinfo=UTC)
        self.assertEqual(describe_time_difference(time1, time2), "2 days")
        
        # Test weeks
        time2 = datetime(2023, 1, 15, 10, 0, 0, tzinfo=UTC)
        self.assertEqual(describe_time_difference(time1, time2), "2 weeks")
        
        # Test months
        time2 = datetime(2023, 3, 1, 10, 0, 0, tzinfo=UTC)
        self.assertEqual(describe_time_difference(time1, time2), "2 months")
        
        # Test years
        time2 = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
        self.assertEqual(describe_time_difference(time1, time2), "2 years")
        
        # Test reverse order (time2 before time1)
        time2 = datetime(2022, 1, 1, 10, 0, 0, tzinfo=UTC)
        self.assertEqual(describe_time_difference(time1, time2), "in 12 months")

    def test_rolling_window(self):
        """Test creating rolling windows"""
        timestamps = [
            datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC),
            datetime(2023, 1, 2, 0, 0, 0, tzinfo=UTC),
            datetime(2023, 1, 3, 0, 0, 0, tzinfo=UTC),
        ]
        
        # Test 1-day windows
        windows = rolling_window(timestamps, '1d')
        self.assertEqual(len(windows), 3)
        
        # Check first window
        self.assertEqual(windows[0][1], timestamps[0])  # End
        self.assertEqual(windows[0][0], timestamps[0] - timedelta(days=1))  # Start
        
        # Test 2-day windows
        windows = rolling_window(timestamps, '2d')
        self.assertEqual(len(windows), 3)
        
        # Check first window
        self.assertEqual(windows[0][1], timestamps[0])  # End
        self.assertEqual(windows[0][0], timestamps[0] - timedelta(days=2))  # Start

    def test_time_buckets(self):
        """Test grouping timestamps into buckets"""
        timestamps = [
            datetime(2023, 1, 1, 12, 30, 0, tzinfo=UTC),
            datetime(2023, 1, 1, 12, 45, 0, tzinfo=UTC),
            datetime(2023, 1, 1, 13, 15, 0, tzinfo=UTC),
            datetime(2023, 1, 1, 13, 45, 0, tzinfo=UTC),
        ]
        
        # Test 1-hour buckets
        buckets = time_buckets(timestamps, '1h')
        self.assertEqual(len(buckets), 2)
        
        bucket_keys = sorted(buckets.keys())
        self.assertEqual(bucket_keys[0], datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC))
        self.assertEqual(bucket_keys[1], datetime(2023, 1, 1, 13, 0, 0, tzinfo=UTC))
        
        self.assertEqual(len(buckets[bucket_keys[0]]), 2)  # 12:30, 12:45
        self.assertEqual(len(buckets[bucket_keys[1]]), 2)  # 13:15, 13:45

    def test_fill_time_gaps(self):
        """Test filling gaps in time series"""
        time_series = [
            (datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC), 10),
            (datetime(2023, 1, 1, 12, 2, 0, tzinfo=UTC), 20),
            (datetime(2023, 1, 1, 12, 5, 0, tzinfo=UTC), 30),
        ]
        
        # Test forward fill
        filled = fill_time_gaps(time_series, '1m', 'ffill')
        self.assertEqual(len(filled), 6)  # 12:00, 12:01, 12:02, 12:03, 12:04, 12:05
        self.assertEqual(filled[1][1], 10)  # 12:01 should have value from 12:00
        self.assertEqual(filled[3][1], 20)  # 12:03 should have value from 12:02
        
        # Test zero fill
        filled = fill_time_gaps(time_series, '1m', 'zero')
        self.assertEqual(len(filled), 6)
        self.assertEqual(filled[1][1], 0)  # 12:01 should be zero
        self.assertEqual(filled[3][1], 0)  # 12:03 should be zero

    def test_get_execution_latency_budget(self):
        """Test getting execution latency budgets"""
        # Test critical priority
        budget = get_execution_latency_budget('data_fetch', 'critical')
        self.assertEqual(budget, 0.001)  # 1ms
        
        # Test normal priority
        budget = get_execution_latency_budget('data_fetch', 'normal')
        self.assertEqual(budget, 0.1)  # 100ms
        
        # Test low priority
        budget = get_execution_latency_budget('data_fetch', 'low')
        self.assertEqual(budget, 1.0)  # 1s
        
        # Test default operation
        budget = get_execution_latency_budget('unknown_operation', 'normal')
        self.assertEqual(budget, 0.5)  # 500ms default for normal priority

    @patch('utils.time_utils.now')
    def test_is_historical_future_time(self, mock_now):
        """Test checking if time is historical or future"""
        # Set fixed current time
        current_time = datetime(2023, 1, 15, 12, 0, 0, tzinfo=UTC)
        mock_now.return_value = current_time
        
        # Test historical time
        past_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)
        self.assertTrue(is_historical_time(past_time))
        self.assertFalse(is_future_time(past_time))
        
        # Test future time
        future_time = datetime(2023, 2, 1, 12, 0, 0, tzinfo=UTC)
        self.assertFalse(is_historical_time(future_time))
        self.assertTrue(is_future_time(future_time))
        
        # Test current time
        self.assertFalse(is_future_time(current_time))
        self.assertFalse(is_historical_time(current_time))

    @patch('utils.time_utils.now')
    def test_time_until_since(self, mock_now):
        """Test calculating time until/since a target time"""
        # Set fixed current time
        current_time = datetime(2023, 1, 15, 12, 0, 0, tzinfo=UTC)
        mock_now.return_value = current_time
        
        # Test time until future date
        future_time = datetime(2023, 1, 16, 12, 0, 0, tzinfo=UTC)
        self.assertEqual(time_until(future_time), timedelta(days=1))
        
        # Test time until past date (should be zero)
        past_time = datetime(2023, 1, 14, 12, 0, 0, tzinfo=UTC)
        self.assertEqual(time_until(past_time), timedelta(0))
        
        # Test time since past date
        self.assertEqual(time_since(past_time), timedelta(days=1))
        
        # Test time since future date (should be zero)
        self.assertEqual(time_since(future_time), timedelta(0))

    def test_localize_timestamp(self):
        """Test localizing Unix timestamp to timezone"""
        # 2023-01-01 12:00:00 UTC as timestamp
        timestamp = 1672574400.0
        
        # Localize to UTC
        dt = localize_timestamp(timestamp, UTC)
        self.assertEqual(dt, datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC))
        
        # Localize to EST
        dt = localize_timestamp(timestamp, EST)
        self.assertEqual(dt.hour, 7)  # UTC-5
        self.assertEqual(dt.tzinfo, EST)
        
        # Localize to JST
        dt = localize_timestamp(timestamp, JST)
        self.assertEqual(dt.hour, 21)  # UTC+9
        self.assertEqual(dt.tzinfo, JST)

    def test_get_trading_timestamps(self):
        """Test generating timestamps within trading hours"""
        start = datetime(2023, 1, 2, 0, 0, 0, tzinfo=UTC)  # Monday
        end = datetime(2023, 1, 6, 23, 59, 59, tzinfo=UTC)  # Friday
        
        # Test hourly timestamps for US equities
        timestamps = get_trading_timestamps(start, end, '1h', TradingCalendar.US_EQUITIES)
        
        # Should only have timestamps during trading hours
        for ts in timestamps:
            # Convert to EST for easier checking
            ts_est = convert_timezone(ts, EST)
            self.assertTrue(9 <= ts_est.hour < 16)  # 9:30 AM to 4:00 PM
            self.assertTrue(0 <= ts_est.weekday() <= 4)  # Monday to Friday
        
        # Test with extended hours
        timestamps = get_trading_timestamps(start, end, '1h', TradingCalendar.US_EQUITIES, include_extended=True)
        
        # Should have more timestamps with extended hours
        self.assertTrue(len(timestamps) > 0)
        
        # Test with crypto (24/7)
        timestamps = get_trading_timestamps(start, end, '1d', TradingCalendar.CRYPTO)
        self.assertEqual(len(timestamps), 5)  # One per day, Monday to Friday

    def test_format_duration(self):
        """Test formatting durations in human-readable format"""
        # Test microseconds
        self.assertEqual(format_duration(0.0005), "500.00 µs")
        
        # Test milliseconds
        self.assertEqual(format_duration(0.5), "500.00 ms")
        
        # Test seconds
        self.assertEqual(format_duration(5), "5.00 s")
        
        # Test minutes and seconds
        self.assertEqual(format_duration(65), "1m 5s")
        
        # Test hours and minutes
        self.assertEqual(format_duration(3665), "1h 1m")
        
        # Test days and hours
        self.assertEqual(format_duration(90000), "1d 1h")

    def test_parse_trading_time(self):
        """Test parsing trading time strings"""
        # Set a reference date
        reference = datetime(2023, 1, 3, 0, 0, 0, tzinfo=UTC)  # Tuesday
        
        # Test parsing 'open'
        dt = parse_trading_time('open', reference, EST)
        self.assertEqual(dt.hour, 9)
        self.assertEqual(dt.minute, 30)
        self.assertEqual(dt.tzinfo, EST)
        
        # Test parsing 'close'
        dt = parse_trading_time('close', reference, EST)
        self.assertEqual(dt.hour, 16)
        self.assertEqual(dt.minute, 0)
        self.assertEqual(dt.tzinfo, EST)
        
        # Test parsing time format (24-hour)
        dt = parse_trading_time('14:30', reference, EST)
        self.assertEqual(dt.hour, 14)
        self.assertEqual(dt.minute, 30)
        self.assertEqual(dt.tzinfo, EST)
        
        # Test parsing time format (12-hour)
        dt = parse_trading_time('2:30 PM', reference, EST)
        self.assertEqual(dt.hour, 14)
        self.assertEqual(dt.minute, 30)
        self.assertEqual(dt.tzinfo, EST)
        
        # Test invalid format
        with self.assertRaises(ValueError):
            parse_trading_time('invalid', reference, EST)

    def test_market_time_conversions(self):
        """Test market time conversions"""
        # Create a US market time (EST)
        us_time = datetime(2023, 1, 3, 9, 30, 0, tzinfo=EST)
        
        # Convert to UTC
        utc_time = market_time_to_utc(us_time, 'us')
        self.assertEqual(utc_time.hour, 14)  # UTC+5 from EST
        self.assertEqual(utc_time.tzinfo, UTC)
        
        # Convert back to market time
        market_time = utc_to_market_time(utc_time, 'us')
        self.assertEqual(market_time, us_time)
        
        # Test with other markets
        jp_time = datetime(2023, 1, 3, 9, 0, 0, tzinfo=JST)
        utc_time = market_time_to_utc(jp_time, 'asia')
        self.assertEqual(utc_time.hour, 0)  # UTC-9 from JST
        
        # Convert back
        market_time = utc_to_market_time(utc_time, 'asia')
        self.assertEqual(market_time, jp_time)

    def test_get_calendar_for_asset(self):
        """Test getting appropriate calendar for assets"""
        # Test equities
        self.assertEqual(get_calendar_for_asset('equity', 'US'), TradingCalendar.US_EQUITIES)
        self.assertEqual(get_calendar_for_asset('equity', 'JP'), TradingCalendar.JAPAN)
        self.assertEqual(get_calendar_for_asset('equity', 'EU'), TradingCalendar.EUROPE)
        
        # Test crypto (region-independent)
        self.assertEqual(get_calendar_for_asset('crypto', 'US'), TradingCalendar.CRYPTO)
        self.assertEqual(get_calendar_for_asset('crypto', 'JP'), TradingCalendar.CRYPTO)
        
        # Test forex (region-independent)
        self.assertEqual(get_calendar_for_asset('forex', 'US'), TradingCalendar.FOREX)
        
        # Test futures
        self.assertEqual(get_calendar_for_asset('future', 'US'), TradingCalendar.US_FUTURES)
        
        # Test unknown asset (should default to US_EQUITIES)
        self.assertEqual(get_calendar_for_asset('unknown', 'US'), TradingCalendar.US_EQUITIES)

    def test_get_market_hours_in_period(self):
        """Test calculating market hours in a period"""
        start = datetime(2023, 1, 2, 0, 0, 0, tzinfo=UTC)  # Monday
        end = datetime(2023, 1, 6, 23, 59, 59, tzinfo=UTC)  # Friday
        
        # Test US equities (6.5 hours per day)
        hours = get_market_hours_in_period(start, end, TradingCalendar.US_EQUITIES)
        self.assertAlmostEqual(hours, 6.5 * 5, places=1)  # 5 trading days
        
        # Test with extended hours
        hours = get_market_hours_in_period(start, end, TradingCalendar.US_EQUITIES, include_extended=True)
        self.assertGreater(hours, 6.5 * 5)  # Should be more hours with extended trading
        
        # Test crypto (24/7)
        hours = get_market_hours_in_period(start, end, TradingCalendar.CRYPTO)
        self.assertAlmostEqual(hours, 24 * 5, places=1)  # 24 hours for 5 days
        
        # Test forex (24/5)
        hours = get_market_hours_in_period(start, end, TradingCalendar.FOREX)
        self.assertAlmostEqual(hours, 24 * 5, places=1)  # 24 hours for 5 days

    def test_align_to_trading_schedule(self):
        """Test aligning datetime to trading schedule"""
        # Test time already in trading hours
        dt = datetime(2023, 1, 3, 14, 30, 0, tzinfo=UTC)  # 9:30 AM EST on Tuesday
        result = align_to_trading_schedule(dt, TradingCalendar.US_EQUITIES)
        self.assertEqual(result, dt)
        
        # Test time before market open (forward alignment)
        dt = datetime(2023, 1, 3, 13, 0, 0, tzinfo=UTC)  # 8:00 AM EST on Tuesday
        result = align_to_trading_schedule(dt, TradingCalendar.US_EQUITIES, 'forward')
        self.assertEqual(result.hour, 14)  # 9:30 AM EST
        self.assertEqual(result.minute, 30)
        
        # Test time after market close (backward alignment)
        dt = datetime(2023, 1, 3, 21, 30, 0, tzinfo=UTC)  # 4:30 PM EST on Tuesday
        result = align_to_trading_schedule(dt, TradingCalendar.US_EQUITIES, 'backward')
        self.assertEqual(result.hour, 21)  # 4:00 PM EST
        self.assertEqual(result.minute, 0)
        
        # Test nearest alignment
        dt = datetime(2023, 1, 3, 13, 45, 0, tzinfo=UTC)  # 8:45 AM EST
        result = align_to_trading_schedule(dt, TradingCalendar.US_EQUITIES, 'nearest')
        self.assertEqual(result.hour, 14)  # 9:30 AM EST (closer than 4:00 PM previous day)
        self.assertEqual(result.minute, 30)

    def test_date_range(self):
        """Test generating date ranges with regular intervals"""
        start = datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC)
        end = datetime(2023, 1, 5, 0, 0, 0, tzinfo=UTC)
        
        # Test daily intervals
        dates = date_range(start, end, '1d')
        self.assertEqual(len(dates), 5)  # 5 days
        self.assertEqual(dates[0], start)
        self.assertEqual(dates[-1], end)
        
        # Test hourly intervals
        dates = date_range(start, end, '24h')
        self.assertEqual(len(dates), 5)  # Same as daily
        
        # Test shorter intervals
        dates = date_range(start, start + timedelta(hours=5), '1h')
        self.assertEqual(len(dates), 6)  # 0h, 1h, 2h, 3h, 4h, 5h

    def test_get_timezones_for_region(self):
        """Test getting timezones for a region"""
        # Test US timezones
        us_timezones = get_timezones_for_region('US')
        self.assertTrue(len(us_timezones) > 0)
        self.assertTrue(any('Eastern' in tz.zone for tz in us_timezones))
        self.assertTrue(any('Pacific' in tz.zone for tz in us_timezones))
        
        # Test Europe timezones
        eu_timezones = get_timezones_for_region('Europe')
        self.assertTrue(len(eu_timezones) > 0)
        self.assertTrue(any('London' in tz.zone for tz in eu_timezones))
        self.assertTrue(any('Paris' in tz.zone for tz in eu_timezones))
        
        # Test Asia timezones
        asia_timezones = get_timezones_for_region('Asia')
        self.assertTrue(len(asia_timezones) > 0)
        self.assertTrue(any('Tokyo' in tz.zone for tz in asia_timezones))
        self.assertTrue(any('Shanghai' in tz.zone for tz in asia_timezones))

    def test_calculate_time_efficiency(self):
        """Test calculating time efficiency ratio"""
        # Test ideal case (execution time equals theoretical minimum)
        efficiency = calculate_time_efficiency(10.0, 10.0)
        self.assertEqual(efficiency, 1.0)
        
        # Test good efficiency (execution time is twice the theoretical minimum)
        efficiency = calculate_time_efficiency(20.0, 10.0)
        self.assertEqual(efficiency, 0.5)
        
        # Test poor efficiency (execution time is 10x the theoretical minimum)
        efficiency = calculate_time_efficiency(100.0, 10.0)
        self.assertEqual(efficiency, 0.1)
        
        # Test boundary cases
        self.assertEqual(calculate_time_efficiency(0, 10.0), 0)
        self.assertEqual(calculate_time_efficiency(10.0, 0), 0)
        self.assertEqual(calculate_time_efficiency(-1, 10.0), 0)

    def test_get_current_market_session(self):
        """Test determining the current market session"""
        # Create calendar mock for testing
        calendar = TradingCalendar.US_EQUITIES
        
        # Test regular hours
        dt = datetime(2023, 1, 3, 14, 30, 0, tzinfo=UTC)  # 9:30 AM EST on Tuesday
        session = get_current_market_session(dt, calendar)
        self.assertEqual(session, 'regular_hours')
        
        # Test after hours
        dt = datetime(2023, 1, 3, 21, 30, 0, tzinfo=UTC)  # 4:30 PM EST on Tuesday
        session = get_current_market_session(dt, calendar)
        self.assertEqual(session, 'after_hours')
        
        # Test pre-market
        dt = datetime(2023, 1, 3, 13, 0, 0, tzinfo=UTC)  # 8:00 AM EST on Tuesday
        session = get_current_market_session(dt, calendar)
        self.assertEqual(session, 'pre_market')
        
        # Test closed (weekend)
        dt = datetime(2023, 1, 1, 14, 30, 0, tzinfo=UTC)  # Sunday
        session = get_current_market_session(dt, calendar)
        self.assertEqual(session, 'closed')
        
        # Test crypto (always regular hours)
        dt = datetime(2023, 1, 1, 14, 30, 0, tzinfo=UTC)  # Sunday
        session = get_current_market_session(dt, TradingCalendar.CRYPTO)
        self.assertEqual(session, 'regular_hours')


class TestDataProcessingFunctions(unittest.TestCase):
    """Test data processing and alignment functions"""

    def test_align_timestamps(self):
        """Test aligning timestamps to timeframe boundaries"""
        timestamps = [
            datetime(2023, 1, 1, 12, 17, 0, tzinfo=UTC),
            datetime(2023, 1, 1, 12, 32, 0, tzinfo=UTC),
            datetime(2023, 1, 1, 12, 59, 0, tzinfo=UTC),
        ]
        
        # Align to 15-minute boundaries
        aligned = align_timestamps(timestamps, '15m')
        self.assertEqual(aligned[0], datetime(2023, 1, 1, 12, 15, 0, tzinfo=UTC))
        self.assertEqual(aligned[1], datetime(2023, 1, 1, 12, 30, 0, tzinfo=UTC))
        self.assertEqual(aligned[2], datetime(2023, 1, 1, 12, 45, 0, tzinfo=UTC))
        
        # Align to 1-hour boundaries
        aligned = align_timestamps(timestamps, '1h')
        self.assertEqual(aligned[0], datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC))
        self.assertEqual(aligned[1], datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC))
        self.assertEqual(aligned[2], datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC))

    def test_create_time_buckets(self):
        """Test creating time buckets"""
        timestamps = [
            datetime(2023, 1, 1, 12, 5, 0, tzinfo=UTC),
            datetime(2023, 1, 1, 12, 15, 0, tzinfo=UTC),
            datetime(2023, 1, 1, 12, 25, 0, tzinfo=UTC),
            datetime(2023, 1, 1, 12, 35, 0, tzinfo=UTC),
            datetime(2023, 1, 1, 12, 45, 0, tzinfo=UTC),
        ]
        
        # Create 30-minute buckets
        buckets = create_time_buckets(timestamps, '30m')
        self.assertEqual(len(buckets), 2)
        
        bucket_times = sorted(buckets.keys())
        self.assertEqual(bucket_times[0], datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC))
        self.assertEqual(bucket_times[1], datetime(2023, 1, 1, 12, 30, 0, tzinfo=UTC))
        
        self.assertEqual(len(buckets[bucket_times[0]]), 3)  # 12:05, 12:15, 12:25
        self.assertEqual(len(buckets[bucket_times[1]]), 2)  # 12:35, 12:45

    def test_calculate_execution_stats(self):
        """Test calculating execution statistics"""
        # Test with execution times
        times = [0.1, 0.2, 0.3, 0.4, 0.5]
        stats = calculate_execution_stats(times)
        
        self.assertEqual(stats['min'], 0.1)
        self.assertEqual(stats['max'], 0.5)
        self.assertAlmostEqual(stats['mean'], 0.3, places=6)
        self.assertAlmostEqual(stats['median'], 0.3, places=6)
        self.assertAlmostEqual(stats['90th_percentile'], 0.46, places=6)
        self.assertEqual(stats['count'], 5)
        self.assertEqual(stats['total'], 1.5)
        
        # Test with empty list
        stats = calculate_execution_stats([])
        self.assertEqual(stats['min'], 0)
        self.assertEqual(stats['max'], 0)
        self.assertEqual(stats['count'], 0)


class TestHolidayAndBusinessDays(unittest.TestCase):
    """Test holiday and business day functions"""

    def test_is_holiday(self):
        """Test holiday detection"""
        # Test common US holidays
        new_years = datetime(2023, 1, 1, tzinfo=UTC)
        self.assertTrue(is_holiday(new_years, TradingCalendar.US_EQUITIES))
        
        mlk_day = datetime(2023, 1, 16, tzinfo=UTC)  # 3rd Monday in January 2023
        self.assertTrue(is_holiday(mlk_day, TradingCalendar.US_EQUITIES))
        
        # Test non-holiday
        regular_day = datetime(2023, 1, 10, tzinfo=UTC)  # Regular Tuesday
        self.assertFalse(is_holiday(regular_day, TradingCalendar.US_EQUITIES))

    def test_get_business_days_between(self):
        """Test counting business days between dates"""
        # Test one week (5 business days)
        start = datetime(2023, 1, 2, tzinfo=UTC)  # Monday
        end = datetime(2023, 1, 6, tzinfo=UTC)    # Friday
        days = get_business_days_between(start, end, TradingCalendar.US_EQUITIES)
        self.assertEqual(days, 5)
        
        # Test including weekend (still 5 business days)
        end = datetime(2023, 1, 8, tzinfo=UTC)    # Sunday
        days = get_business_days_between(start, end, TradingCalendar.US_EQUITIES)
        self.assertEqual(days, 5)
        
        # Test crypto (7 days)
        days = get_business_days_between(start, end, TradingCalendar.CRYPTO)
        self.assertEqual(days, 7)


if __name__ == '__main__':
    unittest.main()