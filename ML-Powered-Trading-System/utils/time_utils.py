"""
time_utils.py - Time and Date Utilities

This module provides utilities for time and date handling throughout the trading system,
with special attention to timezone management, time formatting, and financial market
time calculations.
"""

import logging
from datetime import datetime, timedelta, time
import calendar
import pytz
from enum import Enum
from typing import Optional, Union, Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

# Additional utilities for data time alignment and performance metrics

def align_timestamps(timestamps: List[datetime],
                    timeframe: str) -> List[datetime]:
    """
    Align a list of timestamps to the specified timeframe boundaries.

    Args:
        timestamps: List of datetime objects
        timeframe: Timeframe to align to (e.g., '1m', '1h', '1d')

    Returns:
        List of aligned datetime objects
    """
    return [get_interval_timestamp(dt, timeframe) for dt in timestamps]


def create_time_buckets(data_timestamps: List[datetime],
                      interval: str) -> Dict[datetime, List[datetime]]:
    """
    Group timestamps into time buckets based on the interval.

    Args:
        data_timestamps: List of datetime objects
        interval: Interval for buckets (e.g., '1m', '1h', '1d')

    Returns:
        Dictionary mapping bucket start time to list of timestamps in that bucket
    """
    buckets = {}

    for dt in data_timestamps:
        bucket_time = get_interval_timestamp(dt, interval)
        if bucket_time not in buckets:
            buckets[bucket_time] = []
        buckets[bucket_time].append(dt)

    return buckets


def calculate_execution_stats(execution_times: List[float]) -> Dict[str, float]:
    """
    Calculate statistics for a list of execution times.

    Args:
        execution_times: List of execution times in seconds

    Returns:
        Dictionary with execution statistics
    """
    if not execution_times:
        return {
            'min': 0,
            'max': 0,
            'mean': 0,
            'median': 0,
            '90th_percentile': 0,
            '95th_percentile': 0,
            '99th_percentile': 0,
            'count': 0,
            'total': 0
        }

    import numpy as np

    times = np.array(execution_times)

    return {
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'mean': float(np.mean(times)),
        'median': float(np.median(times)),
        '90th_percentile': float(np.percentile(times, 90)),
        '95th_percentile': float(np.percentile(times, 95)),
        '99th_percentile': float(np.percentile(times, 99)),
        'count': len(times),
        'total': float(np.sum(times))
    }


class Timer:
    """Utility class for timing code execution"""

    def __init__(self, name: str = None):
        """
        Initialize timer.

        Args:
            name: Optional name for this timer
        """
        self.name = name or 'timer'
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        """Start timing when entering context"""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing when exiting context"""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time

        if exc_type is not None:
            logger.warning(f"Timer '{self.name}' exited with exception: {exc_type.__name__}")

    def reset(self):
        """Reset the timer"""
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        self.end_time = None
        self.elapsed = None
        return self

    def stop(self):
        """Stop the timer and return elapsed time"""
        if self.start_time is None:
            raise ValueError("Timer was not started")

        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed

    def get_elapsed(self, reset: bool = False) -> float:
        """
        Get elapsed time.

        Args:
            reset: Whether to reset the timer after getting elapsed time

        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0

        if self.end_time is None:
            # Timer is still running
            current = time.time()
            elapsed = current - self.start_time
        else:
            elapsed = self.elapsed

        if reset:
            self.reset()

        return elapsed


def calculate_time_efficiency(execution_time: float,
                            theoretical_minimum_time: float) -> float:
    """
    Calculate time efficiency ratio.

    Args:
        execution_time: Actual execution time
        theoretical_minimum_time: Theoretical minimum time

    Returns:
        Efficiency ratio (0-1, higher is better)
    """
    if execution_time <= 0 or theoretical_minimum_time <= 0:
        return 0

    return min(1.0, theoretical_minimum_time / execution_time)


def get_current_market_session(dt: datetime,
                            calendar: TradingCalendar) -> str:
    """
    Determine the current market session for a given datetime.

    Args:
        dt: Datetime to check
        calendar: Trading calendar to use

    Returns:
        Session name ('pre_market', 'regular_hours', 'after_hours', 'closed')
    """
    if calendar not in TRADING_HOURS:
        raise ValueError(f"Unknown trading calendar: {calendar}")

    calendar_info = TRADING_HOURS[calendar]

    # Convert to calendar timezone
    dt = convert_timezone(dt, calendar_info['timezone'])

    # Check if it's a trading day
    if dt.weekday() not in calendar_info['trading_days']:
        return 'closed'

    # For 24/7 markets
    if calendar == TradingCalendar.CRYPTO:
        return 'regular_hours'

    # For forex
    if calendar == TradingCalendar.FOREX:
        if dt.weekday() == 5:  # Saturday
            return 'closed'
        return 'regular_hours'

    # Get current time
    current_time = dt.time()

    # Check each session
    if 'regular_hours' in calendar_info:
        regular_start = calendar_info['regular_hours']['start']
        regular_end = calendar_info['regular_hours']['end']

        # Check if in regular hours
        in_regular = False
        if regular_start <= regular_end:
            in_regular = regular_start <= current_time < regular_end
        else:  # Overnight session
            in_regular = current_time >= regular_start or current_time < regular_end

        if in_regular:
            return 'regular_hours'

    # Check extended hours if available
    if 'extended_hours' in calendar_info:
        extended_start = calendar_info['extended_hours']['start']
        extended_end = calendar_info['extended_hours']['end']

        # Pre-market
        if current_time < regular_start:
            if extended_start <= current_time:
                return 'pre_market'

        # After-hours
        if current_time >= regular_end:
            if current_time < extended_end:
                return 'after_hours'

    return 'closed'


def get_calendar_for_asset(asset_type: str, region: str = 'US') -> TradingCalendar:
    """
    Get the appropriate trading calendar for an asset.

    Args:
        asset_type: Type of asset ('equity', 'forex', 'crypto', 'future')
        region: Region code ('US', 'JP', 'EU', etc.)

    Returns:
        Appropriate trading calendar
    """
    asset_type = asset_type.lower()
    region = region.upper()

    if asset_type == 'crypto':
        return TradingCalendar.CRYPTO

    if asset_type == 'forex':
        return TradingCalendar.FOREX

    if asset_type == 'equity':
        if region == 'US':
            return TradingCalendar.US_EQUITIES
        elif region == 'JP':
            return TradingCalendar.JAPAN
        elif region in ['EU', 'UK']:
            return TradingCalendar.EUROPE

    if asset_type == 'future':
        if region == 'US':
            return TradingCalendar.US_FUTURES

    # Default to US equities if no match
    logger.warning(f"No specific calendar for {asset_type} in {region}, using US_EQUITIES")
    return TradingCalendar.US_EQUITIES


def get_market_hours_in_period(start_date: datetime,
                             end_date: datetime,
                             calendar: TradingCalendar,
                             include_extended: bool = False) -> float:
    """
    Calculate the number of market hours in a period.

    Args:
        start_date: Start date
        end_date: End date
        calendar: Trading calendar to use
        include_extended: Whether to include extended hours

    Returns:
        Number of market hours in the period
    """
    # Special cases for 24-hour markets
    if calendar == TradingCalendar.CRYPTO:
        # 24/7 market
        return (end_date - start_date).total_seconds() / 3600

    if calendar == TradingCalendar.FOREX:
        # 24/5 market (closed on weekends)
        days = get_business_days_between(start_date, end_date, calendar)
        return days * 24

    # For markets with specific hours
    calendar_info = TRADING_HOURS[calendar]
    trading_days = get_trading_days(start_date, end_date, calendar)

    # Get session hours
    if include_extended and 'extended_hours' in calendar_info:
        session_start = calendar_info['extended_hours']['start']
        session_end = calendar_info['extended_hours']['end']
    else:
        session_start = calendar_info['regular_hours']['start']
        session_end = calendar_info['regular_hours']['end']

    # Calculate hours per day
    if session_start <= session_end:
        hours_per_day = (session_end.hour - session_start.hour) + \
                        (session_end.minute - session_start.minute) / 60
    else:
        # Overnight session
        hours_per_day = (24 - session_start.hour - session_start.minute/60) + \
                       (session_end.hour + session_end.minute/60)

    return len(trading_days) * hours_per_day


def align_to_trading_schedule(dt: datetime,
                           calendar: TradingCalendar,
                           direction: str = 'nearest') -> datetime:
    """
    Align a datetime to the trading schedule.

    Args:
        dt: Datetime to align
        calendar: Trading calendar to use
        direction: Alignment direction ('nearest', 'forward', 'backward')

    Returns:
        Aligned datetime
    """
    if is_trading_hours(dt, calendar):
        return dt

    if direction == 'forward':
        return get_next_trading_time(dt, calendar)

    # Get the previous trading time
    calendar_info = TRADING_HOURS[calendar]

    # Convert to calendar timezone
    dt = convert_timezone(dt, calendar_info['timezone'])

    # Special cases for 24-hour markets
    if calendar == TradingCalendar.CRYPTO:
        return dt

    # Get trading hours
    session_start = calendar_info['regular_hours']['start']
    session_end = calendar_info['regular_hours']['end']

    current_time = dt.time()
    current_day = dt

    # If after market close, get previous close
    if current_time > session_end:
        previous_close = dt.replace(
            hour=session_end.hour,
            minute=session_end.minute,
            second=0,
            microsecond=0
        )
        return previous_close

    # If before market open, get previous day's close
    if current_time < session_start:
        # Find previous trading day
        previous_day = current_day - timedelta(days=1)
        while previous_day.weekday() not in calendar_info['trading_days']:
            previous_day = previous_day - timedelta(days=1)

        previous_close = previous_day.replace(
            hour=session_end.hour,
            minute=session_end.minute,
            second=0,
            microsecond=0
        )
        return previous_close

    # Handle 'nearest' direction
    if direction == 'nearest':
        forward = get_next_trading_time(dt, calendar)
        backward = align_to_trading_schedule(dt, calendar, 'backward')

        forward_diff = (forward - dt).total_seconds()
        backward_diff = (dt - backward).total_seconds()

        return forward if forward_diff < backward_diff else backward

    return dt


def date_range(start_date: datetime,
             end_date: datetime,
             interval: str) -> List[datetime]:
    """
    Generate a range of dates with regular intervals.

    Args:
        start_date: Start date
        end_date: End date
        interval: Interval string (e.g., '1d', '1h')

    Returns:
        List of datetime objects
    """
    delta = parse_period(interval)
    dates = []

    current = start_date
    while current <= end_date:
        dates.append(current)
        current += delta

    return dates


def get_timezones_for_region(region: str) -> List[pytz.timezone]:
    """
    Get a list of timezones for a geographical region.

    Args:
        region: Region name ('US', 'Europe', 'Asia', etc.)

    Returns:
        List of timezone objects
    """
    region = region.lower()
    timezones = []

    for tz_name in pytz.all_timezones:
        if region in tz_name.lower():
            timezones.append(pytz.timezone(tz_name))

    return timezones


class SamplingPeriodicity(Enum):
    """Standard sampling periodicities for time series data"""
    TICK = "tick"           # Tick data
    MILLISECOND = "ms"      # Milliseconds
    SECOND = "s"            # Seconds
    MINUTE = "m"            # Minutes
    HOUR = "h"              # Hours
    DAY = "d"               # Days
    WEEK = "w"              # Weeks
    MONTH = "M"             # Months
    QUARTER = "Q"           # Quarters
    YEAR = "Y"              # Years


class TimingStatistics:
    """Class for tracking and analyzing execution times"""

    def __init__(self, name: str = "", max_samples: int = 1000):
        """
        Initialize timing statistics tracker.

        Args:
            name: Name for this statistics tracker
            max_samples: Maximum number of samples to keep
        """
        self.name = name
        self.max_samples = max_samples
        self.samples = []
        self.total_time = 0
        self.count = 0
        self.min_time = float('inf')
        self.max_time = 0
        self._lock = threading.RLock()  # For thread safety

    def add_sample(self, execution_time: float) -> None:
        """
        Add a timing sample.

        Args:
            execution_time: Execution time in seconds
        """
        with self._lock:
            # Update statistics
            self.count += 1
            self.total_time += execution_time
            self.min_time = min(self.min_time, execution_time)
            self.max_time = max(self.max_time, execution_time)

            # Add to samples list, keeping it at max size
            self.samples.append(execution_time)
            if len(self.samples) > self.max_samples:
                self.samples.pop(0)

    def get_stats(self) -> Dict[str, float]:
        """
        Get current timing statistics.

        Returns:
            Dictionary with timing statistics
        """
        with self._lock:
            if not self.samples:
                return {
                    'min': 0,
                    'max': 0,
                    'mean': 0,
                    'median': 0,
                    '95th_percentile': 0,
                    '99th_percentile': 0,
                    'count': 0,
                    'total': 0
                }

            import numpy as np

            # Calculate statistics using numpy
            samples_array = np.array(self.samples)

            return {
                'min': float(np.min(samples_array)),
                'max': float(np.max(samples_array)),
                'mean': float(np.mean(samples_array)),
                'median': float(np.median(samples_array)),
                '95th_percentile': float(np.percentile(samples_array, 95)),
                '99th_percentile': float(np.percentile(samples_array, 99)),
                'count': self.count,
                'total': self.total_time
            }

    def reset(self) -> None:
        """Reset all statistics"""
        with self._lock:
            self.samples = []
            self.total_time = 0
            self.count = 0
            self.min_time = float('inf')
            self.max_time = 0

    def __str__(self) -> str:
        """String representation of timing statistics"""
        stats = self.get_stats()
        return (f"TimingStatistics({self.name}): "
                f"count={stats['count']}, "
                f"mean={stats['mean']:.6f}s, "
                f"median={stats['median']:.6f}s, "
                f"95p={stats['95th_percentile']:.6f}s, "
                f"min={stats['min']:.6f}s, "
                f"max={stats['max']:.6f}s")


class TimeSeriesResampler:
    """Utility for resampling time series data"""

    @staticmethod
    def resample(time_series: List[Tuple[datetime, Any]],
                target_interval: str,
                aggregation_method: str = 'last') -> List[Tuple[datetime, Any]]:
        """
        Resample a time series to a different interval.

        Args:
            time_series: List of (timestamp, value) tuples
            target_interval: Target interval (e.g., '1m', '1h')
            aggregation_method: Method to aggregate values ('last', 'first', 'mean', 'sum', 'min', 'max')

        Returns:
            Resampled time series
        """
        if not time_series:
            return []

        # Create buckets
        buckets = {}
        for timestamp, value in time_series:
            bucket_time = get_interval_timestamp(timestamp, target_interval)
            if bucket_time not in buckets:
                buckets[bucket_time] = []
            buckets[bucket_time].append((timestamp, value))

        # Aggregate values in each bucket
        result = []
        for bucket_time, points in sorted(buckets.items()):
            if aggregation_method == 'last':
                # Take the last value in the bucket
                result.append((bucket_time, points[-1][1]))

            elif aggregation_method == 'first':
                # Take the first value in the bucket
                result.append((bucket_time, points[0][1]))

            elif aggregation_method == 'mean':
                # Calculate mean of values
                values = [p[1] for p in points]
                result.append((bucket_time, sum(values) / len(values)))

            elif aggregation_method == 'sum':
                # Sum values
                values = [p[1] for p in points]
                result.append((bucket_time, sum(values)))

            elif aggregation_method == 'min':
                # Take minimum value
                values = [p[1] for p in points]
                result.append((bucket_time, min(values)))

            elif aggregation_method == 'max':
                # Take maximum value
                values = [p[1] for p in points]
                result.append((bucket_time, max(values)))

            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        return result


class PerformanceTimer:
    """
    Advanced timer for measuring and logging code performance.
    Supports hierarchical timing, aggregation, and statistics.
    """

    _instances = {}  # Class-level dictionary to store named instances

    @classmethod
    def get_instance(cls, name: str = "default") -> 'PerformanceTimer':
        """
        Get a named timer instance (singleton pattern).

        Args:
            name: Name for the timer instance

        Returns:
            PerformanceTimer instance
        """
        if name not in cls._instances:
            cls._instances[name] = PerformanceTimer(name)
        return cls._instances[name]

    def __init__(self, name: str = "default"):
        """
        Initialize performance timer.

        Args:
            name: Name for this timer
        """
        self.name = name
        self.timers = {}  # Dict mapping label to TimingStatistics
        self.active_timers = {}  # Dict mapping label to start time
        self.hierarchy = []  # Stack for hierarchical timing
        self._lock = threading.RLock()

    def start(self, label: str) -> None:
        """
        Start timing a section.

        Args:
            label: Label for the section
        """
        with self._lock:
            if label not in self.timers:
                self.timers[label] = TimingStatistics(label)

            self.active_timers[label] = time.time()
            self.hierarchy.append(label)

    def stop(self, label: str = None) -> float:
        """
        Stop timing a section and record the time.

        Args:
            label: Label for the section (if None, use last started timer)

        Returns:
            Elapsed time in seconds
        """
        with self._lock:
            if label is None:
                if not self.hierarchy:
                    raise ValueError("No active timers to stop")
                label = self.hierarchy.pop()
            elif label in self.hierarchy:
                # Remove this and all child timers
                idx = self.hierarchy.index(label)
                removed = self.hierarchy[idx:]
                self.hierarchy = self.hierarchy[:idx]

                # Stop all child timers
                for child in removed[1:]:
                    if child in self.active_timers:
                        self.stop(child)

            if label not in self.active_timers:
                raise ValueError(f"Timer '{label}' was not started")

            start_time = self.active_timers.pop(label)
            elapsed = time.time() - start_time

            self.timers[label].add_sample(elapsed)

            return elapsed

    def get_stats(self, label: str = None) -> Dict[str, Any]:
        """
        Get timing statistics.

        Args:
            label: Label to get statistics for (if None, get all)

        Returns:
            Dictionary with timing statistics
        """
        with self._lock:
            if label is not None:
                if label not in self.timers:
                    raise ValueError(f"No statistics for timer '{label}'")
                return self.timers[label].get_stats()

            # Return all stats
            return {label: timer.get_stats() for label, timer in self.timers.items()}

    def reset(self, label: str = None) -> None:
        """
        Reset timing statistics.

        Args:
            label: Label to reset (if None, reset all)
        """
        with self._lock:
            if label is not None:
                if label in self.timers:
                    self.timers[label].reset()
                if label in self.active_timers:
                    del self.active_timers[label]
            else:
                # Reset all
                for timer in self.timers.values():
                    timer.reset()
                self.active_timers = {}
                self.hierarchy = []

    def summary(self) -> str:
        """
        Get a summary of all timing statistics.

        Returns:
            String summary
        """
        with self._lock:
            lines = [f"=== Performance Timer: {self.name} ==="]

            if not self.timers:
                lines.append("No timing data available")
                return "\n".join(lines)

            # Sort by total time descending
            sorted_timers = sorted(
                self.timers.items(),
                key=lambda x: x[1].get_stats()['total'],
                reverse=True
            )

            for label, timer in sorted_timers:
                stats = timer.get_stats()
                lines.append(f"{label}: count={stats['count']}, "
                            f"mean={stats['mean']:.6f}s, "
                            f"median={stats['median']:.6f}s, "
                            f"95p={stats['95th_percentile']:.6f}s, "
                            f"total={stats['total']:.6f}s")

            return "\n".join(lines)


def get_quarter_start(dt: datetime) -> datetime:
    """
    Get the start date of the quarter containing this date.

    Args:
        dt: Date to check

    Returns:
        First day of the quarter
    """
    quarter = (dt.month - 1) // 3
    return dt.replace(month=quarter*3+1, day=1, hour=0, minute=0, second=0, microsecond=0)


def get_quarter_end(dt: datetime) -> datetime:
    """
    Get the end date of the quarter containing this date.

    Args:
        dt: Date to check

    Returns:
        Last day of the quarter
    """
    quarter = (dt.month - 1) // 3
    if quarter < 3:
        # Not the last quarter
        next_quarter_start = dt.replace(month=quarter*3+4, day=1)
    else:
        # Last quarter, move to next year
        next_quarter_start = dt.replace(year=dt.year+1, month=1, day=1)

    # End is one day before the start of next quarter
    return next_quarter_start - timedelta(days=1)


def get_month_end(dt: datetime) -> datetime:
    """
    Get the last day of the month containing this date.

    Args:
        dt: Date to check

    Returns:
        Last day of the month
    """
    # Get the last day using calendar
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    return dt.replace(day=last_day, hour=23, minute=59, second=59, microsecond=999999)


def get_week_start(dt: datetime, start_day: int = 0) -> datetime:
    """
    Get the start of the week containing this date.

    Args:
        dt: Date to check
        start_day: Start day of week (0=Monday, 6=Sunday)

    Returns:
        First day of the week
    """
    # Calculate days to subtract to get to start day
    days_diff = (dt.weekday() - start_day) % 7
    return (dt - timedelta(days=days_diff)).replace(hour=0, minute=0, second=0, microsecond=0)


def get_week_end(dt: datetime, start_day: int = 0) -> datetime:
    """
    Get the end of the week containing this date.

    Args:
        dt: Date to check
        start_day: Start day of week (0=Monday, 6=Sunday)

    Returns:
        Last day of the week
    """
    # Week end is 6 days after week start
    week_start = get_week_start(dt, start_day)
    return (week_start + timedelta(days=6)).replace(hour=23, minute=59, second=59, microsecond=999999)


def estimate_data_periodicity(timestamps: List[datetime]) -> str:
    """
    Estimate the periodicity of a time series.

    Args:
        timestamps: List of datetime objects

    Returns:
        Estimated periodicity string (e.g., '1m', '1h')
    """
    if not timestamps or len(timestamps) < 2:
        return "unknown"

    # Sort timestamps
    sorted_times = sorted(timestamps)

    # Calculate differences between consecutive timestamps
    deltas = [(sorted_times[i+1] - sorted_times[i]).total_seconds()
              for i in range(len(sorted_times)-1)]

    if not deltas:
        return "unknown"

    # Calculate median difference
    import numpy as np
    median_seconds = float(np.median(deltas))

    # Determine appropriate unit
    if median_seconds < 1:
        # Milliseconds
        return f"{int(median_seconds * 1000)}ms"
    elif median_seconds < 60:
        # Seconds
        return f"{int(median_seconds)}s"
    elif median_seconds < 3600:
        # Minutes
        minutes = int(median_seconds / 60)
        return f"{minutes}m"
    elif median_seconds < 86400:
        # Hours
        hours = int(median_seconds / 3600)
        return f"{hours}h"
    elif median_seconds < 604800:
        # Days
        days = int(median_seconds / 86400)
        return f"{days}d"
    elif median_seconds < 2592000:
        # Weeks
        weeks = int(median_seconds / 604800)
        return f"{weeks}w"
    else:
        # Months or longer
        months = int(median_seconds / 2592000)
        return f"{months}M"


def generate_calendar_periods(start_date: datetime, end_date: datetime, period: str) -> List[Tuple[datetime, datetime]]:
    """
    Generate calendar periods between start and end dates.

    Args:
        start_date: Start date
        end_date: End date
        period: Period type ('day', 'week', 'month', 'quarter', 'year')

    Returns:
        List of (period_start, period_end) tuples
    """
    periods = []
    current = start_date

    period = period.lower()

    while current <= end_date:
        if period == 'day':
            period_start = current.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start.replace(hour=23, minute=59, second=59, microsecond=999999)
            next_start = period_start + timedelta(days=1)

        elif period == 'week':
            period_start = get_week_start(current)
            period_end = get_week_end(current)
            next_start = period_start + timedelta(days=7)

        elif period == 'month':
            period_start = current.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            period_end = get_month_end(current)

            # Next month
            if current.month == 12:
                next_start = current.replace(year=current.year+1, month=1, day=1)
            else:
                next_start = current.replace(month=current.month+1, day=1)

        elif period == 'quarter':
            period_start = get_quarter_start(current)
            period_end = get_quarter_end(current)

            # Next quarter
            quarter = (current.month - 1) // 3
            if quarter < 3:
                next_start = current.replace(month=quarter*3+4, day=1)
            else:
                next_start = current.replace(year=current.year+1, month=1, day=1)

        elif period == 'year':
            period_start = current.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            period_end = current.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
            next_start = current.replace(year=current.year+1, month=1, day=1)

        else:
            raise ValueError(f"Unknown period: {period}")

        # Add period if it overlaps with the date range
        if period_end >= start_date and period_start <= end_date:
            # Clip to the requested date range
            clipped_start = max(period_start, start_date)
            clipped_end = min(period_end, end_date)
            periods.append((clipped_start, clipped_end))

        # Move to next period
        current = next_start

    return periods


def time_segments_overlap(segment1: Tuple[datetime, datetime],
                        segment2: Tuple[datetime, datetime]) -> bool:
    """
    Check if two time segments overlap.

    Args:
        segment1: First segment as (start, end) tuple
        segment2: Second segment as (start, end) tuple

    Returns:
        True if segments overlap, False otherwise
    """
    start1, end1 = segment1
    start2, end2 = segment2

    # Segments overlap if one starts before the other ends
    return start1 <= end2 and start2 <= end1


def get_overlap_duration(segment1: Tuple[datetime, datetime],
                       segment2: Tuple[datetime, datetime]) -> timedelta:
    """
    Calculate the duration of overlap between two time segments.

    Args:
        segment1: First segment as (start, end) tuple
        segment2: Second segment as (start, end) tuple

    Returns:
        Duration of overlap as timedelta
    """
    start1, end1 = segment1
    start2, end2 = segment2

    if not time_segments_overlap(segment1, segment2):
        return timedelta(0)

    # Calculate overlap
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    return overlap_end - overlap_start


def describe_time_difference(time1: datetime, time2: datetime) -> str:
    """
    Generate a human-readable description of the time difference.

    Args:
        time1: First time
        time2: Second time

    Returns:
        Human-readable description
    """
    # Ensure time1 is earlier
    if time1 > time2:
        time1, time2 = time2, time1
        prefix = "in "
    else:
        prefix = ""

    diff = time2 - time1
    seconds = diff.total_seconds()

    if seconds < 60:
        return f"{prefix}{int(seconds)} seconds"

    minutes = seconds / 60
    if minutes < 60:
        return f"{prefix}{int(minutes)} minutes"

    hours = minutes / 60
    if hours < 24:
        return f"{prefix}{int(hours)} hours"

    days = hours / 24
    if days < 7:
        return f"{prefix}{int(days)} days"

    weeks = days / 7
    if weeks < 4:
        return f"{prefix}{int(weeks)} weeks"

    months = days / 30.44  # Average month length
    if months < 12:
        return f"{prefix}{int(months)} months"

    years = days / 365.25  # Average year length
    return f"{prefix}{int(years)} years"


def rolling_window(timestamps: List[datetime], window_size: str) -> List[Tuple[datetime, datetime]]:
    """
    Create rolling windows of a specified size.

    Args:
        timestamps: List of datetime objects
        window_size: Window size (e.g., '1d', '7d')

    Returns:
        List of (window_start, window_end) tuples
    """
    if not timestamps:
        return []

    # Sort timestamps
    sorted_times = sorted(timestamps)
    window_delta = parse_period(window_size)

    windows = []
    for ts in sorted_times:
        window_end = ts
        window_start = window_end - window_delta
        windows.append((window_start, window_end))

    return windows


def time_buckets(timestamps: List[datetime], bucket_size: str) -> Dict[datetime, List[datetime]]:
    """
    Group timestamps into time buckets.

    Args:
        timestamps: List of datetime objects
        bucket_size: Bucket size (e.g., '1h', '1d')

    Returns:
        Dictionary mapping bucket start time to list of timestamps
    """
    buckets = {}

    for ts in timestamps:
        bucket_start = get_interval_timestamp(ts, bucket_size)
        if bucket_start not in buckets:
            buckets[bucket_start] = []
        buckets[bucket_start].append(ts)

    return buckets


def fill_time_gaps(time_series: List[Tuple[datetime, Any]],
                interval: str,
                fill_method: str = 'ffill') -> List[Tuple[datetime, Any]]:
    """
    Fill gaps in a time series.

    Args:
        time_series: List of (timestamp, value) tuples
        interval: Interval between points (e.g., '1h', '1d')
        fill_method: Method to fill gaps ('ffill', 'bfill', 'zero', 'nan')

    Returns:
        Time series with gaps filled
    """
    if not time_series:
        return []

    # Sort time series
    sorted_series = sorted(time_series, key=lambda x: x[0])

    # Calculate the expected timestamps
    start_time = sorted_series[0][0]
    end_time = sorted_series[-1][0]
    delta = parse_period(interval)

    expected_times = []
    current = start_time
    while current <= end_time:
        expected_times.append(current)
        current += delta

    # Convert time series to dict for easy lookup
    time_dict = {ts: val for ts, val in sorted_series}

    # Fill gaps
    result = []
    last_value = None

    for ts in expected_times:
        if ts in time_dict:
            # Timestamp exists in original series
            value = time_dict[ts]
            last_value = value
        else:
            # Gap - fill according to method
            if fill_method == 'ffill' and last_value is not None:
                value = last_value
            elif fill_method == 'bfill':
                # Find next available value
                next_values = [(t, v) for t, v in sorted_series if t > ts]
                value = next_values[0][1] if next_values else None
            elif fill_method == 'zero':
                # Use zero or empty string
                if last_value is not None:
                    value = type(last_value)()  # Empty of same type
                else:
                    value = 0
            elif fill_method == 'nan':
                value = float('nan')
            else:
                raise ValueError(f"Unknown fill method: {fill_method}")

        result.append((ts, value))

    return result


def get_execution_latency_budget(operation_type: str,
                              priority: str = 'normal') -> float:
    """
    Get the target latency budget for an operation.

    Args:
        operation_type: Type of operation ('data_fetch', 'order_create', 'strategy_update', etc.)
        priority: Priority level ('critical', 'high', 'normal', 'low')

    Returns:
        Target latency in seconds
    """
    # Define latency budgets for different operations
    latency_budgets = {
        'critical': {
            'data_fetch': 0.001,           # 1ms
            'order_create': 0.005,         # 5ms
            'order_update': 0.005,         # 5ms
            'order_cancel': 0.005,         # 5ms
            'position_update': 0.001,      # 1ms
            'signal_generation': 0.01,     # 10ms
            'risk_check': 0.001,           # 1ms
            'portfolio_update': 0.01,      # 10ms
            'strategy_update': 0.05,       # 50ms
            'market_data_process': 0.001,  # 1ms
            'tick_processing': 0.0005,     # 0.5ms
            'state_update': 0.001,         # 1ms
            'event_publish': 0.0005,       # 0.5ms
            'default': 0.01                # 10ms
        },
        'high': {
            'data_fetch': 0.01,            # 10ms
            'order_create': 0.05,          # 50ms
            'order_update': 0.05,          # 50ms
            'order_cancel': 0.05,          # 50ms
            'position_update': 0.01,       # 10ms
            'signal_generation': 0.1,      # 100ms
            'risk_check': 0.01,            # 10ms
            'portfolio_update': 0.1,       # 100ms
            'strategy_update': 0.2,        # 200ms
            'market_data_process': 0.01,   # 10ms
            'tick_processing': 0.005,      # 5ms
            'state_update': 0.01,          # 10ms
            'event_publish': 0.005,        # 5ms
            'default': 0.1                 # 100ms
        },
        'normal': {
            'data_fetch': 0.1,             # 100ms
            'order_create': 0.2,           # 200ms
            'order_update': 0.2,           # 200ms
            'order_cancel': 0.2,           # 200ms
            'position_update': 0.1,        # 100ms
            'signal_generation': 0.5,      # 500ms
            'risk_check': 0.1,             # 100ms
            'portfolio_update': 0.5,       # 500ms
            'strategy_update': 1.0,        # 1s
            'market_data_process': 0.1,    # 100ms
            'tick_processing': 0.05,       # 50ms
            'state_update': 0.1,           # 100ms
            'event_publish': 0.05,         # 50ms
            'default': 0.5                 # 500ms
        },
        'low': {
            'data_fetch': 1.0,             # 1s
            'order_create': 1.0,           # 1s
            'order_update': 1.0,           # 1s
            'order_cancel': 1.0,           # 1s
            'position_update': 0.5,        # 500ms
            'signal_generation': 2.0,      # 2s
            'risk_check': 0.5,             # 500ms
            'portfolio_update': 2.0,       # 2s
            'strategy_update': 5.0,        # 5s
            'market_data_process': 0.5,    # 500ms
            'tick_processing': 0.2,        # 200ms
            'state_update': 0.5,           # 500ms
            'event_publish': 0.2,          # 200ms
            'default': 2.0                 # 2s
        }
    }

    # Get the latency budget
    if priority not in latency_budgets:
        priority = 'normal'

    priority_budgets = latency_budgets[priority]

    return priority_budgets.get(operation_type, priority_budgets['default'])


def is_historical_time(dt: datetime) -> bool:
    """
    Check if a datetime is in the past (historical).

    Args:
        dt: Datetime to check

    Returns:
        True if the datetime is in the past, False otherwise
    """
    return dt < now()


def is_future_time(dt: datetime) -> bool:
    """
    Check if a datetime is in the future.

    Args:
        dt: Datetime to check

    Returns:
        True if the datetime is in the future, False otherwise
    """
    return dt > now()


def time_until(dt: datetime) -> timedelta:
    """
    Calculate the time until a future datetime.

    Args:
        dt: Target datetime

    Returns:
        Timedelta until the target datetime (0 if in the past)
    """
    current_time = now(dt.tzinfo)
    if dt <= current_time:
        return timedelta(0)

    return dt - current_time


def time_since(dt: datetime) -> timedelta:
    """
    Calculate the time since a past datetime.

    Args:
        dt: Target datetime

    Returns:
        Timedelta since the target datetime (0 if in the future)
    """
    current_time = now(dt.tzinfo)
    if dt >= current_time:
        return timedelta(0)

    return current_time - dt


def localize_timestamp(timestamp: float, timezone: pytz.timezone) -> datetime:
    """
    Convert Unix timestamp to localized datetime.

    Args:
        timestamp: Unix timestamp
        timezone: Target timezone

    Returns:
        Localized datetime
    """
    dt = datetime.fromtimestamp(timestamp, UTC)
    return dt.astimezone(timezone)


def get_trading_timestamps(start_date: datetime,
                         end_date: datetime,
                         interval: str,
                         calendar: TradingCalendar,
                         include_extended: bool = False) -> List[datetime]:
    """
    Generate timestamps at regular intervals within trading hours.

    Args:
        start_date: Start date
        end_date: End date
        interval: Interval between timestamps (e.g., '1h', '1d')
        calendar: Trading calendar to use
        include_extended: Whether to include extended trading hours

    Returns:
        List of timestamps within trading hours
    """
    timestamps = []
    delta = parse_period(interval)

    # Get list of trading days
    trading_days = get_trading_days(start_date, end_date, calendar)

    for day in trading_days:
        # Get trading hours for this day
        open_time, close_time = get_market_open_close(day, calendar, include_extended)

        if open_time is None or close_time is None:
            continue

        # Adjust to start_date and end_date
        if open_time < start_date:
            open_time = start_date
        if close_time > end_date:
            close_time = end_date

        # Generate timestamps for this day
        current = open_time
        while current <= close_time:
            timestamps.append(current)
            current += delta

    return timestamps


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds as a human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 0.001:
        return f"{seconds*1000000:.2f} µs"
    elif seconds < 1:
        return f"{seconds*1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        minutes = seconds / 60
        seconds %= 60
        return f"{int(minutes)}m {int(seconds)}s"
    elif seconds < 86400:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{int(hours)}h {int(minutes)}m"
    else:
        days = seconds / 86400
        hours = (seconds % 86400) / 3600
        return f"{int(days)}d {int(hours)}h"


def parse_trading_time(time_str: str,
                     reference_date: datetime = None,
                     timezone: pytz.timezone = None) -> datetime:
    """
    Parse a trading time string into a datetime.

    Args:
        time_str: Time string (e.g., '9:30', '16:00', 'open', 'close')
        reference_date: Reference date to use (default: today)
        timezone: Timezone to use (default: UTC)

    Returns:
        Datetime object
    """
    if reference_date is None:
        reference_date = now(timezone)

    if timezone is None:
        timezone = UTC

    # Ensure reference_date has timezone info
    if reference_date.tzinfo is None:
        reference_date = timezone.localize(reference_date)

    # Convert to the target timezone
    reference_date = reference_date.astimezone(timezone)

    # Base date (midnight)
    base_date = reference_date.replace(hour=0, minute=0, second=0, microsecond=0)

    # Parse special keywords
    if time_str.lower() == 'open':
        # Use US market open (9:30 AM Eastern)
        eastern_date = reference_date.astimezone(EST)
        eastern_date = eastern_date.replace(hour=9, minute=30, second=0, microsecond=0)
        return eastern_date.astimezone(timezone)

    elif time_str.lower() == 'close':
        # Use US market close (4:00 PM Eastern)
        eastern_date = reference_date.astimezone(EST)
        eastern_date = eastern_date.replace(hour=16, minute=0, second=0, microsecond=0)
        return eastern_date.astimezone(timezone)

    # Parse time formats
    formats = [
        '%H:%M',        # 14:30
        '%H:%M:%S',     # 14:30:00
        '%I:%M %p',     # 2:30 PM
        '%I:%M:%S %p'   # 2:30:00 PM
    ]

    for fmt in formats:
        try:
            # Parse time
            parsed_time = datetime.strptime(time_str, fmt).time()

            # Combine with reference date
            result = base_date.replace(
                hour=parsed_time.hour,
                minute=parsed_time.minute,
                second=parsed_time.second,
                microsecond=0
            )

            return result

        except ValueError:
            continue

    raise ValueError(f"Could not parse trading time: {time_str}")


def market_time_to_utc(market_time: datetime, market: str = 'us') -> datetime:
    """
    Convert a market local time to UTC.

    Args:
        market_time: Datetime in market local timezone
        market: Market code ('us', 'eu', 'asia')

    Returns:
        Datetime in UTC
    """
    market = market.lower()

    if market == 'us':
        timezone = EST
    elif market == 'eu':
        timezone = GMT
    elif market == 'asia' or market == 'jp':
        timezone = JST
    else:
        raise ValueError(f"Unknown market: {market}")

    # If the time doesn't have timezone info, assume it's in the market timezone
    if market_time.tzinfo is None:
        market_time = timezone.localize(market_time)

    # Convert to UTC
    return market_time.astimezone(UTC)


def utc_to_market_time(utc_time: datetime, market: str = 'us') -> datetime:
    """
    Convert a UTC time to market local time.

    Args:
        utc_time: Datetime in UTC
        market: Market code ('us', 'eu', 'asia')

    Returns:
        Datetime in market local timezone
    """
    market = market.lower()

    if market == 'us':
        timezone = EST
    elif market == 'eu':
        timezone = GMT
    elif market == 'asia' or market == 'jp':
        timezone = JST
    else:
        raise ValueError(f"Unknown market: {market}")

    # If the time doesn't have timezone info, assume it's UTC
    if utc_time.tzinfo is None:
        utc_time = UTC.localize(utc_time)

    # Convert to market timezone
    return utc_time.astimezone(timezone)


# Initialize global performance timer
_global_timer = PerformanceTimer.get_instance("global")

def start_timing(label: str) -> None:
    """
    Start timing a section using the global timer.

    Args:
        label: Label for the section
    """
    _global_timer.start(label)

def stop_timing(label: str = None) -> float:
    """
    Stop timing a section using the global timer.

    Args:
        label: Label for the section (if None, use last started timer)

    Returns:
        Elapsed time in seconds
    """
    return _global_timer.stop(label)

def get_timing_stats(label: str = None) -> Dict[str, Any]:
    """
    Get timing statistics from the global timer.

    Args:
        label: Label to get statistics for (if None, get all)

    Returns:
        Dictionary with timing statistics
    """
    return _global_timer.get_stats(label)

def timing_summary() -> str:
    """
    Get a summary of all timing statistics from the global timer.

    Returns:
        String summary
    """
    return _global_timer.summary()

def reset_timing(label: str = None) -> None:
    """
    Reset timing statistics in the global timer.

    Args:
        label: Label to reset (if None, reset all)
    """
    _global_timer.reset(label)


# Context manager for timing
class TimingContext:
    """Context manager for timing code execution with the global timer"""

    def __init__(self, label: str):
        """
        Initialize timing context.

        Args:
            label: Label for the timing section
        """
        self.label = label

    def __enter__(self):
        """Start timing when entering context"""
        start_timing(self.label)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing when exiting context"""
        stop_timing(self.label)

 Standard timezone constants
UTC = pytz.UTC
EST = pytz.timezone('US/Eastern')  # US Stock Market
JST = pytz.timezone('Asia/Tokyo')  # Japan Stock Market
GMT = pytz.timezone('Europe/London')  # European Markets


class TimeResolution(Enum):
    """Time resolution enum for different system operations"""
    MILLISECOND = 1
    SECOND = 2
    MINUTE = 3
    HOUR = 4
    DAY = 5
    WEEK = 6
    MONTH = 7


class TradingCalendar(Enum):
    """Standard trading calendar types"""
    US_EQUITIES = "us_equities"  # US stock market
    US_FUTURES = "us_futures"    # US futures market
    FOREX = "forex"              # 24/5 forex market
    CRYPTO = "crypto"            # 24/7 crypto market
    JAPAN = "japan"              # Japan stock market
    EUROPE = "europe"            # European markets
    CUSTOM = "custom"            # Custom calendar


# Trading hours for different markets (24-hour format)
TRADING_HOURS = {
    TradingCalendar.US_EQUITIES: {
        'timezone': EST,
        'regular_hours': {'start': time(9, 30), 'end': time(16, 0)},
        'extended_hours': {'start': time(4, 0), 'end': time(20, 0)},
        'trading_days': [0, 1, 2, 3, 4],  # Monday to Friday
    },
    TradingCalendar.US_FUTURES: {
        'timezone': EST,
        'regular_hours': {'start': time(8, 30), 'end': time(15, 0)},
        'extended_hours': {'start': time(17, 0), 'end': time(16, 0)},  # Overnight session
        'trading_days': [0, 1, 2, 3, 4],  # Monday to Friday
    },
    TradingCalendar.FOREX: {
        'timezone': UTC,
        'regular_hours': {'start': time(0, 0), 'end': time(0, 0)},  # 24 hours
        'trading_days': [0, 1, 2, 3, 4, 6],  # Monday to Friday, Sunday evening
    },
    TradingCalendar.CRYPTO: {
        'timezone': UTC,
        'regular_hours': {'start': time(0, 0), 'end': time(0, 0)},  # 24 hours
        'trading_days': [0, 1, 2, 3, 4, 5, 6],  # All days
    },
    TradingCalendar.JAPAN: {
        'timezone': JST,
        'regular_hours': {'start': time(9, 0), 'end': time(15, 0)},
        'trading_days': [0, 1, 2, 3, 4],  # Monday to Friday
    },
    TradingCalendar.EUROPE: {
        'timezone': GMT,
        'regular_hours': {'start': time(8, 0), 'end': time(16, 30)},
        'trading_days': [0, 1, 2, 3, 4],  # Monday to Friday
    },
}


def now(timezone: Optional[pytz.timezone] = None) -> datetime:
    """
    Get current datetime with specified timezone (UTC by default).

    Args:
        timezone: Timezone to use (default: UTC)

    Returns:
        Current datetime with timezone information
    """
    current_time = datetime.now(UTC)
    if timezone:
        current_time = current_time.astimezone(timezone)
    return current_time


def to_timestamp(dt: Union[datetime, str]) -> float:
    """
    Convert datetime or string to Unix timestamp (seconds since epoch).

    Args:
        dt: Datetime object or ISO format string

    Returns:
        Unix timestamp
    """
    if isinstance(dt, str):
        dt = parse_datetime(dt)

    # Ensure we have timezone information
    if dt.tzinfo is None:
        dt = UTC.localize(dt)

    return dt.timestamp()


def from_timestamp(timestamp: float, timezone: Optional[pytz.timezone] = None) -> datetime:
    """
    Convert Unix timestamp to datetime with timezone.

    Args:
        timestamp: Unix timestamp (seconds since epoch)
        timezone: Timezone to use (default: UTC)

    Returns:
        Datetime object
    """
    dt = datetime.fromtimestamp(timestamp, UTC)
    if timezone:
        dt = dt.astimezone(timezone)
    return dt


def parse_datetime(dt_str: str, timezone: Optional[pytz.timezone] = None) -> datetime:
    """
    Parse a datetime string to a datetime object.
    Handles ISO format and common trading platform formats.

    Args:
        dt_str: Datetime string
        timezone: Timezone to use if string doesn't specify one

    Returns:
        Datetime object with timezone information
    """
    try:
        # Try ISO format first
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except ValueError:
        # Try different formats
        formats = [
            '%Y-%m-%d %H:%M:%S',  # 2023-01-01 14:30:00
            '%Y-%m-%d %H:%M',     # 2023-01-01 14:30
            '%Y-%m-%d',           # 2023-01-01
            '%m/%d/%Y %H:%M:%S',  # 01/01/2023 14:30:00
            '%m/%d/%Y %H:%M',     # 01/01/2023 14:30
            '%m/%d/%Y',           # 01/01/2023
        ]

        dt = None
        for fmt in formats:
            try:
                dt = datetime.strptime(dt_str, fmt)
                break
            except ValueError:
                continue

        if dt is None:
            raise ValueError(f"Could not parse datetime string: {dt_str}")

    # Add timezone info if needed
    if dt.tzinfo is None:
        if timezone:
            dt = timezone.localize(dt)
        else:
            dt = UTC.localize(dt)

    return dt


def format_datetime(dt: datetime, fmt: str = 'iso') -> str:
    """
    Format a datetime object as a string.

    Args:
        dt: Datetime object
        fmt: Format to use ('iso', 'date', 'time', 'full', or strftime format string)

    Returns:
        Formatted datetime string
    """
    if fmt == 'iso':
        return dt.isoformat()
    elif fmt == 'date':
        return dt.strftime('%Y-%m-%d')
    elif fmt == 'time':
        return dt.strftime('%H:%M:%S')
    elif fmt == 'full':
        return dt.strftime('%Y-%m-%d %H:%M:%S %Z')
    else:
        return dt.strftime(fmt)


def convert_timezone(dt: datetime, timezone: pytz.timezone) -> datetime:
    """
    Convert datetime to a different timezone.

    Args:
        dt: Datetime object
        timezone: Target timezone

    Returns:
        Datetime object in the target timezone
    """
    if dt.tzinfo is None:
        dt = UTC.localize(dt)

    return dt.astimezone(timezone)


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    """
    Convert a timeframe string to a timedelta.

    Args:
        timeframe: Timeframe string (e.g., '1m', '1h', '1d')

    Returns:
        Timedelta representing the timeframe
    """
    unit = timeframe[-1]
    value = int(timeframe[:-1])

    if unit == 'm':
        return timedelta(minutes=value)
    elif unit == 'h':
        return timedelta(hours=value)
    elif unit == 'd':
        return timedelta(days=value)
    elif unit == 'w':
        return timedelta(weeks=value)
    elif unit == 'M':
        # This is approximate (assuming 30 days per month)
        return timedelta(days=value * 30)
    else:
        raise ValueError(f"Unknown timeframe unit: {unit}")


def ceil_dt(dt: datetime, resolution: TimeResolution) -> datetime:
    """
    Ceil a datetime to the given resolution.

    Args:
        dt: Datetime to ceil
        resolution: Time resolution to ceil to

    Returns:
        Ceiled datetime
    """
    if resolution == TimeResolution.MILLISECOND:
        return dt  # No change needed

    if resolution == TimeResolution.SECOND:
        return dt.replace(microsecond=0) + timedelta(seconds=1) \
            if dt.microsecond > 0 else dt

    if resolution == TimeResolution.MINUTE:
        dt = dt.replace(second=0, microsecond=0) + timedelta(minutes=1) \
            if dt.second > 0 or dt.microsecond > 0 else dt
        return dt

    if resolution == TimeResolution.HOUR:
        dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1) \
            if dt.minute > 0 or dt.second > 0 or dt.microsecond > 0 else dt
        return dt

    if resolution == TimeResolution.DAY:
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1) \
            if dt.hour > 0 or dt.minute > 0 or dt.second > 0 or dt.microsecond > 0 else dt
        return dt

    if resolution == TimeResolution.WEEK:
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        days_to_add = 7 - dt.weekday()
        if days_to_add > 0:
            dt = dt + timedelta(days=days_to_add)
        return dt

    if resolution == TimeResolution.MONTH:
        if dt.day == 1 and dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
            return dt

        year = dt.year
        month = dt.month + 1
        if month > 12:
            month = 1
            year += 1

        return datetime(year, month, 1, tzinfo=dt.tzinfo)

    raise ValueError(f"Unknown resolution: {resolution}")


def floor_dt(dt: datetime, resolution: TimeResolution) -> datetime:
    """
    Floor a datetime to the given resolution.

    Args:
        dt: Datetime to floor
        resolution: Time resolution to floor to

    Returns:
        Floored datetime
    """
    if resolution == TimeResolution.MILLISECOND:
        # Floor to millisecond
        return dt.replace(microsecond=(dt.microsecond // 1000) * 1000)

    if resolution == TimeResolution.SECOND:
        return dt.replace(microsecond=0)

    if resolution == TimeResolution.MINUTE:
        return dt.replace(second=0, microsecond=0)

    if resolution == TimeResolution.HOUR:
        return dt.replace(minute=0, second=0, microsecond=0)

    if resolution == TimeResolution.DAY:
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    if resolution == TimeResolution.WEEK:
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        days_to_subtract = dt.weekday()
        if days_to_subtract > 0:
            dt = dt - timedelta(days=days_to_subtract)
        return dt

    if resolution == TimeResolution.MONTH:
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    raise ValueError(f"Unknown resolution: {resolution}")


def round_dt(dt: datetime, resolution: TimeResolution) -> datetime:
    """
    Round a datetime to the given resolution.

    Args:
        dt: Datetime to round
        resolution: Time resolution to round to

    Returns:
        Rounded datetime
    """
    floor = floor_dt(dt, resolution)
    ceil = ceil_dt(dt, resolution)

    # Calculate the midpoint
    diff_to_floor = (dt - floor).total_seconds()
    diff_to_ceil = (ceil - dt).total_seconds()

    if diff_to_floor <= diff_to_ceil:
        return floor
    else:
        return ceil


def is_trading_hours(dt: datetime,
                     calendar: TradingCalendar,
                     include_extended: bool = False) -> bool:
    """
    Check if the given datetime is during trading hours.

    Args:
        dt: Datetime to check
        calendar: Trading calendar to use
        include_extended: Whether to include extended trading hours

    Returns:
        True if the datetime is during trading hours, False otherwise
    """
    if calendar not in TRADING_HOURS:
        raise ValueError(f"Unknown trading calendar: {calendar}")

    calendar_info = TRADING_HOURS[calendar]

    # Convert to calendar timezone
    dt = convert_timezone(dt, calendar_info['timezone'])

    # Check if it's a trading day
    if dt.weekday() not in calendar_info['trading_days']:
        return False

    # For 24/7 markets like crypto
    if (calendar == TradingCalendar.CRYPTO or
        (calendar == TradingCalendar.FOREX and dt.weekday() != 5)):  # Forex closes on Saturday
        return True

    # Check trading hours
    current_time = dt.time()

    if include_extended and 'extended_hours' in calendar_info:
        start = calendar_info['extended_hours']['start']
        end = calendar_info['extended_hours']['end']
    else:
        start = calendar_info['regular_hours']['start']
        end = calendar_info['regular_hours']['end']

    # Handle overnight sessions
    if start > end:
        return current_time >= start or current_time < end
    else:
        return start <= current_time < end


def get_next_trading_time(dt: datetime,
                         calendar: TradingCalendar,
                         include_extended: bool = False) -> datetime:
    """
    Get the next trading time from the given datetime.
    If the given time is already in trading hours, it is returned unchanged.

    Args:
        dt: Starting datetime
        calendar: Trading calendar to use
        include_extended: Whether to include extended trading hours

    Returns:
        Next trading datetime
    """
    if is_trading_hours(dt, calendar, include_extended):
        return dt

    calendar_info = TRADING_HOURS[calendar]
    dt = convert_timezone(dt, calendar_info['timezone'])

    # For 24/7 markets, the next time would be immediate
    if calendar == TradingCalendar.CRYPTO:
        return dt

    # For forex, next time is Monday if it's Saturday
    if calendar == TradingCalendar.FOREX and dt.weekday() == 5:
        # Move to Monday
        days_to_add = 2
        next_dt = dt + timedelta(days=days_to_add)
        return next_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    # Get trading hours
    if include_extended and 'extended_hours' in calendar_info:
        trading_start = calendar_info['extended_hours']['start']
        trading_end = calendar_info['extended_hours']['end']
    else:
        trading_start = calendar_info['regular_hours']['start']
        trading_end = calendar_info['regular_hours']['end']

    current_time = dt.time()
    current_weekday = dt.weekday()

    # Check if we're before trading hours on a trading day
    if current_weekday in calendar_info['trading_days'] and current_time < trading_start:
        return dt.replace(
            hour=trading_start.hour,
            minute=trading_start.minute,
            second=0,
            microsecond=0
        )

    # Find the next trading day
    days_to_add = 1
    while True:
        next_weekday = (current_weekday + days_to_add) % 7
        if next_weekday in calendar_info['trading_days']:
            break
        days_to_add += 1

    # Create datetime for next trading day at start time
    next_dt = dt + timedelta(days=days_to_add)
    return next_dt.replace(
        hour=trading_start.hour,
        minute=trading_start.minute,
        second=0,
        microsecond=0
    )


def get_market_open_close(dt: datetime,
                         calendar: TradingCalendar,
                         include_extended: bool = False) -> Tuple[datetime, datetime]:
    """
    Get the market open and close times for a given date.

    Args:
        dt: Date to check (time portion is ignored)
        calendar: Trading calendar to use
        include_extended: Whether to include extended trading hours

    Returns:
        Tuple of (market open time, market close time)
    """
    calendar_info = TRADING_HOURS[calendar]
    timezone = calendar_info['timezone']

    # Convert to calendar timezone and set to midnight
    dt = convert_timezone(dt, timezone).replace(hour=0, minute=0, second=0, microsecond=0)

    # Check if it's a trading day
    if dt.weekday() not in calendar_info['trading_days']:
        return (None, None)

    # For 24/7 markets
    if calendar == TradingCalendar.CRYPTO:
        return (dt, dt + timedelta(days=1))

    # For 24/5 markets like forex
    if calendar == TradingCalendar.FOREX and dt.weekday() != 5:  # Not Saturday
        return (dt, dt + timedelta(days=1))

    # Get the trading hours
    if include_extended and 'extended_hours' in calendar_info:
        start = calendar_info['extended_hours']['start']
        end = calendar_info['extended_hours']['end']
    else:
        start = calendar_info['regular_hours']['start']
        end = calendar_info['regular_hours']['end']

    # Handle overnight sessions
    if start > end:
        open_time = dt.replace(hour=start.hour, minute=start.minute)
        close_time = (dt + timedelta(days=1)).replace(hour=end.hour, minute=end.minute)
    else:
        open_time = dt.replace(hour=start.hour, minute=start.minute)
        close_time = dt.replace(hour=end.hour, minute=end.minute)

    return (open_time, close_time)


def get_trading_days(start_date: datetime,
                    end_date: datetime,
                    calendar: TradingCalendar) -> List[datetime]:
    """
    Get all trading days between start_date and end_date (inclusive).

    Args:
        start_date: Start date
        end_date: End date
        calendar: Trading calendar to use

    Returns:
        List of trading days
    """
    if calendar not in TRADING_HOURS:
        raise ValueError(f"Unknown trading calendar: {calendar}")

    calendar_info = TRADING_HOURS[calendar]

    # Convert dates to calendar timezone and set to midnight
    timezone = calendar_info['timezone']
    start = convert_timezone(start_date, timezone).replace(hour=0, minute=0, second=0, microsecond=0)
    end = convert_timezone(end_date, timezone).replace(hour=0, minute=0, second=0, microsecond=0)

    trading_days = []
    current = start

    while current <= end:
        if current.weekday() in calendar_info['trading_days']:
            trading_days.append(current)

        current += timedelta(days=1)

    return trading_days


def parse_period(period: str) -> timedelta:
    """
    Parse a period string into a timedelta.

    Args:
        period: Period string (e.g., '1d', '2w', '3m', '4h')

    Returns:
        Timedelta object
    """
    if not period:
        raise ValueError("Period cannot be empty")

    # Split into value and unit
    value = int(''.join(filter(str.isdigit, period)))
    unit = ''.join(filter(str.isalpha, period)).lower()

    if unit == 'm' or unit == 'min':
        return timedelta(minutes=value)
    elif unit == 'h' or unit == 'hour':
        return timedelta(hours=value)
    elif unit == 'd' or unit == 'day':
        return timedelta(days=value)
    elif unit == 'w' or unit == 'week':
        return timedelta(weeks=value)
    elif unit == 'mo' or unit == 'month':
        # Approximate months as 30 days
        return timedelta(days=value * 30)
    elif unit == 'y' or unit == 'year':
        # Approximate years as 365 days
        return timedelta(days=value * 365)
    else:
        raise ValueError(f"Unknown period unit: {unit}")


def get_interval_timestamp(dt: datetime, interval: str) -> datetime:
    """
    Align a timestamp to the nearest interval boundary.

    Args:
        dt: Datetime to align
        interval: Interval string (e.g., '1m', '1h', '1d')

    Returns:
        Aligned datetime
    """
    if not interval:
        return dt

    unit = interval[-1].lower()
    value = int(interval[:-1])

    if unit == 'm':
        # Minute intervals
        minute = (dt.minute // value) * value
        return dt.replace(minute=minute, second=0, microsecond=0)

    elif unit == 'h':
        # Hour intervals
        hour = (dt.hour // value) * value
        return dt.replace(hour=hour, minute=0, second=0, microsecond=0)

    elif unit == 'd':
        # Day intervals
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    elif unit == 'w':
        # Week intervals - align to Monday
        days_to_subtract = dt.weekday()
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return dt - timedelta(days=days_to_subtract)

    elif unit == 'M':
        # Month intervals - align to first day of month
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    else:
        raise ValueError(f"Unknown interval unit: {unit}")


def generate_time_windows(start_time: datetime,
                        end_time: datetime,
                        window_size: str) -> List[Tuple[datetime, datetime]]:
    """
    Generate time windows between start_time and end_time.

    Args:
        start_time: Start time
        end_time: End time
        window_size: Window size as string (e.g., '1h', '1d')

    Returns:
        List of (window_start, window_end) tuples
    """
    if start_time > end_time:
        raise ValueError("Start time must be before end time")

    delta = parse_period(window_size)
    windows = []

    # Align start_time to window boundary
    current = get_interval_timestamp(start_time, window_size)

    while current < end_time:
        window_end = current + delta
        windows.append((current, window_end))
        current = window_end

    return windows


def is_holiday(dt: datetime, calendar: TradingCalendar) -> bool:
    """
    Check if a given date is a market holiday.

    This is a placeholder function - in a real implementation,
    you would integrate with a more comprehensive holiday calendar.

    Args:
        dt: Date to check
        calendar: Trading calendar to use

    Returns:
        True if the date is a holiday, False otherwise
    """
    # This would typically be implemented with a more comprehensive
    # holiday calendar, possibly from a library like pandas_market_calendars
    # For now, we'll just use a simple check for common US holidays

    if calendar in [TradingCalendar.US_EQUITIES, TradingCalendar.US_FUTURES]:
        # Convert to US Eastern time
        dt_eastern = convert_timezone(dt, EST)

        # New Year's Day
        if dt_eastern.month == 1 and dt_eastern.day == 1:
            return True

        # Martin Luther King Jr. Day (3rd Monday in January)
        if dt_eastern.month == 1 and dt_eastern.weekday() == 0 and 15 <= dt_eastern.day <= 21:
            return True

        # Presidents Day (3rd Monday in February)
        if dt_eastern.month == 2 and dt_eastern.weekday() == 0 and 15 <= dt_eastern.day <= 21:
            return True

        # Good Friday (would need to calculate Easter, complex for a simple example)

        # Memorial Day (last Monday in May)
        if dt_eastern.month == 5 and dt_eastern.weekday() == 0 and dt_eastern.day > 24:
            return True

        # Independence Day
        if dt_eastern.month == 7 and dt_eastern.day == 4:
            return True

        # Labor Day (1st Monday in September)
        if dt_eastern.month == 9 and dt_eastern.weekday() == 0 and dt_eastern.day <= 7:
            return True

        # Thanksgiving Day (4th Thursday in November)
        if dt_eastern.month == 11 and dt_eastern.weekday() == 3 and 22 <= dt_eastern.day <= 28:
            return True

        # Christmas
        if dt_eastern.month == 12 and dt_eastern.day == 25:
            return True

    # For other markets, you would implement similar logic

    return False


def get_business_days_between(start_date: datetime,
                             end_date: datetime,
                             calendar: TradingCalendar) -> int:
    """
    Get the number of business days between start_date and end_date.

    Args:
        start_date: Start date
        end_date: End date
        calendar: Trading calendar to use

    Returns:
        Number of business days
    """
    trading_days = get_trading_days(start_date, end_date, calendar)
    return len(trading_days)


def get_execution_time_metrics(start_time: datetime,
                            end_time: datetime,
                            calendar: TradingCalendar) -> Dict[str, Any]:
    """
    Calculate execution time metrics (total time, trading time, etc.)

    Args:
        start_time: Start time
        end_time: End time
        calendar: Trading calendar to use

    Returns:
        Dictionary with timing metrics
    """
    total_seconds = (end_time - start_time).total_seconds()

    # Get trading days
    trading_days = get_trading_days(start_time, end_time, calendar)

    # Calculate business days
    business_days = len(trading_days)

    # Calculate trading hours (approximation)
    calendar_info = TRADING_HOURS[calendar]

    if 'regular_hours' in calendar_info:
        start_hour = calendar_info['regular_hours']['start']
        end_hour = calendar_info['regular_hours']['end']

        # Handle overnight sessions
        if start_hour > end_hour:
            hours_per_day = 24 - start_hour.hour - start_hour.minute/60 + end_hour.hour + end_hour.minute/60
        else:
            hours_per_day = end_hour.hour + end_hour.minute/60 - start_hour.hour - start_hour.minute/60
    else:
        # 24 hour markets
        hours_per_day = 24

    trading_hours = business_days * hours_per_day

    return {
        'total_time_seconds': total_seconds,
        'total_time_hours': total_seconds / 3600,
        'total_time_days': total_seconds / 86400,
        'business_days': business_days,
        'trading_hours': trading_hours,
        'trading_time_percentage': (trading_hours * 3600 / total_seconds) * 100 if total_seconds > 0 else 0
    }


# Timeframe conversion utilities
def timeframe_to_seconds(timeframe: str) -> int:
    """
    Convert a timeframe string to seconds.

    Args:
        timeframe: Timeframe string (e.g., '1m', '1h', '1d')

    Returns:
        Number of seconds
    """
    delta = timeframe_to_timedelta(timeframe)
    return int(delta.total_seconds())


def seconds_to_timeframe(seconds: int) -> str:
    """
    Convert seconds to a timeframe string using the most appropriate unit.

    Args:
        seconds: Number of seconds

    Returns:
        Timeframe string
    """
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}m"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{hours}h"
    elif seconds < 604800:
        days = seconds // 86400
        return f"{days}d"
    elif seconds < 2592000:  # ~30 days
        weeks = seconds // 604800
        return f"{weeks}w"
    else:
        months = seconds // 2592000
        return f"{months}M"


#