import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import logging
from scipy import stats
import json


class RegimePropertiesCalculator:
    """
    Calculates and tracks statistical properties of different market regimes
    to enable strategy adaptation and provide insights for decision making.
    """
    
    def __init__(
        self,
        memory_decay: float = 0.95,
        min_samples: int = 50,
        property_history_size: int = 10,
        regime_transition_matrix_size: int = 10
    ):
        """
        Initialize the regime properties calculator.
        
        Args:
            memory_decay: Factor for exponential decay of old samples (default: 0.95)
            min_samples: Minimum samples needed for reliable statistics (default: 50)
            property_history_size: Number of snapshots to keep for each regime (default: 10)
            regime_transition_matrix_size: Size of the transition history to maintain (default: 10)
        """
        self.memory_decay = memory_decay
        self.min_samples = min_samples
        self.property_history_size = property_history_size
        self.regime_transition_matrix_size = regime_transition_matrix_size
        
        # Core data structures
        self.regime_stats = defaultdict(dict)
        self.regime_properties_history = defaultdict(list)
        self.regime_samples = defaultdict(int)
        self.feature_stats = defaultdict(lambda: defaultdict(dict))
        self.transitions = []
        self.transition_matrix = None
        
        # Property categories
        self.return_properties = [
            'mean_return', 'return_volatility', 'sharpe_ratio', 'downside_deviation',
            'max_drawdown', 'win_rate', 'profit_factor'
        ]
        
        self.volatility_properties = [
            'volatility_level', 'volatility_of_volatility', 'garch_persistence',
            'range_intensity', 'gap_frequency'
        ]
        
        self.correlation_properties = [
            'internal_correlation', 'external_correlation', 'lead_lag',
            'regime_consistency'
        ]
        
        self.temporal_properties = [
            'avg_duration', 'expected_remaining_duration', 'time_of_day_bias',
            'day_of_week_bias', 'seasonality_score'
        ]
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        
    def update(self, 
              df: pd.DataFrame, 
              regime_label: int, 
              timestamp: Optional[pd.Timestamp] = None) -> Dict:
        """
        Update regime properties with new data.
        
        Args:
            df: DataFrame containing market data (OHLCV and features)
            regime_label: The current regime label (-1 for noise/unknown)
            timestamp: Optional timestamp for this update
            
        Returns:
            Dictionary of updated regime properties
        """
        if regime_label == -1:
            # Skip noise points for property calculation
            return {}
            
        # Get current timestamp if not provided
        if timestamp is None and isinstance(df.index, pd.DatetimeIndex):
            timestamp = df.index[-1]
        elif timestamp is None:
            timestamp = pd.Timestamp.now()
            
        # Ensure required columns exist (use reasonable defaults if not)
        df = self._prepare_data(df)
        
        # Update sample counter with decay for this regime
        self.regime_samples[regime_label] = (
            self.memory_decay * self.regime_samples.get(regime_label, 0) + 1
        )
        
        # Skip full calculation if we don't have enough samples
        if self.regime_samples[regime_label] < 10:
            # Just store basic info 
            self.regime_stats[regime_label]['last_seen'] = timestamp
            self.regime_stats[regime_label]['sample_count'] = self.regime_samples[regime_label]
            return self.regime_stats[regime_label]
        
        # Calculate comprehensive properties
        properties = {}
        
        # 1. Return-based properties
        properties.update(self._calculate_return_properties(df, regime_label))
        
        # 2. Volatility properties
        properties.update(self._calculate_volatility_properties(df, regime_label))
        
        # 3. Correlation properties
        properties.update(self._calculate_correlation_properties(df, regime_label))
        
        # 4. Temporal properties
        properties.update(self._calculate_temporal_properties(df, regime_label, timestamp))
        
        # 5. Feature-specific statistics
        self._update_feature_stats(df, regime_label)
        
        # Core metadata
        properties['last_seen'] = timestamp
        properties['sample_count'] = self.regime_samples[regime_label]
        properties['confidence'] = min(1.0, self.regime_samples[regime_label] / self.min_samples)
        
        # Update regime properties
        self.regime_stats[regime_label] = properties
        
        # Store history (keeping limited snapshots per regime)
        self.regime_properties_history[regime_label].append({
            'timestamp': timestamp,
            'properties': properties.copy()
        })
        
        # Keep only the most recent snapshots
        if len(self.regime_properties_history[regime_label]) > self.property_history_size:
            self.regime_properties_history[regime_label].pop(0)
        
        return properties
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required columns exist with reasonable defaults."""
        df = df.copy()
        
        # Make sure basic price data exists
        if 'close' not in df.columns and 'price' in df.columns:
            df['close'] = df['price']
            
        # Ensure we have returns
        if 'returns' not in df.columns and 'close' in df.columns:
            df['returns'] = df['close'].pct_change().fillna(0)
            
        # Create volatility if not exists
        if 'volatility' not in df.columns and 'returns' in df.columns:
            df['volatility'] = df['returns'].rolling(21, min_periods=1).std().fillna(0)
            
        # Add basic time features if using DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            if 'hour' not in df.columns:
                df['hour'] = df.index.hour
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = df.index.dayofweek
                
        return df
    
    def _calculate_return_properties(self, df: pd.DataFrame, regime_label: int) -> Dict:
        """Calculate return-based properties for the regime."""
        properties = {}
        
        if 'returns' not in df.columns:
            return properties
            
        returns = df['returns'].values
        
        # Basic statistics with memory decay
        old_mean = self.regime_stats.get(regime_label, {}).get('mean_return', 0)
        old_variance = self.regime_stats.get(regime_label, {}).get('return_variance', 0)
        
        # Update mean and variance with exponential decay
        if np.isfinite(returns).any():
            mean_return = self.memory_decay * old_mean + (1 - self.memory_decay) * np.nanmean(returns)
            
            # Update variance using Welford's online algorithm with decay
            delta = np.nanmean(returns) - old_mean
            variance = (self.memory_decay * old_variance + 
                       (1 - self.memory_decay) * np.nanvar(returns, ddof=1) +
                       self.memory_decay * (1 - self.memory_decay) * delta**2)
            
            properties['mean_return'] = mean_return
            properties['return_variance'] = variance
            properties['return_volatility'] = np.sqrt(variance) if variance > 0 else 0
            
            # Sharpe ratio (if sufficient data)
            if properties['return_volatility'] > 0:
                properties['sharpe_ratio'] = mean_return / properties['return_volatility']
            else:
                properties['sharpe_ratio'] = 0
                
            # Downside deviation
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                properties['downside_deviation'] = np.sqrt(np.mean(negative_returns**2))
            else:
                properties['downside_deviation'] = 0
                
            # Win rate
            properties['win_rate'] = np.mean(returns > 0)
            
            # Profit factor
            gain = np.sum(returns[returns > 0])
            loss = np.abs(np.sum(returns[returns < 0]))
            properties['profit_factor'] = gain / loss if loss > 0 else 1.0
            
            # Maximum drawdown (simplified calculation)
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative / running_max) - 1
            properties['max_drawdown'] = np.min(drawdown)
        
        return properties
    
    def _calculate_volatility_properties(self, df: pd.DataFrame, regime_label: int) -> Dict:
        """Calculate volatility-based properties for the regime."""
        properties = {}
        
        # Volatility level
        if 'volatility' in df.columns:
            properties['volatility_level'] = df['volatility'].mean()
            
            # Volatility of volatility (variability of volatility)
            properties['volatility_of_volatility'] = df['volatility'].std() / df['volatility'].mean()
            
            # GARCH persistence approximation (auto-correlation of volatility)
            if len(df) > 10:
                vol_autocorr = df['volatility'].autocorr(lag=1)
                properties['garch_persistence'] = vol_autocorr if np.isfinite(vol_autocorr) else 0
        
        # Price range intensity
        if all(col in df.columns for col in ['high', 'low', 'close']):
            # Normalized daily range
            df['norm_range'] = (df['high'] - df['low']) / df['close']
            properties['range_intensity'] = df['norm_range'].mean()
            
            # Gap frequency - significant price gaps between periods
            if 'open' in df.columns and len(df) > 1:
                df['gap'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)).abs()
                gap_threshold = df['volatility'].mean() * 2
                properties['gap_frequency'] = (df['gap'] > gap_threshold).mean()
                
        return properties
    
    def _calculate_correlation_properties(self, df: pd.DataFrame, regime_label: int) -> Dict:
        """Calculate correlation-based properties for the regime."""
        properties = {}
        
        # Only calculate if we have enough data
        if len(df.columns) < 3 or len(df) < 10:
            return properties
        
        # Internal correlation - correlation between numeric features
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            # Exclude self-correlations (diagonal)
            corr_values = corr_matrix.values
            mask = ~np.eye(corr_values.shape[0], dtype=bool)
            properties['internal_correlation'] = np.nanmean(corr_values[mask])
        
        # Regime consistency - how stable are the properties
        if regime_label in self.regime_properties_history and len(self.regime_properties_history[regime_label]) > 1:
            # Compare current properties with previous snapshot
            prev_props = self.regime_properties_history[regime_label][-2]['properties']
            curr_props = properties
            
            # Get common numeric properties
            common_keys = set(prev_props.keys()) & set(curr_props.keys()) & set(self.return_properties + self.volatility_properties)
            
            if common_keys:
                diffs = [abs(curr_props.get(k, 0) - prev_props.get(k, 0)) / (abs(prev_props.get(k, 1e-8)) + 1e-8) 
                         for k in common_keys if isinstance(prev_props.get(k), (int, float))]
                
                if diffs:
                    # Convert to stability score (1 = very stable, 0 = unstable)
                    mean_rel_diff = np.mean(diffs)
                    properties['regime_consistency'] = np.exp(-mean_rel_diff)
        
        return properties
    
    def _calculate_temporal_properties(self, df: pd.DataFrame, regime_label: int, 
                                      timestamp: pd.Timestamp) -> Dict:
        """Calculate temporal properties for the regime."""
        properties = {}
        
        # Update the last seen timestamp
        last_seen = self.regime_stats.get(regime_label, {}).get('last_seen')
        
        # Average duration calculation
        if regime_label in self.regime_properties_history and len(self.regime_properties_history[regime_label]) > 1:
            # Calculate durations between appearances
            history = self.regime_properties_history[regime_label]
            timestamps = [h['timestamp'] for h in history if 'timestamp' in h]
            
            if len(timestamps) > 1:
                # Calculate durations between consecutive appearances
                durations = [(t2 - t1).total_seconds() for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
                if durations:
                    properties['avg_duration'] = np.mean(durations)
        
        # Time of day bias
        if 'hour' in df.columns:
            # Get mode hour
            hour_counts = df['hour'].value_counts()
            if not hour_counts.empty:
                properties['peak_hour'] = hour_counts.idxmax()
                
                # Calculate time of day bias (0 = uniform, 1 = highly concentrated)
                expected_pct = 1.0 / 24
                hour_pcts = hour_counts / hour_counts.sum()
                max_pct = hour_pcts.max()
                properties['time_of_day_bias'] = (max_pct - expected_pct) / (1 - expected_pct)
        
        # Day of week bias
        if 'day_of_week' in df.columns:
            day_counts = df['day_of_week'].value_counts()
            if not day_counts.empty:
                properties['peak_day'] = day_counts.idxmax()
                
                # Calculate day of week bias (0 = uniform, 1 = highly concentrated)
                expected_pct = 1.0 / 7
                day_pcts = day_counts / day_counts.sum()
                max_pct = day_pcts.max()
                properties['day_of_week_bias'] = (max_pct - expected_pct) / (1 - expected_pct)
        
        return properties
    
    def _update_feature_stats(self, df: pd.DataFrame, regime_label: int) -> None:
        """Update feature-specific statistics for the regime."""
        # Calculate feature statistics
        numeric_cols = df.select_dtypes(include=np.number).columns
        
        for feature in numeric_cols:
            if feature not in df.columns:
                continue
                
            values = df[feature].values
            if not np.isfinite(values).any():
                continue
                
            # Get basic stats
            mean = np.nanmean(values)
            std = np.nanstd(values)
            
            # Update feature stats with memory decay
            old_mean = self.feature_stats[regime_label][feature].get('mean', mean)
            old_std = self.feature_stats[regime_label][feature].get('std', std)
            
            self.feature_stats[regime_label][feature]['mean'] = (
                self.memory_decay * old_mean + (1 - self.memory_decay) * mean
            )
            
            self.feature_stats[regime_label][feature]['std'] = (
                self.memory_decay * old_std + (1 - self.memory_decay) * std
            )
            
            # Additional feature stats (percentiles, skew, kurtosis)
            if len(values) >= 20:
                self.feature_stats[regime_label][feature]['skew'] = stats.skew(values)
                self.feature_stats[regime_label][feature]['kurtosis'] = stats.kurtosis(values)
                self.feature_stats[regime_label][feature]['percentiles'] = {
                    '10': np.nanpercentile(values, 10),
                    '25': np.nanpercentile(values, 25),
                    '50': np.nanpercentile(values, 50),
                    '75': np.nanpercentile(values, 75),
                    '90': np.nanpercentile(values, 90)
                }
    
    def record_transition(self, from_regime: int, to_regime: int, 
                        timestamp: Optional[pd.Timestamp] = None) -> None:
        """
        Record a regime transition to build transition probability matrix.
        
        Args:
            from_regime: Source regime label
            to_regime: Destination regime label
            timestamp: When the transition occurred
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()
            
        self.transitions.append({
            'timestamp': timestamp,
            'from_regime': from_regime,
            'to_regime': to_regime
        })
        
        # Maintain limited size transition history
        if len(self.transitions) > self.regime_transition_matrix_size * 10:
            self.transitions = self.transitions[-self.regime_transition_matrix_size * 10:]
            
        # Update transition matrix
        self._update_transition_matrix()
    
    def _update_transition_matrix(self) -> None:
        """Update the regime transition probability matrix."""
        if not self.transitions:
            return
            
        # Get unique regimes
        all_regimes = set()
        for t in self.transitions:
            all_regimes.add(t['from_regime'])
            all_regimes.add(t['to_regime'])
            
        # Skip noise regime (-1) in the matrix
        regimes = sorted([r for r in all_regimes if r >= 0])
        
        if not regimes:
            return
            
        # Initialize matrix
        matrix = np.zeros((len(regimes), len(regimes)))
        
        # Count transitions
        regime_idx = {r: i for i, r in enumerate(regimes)}
        
        for t in self.transitions:
            # Skip transitions involving noise
            if t['from_regime'] < 0 or t['to_regime'] < 0:
                continue
                
            # Count the transition
            from_idx = regime_idx[t['from_regime']]
            to_idx = regime_idx[t['to_regime']]
            matrix[from_idx, to_idx] += 1
            
        # Convert to probabilities (rows sum to 1)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        prob_matrix = matrix / row_sums
        
        self.transition_matrix = {
            'regimes': regimes,
            'matrix': prob_matrix,
            'last_updated': pd.Timestamp.now()
        }
    
    def get_feature_distribution(self, regime_label: int, feature: str) -> Dict:
        """
        Get the distribution statistics for a specific feature in a regime.
        
        Args:
            regime_label: The regime to get stats for
            feature: Feature name
            
        Returns:
            Distribution statistics or empty dict if not available
        """
        if regime_label in self.feature_stats and feature in self.feature_stats[regime_label]:
            return self.feature_stats[regime_label][feature]
        return {}
    
    def get_transition_probability(self, from_regime: int, to_regime: int) -> float:
        """
        Get probability of transitioning from one regime to another.
        
        Args:
            from_regime: Source regime
            to_regime: Destination regime
            
        Returns:
            Transition probability (0-1)
        """
        if self.transition_matrix is None:
            return 0.