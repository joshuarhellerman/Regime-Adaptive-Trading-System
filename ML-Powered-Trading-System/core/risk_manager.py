"""
Risk Manager Module

This module implements comprehensive risk management for the trading system,
providing position sizing, drawdown protection, risk metrics, and exposure management
with regime-awareness specifically optimized for forex markets.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import os
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration for dynamic risk management"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DirectionalBias(Enum):
    """Market direction classification"""
    UPWARD = "up"
    DOWNWARD = "down"
    NEUTRAL = "neutral"


class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskManager:
    """
    Risk Manager for the trading system.

    Handles all aspects of risk management including position sizing, drawdown protection,
    exposure limits, volatility-based adjustments, and correlation risk with regime-awareness.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the risk manager.

        Args:
            config: Configuration dictionary with risk parameters
        """
        self.config = config or {}

        # Load default config if not provided
        if not self.config:
            self._load_default_config()

        # Core risk parameters
        self.max_drawdown = self.config.get('max_drawdown', 0.10)  # Maximum allowed drawdown (10%)
        self.max_daily_loss = self.config.get('max_daily_loss', 0.05)  # Maximum daily loss (5%)
        self.max_position_size = self.config.get('max_position_size', 0.10)  # Maximum single position (10%)
        self.max_total_exposure = self.config.get('max_total_exposure', 0.50)  # Maximum total exposure (50%)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.01)  # Risk per trade (1%)
        self.max_open_positions = self.config.get('max_open_positions', 5)  # Maximum open positions

        # Forex-specific risk parameters
        self.leverage = self.config.get('leverage', 50)  # Default leverage
        self.max_leverage = self.config.get('max_leverage', 100)  # Maximum allowed leverage
        self.leverage_by_pair = self.config.get('leverage_by_pair', {})  # Pair-specific leverage
        self.currency_exposure_limits = self.config.get('currency_exposure_limits', {})  # Per-currency exposure limits

        # Dynamic risk adjustment
        self.use_dynamic_risk = self.config.get('use_dynamic_risk', True)
        self.dynamic_risk_factors = self.config.get('dynamic_risk_factors', {
            'market_volatility': 1.0,
            'system_performance': 1.0,
            'news_impact': 1.0
        })

        # Volatility-based sizing
        self.use_volatility_sizing = self.config.get('use_volatility_sizing', True)
        self.volatility_lookback = self.config.get('volatility_lookback', 20)  # Days
        self.volatility_risk_factor = self.config.get('volatility_risk_factor', 1.0)

        # Correlation risk
        self.use_correlation_risk = self.config.get('use_correlation_risk', True)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.correlation_lookback = self.config.get('correlation_lookback', 30)  # Days

        # Regime-based risk
        self.use_regime_based_risk = self.config.get('use_regime_based_risk', True)
        self.regime_risk_adjustments = self.config.get('regime_risk_adjustments', {
            # Low volatility adjustments
            'low': {
                'position_size_multiplier': 1.2,  # Increase position size
                'max_positions_adjustment': 1,  # Allow more positions
                'risk_per_trade_multiplier': 1.2  # Take more risk per trade
            },
            # Medium volatility adjustments (neutral)
            'medium': {
                'position_size_multiplier': 1.0,
                'max_positions_adjustment': 0,
                'risk_per_trade_multiplier': 1.0
            },
            # High volatility adjustments
            'high': {
                'position_size_multiplier': 0.7,  # Decrease position size
                'max_positions_adjustment': -1,  # Allow fewer positions
                'risk_per_trade_multiplier': 0.7  # Take less risk per trade
            }
        })

        # Black swan protection
        self.black_swan_risk_reduction = self.config.get('black_swan_risk_reduction',
                                                         0.5)  # 50% risk reduction on black swan detection

        # Performance tracking
        self.peak_equity = None
        self.current_drawdown = 0.0
        self.daily_losses = {}  # Date -> loss amount
        self.positions = {}  # Symbol -> position details
        self.trade_history = []  # List of trade results for analysis
        self.current_risk_level = RiskLevel.MEDIUM

        # Path for saving risk reports
        self.reports_dir = Path(self.config.get('reports_dir', 'data/risk_reports'))
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized RiskManager with max drawdown {self.max_drawdown:.1%}, "
                    f"risk per trade {self.risk_per_trade:.1%}, leverage {self.leverage}x")

    def _load_default_config(self):
        """Load default risk management configuration."""
        self.config = {
            'max_drawdown': 0.10,
            'max_daily_loss': 0.05,
            'max_position_size': 0.10,
            'max_total_exposure': 0.50,
            'risk_per_trade': 0.01,
            'max_open_positions': 5,
            'leverage': 50,
            'max_leverage': 100,
            'leverage_by_pair': {
                'EUR/USD': 50,
                'GBP/USD': 50,
                'USD/JPY': 50,
                'AUD/USD': 50
            },
            'currency_exposure_limits': {
                'USD': 0.80,  # Maximum 80% exposure to USD
                'EUR': 0.50,
                'GBP': 0.50,
                'JPY': 0.50,
                'AUD': 0.30
            },
            'use_dynamic_risk': True,
            'dynamic_risk_factors': {
                'market_volatility': 1.0,
                'system_performance': 1.0,
                'news_impact': 1.0
            },
            'use_volatility_sizing': True,
            'volatility_lookback': 20,
            'volatility_risk_factor': 1.0,
            'use_correlation_risk': True,
            'correlation_threshold': 0.7,
            'correlation_lookback': 30,
            'use_regime_based_risk': True,
            'black_swan_risk_reduction': 0.5,
            'reports_dir': 'data/risk_reports'
        }

    def calculate_position_size(
            self,
            symbol: str,
            equity: float,
            price: float,
            atr: Optional[float] = None,
            stop_loss: Optional[float] = None,
            regime_info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate optimal position size based on risk parameters, volatility, and market conditions.

        Args:
            symbol: Trading symbol (e.g., 'EUR/USD')
            equity: Current account equity
            price: Current market price
            atr: Average True Range (optional, for volatility-based sizing)
            stop_loss: Stop loss price (optional)
            regime_info: Current market regime information (optional)

        Returns:
            Position size in base currency units
        """
        # Get pair-specific leverage
        leverage = self.leverage_by_pair.get(symbol, self.leverage)

        # Apply dynamic risk adjustment if enabled
        risk_per_trade = self.risk_per_trade
        if self.use_dynamic_risk:
            risk_factor = self._calculate_dynamic_risk_factor(regime_info)
            risk_per_trade *= risk_factor

        # Calculate risk amount
        risk_amount = equity * risk_per_trade

        # Calculate position size based on stop loss if provided
        if stop_loss is not None:
            # Calculate risk in price terms
            price_risk = abs(price - stop_loss)

            if price_risk > 0:
                # Position size without leverage
                base_position_size = risk_amount / price_risk

                # Apply leverage
                position_size = base_position_size * leverage

                # Calculate position value
                position_value = position_size * price

                # Apply regime-based adjustment
                if self.use_regime_based_risk and regime_info:
                    volatility_regime = regime_info.get('volatility_regime', 'medium')
                    if volatility_regime in self.regime_risk_adjustments:
                        multiplier = self.regime_risk_adjustments[volatility_regime]['position_size_multiplier']
                        position_size *= multiplier
                        position_value *= multiplier

                # Apply black swan adjustment if detected
                if regime_info and regime_info.get('black_swan_detected', False):
                    position_size *= self.black_swan_risk_reduction
                    position_value *= self.black_swan_risk_reduction

                # Check if position exceeds maximum allowed size
                max_position_value = equity * self.max_position_size
                if position_value > max_position_value:
                    position_size = max_position_value / price

                return position_size

        # If no stop loss is provided but ATR is available, use volatility-based sizing
        if atr is not None and self.use_volatility_sizing:
            # ATR-based stop loss (typically 2-3x ATR)
            atr_multiplier = 2.0
            if regime_info:
                # Adjust multiplier based on volatility regime
                volatility_regime = regime_info.get('volatility_regime', 'medium')
                if volatility_regime == 'high':
                    atr_multiplier = 3.0  # Wider stops in high volatility
                elif volatility_regime == 'low':
                    atr_multiplier = 1.5  # Tighter stops in low volatility

            # Calculate position size based on ATR
            price_risk = atr * atr_multiplier
            if price_risk > 0:
                base_position_size = risk_amount / price_risk
                position_size = base_position_size * leverage

                # Apply regime-based adjustment
                if self.use_regime_based_risk and regime_info:
                    volatility_regime = regime_info.get('volatility_regime', 'medium')
                    if volatility_regime in self.regime_risk_adjustments:
                        position_size *= self.regime_risk_adjustments[volatility_regime]['position_size_multiplier']

                # Apply black swan adjustment if detected
                if regime_info and regime_info.get('black_swan_detected', False):
                    position_size *= self.black_swan_risk_reduction

                # Check against maximum position size
                max_position_value = equity * self.max_position_size
                if position_size * price > max_position_value:
                    position_size = max_position_value / price

                return position_size

        # If neither stop loss nor ATR provided, use fixed percentage of equity
        max_position_value = equity * self.max_position_size
        position_size = (max_position_value / price) * leverage

        # Apply regime-based adjustment
        if self.use_regime_based_risk and regime_info:
            volatility_regime = regime_info.get('volatility_regime', 'medium')
            if volatility_regime in self.regime_risk_adjustments:
                position_size *= self.regime_risk_adjustments[volatility_regime]['position_size_multiplier']

        # Apply black swan adjustment if detected
        if regime_info and regime_info.get('black_swan_detected', False):
            position_size *= self.black_swan_risk_reduction

        return position_size

    def _calculate_dynamic_risk_factor(self, regime_info: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate dynamic risk adjustment factor based on market conditions and system performance.

        Args:
            regime_info: Current market regime information (optional)

        Returns:
            Risk adjustment factor (1.0 is neutral, <1.0 reduces risk, >1.0 increases risk)
        """
        # Get current risk factors
        market_volatility = self.dynamic_risk_factors.get('market_volatility', 1.0)
        system_performance = self.dynamic_risk_factors.get('system_performance', 1.0)
        news_impact = self.dynamic_risk_factors.get('news_impact', 1.0)

        # Calculate combined factor - more sophisticated logic can be implemented here
        combined_factor = market_volatility * system_performance * news_impact

        # Apply risk level adjustments
        if self.current_risk_level == RiskLevel.LOW:
            combined_factor *= 1.2  # Increase risk when risk level is low
        elif self.current_risk_level == RiskLevel.HIGH:
            combined_factor *= 0.7  # Reduce risk when risk level is high
        elif self.current_risk_level == RiskLevel.CRITICAL:
            combined_factor *= 0.3  # Drastically reduce risk in critical conditions

        # Apply regime-based adjustment if regime information provided
        if self.use_regime_based_risk and regime_info:
            # Apply volatility regime adjustment
            volatility_regime = regime_info.get('volatility_regime', 'medium')
            if volatility_regime in self.regime_risk_adjustments:
                combined_factor *= self.regime_risk_adjustments[volatility_regime]['risk_per_trade_multiplier']

            # Apply directional bias adjustment
            direction = regime_info.get('directional_bias')
            trend_strength = regime_info.get('trend_strength', 0)

            # Adjust for trend strength
            if trend_strength > 25 and direction in ['up', 'down']:
                # Stronger trends = more confidence = slightly higher risk
                combined_factor *= (1.0 + (trend_strength - 25) / 100)

            # Apply stability adjustment
            stability = regime_info.get('stability', 0.5)
            combined_factor *= (0.8 + 0.4 * stability)  # 0.8-1.2 range based on stability

        # Apply drawdown-based adjustment
        if self.current_drawdown > 0:
            # Reduce risk as drawdown increases
            drawdown_factor = 1.0 - (self.current_drawdown / self.max_drawdown) * 0.5
            combined_factor *= max(0.5, drawdown_factor)

        # Apply black swan adjustment if detected
        if regime_info and regime_info.get('black_swan_detected', False):
            combined_factor *= self.black_swan_risk_reduction

        # Limit factor range
        return max(0.3, min(1.5, combined_factor))

    def update_risk_factors(self,
                            market_volatility: Optional[float] = None,
                            system_performance: Optional[float] = None,
                            news_impact: Optional[float] = None) -> None:
        """
        Update dynamic risk factors based on current market and system conditions.

        Args:
            market_volatility: Market volatility factor (optional)
            system_performance: System performance factor (optional)
            news_impact: News impact factor (optional)
        """
        if market_volatility is not None:
            self.dynamic_risk_factors['market_volatility'] = market_volatility

        if system_performance is not None:
            self.dynamic_risk_factors['system_performance'] = system_performance

        if news_impact is not None:
            self.dynamic_risk_factors['news_impact'] = news_impact

        logger.debug(f"Updated risk factors: {self.dynamic_risk_factors}")

    def set_risk_level(self, level: RiskLevel) -> None:
        """
        Set the current risk level for dynamic risk management.

        Args:
            level: Risk level to set
        """
        self.current_risk_level = level
        logger.info(f"Risk level set to {level.value}")

    def check_max_drawdown(self, equity: float) -> bool:
        """
        Check if current drawdown exceeds maximum allowed.

        Args:
            equity: Current equity value

        Returns:
            True if drawdown is acceptable, False if exceeded
        """
        # Initialize peak equity if not set
        if self.peak_equity is None:
            self.peak_equity = equity
            return True

        # Update peak if equity is higher
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.current_drawdown = 0.0
            return True

        # Calculate current drawdown
        self.current_drawdown = (self.peak_equity - equity) / self.peak_equity

        # Check if drawdown exceeds maximum
        if self.current_drawdown > self.max_drawdown:
            logger.warning(f"Maximum drawdown exceeded: {self.current_drawdown:.2%} > {self.max_drawdown:.2%}")
            return False

        return True

    def check_daily_loss(self, date: datetime, loss_amount: float, equity: float) -> bool:
        """
        Check if daily loss exceeds maximum allowed.

        Args:
            date: Current date
            loss_amount: Loss amount for this trade
            equity: Current equity

        Returns:
            True if daily loss is acceptable, False if exceeded
        """
        date_key = date.strftime('%Y-%m-%d')

        # Initialize daily loss for this date if not exists
        if date_key not in self.daily_losses:
            self.daily_losses[date_key] = 0.0

        # Add loss to daily total
        self.daily_losses[date_key] += loss_amount

        # Calculate daily loss as percentage of equity
        daily_loss_pct = self.daily_losses[date_key] / equity

        # Check if daily loss exceeds maximum
        if daily_loss_pct > self.max_daily_loss:
            logger.warning(f"Maximum daily loss exceeded: {daily_loss_pct:.2%} > {self.max_daily_loss:.2%}")
            return False

        return True

    def check_max_positions(self, regime_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if number of open positions exceeds maximum allowed, considering regime adjustments.

        Args:
            regime_info: Current market regime information (optional)

        Returns:
            True if number of positions is acceptable, False if exceeded
        """
        open_positions = sum(1 for pos in self.positions.values() if pos.get('size', 0) > 0)

        # Apply regime-based adjustment to max positions
        adjusted_max_positions = self.max_open_positions

        if self.use_regime_based_risk and regime_info:
            volatility_regime = regime_info.get('volatility_regime', 'medium')
            if volatility_regime in self.regime_risk_adjustments:
                adjustment = self.regime_risk_adjustments[volatility_regime]['max_positions_adjustment']
                adjusted_max_positions += adjustment

                # Ensure we have at least 1 position allowed
                adjusted_max_positions = max(1, adjusted_max_positions)

        # Apply black swan adjustment
        if regime_info and regime_info.get('black_swan_detected', False):
            adjusted_max_positions = max(1, int(adjusted_max_positions * self.black_swan_risk_reduction))

        # Check if number of positions exceeds adjusted maximum
        if open_positions >= adjusted_max_positions:
            logger.warning(f"Maximum open positions reached: {open_positions} >= {adjusted_max_positions}")
            return False

        return True

    def check_total_exposure(self, equity: float, regime_info: Optional[Dict[str, Any]] = None) -> Tuple[bool, float]:
        """
        Check if total market exposure exceeds maximum allowed, considering regime adjustments.

        Args:
            equity: Current account equity
            regime_info: Current market regime information (optional)

        Returns:
            Tuple of (is_acceptable, current_exposure_ratio)
        """
        # Calculate total exposure
        total_exposure = sum(
            abs(pos.get('size', 0) * pos.get('avg_price', 0))
            for pos in self.positions.values()
        )

        # Calculate exposure ratio
        exposure_ratio = total_exposure / equity if equity > 0 else 0

        # Apply regime-based adjustment to max exposure
        adjusted_max_exposure = self.max_total_exposure

        if self.use_regime_based_risk and regime_info:
            volatility_regime = regime_info.get('volatility_regime', 'medium')
            if volatility_regime in self.regime_risk_adjustments:
                # Adjust exposure limit based on regime
                if volatility_regime == 'low':
                    adjusted_max_exposure *= 1.2  # Allow 20% more exposure in low volatility
                elif volatility_regime == 'high':
                    adjusted_max_exposure *= 0.8  # Allow 20% less exposure in high volatility

        # Apply black swan adjustment
        if regime_info and regime_info.get('black_swan_detected', False):
            adjusted_max_exposure *= self.black_swan_risk_reduction

        # Check if exposure exceeds adjusted maximum
        if exposure_ratio > adjusted_max_exposure:
            logger.warning(f"Maximum total exposure exceeded: {exposure_ratio:.2%} > {adjusted_max_exposure:.2%}")
            return False, exposure_ratio

        return True, exposure_ratio

    def check_currency_exposure(self, equity: float, regime_info: Optional[Dict[str, Any]] = None) -> Dict[
        str, Tuple[bool, float]]:
        """
        Check if currency-specific exposure exceeds limits, considering regime adjustments.

        Args:
            equity: Current account equity
            regime_info: Current market regime information (optional)

        Returns:
            Dict mapping currency to (is_acceptable, exposure_ratio) tuples
        """
        # Calculate exposure for each currency
        currency_exposure = {}

        # Extract currencies from positions
        for symbol, position in self.positions.items():
            if '/' in symbol:
                base_currency, quote_currency = symbol.split('/')

                position_size = position.get('size', 0)
                position_value = position_size * position.get('avg_price', 0)

                # Add base currency exposure (long = positive, short = negative)
                if base_currency not in currency_exposure:
                    currency_exposure[base_currency] = 0
                currency_exposure[base_currency] += position_value

                # Add quote currency exposure (short = positive, long = negative)
                if quote_currency not in currency_exposure:
                    currency_exposure[quote_currency] = 0
                currency_exposure[quote_currency] -= position_value

        # Apply regime-based adjustment to currency limits
        currency_limit_multiplier = 1.0
        if self.use_regime_based_risk and regime_info:
            volatility_regime = regime_info.get('volatility_regime', 'medium')
            if volatility_regime == 'low':
                currency_limit_multiplier = 1.2  # 20% more exposure allowed in low volatility
            elif volatility_regime == 'high':
                currency_limit_multiplier = 0.8  # 20% less exposure allowed in high volatility

        # Apply black swan adjustment
        if regime_info and regime_info.get('black_swan_detected', False):
            currency_limit_multiplier *= self.black_swan_risk_reduction

        # Check against limits
        results = {}
        for currency, exposure in currency_exposure.items():
            # Get limit for this currency
            base_limit = self.currency_exposure_limits.get(currency, 1.0)
            adjusted_limit = base_limit * currency_limit_multiplier

            # Calculate exposure ratio
            exposure_ratio = abs(exposure) / equity if equity > 0 else 0

            # Check if exposure exceeds limit
            is_acceptable = exposure_ratio <= adjusted_limit

            if not is_acceptable:
                logger.warning(f"Currency exposure limit exceeded for {currency}: "
                               f"{exposure_ratio:.2%} > {adjusted_limit:.2%}")

            results[currency] = (is_acceptable, exposure_ratio)

        return results

    def update_position(self, symbol: str, size: float, avg_price: float) -> None:
        """
        Update position information for risk tracking.

        Args:
            symbol: Trading symbol
            size: Position size
            avg_price: Average entry price
        """
        self.positions[symbol] = {
            'size': size,
            'avg_price': avg_price,
            'updated_at': datetime.now()
        }

        logger.debug(f"Updated position for {symbol}: size={size}, avg_price={avg_price}")

    def record_trade(self,
                     symbol: str,
                     side: str,
                     size: float,
                     entry_price: float,
                     exit_price: Optional[float] = None,
                     pnl: Optional[float] = None,
                     duration: Optional[float] = None,
                     strategy: Optional[str] = None,
                     regime_id: Optional[int] = None) -> None:
        """
        Record a trade for performance analysis.

        Args:
            symbol: Trading symbol
            side: Trade side ('buy' or 'sell')
            size: Position size
            entry_price: Entry price
            exit_price: Exit price (optional, for completed trades)
            pnl: Profit/loss amount (optional, for completed trades)
            duration: Trade duration in seconds (optional)
            strategy: Strategy name (optional)
            regime_id: Market regime ID during trade (optional)
        """
        trade = {
            'symbol': symbol,
            'side': side,
            'size': size,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'entry_time': datetime.now().timestamp(),
            'exit_time': None if exit_price is None else datetime.now().timestamp(),
            'duration': duration,
            'strategy': strategy,
            'regime_id': regime_id
        }

        self.trade_history.append(trade)

    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive risk metrics.

        Returns:
            Dictionary of risk metrics
        """
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown_limit': self.max_drawdown,
            'open_positions': sum(1 for pos in self.positions.values() if pos.get('size', 0) > 0),
            'max_positions_limit': self.max_open_positions,
            'daily_loss': self.daily_losses.get(datetime.now().strftime('%Y-%m-%d'), 0.0),
            'max_daily_loss_limit': self.max_daily_loss,
            'risk_level': self.current_risk_level.value,
            'dynamic_risk_factors': self.dynamic_risk_factors,
            'peak_equity': self.peak_equity,
            'positions': self.positions
        }

    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze trading performance for risk management.

        Returns:
            Dictionary of performance metrics
        """
        # Filter completed trades
        completed_trades = [t for t in self.trade_history if t.get('exit_price') is not None]

        if not completed_trades:
            return {'total_trades': 0}

        # Calculate basic metrics
        winning_trades = [t for t in completed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in completed_trades if t.get('pnl', 0) <= 0]

        total_trades = len(completed_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        total_profit = sum(t.get('pnl', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))

        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Calculate average metrics
        avg_win = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0

        # Expected value per trade
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Per-symbol performance
        symbol_performance = {}
        for symbol in set(t.get('symbol') for t in completed_trades):
            symbol_trades = [t for t in completed_trades if t.get('symbol') == symbol]
            symbol_wins = [t for t in symbol_trades if t.get('pnl', 0) > 0]

            symbol_performance[symbol] = {
                'trades': len(symbol_trades),
                'win_rate': len(symbol_wins) / len(symbol_trades) if symbol_trades else 0,
                'profit': sum(t.get('pnl', 0) for t in symbol_trades)
            }

        # Per-strategy performance
        strategy_performance = {}
        for strategy in set(t.get('strategy') for t in completed_trades if t.get('strategy')):
            strategy_trades = [t for t in completed_trades if t.get('strategy') == strategy]
            strategy_wins = [t for t in strategy_trades if t.get('pnl', 0) > 0]

            strategy_performance[strategy] = {
                'trades': len(strategy_trades),
                'win_rate': len(strategy_wins) / len(strategy_trades) if strategy_trades else 0,
                'profit': sum(t.get('pnl', 0) for t in strategy_trades)
            }

        # Per-regime performance
        regime_performance = {}
        for regime_id in set(t.get('regime_id') for t in completed_trades if t.get('regime_id') is not None):
            regime_trades = [t for t in completed_trades if t.get('regime_id') == regime_id]
            regime_wins = [t for t in regime_trades if t.get('pnl', 0) > 0]

            regime_performance[regime_id] = {
                'trades': len(regime_trades),
                'win_rate': len(regime_wins) / len(regime_trades) if regime_trades else 0,
                'profit': sum(t.get('pnl', 0) for t in regime_trades),
                'avg_profit': sum(t.get('pnl', 0) for t in regime_trades) / len(regime_trades) if regime_trades else 0
            }

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': total_profit - total_loss,
            'symbol_performance': symbol_performance,
            'strategy_performance': strategy_performance,
            'regime_performance': regime_performance
        }
    def adjust_for_regime(self, regime_info: Dict[str, Any]) -> None:
        """
        Adjust risk parameters based on current market regime.

        Args:
            regime_info: Dictionary with regime information
        """
        if not self.use_regime_based_risk or not regime_info:
            return

        # Extract regime information
        volatility_regime = regime_info.get('volatility_regime', 'medium')
        directional_bias = regime_info.get('directional_bias', 'neutral')
        is_trending = regime_info.get('trend_strength', 0) > 25

        # Adjust risk per trade based on volatility regime
        if volatility_regime in self.regime_risk_adjustments:
            self.risk_per_trade = self.config.get('risk_per_trade', 0.01) * self.regime_risk_adjustments[volatility_regime][
                'risk_per_trade_multiplier']
        else:
            self.risk_per_trade = self.config.get('risk_per_trade', 0.01)

        # Adjust risk level based on volatility and black swan
        if regime_info.get('black_swan_detected', False):
            self.current_risk_level = RiskLevel.CRITICAL
        elif volatility_regime == 'high':
            self.current_risk_level = RiskLevel.HIGH
        elif volatility_regime == 'low' and is_trending:
            self.current_risk_level = RiskLevel.LOW
        else:
            self.current_risk_level = RiskLevel.MEDIUM

        logger.info(f"Adjusted risk parameters for regime: risk_per_trade={self.risk_per_trade:.2%}, "
                    f"risk_level={self.current_risk_level.value}, volatility={volatility_regime}, bias={directional_bias}")


    def reset(self) -> None:
        """Reset risk manager state."""
        self.peak_equity = None
        self.current_drawdown = 0.0
        self.daily_losses = {}
        self.positions = {}
        self.trade_history = []
        self.current_risk_level = RiskLevel.MEDIUM

        # Reset to default parameters
        self._load_default_config()

        # Apply configuration
        self.max_drawdown = self.config.get('max_drawdown', 0.10)
        self.max_daily_loss = self.config.get('max_daily_loss', 0.05)
        self.max_position_size = self.config.get('max_position_size', 0.10)
        self.max_total_exposure = self.config.get('max_total_exposure', 0.50)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.01)
        self.max_open_positions = self.config.get('max_open_positions', 5)

        logger.info("Risk manager state reset")


    def validate_trade(self, symbol: str, side: str, size: float, price: float,
                       equity: float, regime_info: Optional[Dict[str, Any]] = None,
                       existing_positions: Dict[str, Dict[str, Any]] = None,
                       price_data: Dict[str, pd.DataFrame] = None) -> Tuple[bool, str]:
        """
        Validate if a potential trade meets all risk management criteria.

        Args:
            symbol: Trading symbol
            side: Trade side ('buy' or 'sell')
            size: Position size
            price: Entry price
            equity: Current account equity
            regime_info: Current market regime information (optional)
            existing_positions: Dictionary of existing positions
            price_data: Price data for correlation analysis

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check max positions
        if existing_positions:
            self.positions = existing_positions

        if not self.check_max_positions(regime_info):
            # Get adjusted max positions
            adjusted_max_positions = self.max_open_positions
            if self.use_regime_based_risk and regime_info:
                volatility_regime = regime_info.get('volatility_regime', 'medium')
                if volatility_regime in self.regime_risk_adjustments:
                    adjustment = self.regime_risk_adjustments[volatility_regime]['max_positions_adjustment']
                    adjusted_max_positions += adjustment

            open_positions = sum(1 for pos in self.positions.values() if pos.get('size', 0) > 0)
            return False, f"Maximum open positions limit reached ({open_positions}/{adjusted_max_positions})"

        # Check position size
        position_value = size * price
        position_ratio = position_value / equity

        # Get adjusted max position size
        adjusted_max_position_size = self.max_position_size
        if self.use_regime_based_risk and regime_info:
            volatility_regime = regime_info.get('volatility_regime', 'medium')
            if volatility_regime in self.regime_risk_adjustments:
                multiplier = self.regime_risk_adjustments[volatility_regime]['position_size_multiplier']
                adjusted_max_position_size *= multiplier

        if position_ratio > adjusted_max_position_size:
            return False, f"Position size exceeds maximum allowed ({position_ratio:.2%} > {adjusted_max_position_size:.2%})"

        # Check total exposure
        is_valid, exposure_ratio = self.check_total_exposure(equity, regime_info)
        if not is_valid:
            # Get adjusted max exposure
            adjusted_max_exposure = self.max_total_exposure
            if self.use_regime_based_risk and regime_info:
                volatility_regime = regime_info.get('volatility_regime', 'medium')
                if volatility_regime == 'low':
                    adjusted_max_exposure *= 1.2
                elif volatility_regime == 'high':
                    adjusted_max_exposure *= 0.8

            return False, f"Total exposure would exceed maximum allowed ({exposure_ratio:.2%} > {adjusted_max_exposure:.2%})"

        # Check currency exposure
        if '/' in symbol:
            base_currency, quote_currency = symbol.split('/')

            # Calculate current currency exposure
            currency_exposure = {}
            for pos_symbol, position in self.positions.items():
                if '/' in pos_symbol:
                    pos_base, pos_quote = pos_symbol.split('/')
                    pos_size = position.get('size', 0)
                    pos_value = pos_size * position.get('avg_price', 0)

                    if pos_base not in currency_exposure:
                        currency_exposure[pos_base] = 0
                    currency_exposure[pos_base] += pos_value

                    if pos_quote not in currency_exposure:
                        currency_exposure[pos_quote] = 0
                    currency_exposure[pos_quote] -= pos_value

            # Add new position
            if base_currency not in currency_exposure:
                currency_exposure[base_currency] = 0
            if quote_currency not in currency_exposure:
                currency_exposure[quote_currency] = 0

            if side == 'buy':
                currency_exposure[base_currency] += position_value
                currency_exposure[quote_currency] -= position_value
            else:  # sell
                currency_exposure[base_currency] -= position_value
                currency_exposure[quote_currency] += position_value

            # Get currency limit multiplier
            currency_limit_multiplier = 1.0
            if self.use_regime_based_risk and regime_info:
                volatility_regime = regime_info.get('volatility_regime', 'medium')
                if volatility_regime == 'low':
                    currency_limit_multiplier = 1.2
                elif volatility_regime == 'high':
                    currency_limit_multiplier = 0.8

            # Check against limits
            for currency, exposure in currency_exposure.items():
                base_limit = self.currency_exposure_limits.get(currency, 1.0)
                adjusted_limit = base_limit * currency_limit_multiplier
                exposure_ratio = abs(exposure) / equity

                if exposure_ratio > adjusted_limit:
                    return False, f"Currency exposure for {currency} would exceed limit ({exposure_ratio:.2%} > {adjusted_limit:.2%})"

        # Check correlation risk if price data provided
        if price_data and self.use_correlation_risk:
            correlation_data = self.calculate_correlation_risk(price_data)
            current_symbols = [s for s in self.positions.keys() if self.positions[s].get('size', 0) > 0]

            if not self.should_diversify(symbol, current_symbols, correlation_data):
                return False, f"Adding {symbol} would exceed correlation threshold with existing positions"

        # Check regime-specific constraints
        if regime_info:
            # Check if the regime is suitable for trading
            if regime_info.get('black_swan_detected', False):
                # Still allow trading during black swan but with warning
                logger.warning(f"Trading during black swan event: {symbol} {side}")

            # Check if this is a low stability regime
            stability = regime_info.get('stability', 0.5)
            if stability < 0.3:
                logger.warning(f"Trading in low stability regime: {symbol} {side}")

            # Check directional bias against trade direction
            directional_bias = regime_info.get('directional_bias')
            trend_strength = regime_info.get('trend_strength', 0)

            if trend_strength > 40:  # Strong trend
                if (directional_bias == 'up' and side == 'sell') or (directional_bias == 'down' and side == 'buy'):
                    logger.warning(f"Trade direction ({side}) is against strong market bias ({directional_bias})")

        # All checks passed
        return True, ""


    def should_diversify(self, new_symbol: str, current_positions: List[str],
                         correlation_data: Dict[str, Dict[str, float]]) -> bool:
        """
        Check if adding a new position would maintain proper diversification.

        Args:
            new_symbol: Symbol for potential new position
            current_positions: List of current position symbols
            correlation_data: Correlation data from calculate_correlation_risk

        Returns:
            True if position would maintain diversification, False otherwise
        """
        if not self.use_correlation_risk or not correlation_data:
            return True

        # Check correlation with existing positions
        for symbol in current_positions:
            # Get correlation between symbols
            corr = None

            if new_symbol in correlation_data and symbol in correlation_data[new_symbol]:
                corr = correlation_data[new_symbol][symbol]
            elif symbol in correlation_data and new_symbol in correlation_data[symbol]:
                corr = correlation_data[symbol][new_symbol]

            # If correlation exceeds threshold, suggest not taking the trade
            if corr is not None and abs(corr) > self.correlation_threshold:
                logger.warning(f"Adding {new_symbol} would exceed correlation threshold "
                               f"(correlation with {symbol}: {corr:.2f})")
                return False

        return True


    def _save_risk_report(self, report: Dict[str, Any]) -> None:
        """
        Save risk report to file.

        Args:
            report: Risk report dictionary
        """
        now = datetime.now()
        filename = f"risk_report_{now.strftime('%Y%m%d_%H%M%S')}.json"
        file_path = self.reports_dir / filename

        try:
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)

            logger.debug(f"Saved risk report to {file_path}")
        except Exception as e:
            logger.error(f"Error saving risk report: {str(e)}")


    def calculate_correlation_risk(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlation risk between different currency pairs.

        Args:
            price_data: Dict mapping symbols to price DataFrames

        Returns:
            Dictionary mapping symbol pairs to correlation values
        """
        if not self.use_correlation_risk or not price_data:
            return {}

        correlations = {}
        symbols = list(price_data.keys())

        # Extract closing prices
        close_data = {}
        for symbol, df in price_data.items():
            if 'close' in df.columns:
                close_data[symbol] = df['close']

        # Calculate correlations
        for i in range(len(symbols)):
            symbol1 = symbols[i]
            if symbol1 not in close_data:
                continue

            correlations[symbol1] = {}

            for j in range(i + 1, len(symbols)):
                symbol2 = symbols[j]
                if symbol2 not in close_data:
                    continue

                # Align data
                s1 = close_data[symbol1]
                s2 = close_data[symbol2]

                if len(s1) > 0 and len(s2) > 0:
                    # Calculate correlation
                    corr = s1.corr(s2)

                    correlations[symbol1][symbol2] = corr

                    # Log high correlations
                    if abs(corr) > self.correlation_threshold:
                        logger.info(f"High correlation detected between {symbol1} and {symbol2}: {corr:.2f}")

        return correlations


    def _generate_risk_recommendations(self, equity: float, regime_info: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generate risk management recommendations based on current state.

        Args:
            equity: Current account equity
            regime_info: Current market regime information (optional)

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check drawdown
        if self.current_drawdown > self.max_drawdown * 0.8:
            recommendations.append(f"WARNING: Approaching maximum drawdown ({self.current_drawdown:.2%})")

        # Check open positions
        open_positions = sum(1 for pos in self.positions.values() if pos.get('size', 0) > 0)
        adjusted_max_positions = self.max_open_positions

        if self.use_regime_based_risk and regime_info:
            volatility_regime = regime_info.get('volatility_regime', 'medium')
            if volatility_regime in self.regime_risk_adjustments:
                adjustment = self.regime_risk_adjustments[volatility_regime]['max_positions_adjustment']
                adjusted_max_positions += adjustment

        if open_positions >= adjusted_max_positions * 0.8:
            recommendations.append(
                f"WARNING: Approaching maximum open positions ({open_positions}/{adjusted_max_positions})")

        # Check total exposure
        _, exposure_ratio = self.check_total_exposure(equity, regime_info)
        adjusted_max_exposure = self.max_total_exposure

        if self.use_regime_based_risk and regime_info:
            volatility_regime = regime_info.get('volatility_regime', 'medium')
            if volatility_regime == 'low':
                adjusted_max_exposure *= 1.2
            elif volatility_regime == 'high':
                adjusted_max_exposure *= 0.8

        if exposure_ratio > adjusted_max_exposure * 0.8:
            recommendations.append(f"WARNING: Approaching maximum total exposure ({exposure_ratio:.2%})")

        # Check daily loss
        today = datetime.now().strftime('%Y-%m-%d')
        daily_loss = self.daily_losses.get(today, 0)
        daily_loss_pct = daily_loss / equity if equity > 0 else 0

        if daily_loss_pct > self.max_daily_loss * 0.7:
            recommendations.append(f"WARNING: Significant daily loss ({daily_loss_pct:.2%})")

        # Performance-based recommendations
        performance = self.analyze_performance()
        win_rate = performance.get('win_rate', 0)
        profit_factor = performance.get('profit_factor', 0)

        if win_rate < 0.4 and performance.get('total_trades', 0) > 10:
            recommendations.append(f"WARNING: Low win rate ({win_rate:.2%}), consider strategy revisions")

        if profit_factor < 1.2 and performance.get('total_trades', 0) > 10:
            recommendations.append(f"WARNING: Low profit factor ({profit_factor:.2f}), consider reducing risk")

        # Add regime-specific recommendations
        if regime_info:
            # Volatility regime recommendations
            volatility_regime = regime_info.get('volatility_regime', 'medium')

            if volatility_regime == 'high':
                recommendations.append("High volatility detected: consider wider stops and reduced position sizes")

            # Trend recommendations
            trend_strength = regime_info.get('trend_strength', 0)
            directional_bias = regime_info.get('directional_bias')

            if trend_strength > 50 and directional_bias in ['up', 'down']:
                recommendations.append(f"Strong {directional_bias} trend detected: favor {directional_bias} trades")

            # Black swan recommendations
            if regime_info.get('black_swan_detected', False):
                recommendations.append("BLACK SWAN EVENT: Extreme caution recommended, reduce all exposure")

            # Regime stability recommendations
            stability = regime_info.get('stability', 0.5)
            if stability < 0.3:
                recommendations.append("Low regime stability: consider reducing position sizes")

        # Add positive recommendations too
        if self.current_drawdown < 0.05 and win_rate > 0.6 and profit_factor > 2.0:
            recommendations.append("System performing well, consider gradual risk increase")

        # Regime-specific strategy recommendations
        if regime_info and 'regime_id' in regime_info:
            regime_id = regime_info['regime_id']
            regime_performance = self.analyze_performance().get('regime_performance', {}).get(regime_id, {})

            if regime_performance:
                regime_win_rate = regime_performance.get('win_rate', 0)
                if regime_win_rate > 0.6 and regime_performance.get('trades', 0) > 5:
                    recommendations.append(f"Current regime is favorable: win rate {regime_win_rate:.2%} in this regime")
                elif regime_win_rate < 0.4 and regime_performance.get('trades', 0) > 5:
                    recommendations.append(
                        f"Current regime is challenging: win rate only {regime_win_rate:.2%} in this regime")

        return recommendations


    def generate_risk_report(self, equity: float, regime_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive risk report.

        Args:
            equity: Current account equity
            regime_info: Current market regime information (optional)

        Returns:
            Dictionary with risk report data
        """
        # Get risk metrics
        risk_metrics = self.get_risk_metrics()

        # Get performance metrics
        performance = self.analyze_performance()

        # Check exposure
        _, exposure_ratio = self.check_total_exposure(equity, regime_info)
        currency_exposure = self.check_currency_exposure(equity, regime_info)

        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'equity': equity,
            'risk_metrics': risk_metrics,
            'performance': performance,
            'exposure': {
                'total_exposure': exposure_ratio,
                'max_exposure_limit': self.max_total_exposure,
                'currency_exposure': {
                    currency: exposure for currency, (_, exposure) in currency_exposure.items()
                },
                'currency_limits': self.currency_exposure_limits
            },
            'positions': self.positions,
            'risk_level': self.current_risk_level.value,
            'regime_info': regime_info,
            'recommendations': self._generate_risk_recommendations(equity, regime_info)
        }

        # Save report
        self._save_risk_report(report)

        return report