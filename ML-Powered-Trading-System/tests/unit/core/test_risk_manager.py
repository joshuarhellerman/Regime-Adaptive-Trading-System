import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from pathlib import Path

# Import the module under test
from risk_manager import RiskManager, RiskLevel, DirectionalBias, VolatilityRegime

class TestRiskManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create a test configuration
        self.test_config = {
            'max_drawdown': 0.15,  # 15%
            'max_daily_loss': 0.05,  # 5%
            'max_position_size': 0.10,  # 10%
            'max_total_exposure': 0.50,  # 50%
            'risk_per_trade': 0.01,  # 1%
            'max_open_positions': 5,
            'leverage': 50,
            'max_leverage': 100,
            'leverage_by_pair': {
                'EUR/USD': 50,
                'GBP/USD': 40,
                'USD/JPY': 50
            },
            'use_dynamic_risk': True,
            'reports_dir': 'test_reports'
        }
        
        # Create risk manager with test config
        self.risk_manager = RiskManager(self.test_config)
        
        # Mock Path.mkdir for reports directory creation
        patcher = patch('risk_manager.Path.mkdir')
        self.addCleanup(patcher.stop)
        self.mock_mkdir = patcher.start()

    def test_initialization(self):
        """Test initialization of RiskManager with default and custom configuration"""
        # Test initialization with custom config
        self.assertEqual(self.risk_manager.max_drawdown, 0.15)
        self.assertEqual(self.risk_manager.risk_per_trade, 0.01)
        self.assertEqual(self.risk_manager.leverage, 50)
        self.assertEqual(self.risk_manager.current_risk_level, RiskLevel.MEDIUM)
        
        # Test initialization with default config
        default_risk_manager = RiskManager()
        self.assertEqual(default_risk_manager.max_drawdown, 0.10)
        self.assertEqual(default_risk_manager.risk_per_trade, 0.01)
        self.assertEqual(default_risk_manager.leverage, 50)

    def test_calculate_position_size_with_stop_loss(self):
        """Test position size calculation with stop loss specified"""
        equity = 10000
        price = 1.2000
        stop_loss = 1.1900  # 100 pip stop
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            symbol='EUR/USD',
            equity=equity,
            price=price,
            stop_loss=stop_loss
        )
        
        # Expected calculation:
        # risk_amount = equity * risk_per_trade = 10000 * 0.01 = 100
        # price_risk = abs(price - stop_loss) = 0.0100
        # base_position_size = risk_amount / price_risk = 100 / 0.0100 = 10000
        # position_size = base_position_size * leverage = 10000 * 50 = 500000 units
        # This should be capped by max_position_size:
        # max_position_value = equity * max_position_size = 10000 * 0.10 = 1000
        # max_position_size_in_units = max_position_value / price = 1000 / 1.2 = 833.33
        expected_size = 1000 / price * 50  # Leveraged position size capped by max_position_size
        
        self.assertAlmostEqual(position_size, expected_size, delta=1)

    def test_calculate_position_size_with_atr(self):
        """Test position size calculation with ATR for volatility-based sizing"""
        equity = 10000
        price = 1.2000
        atr = 0.0020  # 20 pips ATR
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            symbol='EUR/USD',
            equity=equity,
            price=price,
            atr=atr
        )
        
        # Expected calculation:
        # risk_amount = equity * risk_per_trade = 10000 * 0.01 = 100
        # atr_multiplier = 2.0 (default)
        # price_risk = atr * atr_multiplier = 0.0020 * 2.0 = 0.0040
        # base_position_size = risk_amount / price_risk = 100 / 0.0040 = 25000
        # position_size = base_position_size * leverage = 25000 * 50 = 1,250,000 units
        # This should be capped by max_position_size:
        # max_position_value = equity * max_position_size = 10000 * 0.10 = 1000
        # max_position_size_in_units = max_position_value / price = 1000 / 1.2 = 833.33
        expected_size = 1000 / price * 50  # Leveraged position size capped by max_position_size
        
        self.assertAlmostEqual(position_size, expected_size, delta=1)

    def test_calculate_position_size_with_regime_info(self):
        """Test position size calculation with regime information"""
        equity = 10000
        price = 1.2000
        stop_loss = 1.1900
        
        # Test low volatility regime
        regime_info_low = {
            'volatility_regime': 'low',
            'directional_bias': 'up',
            'trend_strength': 30,
            'stability': 0.8
        }
        
        position_size_low = self.risk_manager.calculate_position_size(
            symbol='EUR/USD',
            equity=equity,
            price=price,
            stop_loss=stop_loss,
            regime_info=regime_info_low
        )
        
        # Test high volatility regime
        regime_info_high = {
            'volatility_regime': 'high',
            'directional_bias': 'down',
            'trend_strength': 15,
            'stability': 0.4
        }
        
        position_size_high = self.risk_manager.calculate_position_size(
            symbol='EUR/USD',
            equity=equity,
            price=price,
            stop_loss=stop_loss,
            regime_info=regime_info_high
        )
        
        # Low volatility regime should have larger position size than high volatility
        self.assertGreater(position_size_low, position_size_high)
        
        # Test black swan event
        regime_info_black_swan = {
            'volatility_regime': 'high',
            'black_swan_detected': True
        }
        
        position_size_black_swan = self.risk_manager.calculate_position_size(
            symbol='EUR/USD',
            equity=equity,
            price=price,
            stop_loss=stop_loss,
            regime_info=regime_info_black_swan
        )
        
        # Black swan should result in reduced position size
        self.assertLess(position_size_black_swan, position_size_high)
        self.assertAlmostEqual(position_size_black_swan, position_size_high * self.risk_manager.black_swan_risk_reduction, delta=1)

    def test_dynamic_risk_factor(self):
        """Test dynamic risk factor calculation"""
        # Default risk factors
        self.assertAlmostEqual(self.risk_manager._calculate_dynamic_risk_factor(), 1.0, delta=0.01)
        
        # Update risk factors and test again
        self.risk_manager.update_risk_factors(
            market_volatility=0.8,
            system_performance=1.2,
            news_impact=0.9
        )
        
        # Check new risk factors
        self.assertEqual(self.risk_manager.dynamic_risk_factors['market_volatility'], 0.8)
        self.assertEqual(self.risk_manager.dynamic_risk_factors['system_performance'], 1.2)
        self.assertEqual(self.risk_manager.dynamic_risk_factors['news_impact'], 0.9)
        
        # Calculate combined factor
        expected_factor = 0.8 * 1.2 * 0.9
        # MEDIUM risk level is default (no adjustment)
        self.assertAlmostEqual(self.risk_manager._calculate_dynamic_risk_factor(), expected_factor, delta=0.01)
        
        # Test with high risk level
        self.risk_manager.set_risk_level(RiskLevel.HIGH)
        high_factor = self.risk_manager._calculate_dynamic_risk_factor()
        self.assertAlmostEqual(high_factor, expected_factor * 0.7, delta=0.01)
        
        # Test with low risk level
        self.risk_manager.set_risk_level(RiskLevel.LOW)
        low_factor = self.risk_manager._calculate_dynamic_risk_factor()
        self.assertAlmostEqual(low_factor, expected_factor * 1.2, delta=0.01)

    def test_check_max_drawdown(self):
        """Test maximum drawdown checks"""
        # Initialize with starting equity
        initial_equity = 10000
        self.assertTrue(self.risk_manager.check_max_drawdown(initial_equity))
        self.assertEqual(self.risk_manager.peak_equity, initial_equity)
        self.assertEqual(self.risk_manager.current_drawdown, 0.0)
        
        # Equity increases - should update peak
        increased_equity = 11000
        self.assertTrue(self.risk_manager.check_max_drawdown(increased_equity))
        self.assertEqual(self.risk_manager.peak_equity, increased_equity)
        self.assertEqual(self.risk_manager.current_drawdown, 0.0)
        
        # Equity decreases but within limit
        decreased_equity = 10000
        drawdown = (increased_equity - decreased_equity) / increased_equity
        self.assertTrue(self.risk_manager.check_max_drawdown(decreased_equity))
        self.assertEqual(self.risk_manager.peak_equity, increased_equity)
        self.assertAlmostEqual(self.risk_manager.current_drawdown, drawdown, delta=0.001)
        
        # Equity decreases beyond limit
        critical_equity = 9100  # Just above 15% drawdown limit from 11000
        self.assertTrue(self.risk_manager.check_max_drawdown(critical_equity))
        
        # This would exceed max drawdown
        exceed_equity = 9000  # Above 15% drawdown limit from 11000
        self.assertFalse(self.risk_manager.check_max_drawdown(exceed_equity))
        self.assertGreater(self.risk_manager.current_drawdown, self.risk_manager.max_drawdown)

    def test_check_daily_loss(self):
        """Test daily loss limit checks"""
        equity = 10000
        date = datetime.now()
        
        # No losses yet
        self.assertTrue(self.risk_manager.check_daily_loss(date, 0, equity))
        
        # Small loss
        self.assertTrue(self.risk_manager.check_daily_loss(date, 100, equity))
        
        # Additional small loss
        self.assertTrue(self.risk_manager.check_daily_loss(date, 200, equity))
        
        # Check cumulative loss
        date_key = date.strftime('%Y-%m-%d')
        self.assertEqual(self.risk_manager.daily_losses[date_key], 300)
        
        # Loss approaching limit
        self.assertTrue(self.risk_manager.check_daily_loss(date, 200, equity))
        
        # Loss exceeding limit
        # Current loss: 500, limit is 5% of 10000 = 500
        self.assertTrue(self.risk_manager.check_daily_loss(date, 0, equity))
        self.assertFalse(self.risk_manager.check_daily_loss(date, 501, equity))
        
        # Test that a different date resets the daily loss counter
        tomorrow = date + timedelta(days=1)
        self.assertTrue(self.risk_manager.check_daily_loss(tomorrow, 300, equity))
        tomorrow_key = tomorrow.strftime('%Y-%m-%d')
        self.assertEqual(self.risk_manager.daily_losses[tomorrow_key], 300)

    def test_check_max_positions(self):
        """Test maximum positions check"""
        # No positions initially
        self.assertTrue(self.risk_manager.check_max_positions())
        
        # Add positions up to the limit
        for i in range(5):
            self.risk_manager.update_position(f"PAIR{i}", 1000, 1.0)
            self.assertTrue(self.risk_manager.check_max_positions())
        
        # Add one more position to exceed limit
        self.risk_manager.update_position("PAIR_EXTRA", 1000, 1.0)
        self.assertFalse(self.risk_manager.check_max_positions())
        
        # Test with regime adjustment
        regime_info_low = {'volatility_regime': 'low'}  # Should allow 1 more position
        self.assertTrue(self.risk_manager.check_max_positions(regime_info_low))
        
        # Add another position
        self.risk_manager.update_position("PAIR_EXTRA2", 1000, 1.0)
        self.assertFalse(self.risk_manager.check_max_positions(regime_info_low))
        
        # Test with high volatility regime (should reduce max positions)
        regime_info_high = {'volatility_regime': 'high'}
        # We already have 7 positions, high volatility allows 4 (5-1)
        self.assertFalse(self.risk_manager.check_max_positions(regime_info_high))

    def test_check_total_exposure(self):
        """Test total exposure check"""
        equity = 10000
        
        # No positions initially
        is_valid, exposure_ratio = self.risk_manager.check_total_exposure(equity)
        self.assertTrue(is_valid)
        self.assertEqual(exposure_ratio, 0)
        
        # Add positions
        self.risk_manager.update_position("EUR/USD", 1000, 1.2)  # Value: 1200
        is_valid, exposure_ratio = self.risk_manager.check_total_exposure(equity)
        self.assertTrue(is_valid)
        self.assertAlmostEqual(exposure_ratio, 0.12, delta=0.001)
        
        # Add more positions
        self.risk_manager.update_position("GBP/USD", 2000, 1.5)  # Value: 3000
        is_valid, exposure_ratio = self.risk_manager.check_total_exposure(equity)
        self.assertTrue(is_valid)
        self.assertAlmostEqual(exposure_ratio, 0.42, delta=0.001)
        
        # Exceed limit
        self.risk_manager.update_position("USD/JPY", 1000, 150)  # Value: 150000 (very high to force exceeding)
        is_valid, exposure_ratio = self.risk_manager.check_total_exposure(equity)
        self.assertFalse(is_valid)
        self.assertGreater(exposure_ratio, self.risk_manager.max_total_exposure)
        
        # Test with regime info
        regime_info_low = {'volatility_regime': 'low'}  # Should increase limit by 20%
        is_valid, exposure_ratio = self.risk_manager.check_total_exposure(equity, regime_info_low)
        # 0.5 * 1.2 = 0.6 is the adjusted limit
        if exposure_ratio < 0.6:
            self.assertTrue(is_valid)
        else:
            self.assertFalse(is_valid)

    def test_check_currency_exposure(self):
        """Test currency-specific exposure checks"""
        equity = 10000
        
        # No positions initially
        result = self.risk_manager.check_currency_exposure(equity)
        self.assertEqual(result, {})
        
        # Add positions affecting USD exposure
        self.risk_manager.update_position("EUR/USD", 1000, 1.2)  # Long EUR/Short USD
        self.risk_manager.update_position("GBP/USD", 1000, 1.5)  # Long GBP/Short USD
        
        result = self.risk_manager.check_currency_exposure(equity)
        
        # Check USD exposure (should be significant due to being quote currency in both pairs)
        self.assertIn('USD', result)
        is_valid, exposure_ratio = result['USD']
        # Expected exposure: -(1000*1.2 + 1000*1.5) = -2700
        # Ratio: 2700/10000 = 0.27, which is under the default 0.8 limit for USD
        self.assertTrue(is_valid)
        self.assertAlmostEqual(exposure_ratio, 0.27, delta=0.01)
        
        # Check EUR exposure
        self.assertIn('EUR', result)
        is_valid, exposure_ratio = result['EUR']
        # Expected exposure: 1000*1.2 = 1200
        # Ratio: 1200/10000 = 0.12, which is under the default 0.5 limit for EUR
        self.assertTrue(is_valid)
        self.assertAlmostEqual(exposure_ratio, 0.12, delta=0.01)
        
        # Add positions to exceed EUR limit
        self.risk_manager.update_position("EUR/GBP", 4000, 0.9)  # Long EUR/Short GBP
        
        result = self.risk_manager.check_currency_exposure(equity)
        self.assertIn('EUR', result)
        is_valid, exposure_ratio = result['EUR']
        # Expected exposure: 1000*1.2 + 4000*0.9 = 4800
        # Ratio: 4800/10000 = 0.48, which is close to but under the 0.5 limit
        self.assertTrue(is_valid)
        self.assertAlmostEqual(exposure_ratio, 0.48, delta=0.01)
        
        # Exceed EUR limit
        self.risk_manager.update_position("EUR/JPY", 1000, 130)  # Long EUR/Short JPY
        
        result = self.risk_manager.check_currency_exposure(equity)
        self.assertIn('EUR', result)
        is_valid, exposure_ratio = result['EUR']
        # Expected exposure increased by 1000*130
        # This should exceed the 0.5 limit for EUR
        if exposure_ratio > 0.5:
            self.assertFalse(is_valid)
        
        # Test with regime info
        regime_info_low = {'volatility_regime': 'low'}  # Should increase limits by 20%
        result = self.risk_manager.check_currency_exposure(equity, regime_info_low)
        self.assertIn('EUR', result)
        is_valid, exposure_ratio = result['EUR']
        # With low volatility, limit increases to 0.5 * 1.2 = 0.6
        # If exposure was just over 0.5, it should now be valid
        if exposure_ratio < 0.6:
            self.assertTrue(is_valid)

    def test_update_position(self):
        """Test position updating"""
        # Update a position
        self.risk_manager.update_position("EUR/USD", 1000, 1.2)
        
        # Verify position details
        self.assertIn("EUR/USD", self.risk_manager.positions)
        position = self.risk_manager.positions["EUR/USD"]
        self.assertEqual(position['size'], 1000)
        self.assertEqual(position['avg_price'], 1.2)
        self.assertIn('updated_at', position)
        
        # Update same position
        self.risk_manager.update_position("EUR/USD", 1500, 1.25)
        
        # Verify updated position
        position = self.risk_manager.positions["EUR/USD"]
        self.assertEqual(position['size'], 1500)
        self.assertEqual(position['avg_price'], 1.25)

    def test_record_trade(self):
        """Test trade recording"""
        # Record an entry trade
        self.risk_manager.record_trade(
            symbol="EUR/USD",
            side="buy",
            size=1000,
            entry_price=1.2,
            strategy="trend_following",
            regime_id=1
        )
        
        # Verify trade was recorded
        self.assertEqual(len(self.risk_manager.trade_history), 1)
        trade = self.risk_manager.trade_history[0]
        self.assertEqual(trade['symbol'], "EUR/USD")
        self.assertEqual(trade['side'], "buy")
        self.assertEqual(trade['size'], 1000)
        self.assertEqual(trade['entry_price'], 1.2)
        self.assertIsNone(trade['exit_price'])
        self.assertIsNone(trade['pnl'])
        self.assertEqual(trade['strategy'], "trend_following")
        self.assertEqual(trade['regime_id'], 1)
        
        # Record a completed trade
        self.risk_manager.record_trade(
            symbol="GBP/USD",
            side="sell",
            size=2000,
            entry_price=1.5,
            exit_price=1.48,
            pnl=4000,
            duration=3600,
            strategy="mean_reversion",
            regime_id=2
        )
        
        # Verify second trade
        self.assertEqual(len(self.risk_manager.trade_history), 2)
        trade = self.risk_manager.trade_history[1]
        self.assertEqual(trade['symbol'], "GBP/USD")
        self.assertEqual(trade['exit_price'], 1.48)
        self.assertEqual(trade['pnl'], 4000)
        self.assertEqual(trade['duration'], 3600)

    def test_get_risk_metrics(self):
        """Test retrieving risk metrics"""
        # Setup some test data
        self.risk_manager.peak_equity = 10000
        self.risk_manager.current_drawdown = 0.05
        self.risk_manager.update_position("EUR/USD", 1000, 1.2)
        today = datetime.now().strftime('%Y-%m-%d')
        self.risk_manager.daily_losses[today] = 200
        
        # Get metrics
        metrics = self.risk_manager.get_risk_metrics()
        
        # Verify metrics
        self.assertEqual(metrics['current_drawdown'], 0.05)
        self.assertEqual(metrics['max_drawdown_limit'], 0.15)
        self.assertEqual(metrics['open_positions'], 1)
        self.assertEqual(metrics['max_positions_limit'], 5)
        self.assertEqual(metrics['daily_loss'], 200)
        self.assertEqual(metrics['risk_level'], "medium")
        self.assertEqual(metrics['peak_equity'], 10000)
        self.assertIn('positions', metrics)
        self.assertIn('EUR/USD', metrics['positions'])

    def test_analyze_performance(self):
        """Test performance analysis"""
        # No trades initially
        performance = self.risk_manager.analyze_performance()
        self.assertEqual(performance['total_trades'], 0)
        
        # Record some trades
        # Winning trade
        self.risk_manager.record_trade(
            symbol="EUR/USD",
            side="buy",
            size=1000,
            entry_price=1.2,
            exit_price=1.22,
            pnl=200,
            strategy="trend_following",
            regime_id=1
        )
        
        # Losing trade
        self.risk_manager.record_trade(
            symbol="GBP/USD",
            side="sell",
            size=2000,
            entry_price=1.5,
            exit_price=1.52,
            pnl=-400,
            strategy="mean_reversion",
            regime_id=2
        )
        
        # Another winning trade
        self.risk_manager.record_trade(
            symbol="EUR/USD",
            side="buy",
            size=1500,
            entry_price=1.21,
            exit_price=1.23,
            pnl=300,
            strategy="trend_following",
            regime_id=1
        )
        
        # Analyze performance
        performance = self.risk_manager.analyze_performance()
        
        # Verify metrics
        self.assertEqual(performance['total_trades'], 3)
        self.assertAlmostEqual(performance['win_rate'], 2/3, delta=0.01)
        self.assertEqual(performance['total_profit'], 500)
        self.assertEqual(performance['total_loss'], 400)
        self.assertEqual(performance['net_profit'], 100)
        self.assertAlmostEqual(performance['profit_factor'], 500/400, delta=0.01)
        
        # Check symbol performance
        self.assertIn('symbol_performance', performance)
        self.assertIn('EUR/USD', performance['symbol_performance'])
        self.assertIn('GBP/USD', performance['symbol_performance'])
        
        eur_usd_perf = performance['symbol_performance']['EUR/USD']
        self.assertEqual(eur_usd_perf['trades'], 2)
        self.assertEqual(eur_usd_perf['win_rate'], 1.0)
        self.assertEqual(eur_usd_perf['profit'], 500)
        
        # Check strategy performance
        self.assertIn('strategy_performance', performance)
        self.assertIn('trend_following', performance['strategy_performance'])
        self.assertIn('mean_reversion', performance['strategy_performance'])
        
        trend_perf = performance['strategy_performance']['trend_following']
        self.assertEqual(trend_perf['trades'], 2)
        self.assertEqual(trend_perf['win_rate'], 1.0)
        
        # Check regime performance
        self.assertIn('regime_performance', performance)
        self.assertIn(1, performance['regime_performance'])
        self.assertIn(2, performance['regime_performance'])
        
        regime1_perf = performance['regime_performance'][1]
        self.assertEqual(regime1_perf['trades'], 2)
        self.assertEqual(regime1_perf['win_rate'], 1.0)
        self.assertEqual(regime1_perf['profit'], 500)

    def test_adjust_for_regime(self):
        """Test regime-based risk adjustments"""
        # Initial settings
        initial_risk = self.risk_manager.risk_per_trade
        initial_risk_level = self.risk_manager.current_risk_level
        
        # Test low volatility regime
        regime_info_low = {
            'volatility_regime': 'low',
            'directional_bias': 'up',
            'trend_strength': 30,
            'stability': 0.8
        }
        
        self.risk_manager.adjust_for_regime(regime_info_low)
        
        # Should increase risk per trade for low volatility
        self.assertGreater(self.risk_manager.risk_per_trade, initial_risk)
        self.assertEqual(self.risk_manager.current_risk_level, RiskLevel.LOW)
        
        # Reset
        self.risk_manager.risk_per_trade = initial_risk
        self.risk_manager.current_risk_level = initial_risk_level
        
        # Test high volatility regime
        regime_info_high = {
            'volatility_regime': 'high',
            'directional_bias': 'down',
            'trend_strength': 15,
            'stability': 0.4
        }
        
        self.risk_manager.adjust_for_regime(regime_info_high)
        
        # Should decrease risk per trade for high volatility
        self.assertLess(self.risk_manager.risk_per_trade, initial_risk)
        self.assertEqual(self.risk_manager.current_risk_level, RiskLevel.HIGH)
        
        # Test black swan
        regime_info_black_swan = {
            'volatility_regime': 'high',
            'black_swan_detected': True
        }
        
        self.risk_manager.adjust_for_regime(regime_info_black_swan)
        self.assertEqual(self.risk_manager.current_risk_level, RiskLevel.CRITICAL)

    @patch('risk_manager.logging.info')
    def test_reset(self, mock_log):
        """Test resetting the risk manager state"""
        # Setup some test data
        self.risk_manager.peak_equity = 10000
        self.risk_manager.current_drawdown = 0.05
        self.risk_manager.update_position("EUR/USD", 1000, 1.2)
        self.risk_manager.set_risk_level(RiskLevel.HIGH)
        self.risk_manager.record_trade("EUR/USD", "buy", 1000, 1.2)
        
        # Reset
        self.risk_manager.reset()
        
        # Verify reset state
        self.assertIsNone(self.risk_manager.peak_equity)
        self.assertEqual(self.risk_manager.current_drawdown, 0.0)
        self.assertEqual(self.risk_manager.positions, {})
        self.assertEqual(self.risk_manager.trade_history, [])
        self.assertEqual(self.risk_manager.current_risk_level, RiskLevel.MEDIUM)
        
        # Verify log was called
        mock_log.assert_called_with("Risk manager state reset")

    def test_validate_trade(self):
        """Test trade validation"""
        equity = 10000
        
        # Valid trade
        is_valid, reason = self.risk_manager.validate_trade(
            symbol="EUR/USD",
            side="buy",
            size=1000,
            price=1.2,
            equity=equity
        )
        self.assertTrue(is_valid)
        self.assertEqual(reason, "")
        
        # Test with regime info
        regime_info = {
            'volatility_regime': 'high',
            'directional_bias': 'down',
            'trend_strength': 50,
            'stability': 0.3
        }
        
        is_valid, reason = self.risk_manager.validate_trade(
            symbol="EUR/USD",
            side="buy",  # Buying when trend is down (warning but still valid)
            size=800,    # Smaller size due to high volatility
            price=1.2,
            equity=equity,
            regime_info=regime_info
        )
        self.assertTrue(is_valid)
        
        # Add positions close to the limit
        for i in range(4):
            self.risk_manager.update_position(f"PAIR{i}", 1000, 1.0)
        
        # Test with positions close to limit
        is_valid, reason = self.risk_manager.validate_trade(
            symbol="GBP/USD",
            side="buy",
            size=1000,
            price=1.5,
            equity=equity
        )
        self.assertTrue(is_valid)
        
        # Add one more position to reach limit
        self.risk_manager.update_position("PAIR4", 1000, 1.0)
        
        # Test with max positions reached
        is_valid, reason = self.risk_manager.validate_trade(
            symbol="USD/JPY",
            side="buy",
            size=1000,
            price=150,
            equity=equity
        )
        self.assertFalse(is_valid)
        self.assertIn("Maximum open positions limit reached", reason)
        
        # Reset positions
        self.risk_manager.positions = {}
        
        # Test trade size validation
        is_valid, reason = self.risk_manager.validate_trade(
            symbol="GBP/USD",
            side="buy",
            size=10000,  # Very large size
            price=1.5,
            equity=equity
        )
        self.assertFalse(is_valid)
        self.assertIn("Position size exceeds maximum", reason)
        
        # Test with currency exposure limits
        # First, add a position that creates substantial EUR exposure
        self.risk_manager.update_position("EUR/USD", 5000, 1.2)  # 6000 EUR exposure
        
        # Now try to add more EUR exposure
        is_valid, reason = self.risk_manager.validate_trade(
            symbol="EUR/JPY",
            side="buy",
            size=5000,
            price=130,  # 650,000 EUR exposure
            equity=equity
        )
        self.assertFalse(is_valid)
        self.assertIn("Currency exposure", reason)