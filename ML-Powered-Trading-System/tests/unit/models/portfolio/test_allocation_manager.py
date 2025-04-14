import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from datetime import datetime

from models.portfolio.allocation_manager import AllocationManager, AllocationResult
from models.portfolio.constraints import PortfolioConstraint
from models.portfolio.objectives import PortfolioObjective


class TestAllocationResult(unittest.TestCase):
    """Test the AllocationResult dataclass"""

    def test_allocation_result_creation(self):
        """Test creating an AllocationResult"""
        allocations = {'AAPL': 0.5, 'MSFT': 0.3, 'CASH': 0.2}
        result = AllocationResult(
            allocations=allocations,
            objective_value=0.12,
            constraints_satisfied=True,
            strategy_allocations={'tech': 0.8, 'cash': 0.2},
            sector_allocations={'Technology': 0.8, 'Cash': 0.2}
        )

        self.assertEqual(result.allocations, allocations)
        self.assertEqual(result.objective_value, 0.12)
        self.assertTrue(result.constraints_satisfied)
        self.assertEqual(result.strategy_allocations, {'tech': 0.8, 'cash': 0.2})
        self.assertEqual(result.sector_allocations, {'Technology': 0.8, 'Cash': 0.2})
        self.assertIsInstance(result.timestamp, datetime)
        self.assertEqual(result.metadata, {})

    def test_allocation_result_string_representation(self):
        """Test the string representation of AllocationResult"""
        result = AllocationResult(
            allocations={'AAPL': 1.0},
            objective_value=0.05,
            constraints_satisfied=True
        )
        
        expected_str = "Capital allocation with objective value 0.050000. Constraints satisfied: True"
        self.assertEqual(str(result), expected_str)


class TestAllocationManager(unittest.TestCase):
    """Test the AllocationManager class"""

    def setUp(self):
        """Set up common test fixtures"""
        self.total_capital = 100000.0
        self.cash_buffer_pct = 0.05
        self.manager = AllocationManager(
            total_capital=self.total_capital,
            cash_buffer_pct=self.cash_buffer_pct,
            risk_budget={'strategy1': 0.6, 'strategy2': 0.4},
            strategy_limits={'strategy1': (0.1, 0.7), 'strategy2': (0.2, 0.5)},
            sector_limits={'Technology': (0.0, 0.4), 'Healthcare': (0.1, 0.3)},
            cash_key='CASH'
        )
        
        # Common test data
        self.strategies = {
            'strategy1': {'AAPL': 0.5, 'MSFT': 0.5},
            'strategy2': {'JNJ': 0.7, 'PFE': 0.3}
        }
        
        self.expected_returns = {
            'AAPL': 0.08,
            'MSFT': 0.07,
            'JNJ': 0.05,
            'PFE': 0.04
        }
        
        self.sector_mapping = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'JNJ': 'Healthcare',
            'PFE': 'Healthcare'
        }

    def test_initialization(self):
        """Test initializing the AllocationManager with different parameters"""
        # Test default initialization
        manager = AllocationManager()
        self.assertEqual(manager.total_capital, 100000.0)
        self.assertEqual(manager.cash_buffer_pct, 0.05)
        self.assertEqual(manager.risk_budget, {})
        self.assertEqual(manager.strategy_limits, {})
        self.assertEqual(manager.sector_limits, {})
        self.assertEqual(manager.cash_key, 'CASH')
        
        # Test custom initialization
        self.assertEqual(self.manager.total_capital, self.total_capital)
        self.assertEqual(self.manager.cash_buffer_pct, self.cash_buffer_pct)
        self.assertEqual(self.manager.risk_budget, {'strategy1': 0.6, 'strategy2': 0.4})
        self.assertEqual(self.manager.strategy_limits, {'strategy1': (0.1, 0.7), 'strategy2': (0.2, 0.5)})
        self.assertEqual(self.manager.sector_limits, {'Technology': (0.0, 0.4), 'Healthcare': (0.1, 0.3)})

    def test_allocate_capital_no_strategies(self):
        """Test allocating capital with no strategies"""
        result = self.manager.allocate_capital({}, self.expected_returns)
        
        # Should allocate everything to cash
        self.assertEqual(result.allocations, {'CASH': 1.0})
        self.assertEqual(result.objective_value, 0.0)
        self.assertTrue(result.constraints_satisfied)
        self.assertIsNone(result.strategy_allocations)
        self.assertIsNone(result.sector_allocations)

    def test_allocate_capital_single_strategy(self):
        """Test allocating capital with a single strategy"""
        strategies = {'strategy1': {'AAPL': 0.6, 'MSFT': 0.4}}
        
        result = self.manager.allocate_capital(
            strategies, 
            self.expected_returns
        )
        
        # Check allocations (should be strategy weights with cash buffer)
        allocatable_capital = self.total_capital * (1 - self.cash_buffer_pct)
        expected_aapl = 0.6 * allocatable_capital / self.total_capital
        expected_msft = 0.4 * allocatable_capital / self.total_capital
        expected_cash = self.cash_buffer_pct
        
        self.assertAlmostEqual(result.allocations['AAPL'], expected_aapl, places=6)
        self.assertAlmostEqual(result.allocations['MSFT'], expected_msft, places=6)
        self.assertAlmostEqual(result.allocations['CASH'], expected_cash, places=6)
        self.assertTrue(result.constraints_satisfied)
        
        # Ensure weights sum to 1.0
        allocation_sum = sum(result.allocations.values())
        self.assertAlmostEqual(allocation_sum, 1.0, places=6)

    def test_allocate_capital_multiple_strategies(self):
        """Test allocating capital across multiple strategies"""
        result = self.manager.allocate_capital(
            self.strategies, 
            self.expected_returns
        )
        
        # Ensure all symbols are in the allocation
        expected_symbols = {'AAPL', 'MSFT', 'JNJ', 'PFE', 'CASH'}
        self.assertEqual(set(result.allocations.keys()), expected_symbols)
        
        # Ensure weights sum to 1.0
        allocation_sum = sum(result.allocations.values())
        self.assertAlmostEqual(allocation_sum, 1.0, places=6)
        
        # Check strategy allocations
        self.assertIsNotNone(result.strategy_allocations)
        strategy_allocation_sum = sum(result.strategy_allocations.values())
        self.assertAlmostEqual(strategy_allocation_sum, 0.95, places=6)  # Allow for cash buffer
        
        # Check that the cash allocation is correct
        self.assertAlmostEqual(result.allocations['CASH'], self.cash_buffer_pct, places=6)

    def test_allocate_capital_with_sector_mapping(self):
        """Test allocating capital with sector mapping"""
        result = self.manager.allocate_capital(
            self.strategies, 
            self.expected_returns,
            sector_mapping=self.sector_mapping
        )
        
        # Check sector allocations
        self.assertIsNotNone(result.sector_allocations)
        expected_sectors = {'Technology', 'Healthcare', 'Cash'}
        self.assertEqual(set(result.sector_allocations.keys()), expected_sectors)
        
        # Ensure sector weights sum to 1.0
        sector_sum = sum(result.sector_allocations.values())
        self.assertAlmostEqual(sector_sum, 1.0, places=6)
        
        # Check that cash sector allocation is correct
        self.assertAlmostEqual(result.sector_allocations['Cash'], self.cash_buffer_pct, places=6)

    @patch('models.portfolio.allocation_manager.logging.warning')
    def test_allocate_capital_with_constraints(self, mock_warning):
        """Test allocating capital with constraints"""
        # Create a mock constraint that's satisfied
        satisfied_constraint = MagicMock(spec=PortfolioConstraint)
        satisfied_constraint.is_satisfied.return_value = True
        
        # Create a mock constraint that's not satisfied initially but becomes satisfied after apply
        unsatisfied_constraint = MagicMock(spec=PortfolioConstraint)
        unsatisfied_constraint.is_satisfied.side_effect = [False, True]
        unsatisfied_constraint.apply.return_value = {'AAPL': 0.4, 'MSFT': 0.3, 'JNJ': 0.2, 'PFE': 0.05, 'CASH': 0.05}
        
        constraints = [satisfied_constraint, unsatisfied_constraint]
        
        result = self.manager.allocate_capital(
            self.strategies, 
            self.expected_returns,
            constraints=constraints,
            sector_mapping=self.sector_mapping
        )
        
        # Check that constraints were evaluated
        satisfied_constraint.is_satisfied.assert_called_once()
        unsatisfied_constraint.is_satisfied.assert_called()
        unsatisfied_constraint.apply.assert_called_once()
        
        # Check the updated allocations
        self.assertEqual(result.allocations, 
                         {'AAPL': 0.4, 'MSFT': 0.3, 'JNJ': 0.2, 'PFE': 0.05, 'CASH': 0.05})
        self.assertTrue(result.constraints_satisfied)
        
        # Check that no warning was logged
        mock_warning.assert_not_called()

    @patch('models.portfolio.allocation_manager.logging.warning')
    def test_allocate_capital_with_unsatisfiable_constraint(self, mock_warning):
        """Test allocating capital with a constraint that cannot be satisfied"""
        # Create a mock constraint that cannot be satisfied even after apply
        unsatisfiable_constraint = MagicMock(spec=PortfolioConstraint)
        unsatisfiable_constraint.is_satisfied.return_value = False
        unsatisfiable_constraint.apply.return_value = {'AAPL': 0.4, 'MSFT': 0.3, 'JNJ': 0.2, 'PFE': 0.05, 'CASH': 0.05}
        
        result = self.manager.allocate_capital(
            self.strategies, 
            self.expected_returns,
            constraints=[unsatisfiable_constraint]
        )
        
        # Check that constraint was evaluated and apply was called
        unsatisfiable_constraint.is_satisfied.assert_called()
        unsatisfiable_constraint.apply.assert_called_once()
        
        # Check that constraints_satisfied is False
        self.assertFalse(result.constraints_satisfied)
        
        # Check that a warning was logged
        mock_warning.assert_called_once()

    def test_allocate_capital_with_objective(self):
        """Test allocating capital with an objective function"""
        # Create a mock objective
        mock_objective = MagicMock(spec=PortfolioObjective)
        mock_objective.evaluate.return_value = 0.075  # Mocked objective value
        
        result = self.manager.allocate_capital(
            self.strategies, 
            self.expected_returns,
            objective=mock_objective
        )
        
        # Check that objective was evaluated
        mock_objective.evaluate.assert_called_once()
        
        # Check the objective value
        self.assertEqual(result.objective_value, 0.075)

    def test_allocate_by_expected_return(self):
        """Test the _allocate_by_expected_return method"""
        strategy_returns = {
            'strategy1': 0.075,  # (0.5 * 0.08) + (0.5 * 0.07)
            'strategy2': 0.047   # (0.7 * 0.05) + (0.3 * 0.04)
        }
        
        allocations = self.manager._allocate_by_expected_return(self.strategies, strategy_returns)
        
        # Calculate expected allocations
        allocatable_capital = self.total_capital * (1 - self.cash_buffer_pct)
        total_return = sum(strategy_returns.values())
        expected_s1 = (strategy_returns['strategy1'] / total_return) * allocatable_capital
        expected_s2 = (strategy_returns['strategy2'] / total_return) * allocatable_capital
        
        # Check allocations
        self.assertAlmostEqual(allocations['strategy1'], expected_s1, places=6)
        self.assertAlmostEqual(allocations['strategy2'], expected_s2, places=6)
        
        # Check that allocations sum to allocatable_capital
        allocation_sum = sum(allocations.values())
        self.assertAlmostEqual(allocation_sum, allocatable_capital, places=6)

    def test_allocate_by_expected_return_with_negative_returns(self):
        """Test allocating with negative expected returns"""
        strategy_returns = {
            'strategy1': -0.02,
            'strategy2': 0.05
        }
        
        allocations = self.manager._allocate_by_expected_return(self.strategies, strategy_returns)
        
        # Only strategy2 should get an allocation
        allocatable_capital = self.total_capital * (1 - self.cash_buffer_pct)
        self.assertAlmostEqual(allocations['strategy2'], allocatable_capital, places=6)
        self.assertNotIn('strategy1', allocations)

    def test_allocate_by_expected_return_all_negative(self):
        """Test allocating when all returns are negative"""
        strategy_returns = {
            'strategy1': -0.02,
            'strategy2': -0.03
        }
        
        allocations = self.manager._allocate_by_expected_return(self.strategies, strategy_returns)
        
        # Should use equal allocation
        allocatable_capital = self.total_capital * (1 - self.cash_buffer_pct)
        expected_allocation = allocatable_capital / len(self.strategies)
        
        self.assertAlmostEqual(allocations['strategy1'], expected_allocation, places=6)
        self.assertAlmostEqual(allocations['strategy2'], expected_allocation, places=6)

    @patch('models.portfolio.allocation_manager.AllocationManager._allocate_by_risk_budget')
    def test_allocate_to_strategies_with_risk_model(self, mock_allocate_by_risk_budget):
        """Test allocating to strategies with a risk model"""
        risk_model = MagicMock()
        mock_allocate_by_risk_budget.return_value = {
            'strategy1': 60000.0,
            'strategy2': 35000.0
        }
        
        allocations = self.manager._allocate_to_strategies(
            self.strategies,
            self.expected_returns,
            risk_model
        )
        
        # Check that risk budget allocation was used
        mock_allocate_by_risk_budget.assert_called_once_with(self.strategies, risk_model)
        self.assertEqual(allocations, mock_allocate_by_risk_budget.return_value)
    
    def test_allocate_to_strategies_apply_limits(self):
        """Test applying strategy limits to allocations"""
        # Create a manager with tight strategy limits
        manager = AllocationManager(
            total_capital=self.total_capital,
            cash_buffer_pct=self.cash_buffer_pct,
            strategy_limits={'strategy1': (0.2, 0.3), 'strategy2': (0.6, 0.7)}
        )
        
        # Strategy returns that would naturally allocate more to strategy1
        strategy_returns = {
            'strategy1': 0.10,  # Would get ~70% without limits
            'strategy2': 0.03   # Would get ~30% without limits
        }
        
        # Mock the inner allocation method
        manager._allocate_by_expected_return = MagicMock()
        allocatable_capital = self.total_capital * (1 - self.cash_buffer_pct)
        
        # Return allocations that exceed limits
        manager._allocate_by_expected_return.return_value = {
            'strategy1': 0.75 * allocatable_capital,  # Above max of 0.3
            'strategy2': 0.25 * allocatable_capital   # Below min of 0.6
        }
        
        allocations = manager._allocate_to_strategies(
            self.strategies,
            self.expected_returns
        )
        
        # Check that strategy limits were applied
        expected_s1 = 0.3 * allocatable_capital  # Max limit of 30%
        expected_s2 = 0.7 * allocatable_capital  # Min limit of 60% adjusted to 70% for normalization
        
        # Allow slight differences due to normalization
        self.assertAlmostEqual(allocations['strategy1'] / allocatable_capital, 0.3, places=6)
        self.assertAlmostEqual(allocations['strategy2'] / allocatable_capital, 0.7, places=6)
        
        # Sum should equal allocatable capital
        allocation_sum = sum(allocations.values())
        self.assertAlmostEqual(allocation_sum, allocatable_capital, places=6)


if __name__ == '__main__':
    unittest.main()