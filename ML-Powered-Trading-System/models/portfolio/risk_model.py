"""
Risk model module.

This module provides factor-based risk decomposition and risk forecasting capabilities
for portfolio management. It estimates and decomposes portfolio risk into
systematic and specific components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import scipy.linalg as linalg

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class RiskDecomposition:
    """Container for the results of a risk decomposition analysis."""
    total_risk: float
    systematic_risk: float
    specific_risk: float
    factor_contributions: Dict[str, float]
    asset_contributions: Dict[str, float]
    factor_exposures: Dict[str, Dict[str, float]]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def systematic_pct(self) -> float:
        """The percentage of risk from systematic factors."""
        if self.total_risk <= 0:
            return 0.0
        return (self.systematic_risk / self.total_risk) * 100.0

    @property
    def specific_pct(self) -> float:
        """The percentage of risk from specific (idiosyncratic) factors."""
        if self.total_risk <= 0:
            return 0.0
        return (self.specific_risk / self.total_risk) * 100.0


class RiskModel:
    """
    Factor-based risk model for portfolio risk decomposition and forecasting.

    The risk model decomposes asset returns into systematic (factor-driven)
    and specific (idiosyncratic) components, allowing for better risk management
    and portfolio construction.
    """

    def __init__(self,
                 factor_covariance: Optional[pd.DataFrame] = None,
                 factor_exposures: Optional[pd.DataFrame] = None,
                 specific_variances: Optional[Dict[str, float]] = None,
                 estimation_universe: Optional[List[str]] = None,
                 factor_names: Optional[List[str]] = None):
        """
        Initialize a risk model.

        Args:
            factor_covariance: Covariance matrix of factor returns
            factor_exposures: Asset exposures to risk factors (assets as rows, factors as columns)
            specific_variances: Asset-specific (idiosyncratic) variances
            estimation_universe: List of assets in the estimation universe
            factor_names: List of factor names
        """
        self.factor_covariance = factor_covariance
        self.factor_exposures = factor_exposures
        self.specific_variances = specific_variances or {}
        self.estimation_universe = estimation_universe or []
        self.factor_names = factor_names or []

    def estimate_from_returns(self,
                              returns: pd.DataFrame,
                              factors: pd.DataFrame,
                              min_history: int = 60,
                              shrinkage: float = 0.1) -> None:
        """
        Estimate the risk model parameters from historical returns.

        Args:
            returns: Asset returns with assets as columns, time as rows
            factors: Factor returns with factors as columns, time as rows
            min_history: Minimum number of periods required for estimation
            shrinkage: Covariance shrinkage parameter (0 = sample, 1 = shrinkage target)
        """
        if len(returns) < min_history:
            logger.warning(f"Insufficient history ({len(returns)} < {min_history})")
            return

        # Check for alignment between returns and factors
        if not returns.index.equals(factors.index):
            logger.warning("Returns and factors have misaligned indexes")
            common_index = returns.index.intersection(factors.index)
            if len(common_index) < min_history:
                logger.error("Insufficient aligned data points")
                return
            returns = returns.loc[common_index]
            factors = factors.loc[common_index]

        # Store factor names
        self.factor_names = list(factors.columns)

        # Store estimation universe
        self.estimation_universe = list(returns.columns)

        # Estimate factor covariance with shrinkage
        sample_cov = factors.cov()
        diag_average = np.mean(np.diag(sample_cov))
        shrinkage_target = np.diag(np.diag(sample_cov))  # Diagonal matrix

        # Apply shrinkage
        shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * shrinkage_target
        self.factor_covariance = pd.DataFrame(
            shrunk_cov,
            index=self.factor_names,
            columns=self.factor_names
        )

        # Estimate factor exposures via time-series regression
        self.factor_exposures = pd.DataFrame(index=self.estimation_universe, columns=self.factor_names)
        self.specific_variances = {}

        for asset in self.estimation_universe:
            if asset not in returns.columns:
                continue

            # Estimate factor loadings (betas) via OLS regression
            X = factors.values  # Factors
            y = returns[asset].values  # Asset returns

            # Add constant for intercept
            X_with_const = np.column_stack([np.ones(len(X)), X])

            try:
                # Solve normal equations
                beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]

                # First coefficient is intercept, rest are factor exposures
                alpha, exposures = beta[0], beta[1:]

                # Store factor exposures
                self.factor_exposures.loc[asset] = exposures

                # Calculate residuals
                y_pred = alpha + X @ exposures
                residuals = y - y_pred

                # Estimate specific variance
                specific_variance = np.var(residuals, ddof=len(self.factor_names) + 1)
                self.specific_variances[asset] = specific_variance

            except Exception as e:
                logger.error(f"Error estimating factor exposures for {asset}: {e}")
                # Use defaults if estimation fails
                self.factor_exposures.loc[asset] = 0.0
                self.specific_variances[asset] = returns[asset].var()

        logger.info(f"Risk model estimated with {len(self.estimation_universe)} assets and {len(self.factor_names)} factors")

    def get_asset_covariance_matrix(self, assets: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Construct the asset covariance matrix from the factor model.

        Args:
            assets: List of assets to include (defaults to estimation universe)

        Returns:
            Covariance matrix for the specified assets
        """
        if self.factor_covariance is None or self.factor_exposures is None:
            raise ValueError("Risk model not estimated yet")

        # Default to estimation universe if assets not specified
        assets = assets or self.estimation_universe

        # Filter factor exposures to the requested assets
        valid_assets = [a for a in assets if a in self.factor_exposures.index]

        if not valid_assets:
            raise ValueError("None of the requested assets are in the risk model")

        exposures_subset = self.factor_exposures.loc[valid_assets]

        # Systematic component: B * F * B'
        # B = factor exposures, F = factor covariance
        systematic_cov = exposures_subset @ self.factor_covariance @ exposures_subset.T

        # Specific component (diagonal matrix of specific variances)
        specific_cov = pd.DataFrame(0.0, index=valid_assets, columns=valid_assets)
        for asset in valid_assets:
            if asset in self.specific_variances:
                specific_cov.loc[asset, asset] = self.specific_variances[asset]

        # Total covariance is the sum of systematic and specific components
        asset_cov = systematic_cov + specific_cov

        return asset_cov

    def decompose_portfolio_risk(self,
                                weights: Dict[str, float]) -> RiskDecomposition:
        """
        Decompose the risk of a portfolio into factor and asset contributions.

        Args:
            weights: Dictionary mapping assets to portfolio weights

        Returns:
            RiskDecomposition object with risk decomposition analysis
        """
        if self.factor_covariance is None or self.factor_exposures is None:
            raise ValueError("Risk model not estimated yet")

        # Extract assets in the portfolio
        portfolio_assets = [a for a in weights.keys() if a != 'CASH' and a in self.factor_exposures.index]

        if not portfolio_assets:
            return RiskDecomposition(
                total_risk=0.0,
                systematic_risk=0.0,
                specific_risk=0.0,
                factor_contributions={},
                asset_contributions={},
                factor_exposures={}
            )

        # Create weight vector
        weight_vector = np.array([weights.get(a, 0.0) for a in portfolio_assets])

        # Get factor exposures for portfolio assets
        asset_exposures = self.factor_exposures.loc[portfolio_assets]

        # Calculate portfolio factor exposures (weighted sum of asset exposures)
        portfolio_factor_exposures = asset_exposures.T @ weight_vector

        # Calculate systematic risk
        systematic_variance = portfolio_factor_exposures @ self.factor_covariance @ portfolio_factor_exposures
        systematic_risk = np.sqrt(systematic_variance)

        # Calculate specific risk
        specific_variances = np.array([self.specific_variances.get(a, 0.0) for a in portfolio_assets])
        specific_variance = np.sum((weight_vector ** 2) * specific_variances)
        specific_risk = np.sqrt(specific_variance)

        # Calculate total risk
        total_variance = systematic_variance + specific_variance
        total_risk = np.sqrt(total_variance)

        # Calculate factor contributions to risk
        # Factor contribution = portfolio exposure to factor * factor's contribution to total risk
        factor_marginal_contributions = self.factor_covariance @ portfolio_factor_exposures
        factor_contributions = {}

        for i, factor in enumerate(self.factor_names):
            # Marginal contribution * exposure = total contribution
            factor_contributions[factor] = portfolio_factor_exposures[i] * factor_marginal_contributions[i] / total_risk if total_risk > 0 else 0.0

        # Calculate asset contributions to risk
        asset_contributions = {}

        for i, asset in enumerate(portfolio_assets):
            # Systematic component
            asset_systematic = 0.0
            for j, factor in enumerate(self.factor_names):
                asset_systematic += weight_vector[i] * asset_exposures.iloc[i, j] * factor_marginal_contributions[j]

            # Specific component
            asset_specific = (weight_vector[i] ** 2) * specific_variances[i]

            # Total contribution
            asset_contributions[asset] = (asset_systematic + asset_specific) / total_risk if total_risk > 0 else 0.0

        # Prepare factor exposures for each asset
        factor_exposures_dict = {}
        for asset in portfolio_assets:
            factor_exposures_dict[asset] = {}
            for factor in self.factor_names:
                factor_exposures_dict[asset][factor] = self.factor_exposures.loc[asset, factor]

        return RiskDecomposition(
            total_risk=total_risk,
            systematic_risk=systematic_risk,
            specific_risk=specific_risk,
            factor_contributions=factor_contributions,
            asset_contributions=asset_contributions,
            factor_exposures=factor_exposures_dict
        )

    def get_risk_report(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate a comprehensive risk report for a portfolio.

        Args:
            weights: Dictionary mapping assets to portfolio weights

        Returns:
            Dictionary with risk metrics and analysis
        """
        try:
            # Get risk decomposition
            decomposition = self.decompose_portfolio_risk(weights)

            # Calculated value at risk (VaR) assuming normal distribution
            # Using 95% confidence interval (1.645 standard deviations)
            confidence_level = 0.95
            z_score = 1.645

            # Assume a portfolio value of 1.0 for simplicity (will be scaled by actual value)
            var_95 = z_score * decomposition.total_risk

            # Prepare factor exposures summary
            factor_exposures_summary = {}
            for factor in self.factor_names:
                # Get weighted average exposure to this factor
                total_exposure = 0.0
                total_weight = 0.0

                for asset, weight in weights.items():
                    if asset != 'CASH' and asset in self.factor_exposures.index:
                        total_exposure += weight * self.factor_exposures.loc[asset, factor]
                        total_weight += weight

                factor_exposures_summary[factor] = total_exposure / total_weight if total_weight > 0 else 0.0

            # Identify top risk contributors
            sorted_asset_contributions = sorted(
                decomposition.asset_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            top_asset_contributors = {k: v for k, v in sorted_asset_contributions[:5]}

            sorted_factor_contributions = sorted(
                decomposition.factor_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            top_factor_contributors = {k: v for k, v in sorted_factor_contributions[:5]}

            # Prepare risk report
            report = {
                "total_risk": decomposition.total_risk,
                "systematic_risk": decomposition.systematic_risk,
                "specific_risk": decomposition.specific_risk,
                "systematic_pct": decomposition.systematic_pct,
                "specific_pct": decomposition.specific_pct,
                "value_at_risk_95": var_95,
                "factor_exposures": factor_exposures_summary,
                "top_asset_contributors": top_asset_contributors,
                "top_factor_contributors": top_factor_contributors,
                "timestamp": datetime.now()
            }

            return report

        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now()
            }

    def stress_test_portfolio(self,
                             weights: Dict[str, float],
                             factor_shocks: Dict[str, float]) -> Dict[str, float]:
        """
        Perform stress testing by applying shocks to risk factors.

        Args:
            weights: Dictionary mapping assets to portfolio weights
            factor_shocks: Dictionary mapping factors to shock values (in standard deviations)

        Returns:
            Dictionary with stress test results
        """
        if self.factor_covariance is None or self.factor_exposures is None:
            raise ValueError("Risk model not estimated yet")

        # Extract assets in the portfolio
        portfolio_assets = [a for a in weights.keys() if a != 'CASH' and a in self.factor_exposures.index]

        if not portfolio_assets:
            return {"expected_loss": 0.0}

        # Create weight vector
        weight_vector = np.array([weights.get(a, 0.0) for a in portfolio_assets])

        # Get factor exposures for portfolio assets
        asset_exposures = self.factor_exposures.loc[portfolio_assets]

        # Calculate factor returns from shocks
        factor_returns = {}
        for factor, shock in factor_shocks.items():
            if factor in self.factor_names:
                # Convert shock in standard deviations to return
                factor_volatility = np.sqrt(self.factor_covariance.loc[factor, factor])
                factor_returns[factor] = shock * factor_volatility

        # Calculate expected asset returns under stress
        asset_returns = {}

        for i, asset in enumerate(portfolio_assets):
            # Initialize with zero
            asset_returns[asset] = 0.0

            # Add factor-driven component
            for factor, ret in factor_returns.items():
                factor_idx = self.factor_names.index(factor)
                exposure = asset_exposures.iloc[i, factor_idx]
                asset_returns[asset] += exposure * ret

        # Calculate portfolio loss
        portfolio_return = sum(weights.get(a, 0.0) * asset_returns.get(a, 0.0) for a in portfolio_assets)
        expected_loss = -portfolio_return  # Convert to loss (positive means loss)

        # Calculate component contributions
        factor_contributions = {}
        for factor in factor_shocks.keys():
            if factor in self.factor_names:
                # Calculate how much this factor contributed to the overall loss
                factor_contribution = 0.0
                for i, asset in enumerate(portfolio_assets):
                    factor_idx = self.factor_names.index(factor)
                    exposure = asset_exposures.iloc[i, factor_idx]
                    weight = weights.get(asset, 0.0)
                    factor_contribution += weight * exposure * factor_returns[factor]

                factor_contributions[factor] = -factor_contribution  # Convert to loss contribution

        # Prepare results
        results = {
            "expected_loss": expected_loss,
            "factor_contributions": factor_contributions,
            "asset_returns": asset_returns
        }

        return results

    def save(self, file_path: str) -> None:
        """
        Save the risk model to a file.

        Args:
            file_path: Path to save the model
        """
        model_data = {
            "factor_covariance": self.factor_covariance.to_dict() if self.factor_covariance is not None else None,
            "factor_exposures": self.factor_exposures.to_dict() if self.factor_exposures is not None else None,
            "specific_variances": self.specific_variances,
            "estimation_universe": self.estimation_universe,
            "factor_names": self.factor_names
        }

        try:
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Risk model saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving risk model: {e}")

    @classmethod
    def load(cls, file_path: str) -> 'RiskModel':
        """
        Load a risk model from a file.

        Args:
            file_path: Path to load the model from

        Returns:
            Loaded RiskModel instance
        """
        try:
            import pickle
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)

            # Reconstruct DataFrames
            factor_covariance = None
            if model_data["factor_covariance"]:
                factor_covariance = pd.DataFrame.from_dict(model_data["factor_covariance"])

            factor_exposures = None
            if model_data["factor_exposures"]:
                factor_exposures = pd.DataFrame.from_dict(model_data["factor_exposures"])

            # Create and return model
            model = cls(
                factor_covariance=factor_covariance,
                factor_exposures=factor_exposures,
                specific_variances=model_data["specific_variances"],
                estimation_universe=model_data["estimation_universe"],
                factor_names=model_data["factor_names"]
            )

            logger.info(f"Risk model loaded from {file_path}")
            return model

        except Exception as e:
            logger.error(f"Error loading risk model: {e}")
            return cls()  # Return empty model on error