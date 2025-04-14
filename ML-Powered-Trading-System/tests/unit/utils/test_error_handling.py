"""
Tests for the error handling utilities in the ML-powered trading system.

This module contains unit tests for the error handling functions, classes, and
mechanisms used throughout the system to ensure proper error management,
logging, recovery strategies, and user feedback.
"""
import os
import pytest
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the module under test
# Adjust imports based on your project structure
from ...utils import error_handling
from ...config.base_config import BaseConfig, ConfigError
from ...config.system_config import SystemConfig


class TestErrorCapturing:
    """Tests for error capturing and context management functionality."""

    def test_error_context_manager_captures_exceptions(self):
        """Test that ErrorContext captures and logs exceptions properly."""
        with patch.object(error_handling, 'logger') as mock_logger:
            with pytest.raises(ValueError):
                with error_handling.ErrorContext("test_operation"):
                    raise ValueError("Test error")

            # Verify the error was logged
            mock_logger.error.assert_called_once()
            assert "test_operation" in mock_logger.error.call_args[0][0]
            assert "ValueError" in mock_logger.error.call_args[0][0]

    def test_error_context_manager_with_recovery(self):
        """Test that ErrorContext can execute recovery functions."""
        recovery_mock = MagicMock()

        with patch.object(error_handling, 'logger'):
            with error_handling.ErrorContext("test_operation", on_error=recovery_mock):
                raise ValueError("Test error")

        # Verify recovery function was called
        recovery_mock.assert_called_once()

    def test_error_context_manager_success_case(self):
        """Test that ErrorContext works correctly when no exception occurs."""
        with patch.object(error_handling, 'logger') as mock_logger:
            with error_handling.ErrorContext("test_operation"):
                # No exception here
                pass

            # Verify no error was logged
            mock_logger.error.assert_not_called()


class TestConfigErrorHandling:
    """Tests for configuration-related error handling."""

    def test_config_validation_error_handling(self):
        """Test that ConfigError is raised for invalid configuration values."""
        invalid_config = {"invalid_key": "invalid_value"}

        with pytest.raises(ConfigError) as exc_info:
            error_handling.validate_config_values(SystemConfig, invalid_config)

        assert "Configuration validation failed" in str(exc_info.value)

    def test_config_file_not_found_handling(self):
        """Test handling of missing configuration files."""
        non_existent_path = Path("/path/does/not/exist.yaml")

        with patch.object(error_handling, 'logger') as mock_logger:
            result = error_handling.safe_load_config(non_existent_path, SystemConfig)

        # Should log warning and return default config
        mock_logger.warning.assert_called_once()
        assert isinstance(result, SystemConfig)

    def test_config_syntax_error_handling(self):
        """Test handling of syntax errors in configuration files."""
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "invalid: yaml: content:"

            with patch.object(error_handling, 'logger') as mock_logger:
                with pytest.raises(ConfigError):
                    error_handling.load_config_file(Path("dummy.yaml"), SystemConfig)

            # Verify error was logged
            mock_logger.error.assert_called_once()


class TestAPIErrorHandling:
    """Tests for API-related error handling."""

    def test_api_error_retry_mechanism(self):
        """Test the retry mechanism for API errors."""
        mock_func = MagicMock(side_effect=[
            error_handling.APIError("Rate limit exceeded"),
            error_handling.APIError("Rate limit exceeded"),
            "success"
        ])

        with patch.object(error_handling, 'logger'):
            result = error_handling.retry_on_api_error(
                mock_func, max_retries=3, retry_delay=0.01
            )

        # Function should be called 3 times and eventually succeed
        assert mock_func.call_count == 3
        assert result == "success"

    def test_api_error_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded for API errors."""
        mock_func = MagicMock(side_effect=error_handling.APIError("Persistent error"))

        with patch.object(error_handling, 'logger'):
            with pytest.raises(error_handling.MaxRetriesExceededError):
                error_handling.retry_on_api_error(
                    mock_func, max_retries=3, retry_delay=0.01
                )

        # Function should be called exactly max_retries times
        assert mock_func.call_count == 3


class TestDatabaseErrorHandling:
    """Tests for database-related error handling."""

    def test_database_connection_error_handling(self):
        """Test handling of database connection errors."""
        with patch.object(error_handling, 'logger') as mock_logger:
            with pytest.raises(error_handling.DatabaseError):
                with error_handling.database_connection_handler():
                    raise error_handling.DatabaseConnectionError("Connection failed")

        # Verify error was logged
        mock_logger.error.assert_called_once()

    def test_database_transaction_error_handling(self):
        """Test handling of database transaction errors."""
        mock_db = MagicMock()

        with patch.object(error_handling, 'logger'):
            with pytest.raises(error_handling.DatabaseError):
                with error_handling.database_transaction_handler(mock_db):
                    raise ValueError("Transaction error")

        # Verify transaction was rolled back
        mock_db.rollback.assert_called_once()


class TestDataProcessingErrorHandling:
    """Tests for data processing error handling."""

    def test_data_validation_error_handling(self):
        """Test handling of data validation errors."""
        invalid_data = {"price": "not_a_number"}

        with pytest.raises(error_handling.DataValidationError):
            error_handling.validate_market_data(invalid_data)

    def test_missing_data_error_handling(self):
        """Test handling of missing data errors."""
        incomplete_data = {"timestamp": 1617979142}  # Missing price

        result = error_handling.process_with_missing_data_handler(
            incomplete_data, required_fields=["timestamp", "price", "volume"]
        )

        # Should fill missing fields with None or defaults
        assert "price" in result
        assert result["price"] is None
        assert "volume" in result
        assert result["volume"] is None


class TestErrorLogging:
    """Tests for error logging functionality."""

    def test_error_logger_configuration(self):
        """Test that error logger is properly configured."""
        logger = error_handling.get_error_logger()

        # Check logger level
        assert logger.level == logging.ERROR

        # Check handlers
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)

    def test_error_report_generation(self):
        """Test generation of detailed error reports."""
        error = ValueError("Test error")
        context = {"operation": "test_operation", "data_id": "123"}

        report = error_handling.generate_error_report(error, context)

        # Verify report content
        assert "ValueError" in report
        assert "Test error" in report
        assert "test_operation" in report
        assert "data_id: 123" in report

    def test_error_notification(self):
        """Test error notification mechanism."""
        with patch.object(error_handling, 'send_error_notification') as mock_notify:
            error_handling.notify_critical_error("System failure", {"component": "test"})

            # Verify notification was sent
            mock_notify.assert_called_once()
            assert "System failure" in mock_notify.call_args[0][0]


class TestSystemErrorRecovery:
    """Tests for system error recovery mechanisms."""

    def test_trading_error_recovery(self):
        """Test recovery from trading errors."""
        mock_strategy = MagicMock()

        with patch.object(error_handling, 'logger'):
            error_handling.handle_trading_error(
                ValueError("Trading error"),
                strategy=mock_strategy,
                position_id="pos123"
            )

        # Verify recovery action was taken
        mock_strategy.close_position.assert_called_once_with("pos123")

    def test_model_prediction_error_recovery(self):
        """Test recovery from model prediction errors."""
        mock_model = MagicMock()
        mock_fallback = MagicMock()

        with patch.object(error_handling, 'logger'):
            error_handling.handle_prediction_error(
                ValueError("Prediction error"),
                model=mock_model,
                fallback_model=mock_fallback,
                input_data={"feature1": 0.5}
            )

        # Verify model was reset and fallback was used
        mock_model.reset.assert_called_once()
        mock_fallback.predict.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])