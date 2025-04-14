"""
Main Entry Point

This is the main entry point for the ML-Powered Trading System.
It initializes all components and starts the system.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.system_config import load_config
from core.system import System
from utils.logger import setup_logging

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ML-Powered Trading System")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["paper", "live"],
                       default="paper", help="Trading mode")
    parser.add_argument("--log-level", type=str, 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       default="INFO", help="Logging level")
    return parser.parse_args()

def main():
    """Main function to start the trading system."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting ML-Powered Trading System in {args.mode} mode")

    try:
        # Load configuration
        config = load_config(args.config)

        # Initialize and start the system
        system = System(config, mode=args.mode)
        system.initialize()
        system.start()

        # Keep the main thread running
        system.join()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
        if 'system' in locals():
            system.shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        if 'system' in locals():
            system.emergency_shutdown()
        sys.exit(1)

    logger.info("System shutdown complete")

if __name__ == "__main__":
    main()
