#!/usr/bin/env python
"""
download_historical.py - Historical Market Data Downloader

This script downloads historical market data for specified symbols and timeframes
and stores it in the system's historical repository for regime detection and model training.

Usage:
    python -m scripts.download_historical [options]

Options:
    --symbols SYMBOLS       Comma-separated list of symbols to download
    --timeframes TIMEFRAMES Comma-separated list of timeframes to download (e.g., 1m,1h,1d)
    --start-date START_DATE Start date for data (YYYY-MM-DD)
    --end-date END_DATE     End date for data (YYYY-MM-DD)
    --source SOURCE         Data source (exchange, vendor)
    --config CONFIG         Path to configuration file
    --parallel WORKERS      Number of parallel download workers (default: 4)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

# Import required components
from data.fetchers.exchange_connector import ExchangeConnector, MarketType, DataFrequency, DataType
from data.fetchers.historical_repository import HistoricalRepository
from data.processors.data_normalizer import normalize_market_data
from data.processors.data_integrity import DataIntegrityChecker
from core.event_bus import get_event_bus, EventTopics, create_event, EventPriority
from execution.exchange.connectivity_manager import ConnectivityManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(project_root, "logs", "data_download.log"))
    ]
)

logger = logging.getLogger("historical_downloader")


async def initialize_components(config_path: str) -> Dict[str, Any]:
    """
    Initialize required components.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary of initialized components
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    components = {}
    
    # Initialize event bus
    event_bus = get_event_bus()
    components['event_bus'] = event_bus
    
    # Initialize connectivity manager
    connectivity_manager = ConnectivityManager(config.get('connectivity', {}))
    components['connectivity_manager'] = connectivity_manager
    
    # Initialize exchange connector
    exchange_config = config.get('exchange_connector', {})
    exchange_connector = ExchangeConnector(
        connectivity_manager=connectivity_manager,
        data_normalizer=None,  # Will be set later
        event_bus=event_bus,
        config=exchange_config
    )
    components['exchange_connector'] = exchange_connector
    
    # Initialize historical repository
    repo_config = config.get('historical_repository', {})
    historical_repository = HistoricalRepository(
        time_series_store=None,  # Will be set with mock for this script
        data_normalizer=None,    # Will be set later
        event_bus=event_bus,
        config=repo_config
    )
    components['historical_repository'] = historical_repository
    
    # Initialize data integrity checker
    data_integrity_checker = DataIntegrityChecker()
    components['data_integrity_checker'] = data_integrity_checker
    
    logger.info("Components initialized")
    return components


async def download_symbol_data(
    components: Dict[str, Any],
    symbol: str,
    timeframes: List[str],
    start_date: datetime,
    end_date: datetime,
    source: str,
    market_type: MarketType
) -> Dict[str, Any]:
    """
    Download data for a single symbol across multiple timeframes.
    
    Args:
        components: Dictionary of components
        symbol: Symbol to download
        timeframes: List of timeframes
        start_date: Start date
        end_date: End date
        source: Data source name
        market_type: Market type
        
    Returns:
        Dictionary with download statistics for this symbol
    """
    exchange_connector = components['exchange_connector']
    historical_repository = components['historical_repository']
    data_integrity_checker = components['data_integrity_checker']
    
    stats = {
        "symbol": symbol,
        "total_timeframes": len(timeframes),
        "successful_downloads": 0,
        "failed_downloads": 0,
        "total_bars": 0,
        "data_integrity_issues": 0,
        "start_time": datetime.now().isoformat()
    }
    
    # Convert timeframe strings to DataFrequency enums
    timeframe_enums = []
    for tf in timeframes:
        try:
            timeframe_enums.append(DataFrequency(tf))
        except ValueError:
            logger.warning(f"Invalid timeframe '{tf}', skipping")
    
    # Process each timeframe
    for timeframe in timeframe_enums:
        logger.info(f"Downloading {symbol} {timeframe.value} data from {start_date} to {end_date}")
        
        try:
            # Download data
            data = await exchange_connector.get_historical_data(
                instrument=symbol,
                frequency=timeframe,
                start_time=start_date,
                end_time=end_date,
                data_type=DataType.OHLCV,
                market_type=market_type,
                provider=source
            )
            
            # Check if data was returned
            if data is None or isinstance(data, pd.DataFrame) and data.empty:
                logger.warning(f"No data returned for {symbol} {timeframe.value}")
                stats["failed_downloads"] += 1
                continue
            
            # Ensure we have a DataFrame (API might return dict for multiple instruments)
            if isinstance(data, dict):
                if symbol in data:
                    df = data[symbol]
                else:
                    logger.warning(f"Symbol {symbol} not found in returned data")
                    stats["failed_downloads"] += 1
                    continue
            else:
                df = data
            
            # Check data integrity
            integrity_issues = data_integrity_checker.check(df)
            if integrity_issues:
                logger.warning(f"Data integrity issues for {symbol} {timeframe.value}: {integrity_issues}")
                stats["data_integrity_issues"] += len(integrity_issues)
                
                # Fix data issues
                df = data_integrity_checker.fix(df)
            
            # Normalize data format
            df = normalize_market_data(df, symbol, timeframe.value)
            
            # Save to repository
            success = await historical_repository.store_historical_data(
                data=df,
                instrument=symbol,
                frequency=timeframe.value,
                data_type=DataType.OHLCV,
                market_type=market_type,
                source=source
            )
            
            if success:
                logger.info(f"Successfully saved {len(df)} bars for {symbol} {timeframe.value}")
                stats["successful_downloads"] += 1
                stats["total_bars"] += len(df)
            else:
                logger.error(f"Failed to save data for {symbol} {timeframe.value}")
                stats["failed_downloads"] += 1
            
        except Exception as e:
            logger.error(f"Error downloading {symbol} {timeframe.value}: {str(e)}", exc_info=True)
            stats["failed_downloads"] += 1
    
    stats["end_time"] = datetime.now().isoformat()
    duration = datetime.fromisoformat(stats["end_time"]) - datetime.fromisoformat(stats["start_time"])
    stats["duration_seconds"] = duration.total_seconds()
    
    return stats


async def download_data_parallel(
    components: Dict[str, Any],
    symbols: List[str],
    timeframes: List[str],
    start_date: datetime,
    end_date: datetime,
    source: str,
    market_type: MarketType,
    max_workers: int
) -> List[Dict[str, Any]]:
    """
    Download data for multiple symbols in parallel.
    
    Args:
        components: Dictionary of components
        symbols: List of symbols to download
        timeframes: List of timeframes
        start_date: Start date
        end_date: End date
        source: Data source name
        market_type: Market type
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of download statistics for each symbol
    """
    logger.info(f"Starting parallel download for {len(symbols)} symbols with {max_workers} workers")
    
    # Create task list
    tasks = []
    for symbol in symbols:
        task = download_symbol_data(
            components=components,
            symbol=symbol,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            source=source,
            market_type=market_type
        )
        tasks.append(task)
    
    # Execute tasks in parallel
    results = []
    
    # Process in chunks to avoid overwhelming the system
    chunk_size = max(1, min(max_workers, len(tasks)))
    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i:i+chunk_size]
        chunk_results = await asyncio.gather(*chunk, return_exceptions=True)
        
        for result in chunk_results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with error: {str(result)}")
            else:
                results.append(result)
    
    return results


async def main_async():
    """Async main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download historical market data')
    parser.add_argument('--symbols', type=str, required=True,
                      help='Comma-separated list of symbols')
    parser.add_argument('--timeframes', type=str, default='1h,1d',
                      help='Comma-separated list of timeframes')
    parser.add_argument('--start-date', type=str, required=True,
                      help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                      help='End date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--source', type=str, default='binance',
                      help='Data source name')
    parser.add_argument('--market-type', type=str, default='crypto',
                      choices=['crypto', 'equity', 'forex', 'futures'],
                      help='Market type')
    parser.add_argument('--config', type=str, default='config/data_config.json',
                      help='Path to configuration file')
    parser.add_argument('--parallel', type=int, default=4,
                      help='Number of parallel download workers')
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()
    
    # Parse symbols and timeframes
    symbols = [s.strip() for s in args.symbols.split(',')]
    timeframes = [tf.strip() for tf in args.timeframes.split(',')]
    
    # Parse market type
    market_type_map = {
        'crypto': MarketType.CRYPTO,
        'equity': MarketType.EQUITY,
        'forex': MarketType.FOREX,
        'futures': MarketType.FUTURES
    }
    market_type = market_type_map.get(args.market_type, MarketType.CRYPTO)
    
    # Initialize components
    components = await initialize_components(args.config)
    
    try:
        # Calculate data requirements
        days = (end_date - start_date).days
        approx_bars_per_symbol = sum(
            days // {'1m': 1/1440, '5m': 1/288, '15m': 1/96, '30m': 1/48, 
                   '1h': 1/24, '4h': 1/6, '1d': 1, '1w': 7}.get(tf, 1) 
            for tf in timeframes
        )
        total_bars_estimate = len(symbols) * approx_bars_per_symbol
        
        logger.info(f"Starting historical data download for {len(symbols)} symbols "
                  f"and {len(timeframes)} timeframes")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()} ({days} days)")
        logger.info(f"Estimated bars to download: {total_bars_estimate:,.0f}")
        logger.info(f"Using {args.parallel} parallel workers")
        
        # Start timer
        start_time = time.time()
        
        # Download data in parallel
        results = await download_data_parallel(
            components=components,
            symbols=symbols,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            source=args.source,
            market_type=market_type,
            max_workers=args.parallel
        )
        
        # Calculate overall statistics
        total_successful = sum(r["successful_downloads"] for r in results)
        total_failed = sum(r["failed_downloads"] for r in results)
        total_bars = sum(r["total_bars"] for r in results)
        total_issues = sum(r["data_integrity_issues"] for r in results)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        overall_stats = {
            "total_symbols": len(symbols),
            "total_timeframes": len(timeframes),
            "successful_downloads": total_successful,
            "failed_downloads": total_failed,
            "total_bars": total_bars,
            "data_integrity_issues": total_issues,
            "elapsed_time_seconds": elapsed_time,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "source": args.source,
            "market_type": args.market_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Print summary
        logger.info("Download complete!")
        logger.info(f"Successfully downloaded {total_successful}/{total_successful + total_failed} timeframe-symbol combinations")
        logger.info(f"Total bars downloaded: {total_bars:,}")
        logger.info(f"Data integrity issues fixed: {total_issues}")
        logger.info(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        # Save stats to file
        stats_dir = Path(project_root, "logs", "download_stats")
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_file = stats_dir / f"download_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(stats_file, 'w') as f:
            json.dump({
                "overall": overall_stats,
                "symbols": results
            }, f, indent=2, default=str)
        
        logger.info(f"Statistics saved to {stats_file}")
        
        # Create data ready file for other components to check
        ready_file = Path(project_root, "data", "historical", "data_ready.json")
        with open(ready_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "symbols": symbols,
                "timeframes": timeframes,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_bars": total_bars
            }, f, indent=2)
        
        logger.info(f"Data ready file created at {ready_file}")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()