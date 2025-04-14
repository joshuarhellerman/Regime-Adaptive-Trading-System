#!/usr/bin/env python3
"""
Test runner script for ML-powered trading system.
"""
import argparse
import os
import sys
import subprocess


def run_tests(category=None, module=None, verbose=False, coverage=False):
    """
    Run tests with the specified configuration.

    Args:
        category: Test category (unit, integration, performance)
        module: Specific module to test
        verbose: Whether to show verbose output
        coverage: Whether to generate coverage report
    """
    cmd = ["pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.append("--cov=.")
        cmd.append("--cov-report=term")
        cmd.append("--cov-report=html")

    if category:
        if module:
            cmd.append(f"tests/{category}/{module}")
        else:
            cmd.append(f"tests/{category}")
    elif module:
        cmd.append(f"tests/unit/{module}")
        cmd.append(f"tests/integration/*{module}*")

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for ML-powered trading system")
    parser.add_argument("--category", choices=["unit", "integration", "performance"],
                        help="Test category to run")
    parser.add_argument("--module", help="Specific module to test (e.g., core, data)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")

    args = parser.parse_args()
    run_tests(args.category, args.module, args.verbose, args.coverage)
