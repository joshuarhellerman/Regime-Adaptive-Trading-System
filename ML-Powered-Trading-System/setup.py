"""
Setup file for ml_trading_system
"""

from setuptools import setup, find_packages

setup(
    name="ml_trading_system",
    version="0.1.0",
    description="ML-Powered Trading System",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "requests",
        "websockets",
        "fastapi",
        "uvicorn",
        "pytest",
        "pyarrow",
        "dash",
        "plotly",
        "xgboost",
        "lightgbm",
        "ccxt",
        "joblib",
    ],
    python_requires=">=3.8",
)
