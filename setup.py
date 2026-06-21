from setuptools import setup, find_packages

setup(
    name="fraud_intelligence_ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.6.0",
        "pyod>=1.0.0",
        "imbalanced-learn>=0.9.0",
        "networkx>=2.8.0",
        "pyyaml>=6.0",
        "fastapi>=0.78.0",
        "uvicorn>=0.17.0",
        "mlflow>=1.26.0",
        "pydantic>=1.9.0",
        "httpx>=0.23.0",
        "torch>=1.11.0",
    ],
    author="Antigravity",
    description="Fraud Intelligence AI System - Real-time fraud detection and explanation.",
)
