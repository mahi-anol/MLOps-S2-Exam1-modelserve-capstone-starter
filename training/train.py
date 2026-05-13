#!/usr/bin/env python3
"""
ModelServe — Model Training Entrypoint

This is the reproducible training script required by the exam.
It wraps the src pipeline's model training module.

Usage:
    python training/train.py

Prerequisites:
    - MLflow and Postgres must be running (docker compose up postgres mlflow)
    - Feature engineering must have completed (data/features/ exists)
    - Or run the full pipeline: dvc repro
"""

import sys
from pathlib import Path

# Ensure project root is on the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.Pipelines.Model_Training.model_training import ModelTrainer


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
