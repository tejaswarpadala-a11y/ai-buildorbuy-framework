"""
Configuration constants for Bank Rage Classifier.

Centralizes paths, hyperparameters, and model configurations.
"""

# Model configurations
MODEL_NAME = "roberta-base"
MAX_SEQUENCE_LENGTH = 256
NUM_LABELS = 7

# Training hyperparameters (best configuration from TA-9)
BEST_LEARNING_RATE = 2e-5
BEST_BATCH_SIZE = 16
BEST_NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01

# Data columns
TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Label"
DATE_COLUMN = "Date received"

# Data splits
TRAIN_SIZE = 12000
TEST_SIZE = 3000
HOLDOUT_SIZE = 1000
RANDOM_STATE = 42

# Deployment configuration
CONFIDENCE_THRESHOLD = 0.70  # 95% of predictions above this threshold
HIGH_CONFIDENCE_ROUTING = 0.95  # Route 95% automatically
LOW_CONFIDENCE_REVIEW = 0.05  # 5% to human review

# Metrics thresholds
MIN_ACCURACY = 0.85
MIN_MACRO_F1 = 0.75
MIN_PER_CLASS_F1 = 0.60

# GenAI model names (for benchmarking)
GENAI_MODELS = {
    "gpt-4o": "gpt-4o",
    "gpt-4-turbo": "gpt-4-turbo",
    "claude-sonnet-4": "claude-sonnet-4",
    "claude-haiku-4.5": "claude-haiku-4.5",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-flash-lite": "gemini-flash-lite"
}

# Cost estimates (per 10k texts, as of Feb 2026)
GENAI_COSTS_PER_10K = {
    "gpt-4o": 120.0,
    "gpt-4-turbo": 54.43,
    "claude-sonnet-4": 73.49,
    "claude-haiku-4.5": 57.58,
    "gemini-2.5-flash": 164.0,
    "gemini-flash-lite": 51.20,
    "specialist": 0.0  # Near-zero marginal cost after training
}

# Label priorities (from codebook)
LABEL_PRIORITIES = {
    "Fraud / Security Issue": "Critical",
    "Hidden Fees": "High",
    "Credit Reporting Error": "High",
    "Loan/Mortgage Servicing Issue": "Medium",
    "Account Access / Administration": "Medium",
    "Process Failure / Red Tape": "Low",
    "Other": "Triage"
}

# Evaluation metrics to track
REQUIRED_METRICS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "mcc",
    "per_class_f1"
]

# File naming conventions
MODEL_CHECKPOINT_PREFIX = "specialist_roberta"
PREDICTIONS_FILE_SUFFIX = "_predictions.csv"
METRICS_FILE_SUFFIX = "_metrics.json"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
