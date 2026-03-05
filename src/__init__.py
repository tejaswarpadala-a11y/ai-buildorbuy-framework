"""
Bank Rage Classifier: Root Cause Classification for CFPB Complaints.

A production-ready implementation of fine-tuned RoBERTa specialist model
for classifying consumer financial complaints into operational root causes.
"""

__version__ = "1.0.0"
__author__ = "Teja Padala"

from . import data
from . import models
from . import evaluation
from . import utils

__all__ = ["data", "models", "evaluation", "utils"]
