"""
Data preprocessing utilities for CFPB complaint classification.

Based on DataCleaning__Descriptive_Analytics notebook.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_cfpb_data(
    file_path: str,
    text_column: str = "Consumer complaint narrative",
    date_column: str = "Date received"
) -> pd.DataFrame:
    """
    Load CFPB complaint data from CSV.
    
    Args:
        file_path: Path to CSV file
        text_column: Column containing complaint text
        date_column: Column containing date received
        
    Returns:
        DataFrame with loaded data
    """
    df = pd.read_csv(file_path, low_memory=False)
    
    # Parse dates
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    return df


def clean_complaint_text(df: pd.DataFrame, text_column: str = "Consumer complaint narrative") -> pd.DataFrame:
    """
    Clean complaint text by removing empties and duplicates.
    
    Args:
        df: Input DataFrame
        text_column: Column containing text to clean
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Remove missing values
    df = df.dropna(subset=[text_column])
    
    # Remove empty strings
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column] != ""]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def remove_duplicates(df: pd.DataFrame, text_column: str = "Consumer complaint narrative") -> pd.DataFrame:
    """
    Remove exact duplicate complaints.
    
    Args:
        df: Input DataFrame
        text_column: Column to check for duplicates
        
    Returns:
        DataFrame with duplicates removed
    """
    df = df.copy()
    
    # Drop exact duplicates based on text
    before_count = len(df)
    df = df.drop_duplicates(subset=[text_column], keep='first')
    after_count = len(df)
    
    removed = before_count - after_count
    print(f"Removed {removed} duplicates ({removed/before_count*100:.1f}%)")
    
    df = df.reset_index(drop=True)
    return df


def get_data_summary(df: pd.DataFrame, text_column: str = "Consumer complaint narrative", name: str = "Dataset") -> dict:
    """
    Generate summary statistics for complaint data.
    
    Based on summary_row function from DataCleaning notebook.
    
    Args:
        df: Input DataFrame
        text_column: Column containing complaint text
        name: Name for this dataset
        
    Returns:
        Dictionary with summary statistics
    """
    s = df[text_column].astype(str).fillna("").str.strip()
    
    # Word and character counts
    words = s.str.split().str.len()
    chars = s.str.len()
    
    # Duplicates
    exact_dups = int(s.duplicated(keep="first").sum())
    unique_texts = int(s.nunique())
    dup_rate = round(100 * exact_dups / max(len(df), 1), 1)
    empty_texts = int((s == "").sum())
    
    summary = {
        "name": name,
        "rows": len(df),
        "words_mean": round(words.mean(), 1),
        "words_median": int(words.median()),
        "words_p95": int(words.quantile(0.95)),
        "chars_mean": round(chars.mean(), 1),
        "chars_median": int(chars.median()),
        "chars_p95": int(chars.quantile(0.95)),
        "unique_texts": unique_texts,
        "exact_duplicates": exact_dups,
        "duplicate_rate_pct": dup_rate,
        "empty_texts": empty_texts
    }
    
    return summary


def prepare_for_labeling(
    df: pd.DataFrame,
    text_column: str = "Consumer complaint narrative",
    output_column: str = "sentence"
) -> pd.DataFrame:
    """
    Prepare DataFrame for labeling (GenAI or human).
    
    Args:
        df: Input DataFrame
        text_column: Source column with complaint text
        output_column: Target column name for cleaned text
        
    Returns:
        DataFrame with prepared text column
    """
    df = df.copy()
    df[output_column] = df[text_column].astype(str).str.strip()
    return df


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    
    Args:
        df: Input DataFrame
        test_size: Fraction for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    return train_df, test_df
