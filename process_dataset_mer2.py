import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.model_selection import train_test_split


def split_dataset_real():
    """
    Split the multi-annotator dataset into train, validation, and test sets.
    Performs a simple random split without distribution constraints.
    """
    # Read CSV file
    df = pd.read_csv('xxx/discrete-multi-annotator.csv')  # Replace with your file path

    # Define emotion categories
    emotions = ['worried', 'happy', 'neutral', 'angry', 'surprise', 'sad', 'other', 'unknown']

    # List of annotation column names
    label_columns = [f'discrete{i}' for i in range(1, 11)]  # discrete1 to discrete10

    # Simple random split: 70% train, 15% validation, 15% test
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    print(f"Training set size: {len(train)}")
    print(f"Validation set size: {len(val)}")
    print(f"Test set size: {len(test)}")

    # Save datasets
    train.to_csv('xxx/train_data.csv', index=False)
    val.to_csv('xxx/val_data.csv', index=False)
    test.to_csv('xxx/test_data.csv', index=False)

    print(f"\nFinal dataset sizes:")
    print(f"Training set: {len(train)} samples")
    print(f"Validation set: {len(val)} samples")
    print(f"Test set: {len(test)} samples")

    # Print annotation counts for each annotator
    for col in label_columns:
        print(f"{col}: {df[col].notna().sum()}")


def get_majority_vote(votes):
    """
    Get the majority vote from a list of annotations.

    Args:
        votes (list): List of annotation labels from different annotators

    Returns:
        str: The majority vote label
    """
    # Remove null values (NaN)
    valid_votes = [v for v in votes if pd.notna(v)]
    if not valid_votes:
        return 'unknown'

    # Count occurrences of each label
    vote_counts = Counter(valid_votes)

    # Find the label with the highest count
    max_count = max(vote_counts.values())
    majority_labels = [label for label, count in vote_counts.items() if count == max_count]

    # If multiple labels have the same count, return the one with higher priority in the emotion list
    if len(majority_labels) > 1:
        emotion_priority = ['worried', 'happy', 'neutral', 'angry', 'surprise', 'sad', 'other', 'unknown']
        for emotion in emotion_priority:
            if emotion in majority_labels:
                return emotion

    return majority_labels[0]


if __name__ == "__main__":

    split_dataset_real()
