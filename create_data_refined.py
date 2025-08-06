# FILE 1: create_data_refined.py
# Script to create data_refined.csv from Wisconsin Breast Cancer dataset
# Author: Nishant
# Run this FIRST to create the required CSV file

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

print("Creating data_refined.csv from Wisconsin Breast Cancer dataset...")

# Option 1: Load from sklearn (built-in dataset)
try:
    # Load the built-in breast cancer dataset
    data = load_breast_cancer()
    
    # Create DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Add target column
    df['Diagnosed'] = ['M' if target == 1 else 'B' for target in data.target]
    
    print(f"Dataset loaded from sklearn. Shape: {df.shape}")
    print(f"Features: {len(data.feature_names)}")
    print(f"Target distribution:")
    print(df['Diagnosed'].value_counts())
    
except Exception as e:
    print(f"Error loading from sklearn: {e}")
    print("Creating sample dataset instead...")
    
    # Create sample data if sklearn dataset is not available
    np.random.seed(42)
    n_samples = 569
    
    # Feature names from Wisconsin Breast Cancer dataset
    feature_names = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area',
        'worst smoothness', 'worst compactness', 'worst concavity',
        'worst concave points', 'worst symmetry', 'worst fractal dimension'
    ]
    
    # Generate realistic sample data
    data = {}
    target = np.random.choice(['M', 'B'], n_samples, p=[0.37, 0.63])
    
    for i, feature in enumerate(feature_names):
        # Create features with different correlations to target
        base_value = np.random.normal(0, 1, n_samples)
        if 'mean' in feature or 'worst' in feature:
            # Higher correlation for mean and worst features
            target_effect = np.where(target == 'M', 1.5, -1.5) + np.random.normal(0, 0.5, n_samples)
            data[feature] = base_value + target_effect * (0.3 + np.random.random() * 0.4)
        else:
            # Lower correlation for error features
            data[feature] = base_value + np.where(target == 'M', 0.2, -0.2)
    
    df = pd.DataFrame(data)
    df['Diagnosed'] = target

# Basic preprocessing
print(f"\nPerforming basic preprocessing...")

# Check for missing values
print(f"Missing values: {df.isnull().sum().sum()}")

# Remove any potential duplicates
initial_shape = df.shape
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape} (removed {initial_shape[0] - df.shape[0]} rows)")

# Save to CSV
df.to_csv('data_refined.csv', index=False)
print(f"\ndata_refined.csv created successfully!")
print(f"Final dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Display summary statistics
print(f"\nDataset Summary:")
print(f"Total samples: {len(df)}")
print(f"Total features: {len(df.columns) - 1}")
print(f"Target column: 'Diagnosed'")
print(f"Target distribution:")
print(df['Diagnosed'].value_counts())
print(f"Percentage distribution:")
print(df['Diagnosed'].value_counts(normalize=True) * 100)

print(f"\nFirst few rows of data_refined.csv:")
print(df.head())

print(f"\ndata_refined.csv is ready for your breast cancer classification project!")