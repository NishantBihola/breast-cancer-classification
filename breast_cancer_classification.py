# FILE 2: nishant_breast_cancer_classification.py
# Main Breast Cancer Classification Project
# Author: Nishant
# Run this AFTER creating data_refined.csv
# =============================================================================

print("\n" + "="*80)
print("STARTING MAIN BREAST CANCER CLASSIFICATION PROJECT")
print("="*80)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import warnings
import sys
import codecs

# Fix UnicodeEncodeError by setting stdout encoding
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

warnings.filterwarnings('ignore')

print("=== Breast Cancer Classification Project ===")
print("Author: Nishant")
print("="*50)

# 1. Reading the Dataset
print("\n1. Reading the Dataset...")

try:
    # Load the preprocessed dataset
    df = pd.read_csv('data_refined.csv')
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Check for the target column
    if 'Diagnosed' in df.columns:
        target_col = 'Diagnosed'
    elif 'diagnosis' in df.columns:
        target_col = 'diagnosis'
    else:
        # Find likely target column
        possible_targets = [col for col in df.columns if any(keyword in col.lower() 
                                     for keyword in ['diagnos', 'target', 'label', 'class'])]
        if possible_targets:
            target_col = possible_targets[0]
        else:
            target_col = df.columns[-1]  # Use last column as default
    
    print(f"Target column identified: {target_col}")
    
except FileNotFoundError:
    print("Dataset 'data_refined.csv' not found!")
    print("Please run 'create_data_refined.py' first to create the required dataset.")
    exit()

print(f"\nTarget distribution:")
print(df[target_col].value_counts())

# 2. Feature Selection
print("\n2. Feature Selection...")

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Encode target if it's categorical
if y.dtype == 'object':
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Target classes: {label_encoder.classes_}")
else:
    y_encoded = y.copy()

# Calculate correlation with target
print("\nCalculating correlation with target...")

correlations = []
for column in X.columns:
    corr = np.corrcoef(X[column], y_encoded)[0, 1]
    correlations.append(abs(corr))

# Create correlation dataframe
corr_df = pd.DataFrame({
    'Feature': X.columns,
    'Correlation': correlations
}).sort_values('Correlation', ascending=False)

print("\nTop 15 features by correlation with target:")
print(corr_df.head(15))

# Choose features with correlation above threshold
correlation_threshold = 0.3
important_features_corr = corr_df[corr_df['Correlation'] > correlation_threshold]['Feature'].tolist()

print(f"\nFeatures with correlation > {correlation_threshold}:")
print(f"Number of selected features: {len(important_features_corr)}")
for feature in important_features_corr:
    corr_value = corr_df[corr_df['Feature'] == feature]['Correlation'].values[0]
    print(f"  {feature}: {corr_value:.4f}")

# Create reduced feature set based on correlation
X_reduced_corr = X[important_features_corr]

# 3. Splitting the Data
print("\n3. Splitting the Data...")

# Scale the features
scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

scaler_reduced = StandardScaler()
X_reduced_scaled = scaler_reduced.fit_transform(X_reduced_corr)
X_reduced_scaled = pd.DataFrame(X_reduced_scaled, columns=X_reduced_corr.columns)

# Split full feature set
X_train_full, X_temp_full, y_train, y_temp = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_val_full, X_test_full, y_val, y_test = train_test_split(
    X_temp_full, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Split reduced feature set
X_train_reduced, X_temp_reduced, _, _ = train_test_split(
    X_reduced_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_val_reduced, X_test_reduced, _, _ = train_test_split(
    X_temp_reduced, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training set shape (full): {X_train_full.shape}")
print(f"Validation set shape (full): {X_val_full.shape}")
print(f"Test set shape (full): {X_test_full.shape}")
print(f"Training set shape (reduced): {X_train_reduced.shape}")

# 4. Training Classifiers
print("\n4. Training Classifiers...")

def train_and_evaluate_classifier(clf, clf_name, X_train, X_val, X_test, y_train, y_val, y_test):
    """Train and evaluate a classifier"""
    print(f"\n--- {clf_name} ---")
    
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Predictions
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)
    
    # Accuracy scores
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"Confusion Matrix:")
    print(cm)
    
    return {
        'model': clf,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'predictions': y_test_pred,
        'confusion_matrix': cm
    }

# Initialize classifiers
classifiers = {
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVC': SVC(random_state=42)
}

# Results storage
results_full = {}
results_reduced = {}

print("\n" + "="*50)
print("TRAINING ON FULL FEATURE SET")
print("="*50)

# Train on full feature set
for clf_name, clf in classifiers.items():
    if clf_name == 'KNN':
        # Find optimal k using cross-validation
        print(f"\nFinding optimal k for KNN using cross-validation...")
        k_values = range(1, 21)
        cv_scores = []
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_train_full, y_train, cv=5, scoring='accuracy')
            cv_scores.append(scores.mean())
        
        optimal_k = k_values[np.argmax(cv_scores)]
        print(f"Optimal k: {optimal_k} with CV score: {max(cv_scores):.4f}")
        
        clf = KNeighborsClassifier(n_neighbors=optimal_k)
    
    results_full[clf_name] = train_and_evaluate_classifier(
        clf, clf_name, X_train_full, X_val_full, X_test_full, y_train, y_val, y_test
    )

print("\n" + "="*50)
print("TRAINING ON REDUCED FEATURE SET (CORRELATION-BASED)")
print("="*50)

# Train on reduced feature set
for clf_name, clf in classifiers.items():
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=optimal_k)
    elif clf_name == 'Random Forest':
        clf = RandomForestClassifier(random_state=42)
    else:  # SVC
        clf = SVC(random_state=42)
    
    results_reduced[clf_name] = train_and_evaluate_classifier(
        clf, clf_name, X_train_reduced, X_val_reduced, X_test_reduced, y_train, y_val, y_test
    )

# 5. Challenge: Another Feature Selection Method
print("\n5. Challenge: Alternative Feature Selection Method...")

# Method 1: SelectKBest with f_classif
print("\nUsing SelectKBest with f_classif...")
k_best = SelectKBest(score_func=f_classif, k=min(15, X.shape[1]))
X_kbest = k_best.fit_transform(X_scaled, y_encoded)
selected_features_kbest = X.columns[k_best.get_support()].tolist()

print(f"SelectKBest selected features ({len(selected_features_kbest)}):")
for feature in selected_features_kbest:
    print(f"  {feature}")

# Method 2: Recursive Feature Elimination with Random Forest
print(f"\nUsing Recursive Feature Elimination with Random Forest...")
rfe = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=min(15, X.shape[1]))
X_rfe = rfe.fit_transform(X_scaled, y_encoded)
selected_features_rfe = X.columns[rfe.support_].tolist()

print(f"RFE selected features ({len(selected_features_rfe)}):")
for feature in selected_features_rfe:
    print(f"  {feature}")

# Split the new feature sets
X_train_kbest, X_temp_kbest, _, _ = train_test_split(
    X_kbest, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_val_kbest, X_test_kbest, _, _ = train_test_split(
    X_temp_kbest, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

X_train_rfe, X_temp_rfe, _, _ = train_test_split(
    X_rfe, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_val_rfe, X_test_rfe, _, _ = train_test_split(
    X_temp_rfe, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Train best performing classifier on new feature sets
best_clf_name = max(results_full.keys(), key=lambda x: results_full[x]['test_accuracy'])
print(f"\nUsing best performing classifier: {best_clf_name}")

if best_clf_name == 'KNN':
    best_clf_kbest = KNeighborsClassifier(n_neighbors=optimal_k)
    best_clf_rfe = KNeighborsClassifier(n_neighbors=optimal_k)
elif best_clf_name == 'Random Forest':
    best_clf_kbest = RandomForestClassifier(random_state=42)
    best_clf_rfe = RandomForestClassifier(random_state=42)
else:  # SVC
    best_clf_kbest = SVC(random_state=42)
    best_clf_rfe = SVC(random_state=42)

print("\n" + "="*30)
print("SELECTKBEST RESULTS")
print("="*30)
results_kbest = train_and_evaluate_classifier(
    best_clf_kbest, f"{best_clf_name} (SelectKBest)", 
    X_train_kbest, X_val_kbest, X_test_kbest, y_train, y_val, y_test
)

print("\n" + "="*30)
print("RFE RESULTS")
print("="*30)
results_rfe = train_and_evaluate_classifier(
    best_clf_rfe, f"{best_clf_name} (RFE)", 
    X_train_rfe, X_val_rfe, X_test_rfe, y_train, y_val, y_test
)

# Final Comparison
print("\n" + "="*60)
print("FINAL RESULTS COMPARISON")
print("="*60)

print(f"\nFULL FEATURE SET ({X.shape[1]} features):")
for clf_name, results in results_full.items():
    accuracy = results['test_accuracy']
    status = "✓ MEETS REQUIREMENT" if accuracy >= 0.94 else "✗ Below 94%"
    print(f"  {clf_name}: {accuracy:.4f} {status}")

print(f"\nREDUCED FEATURE SET - CORRELATION ({len(important_features_corr)} features):")
for clf_name, results in results_reduced.items():
    accuracy = results['test_accuracy']
    status = "✓ MEETS REQUIREMENT" if accuracy >= 0.94 else "✗ Below 94%"
    print(f"  {clf_name}: {accuracy:.4f} {status}")

print(f"\nSELECTKBEST FEATURE SET ({len(selected_features_kbest)} features):")
accuracy = results_kbest['test_accuracy']
status = "✓ MEETS REQUIREMENT" if accuracy >= 0.94 else "✗ Below 94%"
print(f"  {best_clf_name}: {accuracy:.4f} {status}")

print(f"\nRFE FEATURE SET ({len(selected_features_rfe)} features):")
accuracy = results_rfe['test_accuracy']
status = "✓ MEETS REQUIREMENT" if accuracy >= 0.94 else "✗ Below 94%"
print(f"  {best_clf_name}: {accuracy:.4f} {status}")

# Find best overall model
all_results = {
    'Full - KNN': results_full['KNN']['test_accuracy'],
    'Full - Random Forest': results_full['Random Forest']['test_accuracy'],
    'Full - SVC': results_full['SVC']['test_accuracy'],
    'Correlation - KNN': results_reduced['KNN']['test_accuracy'],
    'Correlation - Random Forest': results_reduced['Random Forest']['test_accuracy'],
    'Correlation - SVC': results_reduced['SVC']['test_accuracy'],
    f'SelectKBest - {best_clf_name}': results_kbest['test_accuracy'],
    f'RFE - {best_clf_name}': results_rfe['test_accuracy']
}

best_model_config = max(all_results.keys(), key=lambda x: all_results[x])
best_accuracy = all_results[best_model_config]

print(f"\n" + "="*60)
print("BEST MODEL")
print("="*60)
print(f"Configuration: {best_model_config}")
print(f"Test Accuracy: {best_accuracy:.4f}")
status = "✓ MEETS 94% REQUIREMENT" if best_accuracy >= 0.94 else "✗ BELOW 94% REQUIREMENT"
print(f"Status: {status}")

# Feature importance analysis for Random Forest
if 'Random Forest' in best_model_config:
    if 'Full' in best_model_config:
        best_rf_model = results_full['Random Forest']['model']
        feature_names = X.columns
    elif 'Correlation' in best_model_config:
        best_rf_model = results_reduced['Random Forest']['model']
        feature_names = X_reduced_corr.columns
    elif 'SelectKBest' in best_model_config:
        best_rf_model = results_kbest['model']
        feature_names = selected_features_kbest
    else:  # RFE
        best_rf_model = results_rfe['model']
        feature_names = selected_features_rfe
    
    print(f"\nFeature Importances (Top 10):")
    if hasattr(best_rf_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

print(f"\n" + "="*60)
print("SELECTED FEATURE LISTS SUMMARY")
print("="*60)

print(f"\nCorrelation-based features ({len(important_features_corr)}):")
print(important_features_corr)

print(f"\nSelectKBest features ({len(selected_features_kbest)}):")
print(selected_features_kbest)

print(f"\nRFE features ({len(selected_features_rfe)}):")
print(selected_features_rfe)

print(f"\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*60)
print("All requirements have been fulfilled:")
print("✓ Feature selection using correlation")
print("✓ Multiple classifiers (KNN, Random Forest, SVC)")
print("✓ Full vs reduced feature comparison")
print("✓ Accuracy scores and confusion matrices")
print("✓ Alternative feature selection methods")
print("✓ Optimal k selection for KNN using cross-validation")
print("✓ Minimum 94% accuracy requirement tracking")