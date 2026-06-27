import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_val_score
from sklearn.metrics import roc_curve, auc

# The Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from save_metrics import save_model_metrics

# --- SMART PATHING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "metadata", "training data (70%)", "train.csv")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def main():
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)

    print("Loading training data for advanced modeling...")
    try:
        df = pd.read_csv(TRAIN_CSV)
    except FileNotFoundError:
        print("Error: Could not find train.csv. Run create_splits.py first!")
        return

    # --- 1. CREATE NUMERIC LABELS ---
    # XGBoost and ROC Curves require numeric labels: Healthy = 0, Dementia = 1
    df['Target'] = df['Label'].apply(lambda x: 0 if x == 'Healthy' else 1)

    # --- 2. PREPARE X, y, AND GROUPS ---
    # Grab all columns EXCEPT the text/metadata ones
    feature_cols = [col for col in df.columns if col not in ['Filename', 'join_key', 'Label', 'Target']]
    
    X = df[feature_cols].fillna(0).values
    y = df['Target'].values
    groups = df['join_key'].values

    # Setting up the Golden Standard CV: Balances classes AND isolates speakers
    cv = StratifiedGroupKFold(n_splits=5)
    
# --- 1. CREATE NUMERIC LABELS ---
    # XGBoost and ROC Curves require numeric labels: Healthy = 0, Dementia = 1
    df['Target'] = df['Label'].apply(lambda x: 0 if x == 'Healthy' else 1)

    # --- 2. PREPARE X, y, AND GROUPS ---
    # Grab all columns EXCEPT the text/metadata ones
    feature_cols = [col for col in df.columns if col not in ['Filename', 'join_key', 'Label', 'Target']]
    
    X = df[feature_cols].fillna(0).values
    y = df['Target'].values
    groups = df['join_key'].values

    # Setting up the Golden Standard CV: Balances classes AND isolates speakers
    cv = StratifiedGroupKFold(n_splits=5)

    # --- 3. KNN ELBOW PLOT (Finding the ideal neighbors) ---
    print("\nCalculating KNN Elbow Plot...")
    k_values = range(1, 21)
    knn_cv_aucs = []

    for k in k_values:
        # We use a Pipeline to ensure scaling happens SAFELY inside each fold
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=k))
        ])
        scores = cross_val_score(pipe, X, y, cv=cv, groups=groups, scoring='roc_auc')
        knn_cv_aucs.append(scores.mean())

    best_k = k_values[np.argmax(knn_cv_aucs)]
    best_knn_auc = max(knn_cv_aucs)
    print(f"Optimal K found: {best_k} (AUC: {best_knn_auc:.3f})")

    # Plot the Elbow
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, knn_cv_aucs, marker='o', linestyle='-', color='#3498db', linewidth=2)
    plt.axvline(x=best_k, color='#e74c3c', linestyle='--', label=f'Best K ({best_k})')
    plt.title("KNN Elbow Plot (Cross-Validated AUC)", fontsize=14)
    plt.xlabel("Number of Neighbors (K)")
    plt.ylabel("Mean ROC AUC Score")
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    elbow_path = os.path.join(REPORTS_DIR, "knn_elbow_plot.png")
    plt.savefig(elbow_path, dpi=300, bbox_inches='tight')
    print(f"Saved Elbow plot to {elbow_path}")

    # --- 4. TRAIN AND EVALUATE ALL MODELS ---
    print("\nEvaluating all classifiers via Stratified Group K-Fold...")
    
    # Calculate scale_pos_weight for XGBoost to handle the 129 vs 51 imbalance
    imbalance_ratio = sum(y == 0) / sum(y == 1)

    models = {
        "Regularized Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        f"K-Nearest Neighbors (k={best_k})": KNeighborsClassifier(n_neighbors=best_k),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
        "XGBoost": xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=imbalance_ratio, random_state=42)
    }

    plt.figure(figsize=(10, 8))
    
    # Plot random guessing baseline
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing (AUC = 0.500)')

    for name, model in models.items():
        print(f"Training {name}...")
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model)
        ])
        
        # Get out-of-fold probability predictions for every single audio file
        y_proba = cross_val_predict(pipe, X, y, cv=cv, groups=groups, method='predict_proba')[:, 1]
        
        # 1. Get the actual hard predictions (0 or 1) instead of just probabilities
        y_pred = cross_val_predict(pipe, X, y, cv=cv, groups=groups, method='predict')
        
        # 2. Call our Canvas utility to save the metrics and confusion matrix
        save_model_metrics(y, y_pred, MODELS_DIR, REPORTS_DIR, model_name=name)
        
        # Calculate ROC and AUC
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot the curve
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    # --- 5. FORMAT AND SAVE THE ROC GRAPH ---
    plt.title("ROC Curves: Model Separability Test (Speakcare)", fontsize=15, pad=15)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    roc_path = os.path.join(REPORTS_DIR, "roc_auc_comparison.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Saved Master ROC Curve to {roc_path}")

if __name__ == "__main__":
    main()