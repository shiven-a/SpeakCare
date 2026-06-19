import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- SMART PATHING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# File Paths
FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "extracted_features.csv")
DEMENTIA_CSV = os.path.join(PROJECT_ROOT, "master_key.csv") 
HEALTHY_CSV = os.path.join(PROJECT_ROOT, "healthy_master_key.csv") 
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

def main():
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)

    print("Loading extracted features and master keys...")
    try:
        features_df = pd.read_csv(FEATURES_CSV)
        dementia_df = pd.read_csv(DEMENTIA_CSV)
        healthy_df = pd.read_csv(HEALTHY_CSV)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # --- 1. CLEAN AND UNIFY THE DATA ---
    print("Unifying Healthy and Dementia master keys...")
    
    # Standardize names by removing spaces and making lowercase (e.g., "Abe Burrows" -> "abeburrows")
    dementia_df['join_key'] = dementia_df['name'].astype(str).str.lower().str.replace(' ', '', regex=False)
    healthy_df['join_key'] = healthy_df['name'].astype(str).str.lower().str.replace(' ', '', regex=False)
    
    # The audio files might have spaces in the names now too (e.g., "Mel brooks_1.wav" -> "melbrooks")
    features_df['join_key'] = features_df['Filename'].str.split('_').str[0].str.lower().str.replace(' ', '', regex=False)

    # Grab labels from Dementia sheet
    dementia_subset = dementia_df[['join_key', 'dementia type']].rename(columns={'dementia type': 'Label'})
    
    # Grab names from Healthy sheet and explicitly label them "Healthy"
    healthy_subset = healthy_df[['join_key']].copy()
    healthy_subset['Label'] = 'Healthy'

    # Stack them together into one master dataset
    unified_master = pd.concat([dementia_subset, healthy_subset], ignore_index=True).drop_duplicates(subset=['join_key'])

    # --- 2. MERGE WITH AUDIO MATH ---
    df = pd.merge(features_df, unified_master, on='join_key', how='inner')
    
    print(f"Success! Matched {len(df)} audio files to their diagnoses.")
    print("Class breakdown:")
    print(df['Label'].value_counts())

    # Create a binary label strictly for the baseline model (Sick vs. Healthy)
    df['Binary_Label'] = df['Label'].apply(lambda x: 'Healthy' if x == 'Healthy' else 'Dementia')

    # --- 3. PREPARE X AND y ---
    feature_cols = [col for col in features_df.columns if col not in ['Filename', 'join_key']]
    X = df[feature_cols].fillna(0)
    y = df['Binary_Label']

    # --- 4. STANDARD SCALER ---
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 5. PCA ---
    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    var_1 = pca.explained_variance_ratio_[0] * 100
    var_2 = pca.explained_variance_ratio_[1] * 100

    # --- 6. PLOT ---
    print("Plotting and saving graph...")
    plt.figure(figsize=(10, 7))
    
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1], 
        hue=y, 
        palette={"Healthy": "#2ecc71", "Dementia": "#e74c3c"}, 
        alpha=0.8, s=100, edgecolor='w'
    )
    
    plt.title("PCA of Audio Features (Speakcare Baseline)", fontsize=15, pad=15)
    plt.xlabel(f"Principal Component 1 ({var_1:.1f}% Variance Explained)", fontsize=12)
    plt.ylabel(f"Principal Component 2 ({var_2:.1f}% Variance Explained)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_path = os.path.join(REPORTS_DIR, "pca_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nTask 3 Complete! Saved PCA plot to: {output_path}")

if __name__ == "__main__":
    main()