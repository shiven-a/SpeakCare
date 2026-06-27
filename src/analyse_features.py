import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# --- SMART PATHING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# File Paths
TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "metadata", "training data (70%)", "train.csv")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

def main():
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)

    print("Loading training data for dimensionality reduction...")
    try:
        df = pd.read_csv(TRAIN_CSV)
    except FileNotFoundError:
        print("Error: Could not find train.csv. Run create_splits.py first!")
        return
    
# Create the binary label for the 1D test
    df['Binary_Label'] = df['Label'].apply(lambda x: 'Healthy' if x == 'Healthy' else 'Dementia')

    # Grab the math (ignore the string/metadata columns)
    feature_cols = [col for col in df.columns if col not in ['Filename', 'join_key', 'Label', 'Binary_Label']]
    X = df[feature_cols].fillna(0).values

    # --- 2. STANDARD SCALER ---
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 3. LDA (BINARY 1D DENSITY PLOT) ---
    print("Running 1D LDA for Baseline (Healthy vs. Dementia)...")
    lda_1d = LDA(n_components=1)
    
    # Notice we feed it 'Binary_Label' so it explicitly knows what to separate
    X_lda_1d = lda_1d.fit_transform(X_scaled, df['Binary_Label'])

    plt.figure(figsize=(10, 5))
    sns.kdeplot(
        x=X_lda_1d[:, 0], 
        hue=df['Binary_Label'], 
        fill=True, 
        palette={"Healthy": "#2ecc71", "Dementia": "#e74c3c"}, 
        alpha=0.6, linewidth=2
    )
    plt.title("1D LDA Density: Baseline Separability (Sick vs. Healthy)", fontsize=14, pad=15)
    plt.xlabel("Linear Discriminant 1 (Maximized Between-Class Scatter)")
    plt.ylabel("Density of Audio Files")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    out_1d = os.path.join(REPORTS_DIR, "lda_1d_binary.png")
    plt.savefig(out_1d, dpi=300, bbox_inches='tight')

    # --- 4. LDA (MULTI-CLASS 2D SCATTER PLOT) ---
    print("Running 2D LDA for Sub-types...")
    # Because there are 6 classes in 'Label', we can extract 2 components
    lda_2d = LDA(n_components=2)
    X_lda_2d = lda_2d.fit_transform(X_scaled, df['Label'])

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=X_lda_2d[:, 0], y=X_lda_2d[:, 1], 
        hue=df['Label'], 
        palette="bright", 
        alpha=0.8, s=100, edgecolor='w'
    )
    plt.title("2D LDA Scatter: Multi-Class Separability", fontsize=15, pad=15)
    plt.xlabel("Linear Discriminant 1")
    plt.ylabel("Linear Discriminant 2")
    plt.legend(title="Diagnosis", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    out_2d = os.path.join(REPORTS_DIR, "lda_2d_multiclass.png")
    plt.savefig(out_2d, dpi=300, bbox_inches='tight')

    print(f"\nSuccess! Saved 1D Density to: {out_1d}")
    print(f"Success! Saved 2D Scatter to: {out_2d}")

if __name__ == "__main__":
    main()