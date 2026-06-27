import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# --- SMART PATHING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Input Files
FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "extracted_features.csv")
DEMENTIA_CSV = os.path.join(PROJECT_ROOT, "master_key.csv") 
HEALTHY_CSV = os.path.join(PROJECT_ROOT, "healthy_master_key.csv") 

# Output Folders
METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")
TRAIN_DIR = os.path.join(METADATA_DIR, "training data (70%)")
VAL_DIR = os.path.join(METADATA_DIR, "validation data (10%)")
TEST_DIR = os.path.join(METADATA_DIR, "test data (20%)")

def main():
    # 1. Create the specific output folders if they don't exist
    for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    print("Loading and unifying datasets...")
    features_df = pd.read_csv(FEATURES_CSV)
    dementia_df = pd.read_csv(DEMENTIA_CSV)
    healthy_df = pd.read_csv(HEALTHY_CSV)

    # 2. Clean names and merge into one master dataframe
    dementia_df['join_key'] = dementia_df['name'].astype(str).str.lower().str.replace(' ', '', regex=False)
    healthy_df['join_key'] = healthy_df['name'].astype(str).str.lower().str.replace(' ', '', regex=False)
    features_df['join_key'] = features_df['Filename'].str.split('_').str[0].str.lower().str.replace(' ', '', regex=False)

    dementia_subset = dementia_df[['join_key', 'dementia type']].rename(columns={'dementia type': 'Label'})
    healthy_subset = healthy_df[['join_key']].copy()
    healthy_subset['Label'] = 'Healthy'

    unified_master = pd.concat([dementia_subset, healthy_subset], ignore_index=True).drop_duplicates(subset=['join_key'])
    df = pd.merge(features_df, unified_master, on='join_key', how='inner')
    
    # 3. Perform Speaker-Independent Splits
    print("Splitting data securely by speaker...")
    
    # First split: 70% Train, 30% Temp (Val + Test)
    gss_train_test = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss_train_test.split(df, groups=df['join_key']))
    
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    # Second split: From the 30% temp data, take 1/3 for Val (10%) and 2/3 for Test (20%)
    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=0.66, random_state=42) 
    val_idx, test_idx = next(gss_val_test.split(temp_df, groups=temp_df['join_key']))

    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    # 4. Save to CSVs
    train_out = os.path.join(TRAIN_DIR, "train.csv")
    val_out = os.path.join(VAL_DIR, "val.csv")
    test_out = os.path.join(TEST_DIR, "test.csv")

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    # 5. Output Verification
    print("-" * 50)
    print("SPLIT RESULTS (Rows of Audio Data):")
    print(f"Training data (70%):   {len(train_df)} files -> Saved to {train_out}")
    print(f"Validation data (10%): {len(val_df)} files -> Saved to {val_out}")
    print(f"Test data (20%):       {len(test_df)} files -> Saved to {test_out}")
    print("-" * 50)
    
    # Verify no speaker overlap
    train_speakers = set(train_df['join_key'])
    val_speakers = set(val_df['join_key'])
    test_speakers = set(test_df['join_key'])

    if train_speakers.intersection(val_speakers) or train_speakers.intersection(test_speakers):
        print("WARNING: Data Leakage Detected! Speakers overlap.")
    else:
        print("Verification: Strict speaker independence confirmed. Zero data leakage.")

if __name__ == "__main__":
    main()