import os
import glob
import librosa
import numpy as np
import pandas as pd

# --- SMART PATHING ---
# 1. Find out exactly where features.py lives (the src/ folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the main project folder
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 3. Define the exact path to the data and the output file
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "mini")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "extracted_features.csv")

def extract_features(file_path):
    print(f"Processing {os.path.basename(file_path)}...")
    
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None) 

    # --- FEATURE 1: MFCCs ---
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    M = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    mfccs_mean = np.mean(M, axis=1) 

    # --- FEATURE 2: Pitch ---
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=300)
    pitch_mean = np.nanmean(f0) if np.any(voiced_flag) else 0.0

# --- FEATURE 3: Energy (Root Mean Square) ---
    rms = librosa.feature.rms(y=y)
    energy_mean = np.mean(rms)

    # --- NEW: Spectral Centroid & Zero Crossing Rate (From Kaggle Link) ---
    # Centroid: Indicates the 'brightness' of the voice
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)
    
    # ZCR: Indicates the amount of noise/harshness in the speech
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # --- FEATURE 4: Pauses ---
    non_mute_intervals = librosa.effects.split(y, top_db=20)
    total_duration = librosa.get_duration(y=y, sr=sr)
    speaking_duration = sum([(end - start) / sr for start, end in non_mute_intervals])
    pause_duration = total_duration - speaking_duration

    # Package basic features into a dictionary
    features_dict = {
        "Filename": os.path.basename(file_path),
        "Total Duration (s)": round(total_duration, 2),
        "Pause Duration (s)": round(pause_duration, 2),
        "Mean Energy": round(energy_mean, 4),
        "Mean Pitch (Hz)": round(pitch_mean, 2),
        "Spectral Centroid": round(centroid_mean, 2),
        "Zero Crossing Rate": round(zcr_mean, 4),
    }

    # Add the 39 MFCC features
    for i, val in enumerate(mfccs_mean):
        if i < 13:
            features_dict[f"MFCC_{i+1}"] = val
        elif i < 26:
            features_dict[f"MFCC_Delta_{i-12}"] = val
        else:
            features_dict[f"MFCC_Delta2_{i-25}"] = val

    return features_dict

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Could not find the folder '{DATA_DIR}'.")
        print("Please check your folder structure!")
        return

    search_pattern = os.path.join(DATA_DIR, "**", "*.wav")
    wav_files = glob.glob(search_pattern, recursive=True)

    if not wav_files:
        print(f"No WAV files found in '{DATA_DIR}'. Please add some and try again!")
        return

    all_features = []

    for file in wav_files:
        features = extract_features(file)
        all_features.append(features)

    df = pd.DataFrame(all_features)
    
    print("\nExtraction Complete! Here is a preview:")
    print("-" * 60)
    print(df.iloc[:, :7].head()) 

    # Save output using the smart path we defined at the top
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved all 44 features for each file to:\n{OUTPUT_CSV}")

if __name__ == "__main__":
    main()