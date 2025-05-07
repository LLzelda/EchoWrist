import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1) Load your echo‐profiles
feat_path = '/Users/mingl/Desktop/info6120/group_project/audios/session_01/audio_001_fmcw_16bit_diff_profiles.npy'
X = np.load(feat_path)      # shape → (2400, 49274)
N, D = X.shape
print(f"Loaded X: {X.shape}")

# 2) Load the true per‐frame timestamps
frame_time_path = '/Users/mingl/Desktop/info6120/group_project/audios/session_01/record_20250429_044416_327788_frame_time.txt'
audio_ts_all   = np.loadtxt(frame_time_path)   # e.g. (6822,)
# If your profiles were down‐sampled to 2400 frames, pick the first N timestamps:
audio_ts = audio_ts_all[:N]
assert len(audio_ts) == N, f"Expected {N} timestamps, got {len(audio_ts)}"

# 3) Parse your 72‐line records file, but keep only the 36 Pos-start rows
gt_path = '/Users/mingl/Desktop/info6120/group_project/audios/session_01/record_20250429_044416_327788_records.txt'
events = []
with open(gt_path, newline='') as f:
    for row in csv.reader(f):
        if len(row) < 4:
            continue
        label = row[3]
        # skip hold entries
        if label.startswith('hold-'):
            continue
        start_ts = float(row[1])
        end_ts   = float(row[2])
        cls      = int(label.split('-',1)[1]) - 1   # Pos-1→0,…Pos-9→8
        events.append((start_ts, end_ts, cls))

assert len(events) == 36, f"Expected 36 start events, got {len(events)}"

# 4) Build the frame‐level label vector
y_full = np.full(N, -1, dtype=int)
for st, en, cls in events:
    # find nearest frame indices
    i0 = np.argmin(np.abs(audio_ts - st))
    i1 = np.argmin(np.abs(audio_ts - en))
    y_full[i0:i1+1] = cls

# 5) Filter to only the labeled frames
mask  = y_full >= 0
X_lab = X[mask]
y_lab = y_full[mask]
print(f"Labeled frames: {X_lab.shape[0]}, classes: {np.unique(y_lab)}")

# 6) Stratified train/val/test split
X_trval, X_test, y_trval, y_test = train_test_split(
    X_lab, y_lab, test_size=0.20, stratify=y_lab, random_state=42)
X_train, X_val, y_train, y_val    = train_test_split(
    X_trval, y_trval, test_size=0.25, stratify=y_trval, random_state=42)
print(f"Splits → train:{len(y_train)}, val:{len(y_val)}, test:{len(y_test)}")

# 7) Random Forest baseline
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# 8) Evaluate & plot confusion matrix
print(f"\nVal Acc: {clf.score(X_val, y_val):.3f}\n")
y_pred = clf.predict(X_test)
print("Test report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.colorbar()
plt.tight_layout()
plt.show()
