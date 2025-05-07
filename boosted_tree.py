import numpy as np
import csv
from sklearn.decomposition   import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble        import HistGradientBoostingClassifier
from sklearn.metrics         import classification_report, confusion_matrix
import matplotlib.pyplot     as plt

# —— 1) Load your raw echo profiles —— 
X = np.load('/Users/mingl/Desktop/info6120/group_project/audios/session_01/audio_001_fmcw_16bit_diff_profiles.npy')
N, D = X.shape
print(f"Loaded X: {X.shape}")

# —— 2) Build full-frame labels y_full from your 72 events —— 
events = []
with open('/Users/mingl/Desktop/info6120/group_project/audios/session_01/record_20250429_044416_327788_records.txt', newline='') as f:
    for row in csv.reader(f):
        if len(row) < 4: continue
        events.append((float(row[1]), float(row[2]), row[3].replace('hold-','')))
events.sort(key=lambda x: x[0])
starts, ends, labs = zip(*events)

# linearly map each frame to a timestamp
audio_ts = np.linspace(starts[0], ends[-1], N)
y_full   = np.full(N, -1, dtype=int)
for st,en,lab in zip(starts, ends, labs):
    i0 = np.argmin(np.abs(audio_ts - st))
    i1 = np.argmin(np.abs(audio_ts - en))
    label = int(lab.split('-',1)[1]) - 1
    y_full[i0:i1+1] = label

# —— 3) Extract only labeled frames —— 
mask = y_full >= 0
X_lab = X[mask]
y_lab = y_full[mask]
print(f"Labeled data: {X_lab.shape[0]} samples, classes = {np.unique(y_lab)}")

# —— 4) PCA down to 100 dims —— 
pca = PCA(n_components=100, whiten=True, random_state=42)
X_pca = pca.fit_transform(X_lab)

# —— 5) 60/20/20 stratified split —— 
X_trval, X_test, y_trval, y_test = train_test_split(
    X_pca, y_lab, test_size=0.20, stratify=y_lab, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_trval, y_trval, test_size=0.25, stratify=y_trval, random_state=42)

print(f"Splits: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

# —— 6) HistGradientBoostingClassifier with early stopping —— 
clf = HistGradientBoostingClassifier(
    max_iter=200,
    max_depth=6,
    learning_rate=0.1,
    early_stopping=True,
    random_state=42
)
clf.fit(X_train, y_train)

# —— 7) Evaluate on test set —— 
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
cm = confusion_matrix(y_test, y_pred)

# —— 8) Plot confusion matrix —— 
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix (HistGBM)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.colorbar()
plt.tight_layout()
plt.show()

# —— 9) Plot cumulative PCA variance —— 
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_)*100)
plt.axhline(95, color='gray', linestyle='--')
plt.title('Cumulative Variance Explained by PCA')
plt.xlabel('Number of Components')
plt.ylabel('Variance Explained (%)')
plt.show()
