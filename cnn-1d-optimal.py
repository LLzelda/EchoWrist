import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import csv

# ---- 1) Load and label data ----
feat_path = '/Users/mingl/Desktop/info6120/group_project/audios/session_01/audio_001_fmcw_16bit_diff_profiles.npy'
gt_path   = '/Users/mingl/Desktop/info6120/group_project/audios/session_01/record_20250429_044416_327788_records.txt'

X = np.load(feat_path)  # (2400, 49274)
N, D = X.shape
print(f"Loaded X: {X.shape}")

# Read events
events = []
with open(gt_path, newline='') as f:
    for row in csv.reader(f):
        if len(row) < 4:
            continue
        events.append((float(row[1]), float(row[2]), row[3].replace('hold-','')))
events.sort(key=lambda x: x[0])
starts, ends, labs = zip(*events)

# Build time axis and full-length label vector
audio_ts = np.linspace(starts[0], ends[-1], N)
y_full   = np.full(N, -1, dtype=int)
for st, en, lab in zip(starts, ends, labs):
    i0 = np.argmin(np.abs(audio_ts - st))
    i1 = np.argmin(np.abs(audio_ts - en))
    y_full[i0:i1+1] = int(lab.split('-',1)[1]) - 1

mask    = y_full >= 0
X_lab   = X[mask]
y_lab   = y_full[mask]
print(f"Dataset: {X_lab.shape[0]} samples, classes: {np.unique(y_lab)}")

# ---- 2) Split data ----
X_trval, X_test, y_trval, y_test = train_test_split(
    X_lab, y_lab, test_size=0.20, stratify=y_lab, random_state=42)
X_train, X_val, y_train, y_val   = train_test_split(
    X_trval, y_trval, test_size=0.25, stratify=y_trval, random_state=42)

# ---- 3) Data augmentation dataset ----
class NoisyDataset(Dataset):
    def __init__(self, X, y, noise_std=0.01):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.noise_std = noise_std
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        x = self.X[idx] + torch.randn_like(self.X[idx]) * self.noise_std
        return x.unsqueeze(0), self.y[idx]

class PlainDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]

def make_loader(X, y, batch_size=8, augment=False):
    ds = NoisyDataset(X, y) if augment else PlainDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=augment)

train_loader = make_loader(X_train, y_train, augment=True)
val_loader   = make_loader(X_val,   y_val)
test_loader  = make_loader(X_test,  y_test)

# ---- 4) Improved 1D-CNN ----
class ImprovedCNN1D(nn.Module):
    def __init__(self, seq_len, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seq_len)
            feat  = self.features(dummy)
        feat_dim = feat.numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ---- 5) Training function ----
def train_model(model, train_loader, val_loader, device):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    epochs_no_improve = 0
    train_accs, val_accs = [], []

    for epoch in range(1, 31):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        train_accs.append((total - (total-correct)) / len(train_loader.dataset))  # approximate
        val_accs.append(val_acc)
        scheduler.step(val_acc)

        print(f"Epoch {epoch:02d}: Val Acc = {val_acc:.3f}")

        # early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_cnn1d.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 7:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load('best_cnn1d.pt'))
    return model, train_accs, val_accs

# ---- 6) Run training ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedCNN1D(seq_len=D, n_classes=len(np.unique(y_lab)))
model, train_accs, val_accs = train_model(model, train_loader, val_loader, device)

# ---- 7) Test evaluation ----
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        all_preds.append(model(xb).argmax(1).cpu())
        all_labels.append(yb)
all_preds  = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

print("\nTest Report:")
print(classification_report(all_labels, all_preds, zero_division=0))
cm = confusion_matrix(all_labels, all_preds)

# ---- 8) Plot results ----
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(train_accs, label='Train')
plt.plot(val_accs,   label='Val')
plt.title("Train vs Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
