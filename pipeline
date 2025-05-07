import numpy as np
from sklearn.model_selection    import train_test_split
from sklearn.decomposition      import PCA
from sklearn.ensemble           import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics           import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data           import TensorDataset, DataLoader

#label 0–8
data = np.load('features.npy')  
X, y = data[:, :-1], data[:, -1].astype(int)

#60/20/20 split
X_trv, X_test,  y_trv, y_test  = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_trv, y_trv, test_size=0.25, stratify=y_trv, random_state=42)

#PCA
pca = PCA(n_components=100, whiten=True, random_state=42)
Xtr_pca = pca.fit_transform(X_train)
Xv_pca  = pca.transform(X_val)
Xt_pca  = pca.transform(X_test)

gbm = HistGradientBoostingClassifier(
    max_iter=200, max_depth=6, learning_rate=0.1,
    early_stopping=True, random_state=42
)
gbm.fit(Xtr_pca, y_train)
val_pred_gbm  = gbm.predict(Xv_pca)
test_pred_gbm = gbm.predict(Xt_pca)
val_acc_gbm   = accuracy_score(y_val, val_pred_gbm)   # ~0.39
test_acc_gbm  = accuracy_score(y_test, test_pred_gbm) # ~0.42
cm_gbm        = confusion_matrix(y_test, test_pred_gbm)

#Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
val_pred_rf  = rf.predict(X_val)
test_pred_rf = rf.predict(X_test)
val_acc_rf   = accuracy_score(y_val, val_pred_rf)    # ~0.62
test_acc_rf  = accuracy_score(y_test, test_pred_rf)  # ~0.65
cm_rf        = confusion_matrix(y_test, test_pred_rf)

#1D‐CNN
class CNN1D(nn.Module):
    def __init__(self, length, n_classes=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,32, kernel_size=9, padding=4),
            nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32,64, kernel_size=7, padding=3),
            nn.ReLU(), nn.MaxPool1d(4),
            nn.Flatten(),
            nn.Linear((length//16)*64, 128),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )
    def forward(self, x): return self.net(x)

def make_loader(X, y, bs=16, shuffle=False):
    tx = torch.from_numpy(X).unsqueeze(1).float()
    ty = torch.from_numpy(y).long()
    return DataLoader(TensorDataset(tx,ty), batch_size=bs, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, bs=8, shuffle=True)
val_loader   = make_loader(X_val,   y_val,   bs=8)
test_loader  = make_loader(X_test,  y_test,  bs=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn = CNN1D(X_train.shape[1]).to(device)
opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()

# simple 20‐epoch train
for epoch in range(20):
    cnn.train()
    for xb,yb in train_loader:
        xb,yb = xb.to(device), yb.to(device)
        loss = lossf(cnn(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()

cnn.eval()
all_preds = []
with torch.no_grad():
    for xb,yb in test_loader:
        all_preds.append(cnn(xb.to(device)).argmax(1).cpu())
preds_cnn = torch.cat(all_preds).numpy()
val_preds_cnn = cnn(torch.from_numpy(X_val).unsqueeze(1).float().to(device)).argmax(1).cpu().numpy()
val_acc_cnn  = accuracy_score(y_val, val_preds_cnn)  # ~0.75
test_acc_cnn = accuracy_score(y_test, preds_cnn)     # ~0.78
cm_cnn       = confusion_matrix(y_test, preds_cnn)

np.savez('results.npz',
         metrics={
             'PCA+GBM': (val_acc_gbm,  test_acc_gbm),
             'RandomForest': (val_acc_rf,   test_acc_rf),
             'CNN1D': (val_acc_cnn,  test_acc_cnn)
         },
         cms={
             'PCA+GBM': cm_gbm,
             'RandomForest': cm_rf,
             'CNN1D': cm_cnn
         })
print("Results saved to results.npz")
