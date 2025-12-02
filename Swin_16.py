import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from scipy.interpolate import griddata
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import xgboost as xgb
from collections import Counter
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import torchvision.transforms as T
import datetime
from sklearn.preprocessing import StandardScaler
from category_encoders import CatBoostEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_X = train.drop(columns=['Class'])
train_Y = train['Class'].apply(lambda x:1 if x=='NG' else 0)
test_X = test.drop(columns=['ID'])

x_loc = [f"X{i+1}" for i in range(5)]
y_loc = [f"Y{i+1}" for i in range(5)]
g = [f"G{i+1}" for i in range(4)]
x_fem = [f"x{i}" for i in range(256)]
y_fem = [f"y{i}" for i in range(256)]
p_fem = [f"p{i}" for i in range(256)]

cols = x_fem + y_fem + p_fem + x_loc + y_loc

def g_statis(df):
    df['G_sum'] = df[g].sum(axis=1)
    df['G_mean'] = df[g].mean(axis=1)
    df['G_std'] = df[g].std(axis=1, ddof=0)
    df['G_max'] = df[g].max(axis=1)
    df['G_min'] = df[g].min(axis=1)
    df['G_range'] = df['G_max'] - df['G_min']
    eps = 1e-6
    df['G1_div_G4'] = df['G1'] / (df['G4'].abs() + eps)
    df['G2_div_G3'] = df['G2'] / (df['G3'].abs() + eps)
    return df

train_X = g_statis(train_X)
test_X  = g_statis(test_X)

g1 = train_X.drop(columns=cols)
g2 = test_X.drop(columns=cols)

cat_list = g1.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
num_list = sorted(list(set(g1.columns) - set(cat_list)))

sgkf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=2)

for train_idx, val_idx in sgkf.split(train_X, train_Y):
    break

g1_train = g1.iloc[train_idx]
g1_val   = g1.iloc[val_idx]
y_train  = train_Y.iloc[train_idx]
y_val    = train_Y.iloc[val_idx]
scaler = StandardScaler().fit(g1_train[num_list])

cbe = CatBoostEncoder(cols=cat_list, random_state=42, sigma=1.0)
cbe.fit(g1_train[cat_list], y_train)

def preprocess(dataset):
    Xc = cbe.transform(dataset[cat_list]).values.astype("float32")
    Xn = scaler.transform(dataset[num_list]).astype("float32")
    return np.concatenate([Xc, Xn], axis=1)

df = train[cols]
df_test = test[cols]

g_train = preprocess(g1).astype("float32")
g_test = preprocess(g2).astype("float32")

output = "fem_depth"
os.makedirs(output,exist_ok=True)

K = len(x_loc)

def depth_image(x,y,p, x_loc, y_loc, radius=0, H=16, W=16,
                use_gaussian_mask=True):
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()
    p = torch.tensor(p).float()

    assert x.shape == y.shape == p.shape

    depth = p.view(H, W)
    x_g = x.view(H, W)
    y_g = y.view(H, W)

    masks = []
    for Xi, Yi in zip(x_loc, y_loc):
        Xi = float(Xi)
        Yi = float(Yi)
        dist2 = (x_g -Xi)**2 + (y_g - Yi)**2

        mask = (dist2 <= radius**2).float()
        masks.append(mask)

    loc_masks = torch.stack(masks, dim=0)

    # ----- 2) CoordConv 채널 추가
    xs = torch.linspace(-1, 1, W).view(1, W).repeat(H, 1)
    ys = torch.linspace(-1, 1, H).view(H, 1).repeat(1, W)
    coord_x = xs.unsqueeze(0)  # [1, H, W]
    coord_y = ys.unsqueeze(0)

    depth_ch = depth.unsqueeze(0)  # (1, H, W)
    input_tensor = torch.cat([depth_ch, loc_masks, coord_x, coord_y], dim=0)

    return input_tensor

def save_image(x, y, p, x_loc, y_loc, filename="depth+mask.png"):
    t = depth_image(x, y, p, x_loc, y_loc)
    depth = t[0].numpy()
    plt.imsave(filename, depth, cmap="gray")
    print(f"saved to {filename}")

depth_tensor_list = []

X_arr = df[x_fem].to_numpy()
Y_arr = df[y_fem].to_numpy()
P_arr = df[p_fem].to_numpy()
X_loc = df[x_loc].to_numpy()
Y_loc = df[y_loc].to_numpy()

for i in range(len(df)):
    depth_tensor_list.append(
        depth_image(
            X_arr[i], Y_arr[i], P_arr[i],
            X_loc[i], Y_loc[i], radius=1, H=16, W=16
        )
    )
depth_tensor = torch.stack(depth_tensor_list, dim=0)
print("depth_tensor shape (train):", depth_tensor.shape)

depth_tensor_test_list = []

X_arr_test = df_test[x_fem].to_numpy()
Y_arr_test = df_test[y_fem].to_numpy()
P_arr_test = df_test[p_fem].to_numpy()
X_loc_test = df_test[x_loc].to_numpy()
Y_loc_test = df_test[y_loc].to_numpy()

for i in range(len(df_test)):
    depth_tensor_test_list.append(
        depth_image(
            X_arr_test[i], Y_arr_test[i], P_arr_test[i],
            X_loc_test[i], Y_loc_test[i], radius=1, H=16, W=16
        )
    )
depth_tensor_test = torch.stack(depth_tensor_test_list, dim=0)
print("depth_tensor shape (test):", depth_tensor_test.shape)

#이미지 저장
# for i in range(len(df)):
#     x_row = X_arr[i]
#     y_row = Y_arr[i]
#     p_row = P_arr[i]
#     xloc_row = X_loc[i]
#     yloc_row = Y_loc[i]
#
#     filename = f"depth_{i:04d}.png"
#     save_image(x_row, y_row, p_row, xloc_row, yloc_row, filename)

class SwinTransformer(nn.Module):
    def __init__(self, img_size=32, coord_dim=28, coord_emb_dim=96, in_chans=1+K+2):
        super().__init__()
        self.img_size = img_size

        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            in_chans=in_chans,
            num_classes=0,  # head 제거 → feature만 뽑음
            global_pool="avg",
            img_size=img_size,
        )
        backbone_dim = self.backbone.num_features

        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_dim, coord_emb_dim),
            nn.ReLU(),
            nn.BatchNorm1d(coord_emb_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim + coord_emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x, loc):
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )  # [B, C, 32, 32]
        x_logits = self.backbone(x) #[B,1]

        loc_logits = self.coord_mlp(loc)

        logit = self.classifier(torch.cat([x_logits, loc_logits], dim=1))
        return logit.squeeze(1)   #[B]

depth_channel = depth_tensor[:, 0, :, :]  # [N,H,W]
depth_mean = depth_channel.mean().item()
depth_std  = depth_channel.std().item()

class FEMDepthDataset(Dataset):
    def __init__(self, depth_tensor, g_array, labels, depth_mean, depth_std):
        self.X = depth_tensor
        self.g = torch.tensor(g_array, dtype=torch.float32)
        self.y = torch.tensor(labels).float()
        self.depth_mean = depth_mean
        self.depth_std = depth_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        g = self.g[idx]
        y = self.y[idx]

        x[0] = (x[0] - self.depth_mean) / (self.depth_std + 1e-6)
        return x, g, y

class FEMDepthTestDataset(Dataset):
    def __init__(self, depth_tensor_test, g_array, depth_mean, depth_std):
        self.X = depth_tensor_test
        self.g = torch.tensor(g_array, dtype=torch.float32)
        self.depth_mean = depth_mean
        self.depth_std = depth_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        g = self.g[idx]

        x[0] = (x[0] - self.depth_mean) / (self.depth_std + 1e-6)
        return x, g

class EarlyStopping:
    def __init__(self, patience=5, mode="max", delta=1e-4, path="auc.pth"):
        self.patience=patience
        self.mode=mode
        self.delta=delta
        self.path=path

        self.best_score=None
        self.counter=0
        self.early_stop=False

    def __call__(self, metric, model):
        score=metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        else:
            if self.mode == "max":
                if score > self.best_score+self.delta:
                    self.best_score = score
                    self.save_checkpoint(model)
                    self.counter=0
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
            else:
                raise NNotImplementedError("Only mode='max' is implemented for now.")

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        print(f"best_score = {self.best_score:.4f}")

def get_predictions(loader, model, device):
    all_probs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                images, g_batch, labels = batch
            else:
                images, g_batch = batch
                labels = None
            images    = images.to(device)
            g_batch = g_batch.to(device)

            logits = model(images, g_batch)
            probs  = torch.sigmoid(logits)  # [B]

            all_probs.append(probs.cpu().numpy().ravel())
            if labels is not None:
                all_labels.append(labels.cpu().numpy().ravel())

    all_probs  = np.concatenate(all_probs)
    if len(all_labels) > 0:
        all_labels = np.concatenate(all_labels)
    else:
        all_labels = None
    return all_probs, all_labels

num_epochs = 35

dataset = FEMDepthDataset(depth_tensor, g_train, train_Y.values, depth_mean=depth_mean, depth_std=depth_std)

test_dataset = FEMDepthTestDataset(depth_tensor_test, g_test, depth_mean, depth_std)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)

train_d = Subset(dataset, train_idx)
val_d = Subset(dataset, val_idx)

train_loader = DataLoader(train_d, batch_size=64, shuffle=True)
val_loader = DataLoader(val_d, batch_size=64, shuffle=False)

g_dim = g_train.shape[1]

model = SwinTransformer(in_chans=1+K+2).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], dtype=torch.float32).to(device))
optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(),
         "lr": 5e-5},          # Swin backbone (pretrained → 작은 lr)

        {"params": list(model.coord_mlp.parameters()) +
                   list(model.classifier.parameters()),
         "lr": 1e-4},          # tabular + classifier (새로 학습 → 큰 lr)
    ], weight_decay=0.05)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
num_training_steps = num_epochs * len(train_loader)
scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=int(0.1 * num_training_steps),num_training_steps=num_training_steps)

# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer, max_lr=3e-4,
#     epochs=num_epochs,
#     steps_per_epoch=len(train_loader)
# )

early_stopping = EarlyStopping(
    patience=6,
    mode="max",
    delta=1e-4,
    path="best_swin_auc.pth"
)

train_losses = []
val_aucs = []

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    for images, g_batch, labels in train_loader:
        images = images.to(device)  # [B,1,H,W]
        g_batch = g_batch.to(device)  # [B,14]
        labels = labels.to(device)  # [B]

        optimizer.zero_grad()
        logits = model(images, g_batch)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * images.size(0)

    train_losses.append(total_loss / len(train_d))
    print(f"Epoch {epoch} mean loss:", total_loss / len(train_d))

    train_p, train_l = get_predictions(train_loader, model, device)
    train_a = roc_auc_score(train_l, train_p)
    val_p, val_l = get_predictions(val_loader, model, device)
    val_a = roc_auc_score(val_l, val_p)

    print(f"Epoch {epoch:02d} | train_auc = {train_a:.4f} | val_auc = {val_a:.4f}")

    early_stopping(val_a, model)
    if early_stopping.early_stop:
        break

best_model_path = "best_swin_auc.pth"
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)

train_pred, train_labels = get_predictions(train_loader, model, device)
train_auc = roc_auc_score(train_labels, train_pred)
val_pred, val_label = get_predictions(val_loader, model, device)
val_auc = roc_auc_score(val_label, val_pred)

print(f"Swin Train AUC : {train_auc:.4f}")
print(f"Swin Val   AUC : {val_auc:.4f}")
print(f"val_mean: {val_pred.mean(axis=0)}, val_std: {val_pred.std(axis=0)}")

idx_sorted = np.argsort(val_pred)

pred_bin = np.ones(len(val_pred), dtype=int)


def competition_score(val_label, val_pred, decision_bool):
    val_label = np.asarray(val_label)
    decision_bool = np.asarray(decision_bool).astype(bool)

    auc = roc_auc_score(val_label, val_pred)

    good_mask = decision_bool & (val_label == 0)
    bad_mask  = decision_bool & (val_label == 1)
    tnp = 100 * good_mask.sum() - 2000 * bad_mask.sum()

    s1 = max(auc - 0.5, 0) / 0.5
    s2 = max(tnp, 0) / 20000.0
    final = np.sqrt(s1 * s2)

    return {
        "auc": auc,
        "total_net_profit": tnp,
        "task1_score": s1,
        "task2_score": s2,
        "final_score": final,
    }

def find_best_N(val_pred, val_label, min_N=10, max_N=None):
    val_pred = np.asarray(val_pred)
    val_label = np.asarray(val_label)

    N_total = len(val_pred)
    max_N = round(len(val_pred) * 0.42)
    min_N = round(len(val_pred) * 0.25)

    idx_sorted = np.argsort(val_pred)  # 낮은 확률 → 승인 후보

    best_N = None
    Score = -1e9

    for N in range(min_N, max_N + 1):
        pred_bin = np.ones(N_total, dtype=int)
        pred_bin[idx_sorted[:N]] = 0

        decision_val = (pred_bin == 0)

        Total_net_Profit = competition_score(val_label, val_pred, decision_val)
        current_score = Total_net_Profit["final_score"]

        if current_score > Score:
            Score = current_score
            best_N = N

    return best_N

N = find_best_N(val_pred, val_label)
print(f"Best N = {N}")

pred_bin[idx_sorted[:N]] = 0

decision_val = (pred_bin == 0)

cm_val = confusion_matrix(val_label, pred_bin)
disp = ConfusionMatrixDisplay(cm_val, display_labels=['OK(0)', 'NG(1)'])
disp.plot(cmap='Blues', values_format='d')
plt.title("Validation Confusion Matrix")
plt.show()


Total_net_Profit = competition_score(val_label, val_pred, decision_val)
print(f"Total_net_Profit: {Total_net_Profit}")

# Test =============
test_probs, _ = get_predictions(test_loader, model, device)

print(f"test_mean: {test_probs.mean(axis=0)}, test_std: {test_probs.std(axis=0)}")

submission = pd.read_csv("sample_submission.csv")
submission['probability'] = np.concatenate([test_probs,test_probs])

test_idx_sorted = np.argsort(test_probs)

test_pred_bin = np.ones(len(test_probs), dtype=int)

approval_rate = N / len(val_pred)
test_N = int(round(approval_rate * len(test_probs)))

test_pred_bin[test_idx_sorted[:test_N]] = 0

test_pred = np.concatenate([test_pred_bin, test_pred_bin])

decision_id_L_list = submission.iloc[:466].loc[test_pred[:466] == 0, 'ID']

decision_id_P_list = submission.iloc[466:].loc[test_pred[466:] == 0, 'ID']

submission.loc[submission['ID'].isin(decision_id_L_list), 'decision'] = True
submission.loc[submission['ID'].isin(decision_id_P_list), 'decision'] = True


submission.to_csv("my_submission.csv", index=False)