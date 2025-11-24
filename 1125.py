import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from scipy.interpolate import griddata
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_Y = train['Class'].apply(lambda x:1 if x=='NG' else 0)

x_loc = [f"X{i+1}" for i in range(5)]
y_loc = [f"Y{i+1}" for i in range(5)]
g = [f"G{i+1}" for i in range(4)]
x_fem = [f"x{i}" for i in range(256)]
y_fem = [f"y{i}" for i in range(256)]
p_fem = [f"p{i}" for i in range(256)]

cols = x_fem + y_fem + p_fem + x_loc + y_loc

groups = cols + g
indices = train[groups].astype(str).agg('_'.join, axis=1).values

sgkf = StratifiedGroupKFold(n_splits = 5, shuffle=True, random_state=2)

output = "fem_depth"
os.makedirs(output,exist_ok=True)

K = len(x_loc)

df = train[cols]
df_test = test[cols]

g_train = train[g].to_numpy().astype("float32")
g_test = test[g].to_numpy().astype("float32")

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
            X_loc[i], Y_loc[i], radius=0, H=16, W=16
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
            X_loc_test[i], Y_loc_test[i], radius=0, H=16, W=16
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
    def __init__(self, img_size=224, coord_dim=4, coord_emb_dim=64, in_chans=1+K+2):
        super().__init__()
        self.img_size = img_size

        # model option: 'swin_tiny_patch4_window7_224', 'convnext_base','vit_base_patch16_224',"regnetx_016"
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            in_chans=in_chans,
            num_classes=0,  # head 제거 → feature만 뽑음
            global_pool="avg",
        )
        backbone_dim = self.backbone.num_features

        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_dim, coord_emb_dim),
            nn.BatchNorm1d(coord_emb_dim),
            nn.ReLU(),
            nn.Linear(coord_emb_dim, coord_emb_dim),
            nn.BatchNorm1d(coord_emb_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim + coord_emb_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def forward(self, x, loc):
        # h,w를 224*224로 키우기
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            )
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
            if len(batch) ==3:
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

# ========================XGBOOST===================================
train_X = train.drop(columns=['Class']).iloc[:,:-256*3]
test_X = test.drop(columns=['ID']).iloc[:,:-256*3]
cat_list = train_X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
num_list = sorted(list(set(train_X.columns) - set(cat_list)- set(groups)))

OE = OneHotEncoder(min_frequency=0.01, handle_unknown='infrequent_if_exist', sparse_output=False)
OE.fit(train_X[cat_list])

def preprocess(dataset):
    Xc = OE.transform(dataset[cat_list])
    Xn = np.array(dataset[num_list])
    return np.concatenate([Xc, Xn], axis=1)

num_epochs = 30

dataset = FEMDepthDataset(depth_tensor, g_train, train_Y.values, depth_mean=depth_mean, depth_std=depth_std)

test_dataset = FEMDepthTestDataset(depth_tensor_test, g_test, depth_mean, depth_std)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)

counter = Counter(train_Y)
neg, pos = counter[0], counter[1]
scale_factor = neg / pos

for train_idx, val_idx in sgkf.split(np.zeros(len(train_Y)), train_Y, indices):
    break

train_d = Subset(dataset, train_idx)
val_d = Subset(dataset, val_idx)

train_loader = DataLoader(train_d, batch_size=64, shuffle=True)
val_loader = DataLoader(val_d, batch_size=64, shuffle=False)

model = SwinTransformer(in_chans=1+K+2).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], dtype=torch.float32).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)

num_training_steps = num_epochs * len(train_loader)
scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=int(0.1 * num_training_steps),num_training_steps=num_training_steps)

early_stopping = EarlyStopping(
    patience=5,
    mode="max",
    delta=1e-4,
    path="best_swin_auc.pth"
)

train_losses = []
val_aucs = []

for name, param in model.backbone.named_parameters():
    if "layers.0" in name or "layers.1" in name:
        param.requires_grad = False

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    for images, g_batch, labels in train_loader:
        images = images.to(device) #[B,1,H,W]
        g_batch = g_batch.to(device) # [B,14]
        labels = labels.to(device) #[B]

        optimizer.zero_grad()
        logits = model(images, g_batch)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * images.size(0)

    train_losses.append(total_loss / len(train_d))

    val_pred, val_label = get_predictions(val_loader, model, device)
    val_auc = roc_auc_score(val_label, val_pred)

    print(f"Epoch {epoch:02d} | train_loss = {total_loss / len(train_d):.4f} | val_auc = {val_auc:.4f}")

    early_stopping(val_auc, model)
    if early_stopping.early_stop:
        break

best_model_path = "best_swin_auc.pth"
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)
model.eval()

train_probs_fold, train_labels_fold = get_predictions(train_loader, model, device)
val_probs_fold, val_labels_fold = get_predictions(val_loader, model, device)

swin_train_auc = roc_auc_score(train_labels_fold, train_probs_fold)
swin_val_auc = roc_auc_score(val_labels_fold, val_probs_fold)
print(f" Swin Train AUC : {swin_train_auc:.4f}")
print(f" Swin Val   AUC : {swin_val_auc:.4f}")

# test 예측 (이 fold 모델 기준)
test_probs, _ = get_predictions(test_loader, model, device)

# ---- XGBoost ----
X_train_processed = preprocess(train_X.iloc[train_idx])
X_val_processed   = preprocess(train_X.iloc[val_idx])
y_train_fold = train_Y.iloc[train_idx].values
y_val_fold = train_Y.iloc[val_idx].values

xgb_model = XGBClassifier(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.01,
    subsample=0.6,
    colsample_bytree=0.6,
min_child_weight=5,
    reg_lambda=50,
    scale_pos_weight=scale_factor,
    eval_metric='logloss',
    tree_method='hist',
    random_state=42
)

xgb_model.fit(X_train_processed, y_train_fold)
xgb_train_pred = xgb_model.predict_proba(X_train_processed)[:, 1]
xgb_val_pred = xgb_model.predict_proba(X_val_processed)[:, 1]

xgb_train_auc = roc_auc_score(y_train_fold, xgb_train_pred)
xgb_val_auc = roc_auc_score(y_val_fold, xgb_val_pred)

print(f"XGB  Train AUC : {xgb_train_auc:.4f}")
print(f" XGB  Val   AUC : {xgb_val_auc:.4f}")

xgb_test_pred = xgb_model.predict_proba(preprocess(test_X))[:, 1]

print(f"corr(Swin_val, XGB_val): {np.corrcoef(val_probs_fold, xgb_val_pred)[0,1]:.4f}")

meta_train_X = np.vstack([
    val_probs_fold,
    xgb_val_pred,
]).T
meta_train_y = y_val_fold

meta_model = LogisticRegression(
    penalty="l2",
    C=1.0,
    class_weight="balanced",
    max_iter=1000,
    solver="lbfgs"
)

meta_model.fit(meta_train_X, meta_train_y)
meta_val_pred = meta_model.predict_proba(meta_train_X)[:,1]
meta_val_auc = roc_auc_score(meta_train_y, meta_val_pred)
print(f" Stack Val AUC  : {meta_val_auc:.4f}")

# ====================Test============================

meta_test_X = np.vstack([
    test_probs,
    xgb_test_pred,
]).T
stacked_test_pred = meta_model.predict_proba(meta_test_X)[:, 1]

submission = pd.read_csv("sample_submission.csv")
submission['probability'] = np.concatenate([stacked_test_pred,stacked_test_pred])

decision_id_L_list = submission.iloc[:466].sort_values('probability').iloc[:150]['ID']
decision_id_P_list = submission.iloc[466:].sort_values('probability').iloc[:150]['ID']

submission.loc[submission['ID'].isin(decision_id_L_list), 'decision'] = True
submission.loc[submission['ID'].isin(decision_id_P_list), 'decision'] = True


submission.to_csv("my_submission.csv", index=False)


