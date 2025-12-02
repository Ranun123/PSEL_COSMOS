import pandas as pd
import numpy as np
import os
import timm
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostClassifier
from category_encoders import CatBoostEncoder
from xgboost import XGBClassifier
from scipy.stats import skew, kurtosis
import warnings

from Swin_grid import SwinTransformer, extract_latent, get_swin_latent_features, depth_tensor, depth_tensor_test, depth_mean, depth_std, g_train, g_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

warnings.filterwarnings("ignore")

model = SwinTransformer(in_chans=8).to(device)
model.load_state_dict(torch.load("best_swin_auc.pth", map_location=device))
model.eval()

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

train_X = train.drop(columns=['Class']).iloc[:,:-256*3]
train_y = train['Class'].apply(lambda x: 1 if x == 'NG' else 0)
test_X = test.drop(columns=['ID']).iloc[:,:-256*3]

X_img_train, X_img_test, y_from_loader = get_swin_latent_features(
        model, device,
        depth_tensor, g_train, train_y,
        depth_tensor_test, g_test,
        depth_mean, depth_std,
        batch_size=64,
    )

cat_list = train_X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
num_list = sorted(list(set(train_X.columns) - set(cat_list)))

def add_row_stats(df_num: pd.DataFrame) -> pd.DataFrame:
    df = df_num.copy()
    df_stats = pd.DataFrame(index=df.index)
    df_stats['r_mean'] = df.mean(axis=1)
    df_stats['r_std'] = df.std(axis=1)
    df_stats['r_min'] = df.min(axis=1)
    df_stats['r_max'] = df.max(axis=1)
    df_stats['r_median'] = df.median(axis=1)
    df_stats['r_range'] = df_stats['r_max'] - df_stats['r_min']
    df_stats['r_skew'] = df.apply(lambda r: skew(r), axis=1)
    df_stats['r_kurtosis'] = df.apply(lambda r: kurtosis(r), axis=1)
    df_stats['r_std_to_mean'] = df_stats['r_std'] / (df_stats['r_mean'].replace(0, np.nan)).abs()
    df_stats['r_max_to_min'] = df_stats['r_max'] / (df_stats['r_min'].replace(0, np.nan)).abs()
    df_stats = df_stats.fillna(0)
    return pd.concat([df, df_stats], axis=1)

X_num = train_X[num_list].apply(pd.to_numeric)
X_test_num = test_X[num_list].apply(pd.to_numeric)

X_num_stats = add_row_stats(X_num)
X_test_num_stats = add_row_stats(X_test_num)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scale_pos_weight = 5.7

model_cat = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        verbose=0,
        class_weights=[1.0, float(scale_pos_weight)],
        task_type="CPU",        # GPU 쓰면 비결정성 있을 수 있음
        bootstrap_type="No"
    )

model_xgb = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=float(scale_pos_weight),
        tree_method="hist",
        random_state=42,
        verbosity=0
    )

oof_preds = np.zeros(len(train_y))
test_preds = np.zeros(len(test_X))

for fold, (tr_idx, val_idx) in enumerate(kf.split(train_X, train_y), 1):
    print(f"\n===== Fold {fold} =====")

    X_tr_num = X_num_stats.iloc[tr_idx]
    X_val_num = X_num_stats.iloc[val_idx]

    X_tr_cat = train_X[cat_list].iloc[tr_idx]
    X_val_cat = train_X[cat_list].iloc[val_idx]

    y_tr = train_y.iloc[tr_idx].values
    y_val = train_y.iloc[val_idx].values

    cbe = CatBoostEncoder(cols=cat_list, random_state=42, sigma=1.0)
    cbe.fit(X_tr_cat, y_tr)

    scaler = StandardScaler().fit(X_tr_num)

    # 3) transform (train / val / test)
    X_tr = np.concatenate([
        cbe.transform(X_tr_cat).values.astype("float32"),
        scaler.transform(X_tr_num).astype("float32")
    ], axis=1)

    X_val = np.concatenate([
        cbe.transform(X_val_cat).values.astype("float32"),
        scaler.transform(X_val_num).astype("float32")
    ], axis=1)

    X_test_fold = np.concatenate([
        cbe.transform(test_X[cat_list]).values.astype("float32"),
        scaler.transform(X_test_num_stats).astype("float32")
    ], axis=1)

    X_tr_img = X_img_train[tr_idx]
    X_val_img = X_img_train[val_idx]

    X_tr = np.concatenate([X_tr, X_tr_img], axis=1)
    X_val = np.concatenate([X_val, X_val_img], axis=1)
    X_test_fold = np.concatenate([X_test_fold, X_img_test], axis=1)

    m_cat = model_cat.fit(X_tr, y_tr)
    m_xgb = model_xgb.fit(X_tr, y_tr)

    p_cat_tr = m_cat.predict_proba(X_tr)[:, 1]
    p_xgb_tr = m_xgb.predict_proba(X_tr)[:, 1]
    p_tr_ens = (p_cat_tr + p_xgb_tr) / 2.0

    p_cat = m_cat.predict_proba(X_val)[:, 1]
    p_xgb = m_xgb.predict_proba(X_val)[:, 1]
    p_ens = (p_cat + p_xgb) / 2.0

    oof_preds[val_idx] = p_ens

    # --- Compute AUC ---
    train_auc = roc_auc_score(y_tr, p_tr_ens)
    val_auc = roc_auc_score(y_val, p_ens)

    print(f"Fold {fold} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

    # test
    p_cat_test = m_cat.predict_proba(X_test_fold)[:, 1]
    p_xgb_test = m_xgb.predict_proba(X_test_fold)[:, 1]
    test_preds += (p_cat_test + p_xgb_test) / 2.0 / kf.n_splits


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

global_auc = roc_auc_score(train_y, oof_preds)
print(f"\nOverall OOF AUC: {global_auc:.4f}")

N = find_best_N(oof_preds, train_y.values)
print(f"Best N = {N}")

idx_sorted = np.argsort(oof_preds)
pred_bin = np.ones(len(oof_preds), dtype=int)
pred_bin[idx_sorted[:N]] = 0

decision_val = (pred_bin == 0)

cm_val = confusion_matrix(train_y, pred_bin)
disp = ConfusionMatrixDisplay(cm_val, display_labels=['OK(0)', 'NG(1)'])
disp.plot(cmap='Blues', values_format='d')
plt.title("Validation Confusion Matrix")
plt.show()


Total_net_Profit = competition_score(train_y, oof_preds, decision_val)
print(f"Total_net_Profit: {Total_net_Profit}")

# Test =============
submission = pd.read_csv("sample_submission.csv")
submission['probability'] = np.concatenate([test_preds,test_preds])

test_idx_sorted = np.argsort(test_preds)

test_pred_bin = np.ones(len(test_preds), dtype=int)

approval_rate = N / len(oof_preds)
test_N = int(round(approval_rate * len(test_preds)))
print(f"test N: {test_N}")
test_pred_bin[test_idx_sorted[:test_N]] = 0

test_pred = np.concatenate([test_pred_bin, test_pred_bin])

decision_id_L_list = submission.iloc[:466].loc[test_pred[:466] == 0, 'ID']
decision_id_P_list = submission.iloc[466:].loc[test_pred[466:] == 0, 'ID']

submission.loc[submission['ID'].isin(decision_id_L_list), 'decision'] = True
submission.loc[submission['ID'].isin(decision_id_P_list), 'decision'] = True


submission.to_csv("my_submission.csv", index=False)


