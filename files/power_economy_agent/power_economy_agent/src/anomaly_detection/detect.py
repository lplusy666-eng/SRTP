"""
VAE 异常检测：训练 + 打分 + 事件聚合
====================================
输出 outputs/anomalies.csv：逐日异常分数、是否异常、聚合后的异常事件区间。
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import abspath, get_logger, ensure_dir, load_config
from anomaly_detection.vae_model import TimeSeriesVAE, vae_loss

log = get_logger("vae")


# ---------- 构造滑窗样本 ----------
def make_windows(df: pd.DataFrame, feature_cols, window: int):
    """返回 (N, window*n_feat) 的窗口矩阵，以及每个窗口中心日期索引。"""
    feats = df[feature_cols].values.astype(np.float32)
    # 标准化（按列）
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + 1e-8
    feats = (feats - mu) / sd

    X, centers = [], []
    for i in range(len(feats) - window + 1):
        X.append(feats[i:i + window].reshape(-1))
        centers.append(i + window // 2)
    return np.array(X, dtype=np.float32), np.array(centers)


# ---------- 训练 ----------
def train_vae(cfg: dict):
    torch.manual_seed(42)
    np.random.seed(42)
    dec_csv = abspath(cfg["decoupling"]["output_csv"])
    df = pd.read_csv(dec_csv, parse_dates=["date"])

    feature_cols = ["economic_deviation", "economic_residual"]
    window = cfg["vae"]["window"]
    X, centers = make_windows(df, feature_cols, window)
    input_dim = X.shape[1]
    log.info("VAE 训练样本: %s, 输入维度=%d", X.shape, input_dim)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TimeSeriesVAE(input_dim,
                          hidden_dim=cfg["vae"]["hidden_dim"],
                          latent_dim=cfg["vae"]["latent_dim"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["vae"]["lr"])

    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=cfg["vae"]["batch_size"], shuffle=True)

    model.train()
    for ep in range(cfg["vae"]["epochs"]):
        total = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss, _ = vae_loss(recon, xb, mu, logvar)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        if (ep + 1) % 10 == 0 or ep == 0:
            log.info("epoch %2d/%d  loss=%.4f", ep + 1, cfg["vae"]["epochs"], total / len(ds))

    ensure_dir(abspath(cfg["vae"]["model_path"]))
    torch.save({"state_dict": model.state_dict(),
                "input_dim": input_dim,
                "feature_cols": feature_cols,
                "window": window}, abspath(cfg["vae"]["model_path"]))
    log.info("VAE 模型已保存: %s", abspath(cfg["vae"]["model_path"]))
    return model, df, X, centers, feature_cols, window


# ---------- 打分与事件聚合 ----------
def detect(cfg: dict):
    model, df, X, centers, feature_cols, window = train_vae(cfg)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        recon, mu, logvar = model(torch.from_numpy(X).to(device))
        err = ((recon - torch.from_numpy(X).to(device)) ** 2).mean(dim=1).cpu().numpy()

    # 映射回逐日分数
    scores = np.full(len(df), np.nan)
    scores[centers] = err
    vae_score = pd.Series(scores).interpolate().bfill().ffill().values

    # 统计偏离度（稳健 z 分数），对平缓且持续的经济冲击更敏感
    dev = df["economic_deviation"].abs().values
    med = np.median(dev)
    mad = np.median(np.abs(dev - med)) + 1e-8
    dev_z = np.abs(dev - med) / (1.4826 * mad)

    # 归一化后融合（VAE 抓形态突变，偏离度抓持续偏移）
    def _norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-12)
    s = 0.5 * _norm(vae_score) + 0.5 * _norm(dev_z)

    thr = np.nanquantile(s, cfg["vae"]["anomaly_quantile"])
    is_anom = (s > thr).astype(int)

    df_out = df[["date"]].copy()
    df_out["anomaly_score"] = s
    df_out["is_anomaly"] = is_anom
    if "_true_anomaly" in df.columns:
        df_out["_true_anomaly"] = df["_true_anomaly"]

    # 聚合成事件区间
    events = _aggregate_events(df_out)
    ensure_dir(abspath(cfg["vae"]["anomaly_csv"]))
    df_out.to_csv(abspath(cfg["vae"]["anomaly_csv"]), index=False)
    log.info("异常检测完成: 阈值=%.4f, 检出异常日=%d, 事件数=%d",
             thr, int(is_anom.sum()), len(events))

    # 若有真值则评估
    if "_true_anomaly" in df_out.columns:
        _evaluate(df_out)
    return df_out, events


def _aggregate_events(df_out: pd.DataFrame, min_len: int = 5):
    """把连续异常日聚合成事件区间。"""
    events = []
    in_evt = False
    start = None
    for i, row in df_out.iterrows():
        if row["is_anomaly"] == 1 and not in_evt:
            in_evt, start = True, i
        elif row["is_anomaly"] == 0 and in_evt:
            in_evt = False
            if i - start >= min_len:
                seg = df_out.iloc[start:i]
                events.append({
                    "start": str(seg["date"].iloc[0].date()),
                    "end": str(seg["date"].iloc[-1].date()),
                    "peak_score": float(seg["anomaly_score"].max()),
                    "duration_days": int(i - start),
                })
    if in_evt and len(df_out) - start >= min_len:
        seg = df_out.iloc[start:]
        events.append({
            "start": str(seg["date"].iloc[0].date()),
            "end": str(seg["date"].iloc[-1].date()),
            "peak_score": float(seg["anomaly_score"].max()),
            "duration_days": int(len(df_out) - start),
        })
    return events


def _evaluate(df_out: pd.DataFrame):
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    y = df_out["_true_anomaly"].values
    p = df_out["is_anomaly"].values
    try:
        auc = roc_auc_score(y, df_out["anomaly_score"].values)
    except Exception:
        auc = float("nan")
    log.info("评估(vs真值): P=%.2f R=%.2f F1=%.2f AUC=%.3f",
             precision_score(y, p, zero_division=0),
             recall_score(y, p, zero_division=0),
             f1_score(y, p, zero_division=0), auc)


if __name__ == "__main__":
    cfg = load_config()
    out, events = detect(cfg)
    print(out.head())
    for e in events:
        print(e)
