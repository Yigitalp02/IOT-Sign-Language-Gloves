import csv, numpy as np
import pandas as pd

# ---------- Config ----------
USERS = 6
REPS_PER_CLASS = 10
CLASSES = {
    "A":[0.9,0.9,0.9,0.9,0.2],
    "B":[0.1,0.2,0.2,0.2,0.2],
    "C":[0.7,0.6,0.5,0.4,0.3],
    "D":[0.2,0.8,0.8,0.8,0.2],
}
FS = 100       # Hz
HOLD_MS = 800  # 0.8 s
REST_MS = 800  # 0.8 s
WINDOW_LEN = int(0.8 * FS)

def gen_calib(n_users=USERS):
    b = np.random.normal(500, 30, (n_users,5))
    m = b + np.random.normal(350, 40, (n_users,5))
    return b, m

def norm_to_raw(norm, b, m):
    return b + norm*(m - b)

def make_csv(path="data/synthetic_stream_v1.csv"):
    baselines, maxbends = gen_calib()
    cols = ["timestamp_ms","user_id","session_id","class_label"] + \
           [f"ch{k}_raw" for k in range(5)] + [f"ch{k}_norm" for k in range(5)] + \
           [f"baseline_ch{k}" for k in range(5)] + [f"maxbend_ch{k}" for k in range(5)] + \
           ["glove_fit","sensor_map_ref","notes"]
    sensor_map_ref = "CH0=index, CH1=middle, CH2=ring, CH3=little, CH4=thumb"
    ts = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for uid in range(USERS):
            b = baselines[uid]; m = maxbends[uid]
            for cls, center in CLASSES.items():
                for rep in range(REPS_PER_CLASS):
                    # REST
                    for _ in range(int(REST_MS*FS/1000)):
                        rest = np.clip(np.random.normal(0.05,0.03,5),0,1)
                        raw = norm_to_raw(rest, b, m)
                        norm = (raw - b) / (m - b + 1e-6)
                        row = [ts, f"U{uid:03d}", "S1", "REST"] + \
                              list(raw.round(2)) + list(norm.round(3)) + \
                              list(b.round(2)) + list(m.round(2)) + \
                              ["normal-fit", sensor_map_ref, ""]
                        w.writerow(row)
                        ts += int(1000/FS)
                    # HOLD (pose)
                    for _ in range(int(HOLD_MS*FS/1000)):
                        pose = np.clip(np.array(center) + np.random.normal(0,0.06,5),0,1)
                        raw = norm_to_raw(pose, b, m)
                        norm = (raw - b) / (m - b + 1e-6)
                        row = [ts, f"U{uid:03d}", "S1", cls] + \
                              list(raw.round(2)) + list(norm.round(3)) + \
                              list(b.round(2)) + list(m.round(2)) + \
                              ["normal-fit", sensor_map_ref, ""]
                        w.writerow(row)
                        ts += int(1000/FS)

def load_and_segment(path="data/synthetic_stream_v1.csv", window_len=WINDOW_LEN):
    df = pd.read_csv(path)
    df = df[df["class_label"].isin(CLASSES.keys())]
    feats, labels = [], []
    for (u,c), g in df.groupby(["user_id","class_label"]):
        g = g.reset_index(drop=True)
        for i in range(0, len(g)-window_len+1, window_len):
            block = g.iloc[i:i+window_len]
            X = block[[f"ch{k}_norm" for k in range(5)]].to_numpy()
            feats.append(X.mean(axis=0))
            labels.append(c)
    return np.array(feats), np.array(labels)

def train_eval(X, y, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    split = int(0.8*len(X))
    Xtr, Xte = X[idx[:split]], X[idx[split:]]
    ytr, yte = y[idx[:split]], y[idx[split:]]
    classes = sorted(set(ytr))
    centroids = {c: Xtr[ytr==c].mean(axis=0) for c in classes}
    def predict(Z):
        preds=[]
        for z in Z:
            d = {c: np.linalg.norm(z-centroids[c]) for c in classes}
            preds.append(min(d, key=d.get))
        return np.array(preds)
    yhat = predict(Xte)
    acc = (yhat==yte).mean()
    return acc, classes

if __name__ == "__main__":
    make_csv()
    X, y = load_and_segment()
    acc, classes = train_eval(X, y)
    print(f"Classes: {classes}")
    print(f"Nearest-centroid accuracy on synthetic windows: {acc*100:.1f}%")
