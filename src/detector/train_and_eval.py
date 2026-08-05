"""
train_and_eval.py — train the paper's Full-CAD injection classifier.

See the detector-training and metric-definition appendix sections.

Trains a small FNN on a train .pt, evaluates on a test .pt. Evaluation masks out
the boundary sentences of each injection span (keeps only the core
[start+3, end-5]) so the metric reflects clear-cut injections, not span edges.

The classifier architecture matches JinaClassifier in detector.variants
(d_hid=256, dropout=0.3), so the saved checkpoint loads directly at inference time
inside FullCADDefense. Inputs are the paper's query-aware context-delta embeddings
produced by detector.build_embeddings.

Usage:
  python -m detector.train_and_eval \
    outputs/embeddings/train_aug.pt outputs/embeddings/test.pt \
    --test_dir data/testing_data_original \
    --test_domains banking travel slack \
    --save_path outputs/models/classifier_aug.pt
"""

import os, re, json, glob, sys, argparse, random, torch
import torch.nn as nn
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def data_loader_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument("train_pt")
parser.add_argument("test_pt")
parser.add_argument("--test_domains", nargs="+", default=["banking", "travel", "slack"],
                    help="Which test domains to build the eval mask from.")
parser.add_argument("--test_dir", default="data/testing_data_original",
                    help="Directory holding the test-domain JSON subfolders (for eval mask).")
parser.add_argument("--save_path", default="outputs/models/classifier_aug.pt",
                    help="Where to save the trained classifier.")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
seed_everything(args.seed)

train_data = torch.load(resolve(args.train_pt), weights_only=True)
test_data  = torch.load(resolve(args.test_pt),  weights_only=True)
X_train, y_train = train_data["X"].to(torch.float32), train_data["y"]
X_test,  y_test  = test_data["X"].to(torch.float32),  test_data["y"]

print(f"Train: {tuple(X_train.shape)} | benign={(y_train==0).sum()}, inject={(y_train==1).sum()}")
print(f"Test:  {tuple(X_test.shape)}  | benign={(y_test==0).sum()}, inject={(y_test==1).sum()}")


# ── Build eval mask from test JSONs ──

def parse_segments(raw):
    if isinstance(raw, list): return raw
    matches = re.findall(r'\[(\d+)\]\s?(.*?)(?=\n\[\d+\]|$)', raw, re.DOTALL)
    segments = {int(idx): content.strip() for idx, content in matches}
    return [segments[k] for k in sorted(segments.keys())]


def build_eval_mask(test_dir, domains):
    """Mask over test rows, iterating JSONs in the same global sorted order
    build_embeddings.py uses, so it aligns row-for-row with the .pt. Boundary injection
    sentences are dropped; files outside the requested domains are excluded."""
    keep_domains = set(domains)
    mask = []
    for jf in sorted(glob.glob(os.path.join(test_dir, "**/*.json"), recursive=True)):
        domain = os.path.relpath(jf, test_dir).split(os.sep)[0]
        if domain not in keep_domains:
            continue
        with open(jf) as f: data = json.load(f)
        n = len(parse_segments(data["sentence_segments"]))
        s, e = data["injection_span_unit"]["start"], data["injection_span_unit"]["end"]
        core_s, core_e = s + 3, e - 5
        for i in range(n):
            mask.append(not (s <= i <= e and (i < core_s or i > core_e)))
    return np.array(mask)


eval_mask = build_eval_mask(resolve(args.test_dir), args.test_domains)
if len(eval_mask) != len(y_test):
    print(f"ERROR: mask size ({len(eval_mask)}) != test size ({len(y_test)})")
    print(f"Check that {args.test_dir} JSONs match the .pt file.")
    sys.exit(1)

n_total_inj = (y_test.numpy() == 1).sum()
n_masked = (~eval_mask & (y_test.numpy() == 1)).sum()
print(f"Eval mask: {n_masked} boundary injection sentences masked out of {n_total_inj}")


# ── Classifier (matches JinaClassifier in detector.variants) ──
class Classifier(nn.Module):
    def __init__(self, d_in, d_hid=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hid, d_hid // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hid // 2, 2),
        )
    def forward(self, x): return self.net(x)


# ── Train ──
model = Classifier(X_train.shape[1]).to(DEVICE)
# Inverse-frequency class weights compensate the benign/injection imbalance of
# sentence-level labels (few injected sentences per document).
counts = torch.bincount(y_train)
weights = (1.0 / counts.float()); weights /= weights.sum()
criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
# Training recipe from the "Detector Representation and Ablations" appendix: AdamW, lr 1e-3,
# 30 epochs (default), batch size 64.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
# Dedicated DataLoader generator: results are averaged over training seeds
# ("Statistical Variation Across Training Seeds" appendix), so shuffling must
# be reproducible per seed and must not consume the model's dropout RNG.
loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=64, shuffle=True,
    generator=data_loader_generator(args.seed + 30_007),
)

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        loss = criterion(model(xb), yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(loader):.4f}")


# ── Eval (masked) ──
model.eval()
with torch.no_grad():
    probs = torch.softmax(model(X_test.to(DEVICE)), dim=1)[:, 1].cpu().numpy()

y_np = y_test.numpy()
m_probs, m_labels = probs[eval_mask], y_np[eval_mask]

auprc   = average_precision_score(m_labels, m_probs)
roc_auc = roc_auc_score(m_labels, m_probs)
fpr_curve, tpr_curve, _ = roc_curve(m_labels, m_probs)

# Confusion at threshold 0.5 (not tuned on test) -> TPR / FPR
# tau = 0.5 is the fixed deployment decision rule p_j > tau (paper, Method
# section); reported here so offline TPR/FPR match the deployed detector.
pred = (m_probs >= 0.5).astype(int)
P, N = m_labels == 1, m_labels == 0
tp = int((pred[P] == 1).sum()); fn = int((pred[P] == 0).sum())
fp = int((pred[N] == 1).sum()); tn = int((pred[N] == 0).sum())
tpr = tp / (tp + fn + 1e-9); fpr = fp / (fp + tn + 1e-9)
best_thresh = 0.5

print(f"\n{'='*50}")
print(f"ROC-AUC: {roc_auc:.4f}   AUPRC: {auprc:.4f}")
print(f"@ threshold 0.5 | TPR={tpr:.4f} ({tp}/{tp+fn})   FPR={fpr:.4f} ({fp}/{fp+tn})")
print(f"{'='*50}")

# ── Draw ROC curve (TPR vs FPR) each run ──
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_curve, tpr_curve, lw=2, label=f"ROC (AUC={roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
ax.scatter([fpr], [tpr], c="crimson", zorder=5, label=f"thr 0.5  TPR={tpr:.2f} FPR={fpr:.2f}")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_xlim(-0.01, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
ax.set_title(f"ROC — {os.path.basename(args.train_pt)}\nAUC={roc_auc:.3f}  AUPRC={auprc:.3f}")
ax.legend(loc="lower right")
plt.tight_layout()
roc_png = os.path.splitext(resolve(args.save_path))[0] + "_roc.png"
os.makedirs(os.path.dirname(roc_png) or ".", exist_ok=True)
plt.savefig(roc_png, dpi=150)
print(f"ROC curve saved → {roc_png}")


# ── Save ──
save_path = resolve(args.save_path)
os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "input_dim": X_train.shape[1],
    "d_hid": 256,
    "dropout": 0.3,
    "best_threshold": float(best_thresh),
    "seed": args.seed,
}, save_path)
print(f"\nModel saved → {save_path}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
