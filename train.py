"""Train the paper's 1D-CNN on synthetic TENG grasp signals and compare with SVM / RF.

Usage:  python train.py --device auxetic          (Figs. 6j-n, S19)
        python train.py --device non_auxetic      (Fig. S18)
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import TengCNN
from teng_signals import CLASS_NAMES, FS, load_comsol_table, make_dataset

HERE = Path(__file__).parent


class RandomForest:
    """Bootstrap-aggregated decision trees with sqrt(n_features) splits, i.e. a random forest.

    Built on DecisionTreeClassifier because Windows Smart App Control blocks the compiled
    sklearn.ensemble module on this machine.
    """

    def __init__(self, n_estimators=300, seed=0):
        self.n_estimators, self.seed = n_estimators, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.n_classes = int(y.max()) + 1
        self.trees = []
        for i in range(self.n_estimators):
            idx = rng.integers(0, len(y), len(y))
            tree = DecisionTreeClassifier(max_features="sqrt", random_state=self.seed + i)
            self.trees.append(tree.fit(X[idx], y[idx]))
        return self

    def predict_proba(self, X):
        prob = np.zeros((len(X), self.n_classes))
        for tree in self.trees:
            prob[:, tree.classes_] += tree.predict_proba(X)
        return prob / len(self.trees)


def train_cnn(Xtr, ytr, Xte, yte, epochs, seed):
    torch.manual_seed(seed)
    model = TengCNN(length=Xtr.shape[1], n_classes=len(CLASS_NAMES))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    to_t = lambda X: torch.from_numpy(X[:, None, :])
    loader = DataLoader(TensorDataset(to_t(Xtr), torch.from_numpy(ytr)), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    Xte_t, yte_t = to_t(Xte), torch.from_numpy(yte)
    hist = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for ep in range(epochs):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            tot += len(yb)
        model.eval()
        with torch.no_grad():
            out = model(Xte_t)
            hist["test_loss"].append(loss_fn(out, yte_t).item())
            hist["test_acc"].append((out.argmax(1) == yte_t).float().mean().item())
        hist["train_loss"].append(loss_sum / tot)
        hist["train_acc"].append(correct / tot)
        if (ep + 1) % 10 == 0:
            print(f"epoch {ep + 1:3d}  train loss {hist['train_loss'][-1]:.4f}  "
                  f"train acc {hist['train_acc'][-1]:.3f}  test acc {hist['test_acc'][-1]:.3f}", flush=True)
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(Xte_t), 1).numpy()
        emb = model.embed(Xte_t).numpy()
    return model, hist, prob, emb


def roc_micro(y, prob):
    yb = label_binarize(y, classes=range(len(CLASS_NAMES)))
    fpr, tpr, _ = roc_curve(yb.ravel(), prob.ravel())
    return fpr, tpr, auc(fpr, tpr)


def plot_confusion(cm, title, path):
    cmn = cm / cm.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cmn[i, j] >= 0.5:
                ax.text(j, i, f"{cmn[i, j]:.1f}", ha="center", va="center",
                        color="white" if cmn[i, j] > 60 else "black", fontsize=9)
    ax.set_xticks(range(6), [str(i + 1) for i in range(6)])
    ax.set_yticks(range(6), [str(i + 1) for i in range(6)])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="% of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auxetic", choices=["auxetic", "non_auxetic"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n_per_class", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.set_num_threads(4)

    out = HERE / "results" / args.device
    out.mkdir(parents=True, exist_ok=True)
    table = load_comsol_table(HERE / "data" / "comsol_fe_voc_c_vs_gap.csv")

    t0 = time.time()
    X, y = make_dataset(table, args.device, args.n_per_class, seed=args.seed)
    np.savez_compressed(out / "dataset.npz", X=X, y=y)
    print(f"dataset {X.shape} in {time.time() - t0:.1f} s", flush=True)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)
    scale = Xtr.std()
    Xtr, Xte = (Xtr / scale).astype(np.float32), (Xte / scale).astype(np.float32)

    # Fig. 6j analogue: one example per class
    fig, axes = plt.subplots(6, 1, figsize=(9, 9), sharex=True)
    t = np.arange(X.shape[1]) / FS
    for c, ax in enumerate(axes):
        ax.plot(t, X[y == c][0], lw=0.8, color="#1f6fb4" if c < 3 else "#c0392b")
        ax.set_ylabel("V", fontsize=8)
        ax.set_title(CLASS_NAMES[c], fontsize=9, loc="left")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Synthetic TENG grasp signals ({args.device.replace('_', '-')} device)")
    fig.tight_layout()
    fig.savefig(out / "fig_signals.png", dpi=150)
    plt.close(fig)

    t0 = time.time()
    model, hist, prob, emb = train_cnn(Xtr, ytr, Xte, yte, args.epochs, args.seed)
    cnn_time = time.time() - t0
    torch.save(model.state_dict(), out / "cnn.pt")
    pred = prob.argmax(1)
    acc_cnn = accuracy_score(yte, pred)

    svm = CalibratedClassifierCV(SVC(kernel="rbf", C=10, gamma="scale"), ensemble=False).fit(Xtr, ytr)
    rf = RandomForest(n_estimators=300, seed=args.seed).fit(Xtr, ytr)
    prob_svm, prob_rf = svm.predict_proba(Xte), rf.predict_proba(Xte)
    acc_svm, acc_rf = accuracy_score(yte, prob_svm.argmax(1)), accuracy_score(yte, prob_rf.argmax(1))

    # Fig. 6l / 6m: accuracy and loss curves
    ep = np.arange(1, args.epochs + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, np.array(hist["train_acc"]) * 100, label="Train")
    a1.plot(ep, np.array(hist["test_acc"]) * 100, label="Test")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Accuracy (%)"); a1.legend(); a1.set_title("Accuracy")
    a2.plot(ep, hist["train_loss"], label="Train")
    a2.plot(ep, hist["test_loss"], label="Test")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Cross-entropy loss"); a2.legend(); a2.set_title("Loss")
    fig.tight_layout(); fig.savefig(out / "fig_training_curves.png", dpi=150); plt.close(fig)

    # Fig. 6k / S19b-c: confusion matrices
    plot_confusion(confusion_matrix(yte, pred), f"CNN, accuracy {acc_cnn * 100:.1f}%", out / "fig_confusion_cnn.png")
    plot_confusion(confusion_matrix(yte, prob_svm.argmax(1)), f"SVM, accuracy {acc_svm * 100:.1f}%", out / "fig_confusion_svm.png")
    plot_confusion(confusion_matrix(yte, prob_rf.argmax(1)), f"Random forest, accuracy {acc_rf * 100:.1f}%", out / "fig_confusion_rf.png")

    # Fig. S19a + S19d: accuracy bars and micro-averaged ROC
    rocs = {name: roc_micro(yte, pr) for name, pr in [("CNN", prob), ("RF", prob_rf), ("SVM", prob_svm)]}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    names, accs = ["SVM", "RF", "CNN"], [acc_svm, acc_rf, acc_cnn]
    bars = a1.bar(names, np.array(accs) * 100, color=["#9aa5b1", "#9aa5b1", "#1f6fb4"])
    for b, a in zip(bars, accs):
        a1.text(b.get_x() + b.get_width() / 2, a * 100 + 1, f"{a * 100:.1f}%", ha="center")
    a1.set_ylim(0, 110); a1.set_ylabel("Test accuracy (%)"); a1.set_title("Classifier comparison")
    for name, (fpr, tpr, a) in rocs.items():
        a2.plot(fpr, tpr, label=f"{name} (AUC = {a:.3f})")
    a2.plot([0, 1], [0, 1], "k--", lw=0.8)
    a2.set_xlabel("False positive rate"); a2.set_ylabel("True positive rate")
    a2.set_title("ROC, micro-averaged over 6 classes"); a2.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig_comparison_roc.png", dpi=150); plt.close(fig)

    # Fig. 6n: t-SNE of the CNN's FC1 features
    z = TSNE(n_components=2, perplexity=min(30, len(emb) - 1), random_state=args.seed).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for c in range(6):
        ax.scatter(z[yte == c, 0], z[yte == c, 1], s=10, label=CLASS_NAMES[c])
    ax.set_title("t-SNE of CNN features (test set)"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7, markerscale=1.5)
    fig.tight_layout(); fig.savefig(out / "fig_tsne.png", dpi=150); plt.close(fig)

    metrics = {
        "device": args.device, "n_train": int(len(ytr)), "n_test": int(len(yte)), "epochs": args.epochs,
        "accuracy": {"cnn": acc_cnn, "svm": acc_svm, "rf": acc_rf},
        "auc_micro": {k: v[2] for k, v in rocs.items()},
        "cnn_final_train_acc": hist["train_acc"][-1], "cnn_final_train_loss": hist["train_loss"][-1],
        "cnn_epoch_reaching_95pct_test": next((i + 1 for i, a in enumerate(hist["test_acc"]) if a >= 0.95), None),
        "cnn_per_class_acc": (confusion_matrix(yte, pred).diagonal() / np.bincount(yte)).tolist(),
        "cnn_train_time_s": cnn_time,
        "cnn_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
