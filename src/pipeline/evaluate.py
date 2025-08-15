"""
Evaluate a trained classifier on a class-structured test set.

What this script does
---------------------
1) Loads the resized test dataset (from `resize.py`) and image transforms.
2) **Aligns class encoding** using `index_remap.json`:
   - If mapping is missing/unreadable → WARN, proceed with dataset order.
   - If same set but different order → re-map dataset to expected order (WARN).
   - If sets differ → WARN, proceed (metrics may be misleading).
3) Loads the trained model weights and runs evaluation.
4) Saves:
   - Confusion matrices (counts + row-normalized),
   - A `.txt` classification report,
   - Image galleries (top mistakes & top correct per true class),
   - Grad-CAM overlays for both groups,
   - `evaluation_summary.json` with headline metrics.

Typical pipeline order
----------------------
fetch → split → resize → validate → train → **evaluate**

Examples
--------
python -m src.pipeline.evaluate \
  --eval-in data/testing_resized \
  --trained-model models/best_valF1_0.9123_epoch14.pth \
  --model resnet18 --image-size 224 --batch-size 64
"""



from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from src.utils.logging_utils import get_logger, configure_logging
from src.utils.parser_utils import add_common_logging_args, add_common_eval_args
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, resnet34, resnet50
from torchvision import transforms

import argparse, numpy as np, json, time, os
from pathlib import Path
from src.utils.paths import OUTPUTS_DIR
from src.core.env import bootstrap_env, log_env_once
from src.core.mapping import read_index_remap, expected_classes_from_remap
from src.core.transforms import build_transforms 

import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from PIL import Image, ImageOps
from datetime import datetime, timezone
import os as _os

log = get_logger(__name__)


def make_parser_evaluate() -> argparse.ArgumentParser:
    """
    Evaluate a trained classifier on a class-structured test folder.

    Examples
    --------
    python -m src.pipeline.evaluate \
      --eval-in data/testing_resized \
      --trained-model models/best_valF1_0.9123_epoch14.pth \
      --model resnet18 --image-size 224 --batch-size 64
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier on a test set.")
    parser.add_argument("--image-size", type=int, default=224, help="Square size used in preprocessing")
    parser.add_argument("--model", choices=["resnet18","resnet34","resnet50"], default="resnet18")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=OUTPUTS_DIR / "mappings" / "latest.json",
        help="Path to index remap mapping JSON (default: OUTPUTS_DIR/mappings/latest.json)",
    )
    add_common_eval_args(parser)     # --eval-in, --eval-out, --trained-model
    add_common_logging_args(parser)  # --log-level, --log-file
    return parser


def _worker_init_fn(worker_id: int):
    """Deterministically seed each DataLoader worker."""
    base_seed = torch.initial_seed() % (2**32)
    np.random.seed(base_seed + worker_id)


def make_eval_loader(dataset, batch_size: int, num_workers: int, seed: int) -> DataLoader:
    """
    Deterministic, non-shuffled DataLoader for evaluation.

    Returns
    -------
    DataLoader
    """
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_fn,
        generator=g,
    )
    log.info("eval_loader_created", extra={
        "num_samples": len(dataset), "batch_size": batch_size, "num_workers": num_workers
    })
    return loader


def build_model(model_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    Construct a ResNet and replace the classification head for `num_classes`.
    """
    name = model_name.lower()
    if name == "resnet18":
        model = resnet18(weights=None if not pretrained else None)
    elif name == "resnet34":
        model = resnet34(weights=None if not pretrained else None)
    elif name == "resnet50":
        model = resnet50(weights=None if not pretrained else None)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    log.info("model_built", extra={"arch": model_name, "num_classes": num_classes, "pretrained": pretrained})
    return model


def load_weights(model: nn.Module, weights_path: str | Path, device: torch.device) -> None:
    """
    Load a checkpoint into `model` (handles common 'module.' prefixes).
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        log.error("weights_file_not_found", extra={"weights_file": str(weights_path)})
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    log.info("weights_loaded", extra={"weights_file": str(weights_path)})


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Compute accuracy, precision, recall, F1 and return raw predictions.

    Returns
    -------
    (acc, prec, rec, f1, y_true, y_pred)
    """
    model.eval()
    all_preds, all_labels = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(yb.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    log.info("evaluation_metrics", extra={
        "acc": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4),
        "samples": len(y_true)
    })
    return acc, prec, rec, f1, y_true, y_pred


def verify_class_order_with_index_remap(ds: ImageFolder, mapping_path: Path) -> list[str]:
    """
    Align `ds.classes` to the mapping.

    Behavior
    --------
    - If mapping missing/unreadable → WARNING, return `ds.classes`.
    - If orders already match        → INFO,    return `ds.classes`.
    - If same set, diff order        → re-map `class_to_idx`, `samples`, `targets`; WARNING.
    - If sets differ                 → WARNING, return `ds.classes` (metrics may be misleading).

    Supports mapping formats:
      A) {"class_names": ["glioma", "meningioma", ...]}
      B) {"0":"glioma","1":"meningioma", ...}  (preferred; created by `split.py`)
    """
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        log.warning("mapping_missing", extra={"expected_path": str(mapping_path)})
        return ds.classes

    # NEW: use shared mapping reader if dict form; fall back to legacy "class_names"
    try:
        obj = json.loads(mapping_path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "class_names" in obj:
            expected = list(obj["class_names"])
        else:
            # preferred core path: str-int keys -> class
            idx_to_class = read_index_remap(mapping_path)
            expected = expected_classes_from_remap(idx_to_class)
    except Exception as e:
        log.warning("mapping_read_failed", extra={"path": str(mapping_path), "error": str(e)})
        return ds.classes

    if ds.classes == expected:
        log.info("class_mapping_verified", extra={"classes": expected})
        return ds.classes

    if set(ds.classes) == set(expected):
        old = list(ds.classes)
        new_class_to_idx = {c: i for i, c in enumerate(expected)}
        ds.class_to_idx = new_class_to_idx
        # rebuild samples/targets using folder names
        new_samples = []
        for p, _ in ds.samples:
            cls = Path(p).parent.name
            if cls not in new_class_to_idx:
                log.error("sample_class_not_in_mapping", extra={"path": p, "cls": cls})
                continue
            new_samples.append((p, new_class_to_idx[cls]))
        ds.samples = new_samples
        ds.targets = [y for _, y in new_samples]
        ds.classes = expected
        log.warning("class_order_mismatch_fixed", extra={"old_order": old, "new_order": expected})
        return ds.classes

    log.warning("class_set_mismatch", extra={
        "dataset_classes": ds.classes, "mapping_classes": expected,
        "note": "Proceeding without remap; metrics may be wrong."
    })
    return ds.classes


def _slugify(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in text).strip("-")


def report_on_loader(model, loader, class_names, device, out_dir: Path, title: str = "Test"):
    """
    Run inference on a DataLoader and save confusion matrices + a classification report.

    Saves two figures to `out_dir`:
      1) {title}_confusion_counts.png
      2) {title}_confusion_row-normalized.png

    Returns
    -------
    (y_true, y_pred)
    """
    model.eval()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(yb.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    # classification report (to disk only)
    try:
        cr_txt = classification_report(y_true, y_pred, target_names=class_names, digits=3, zero_division=0)
        cr_path = out_dir / f"{_slugify(title)}_classification_report.txt"
        cr_path.write_text(cr_txt, encoding="utf-8")
        log.info("classification_report_saved", extra={"path": str(cr_path)})
    except Exception as e:
        log.warning("classification_report_failed", extra={"error": str(e)})

    # confusion matrices
    mats = [
        (None, "Counts", f"{_slugify(title)}_confusion_counts.png"),
        ("true", "Row-normalized (Recall)", f"{_slugify(title)}_confusion_row-normalized.png"),
    ]
    ticks = np.arange(len(class_names))

    for normalize, name, fname in mats:
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, interpolation="nearest")
        plt.title(f"{title} — Confusion Matrix ({name})")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(ticks, class_names, rotation=45, ha="right")
        plt.yticks(ticks, class_names)
        # annotate
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                txt = f"{cm[i, j]:.2f}" if normalize else str(int(cm[i, j]))
                plt.text(j, i, txt, ha="center", va="center")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        save_path = out_dir / fname
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("confusion_matrix_saved", extra={"normalize": normalize or "none", "path": str(save_path)})

    return y_true, y_pred


def get_paths_for_dataset(ds):
    """Return file paths in a Dataset or Subset, aligned with samples order."""
    if isinstance(ds, Subset):
        base, idxs = ds.dataset, ds.indices
        return [base.samples[i][0] for i in idxs]
    else:
        return [s[0] for s in ds.samples]


@torch.inference_mode()
def infer_collect(model, loader, device):
    """
    Run forward passes and collect y_true, y_pred, and per-class probabilities.
    """
    model.eval()
    all_y, all_pred, all_prob = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        all_y.append(yb.cpu().numpy())
        all_pred.append(probs.argmax(1).cpu().numpy())
        all_prob.append(probs.cpu().numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    probs  = np.concatenate(all_prob)
    return y_true, y_pred, probs


def top_mistakes_per_true_class(y_true, y_pred, probs, paths, max_per_class: int, n_classes: int):
    """
    For each true class, select up to `max_per_class` most confident *wrong* predictions.
    """
    out = []
    for c in range(n_classes):
        mask = (y_true == c) & (y_pred != c)
        conf = probs[mask, y_pred[mask]] if mask.any() else np.array([])
        idxs = np.argsort(-conf)[:max_per_class] if conf.size else []
        for i in idxs:
            sel = np.flatnonzero(mask)[i]
            out.append({"true": int(y_true[sel]), "pred": int(y_pred[sel]), "conf": float(conf[i]), "path": paths[sel]})
    return out


def top_correct_per_true_class(y_true, y_pred, probs, paths, max_per_class: int, n_classes: int):
    """
    For each true class, select up to `max_per_class` most confident *correct* predictions.
    """
    out = []
    for c in range(n_classes):
        mask = (y_true == c) & (y_pred == c)
        conf = probs[mask, c] if mask.any() else np.array([])
        idxs = np.argsort(-conf)[:max_per_class] if conf.size else []
        for i in idxs:
            sel = np.flatnonzero(mask)[i]
            out.append({"true": c, "pred": c, "conf": float(conf[i]), "path": paths[sel]})
    return out


def show_calls_gallery(predictions, class_names, cols: int, title: str, save_dir: str | None):
    """
    Render a gallery image for a set of predictions (correct or mistakes).
    """
    rows = int(np.ceil(len(predictions) / cols))
    plt.figure(figsize=(3*cols, 3.3*rows))
    for i, m in enumerate(predictions, 1):
        ax = plt.subplot(rows, cols, i)
        img = ImageOps.exif_transpose(Image.open(m["path"])).convert("L")
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{class_names[m['true']]} → {class_names[m['pred']]}\nconf={m['conf']:.2f}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(save_dir, f"{_slugify(title)}.png")
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        log.info("gallery.saved", extra={"path": out_path, "n_items": len(predictions)})
    plt.close()


def _pick_last_conv_layer(model: nn.Module):
    # crude heuristic for resnets; if needed, allow override via CLI later
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return module
    raise RuntimeError("Could not find a convolutional layer for Grad-CAM.")

def show_gradcam_gallery(
    predictions, class_names, model, device, ds_for_transform, cols: int, title: str, target_layer=None, save_dir: Path | None = None
):
    """
    Render Grad-CAM overlays for a set of predictions (paths + indices).
    """
    target_layer = target_layer or _pick_last_conv_layer(model)
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=device.type == "cuda")

    rows = int(np.ceil(len(predictions) / cols)) or 1
    plt.figure(figsize=(3.5*cols, 3.8*rows))
    for i, m in enumerate(predictions, 1):
        ax = plt.subplot(rows, cols, i)
        img = ImageOps.exif_transpose(Image.open(m["path"])).convert("RGB")
        img_arr = np.array(img).astype(np.float32) / 255.0
        x = ds_for_transform.transform(img).unsqueeze(0).to(device)
        grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(m["pred"])])[0]
        overlay = show_cam_on_image(img_arr, grayscale_cam, use_rgb=True)
        ax.imshow(overlay)
        ax.set_title(f"{class_names[m['true']]} → {class_names[m['pred']]}\nconf={m['conf']:.2f}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(save_dir, f"{_slugify(title)}.png")
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        log.info("gradcam.saved", extra={"path": out_path, "n_items": len(predictions)})
    plt.close()

def main(argv=None):
    """
    CLI entry:
    1) Parse args and configure run/stage-aware logging.
    2) Build test dataset and align class encoding using the mapping.
    3) Evaluate and write artifacts (confusion matrices, reports, galleries, Grad-CAM, summary).
    """
    # 1) Parse CLI
    parser = make_parser_evaluate()
    args = parser.parse_args(argv)

    # 2) Configure logging with run_id + stage='evaluate'
    run_id = _os.getenv("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    if args.log_file:
        configure_logging(log_level=args.log_level, file_mode="fixed", log_file=args.log_file, run_id=run_id, stage="evaluate")
    else:
        configure_logging(log_level=args.log_level, file_mode="auto", run_id=run_id, stage="evaluate")
    log.info("evaluate.start", extra={"args": {k: (str(v) if isinstance(v, Path) else v) for k,v in vars(args).items()}})

    bootstrap_env(seed=args.seed)
    log_env_once()

    # 3) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device_selected", extra={"device": str(device)})

    # 4) Data & transforms (test-only)
    tfs = build_transforms(args.image_size)
    test_root = Path(args.eval_in)
    if not test_root.exists():
        raise FileNotFoundError(f"--eval-in not found: {test_root}")
    test_ds = ImageFolder(test_root, transform=tfs["test"])

    # 5) Verify/align class encoding using mapping (warn and proceed if missing)
    class_names = verify_class_order_with_index_remap(test_ds, args.mapping_path)
    log.info("class_info", extra={"num_classes": len(class_names), "class_names": class_names})

    # 6) DataLoader
    test_loader = make_eval_loader(test_ds, args.batch_size, args.num_workers, args.seed)

    # 7) Model + weights (mirror training; pretrained=False)
    num_classes = len(class_names)
    model = build_model(args.model, num_classes=num_classes, pretrained=False)
    load_weights(model, args.trained_model, device)  # strict=True inside

    # 8) Evaluate
    t0 = time.time()
    test_acc, test_prec, test_rec, test_f1, y_true, y_pred = evaluate(model, test_loader, device)
    dt = time.time() - t0
    log.info("evaluation_metrics", extra={
        "acc": round(float(test_acc), 4),
        "precision_macro": round(float(test_prec), 4),
        "recall_macro": round(float(test_rec), 4),
        "f1_macro": round(float(test_f1), 4),
        "elapsed_sec": round(dt, 2),
        "samples": len(y_true)
    })
    print(f"TEST — Acc={test_acc:.3f}  P={test_prec:.3f}  R={test_rec:.3f}  F1={test_f1:.3f}")

    # 8b) Confusion matrices + classification report
    report_on_loader(
        model=model,
        loader=test_loader,
        class_names=class_names,
        device=device,
        out_dir=Path(args.eval_out),
        title="Test",
    )

    # 8c) Full inference outputs (for galleries/Grad-CAM)
    paths = get_paths_for_dataset(test_ds)
    y_true2, y_pred2, probs = infer_collect(model, test_loader, device)
    assert np.array_equal(y_true, y_true2) and np.array_equal(y_pred, y_pred2), \
        "Mismatch between evaluate() and infer_collect() results."

    # 8d) Top examples per true class
    max_per_class = 6
    mistakes = top_mistakes_per_true_class(
        y_true, y_pred, probs, paths, max_per_class, n_classes=len(class_names)
    )
    corrects = top_correct_per_true_class(
        y_true, y_pred, probs, paths, max_per_class, n_classes=len(class_names)
    )

    # 8e) Save galleries
    gallery_dir = Path(args.eval_out) / "galleries"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    show_calls_gallery(mistakes, class_names, cols=6, title="misclassifications", save_dir=gallery_dir)
    show_calls_gallery(corrects, class_names, cols=6, title="top-correct", save_dir=gallery_dir)

    # 8f) Grad-CAM overlays
    gradcam_dir = Path(args.eval_out) / "gradcam"
    show_gradcam_gallery(
        predictions=mistakes, class_names=class_names, model=model, device=device,
        ds_for_transform=test_ds, cols=4, title="misclass_gradcam", target_layer=None, save_dir=gradcam_dir
    )
    show_gradcam_gallery(
        predictions=corrects, class_names=class_names, model=model, device=device,
        ds_for_transform=test_ds, cols=4, title="correct_class_gradcam", target_layer=None, save_dir=gradcam_dir
    )

    # 9) Persist a small JSON summary
    out_dir = Path(args.eval_out); out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    fname = f"evaluation_summary_{(run_id or 'no-runid')}_{ts}.json"
    summary_path = out_dir / fname
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "class_names": class_names,
            "metrics": {
                "acc": round(float(test_acc), 6),
                "precision_macro": round(float(test_prec), 6),
                "recall_macro": round(float(test_rec), 6),
                "f1_macro": round(float(test_f1), 6),
            },
        }, f, indent=2, ensure_ascii=False)
    log.info("evaluation_summary_written", extra={"path": str(summary_path)})

    try:
        latest_path = out_dir / "evaluation_summary_latest.json"
        latest_path.write_text(Path(summary_path).read_text(encoding="utf-8"), encoding="utf-8")
        log.info("evaluation_summary_latest_updated", extra={"path": str(latest_path)})
    except Exception as e:
        log.warning("evaluation_summary_latest_update_failed", extra={"error": str(e)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())