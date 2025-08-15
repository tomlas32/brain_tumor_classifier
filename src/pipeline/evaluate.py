import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate a model on a given DataLoader.

    Args:
        model: The trained model (nn.Module).
        loader: DataLoader for validation/test set.
        device: torch.device ("cpu" or "cuda").

    Returns:
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
        "acc": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "samples": len(y_true)
    })

    return acc, prec, rec, f1, y_true, y_pred