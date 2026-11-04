import time

import pandas as pd
import torch

from utils.metrics import MetricTracker


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for image, target, _ in loader:
        image, target = image.to(device), target.to(device)
        loss = loss_fn(model(image), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(image)
        n += len(image)
    return total / n


@torch.no_grad()
def evaluate(model, loader, device, loss_fn=None, use_fov=True, return_preds=False):
    """Metrikker (gennemsnit over billeder) og evt. loss på et helt sæt.

    Returnerer (results, tracker, preds), hvor preds er en liste af bool-masker hvis return_preds.
    """
    model.eval()
    tracker, preds = MetricTracker(), []
    total, n = 0.0, 0
    for image, target, fov in loader:
        image, target, fov = image.to(device), target.to(device), fov.to(device)
        logits = model(image)
        if loss_fn is not None:
            total += loss_fn(logits, target).item() * len(image)
            n += len(image)
        tracker.update(logits, target, fov if use_fov else None)
        if return_preds:
            preds.extend((logits > 0).cpu())
    results = tracker.compute()
    if loss_fn is not None:
        results["loss"] = total / n
    return results, tracker, preds


def fit(model, train_loader, val_loader, loss_fn, val_loss_fn, optimizer, epochs, device, run_dir, use_fov=True):
    """Træner og evaluerer på val efter hver epoch. Gemmer bedste model (val-Dice) som best.pt."""
    history, best_dice = [], -1.0
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val, _, _ = evaluate(model, val_loader, device, val_loss_fn, use_fov)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()},
               "seconds": time.time() - start}
        history.append(row)
        improved = val["dice"] > best_dice
        if improved:
            best_dice = val["dice"]
            torch.save(model.state_dict(), run_dir / "best.pt")
        # Skrives hver epoch, så en afbrudt kørsel stadig har sin historik.
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val['loss']:.4f} | "
              f"val dice {val['dice']:.4f} | {row['seconds']:.1f}s" + (" *" if improved else ""), flush=True)
    torch.save(model.state_dict(), run_dir / "last.pt")
    return history
