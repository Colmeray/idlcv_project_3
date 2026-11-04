"""Figurer til rapporten."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def error_map(pred, target, fov):
    """RGB-billede: TP hvid, FP rød, FN blå, TN sort. Uden for FOV grå."""
    pred, target = np.asarray(pred, bool), np.asarray(target) > 0.5
    rgb = np.zeros(pred.shape + (3,))
    rgb[pred & target] = (1, 1, 1)
    rgb[pred & ~target] = (1, 0, 0)
    rgb[~pred & target] = (0, 0.4, 1)
    rgb[~np.asarray(fov, bool)] = (0.3, 0.3, 0.3)
    return rgb


def save_examples(dataset, preds, path, n_examples=4):
    """Billede, ground truth, forudsigelse og fejlkort side om side for de første n billeder."""
    n = min(n_examples, len(preds))
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n), squeeze=False)
    for row in range(n):
        image, target, fov = dataset[row]
        panels = [(image.permute(1, 2, 0).numpy(), dataset.ids[row]), (target.numpy(), "ground truth"),
                  (preds[row].numpy(), "forudsigelse"), (error_map(preds[row], target, fov), "TP / FP (rød) / FN (blå)")]
        for ax, (img, title) in zip(axes[row], panels):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_click_examples(images, targets, clicks_by_strategy, n_clicks, path):
    """Ét billede per række, én kolonne per klik-strategi: positive klik grønne, negative røde."""
    strategies = list(clicks_by_strategy)
    fig, axes = plt.subplots(len(images), len(strategies), figsize=(4 * len(strategies), 3 * len(images)),
                             squeeze=False)
    for row, (sample_id, image) in enumerate(images.items()):
        for ax, strategy in zip(axes[row], strategies):
            clicks = clicks_by_strategy[strategy][sample_id]
            ax.imshow(image)
            ax.contour(targets[sample_id], levels=[0.5], colors="yellow", linewidths=1)
            for key, color in (("positive", "lime"), ("negative", "red")):
                points = np.array(clicks[key][:n_clicks])
                ax.scatter(points[:, 1], points[:, 0], c=color, s=18, edgecolors="black", linewidths=0.5)
            ax.set_title(f"{sample_id}: {strategy}, {n_clicks} klik", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_dice_vs_clicks(results, path):
    """Test-Dice mod antal klik per strategi, med fuldt superviseret U-Net (BCE) som reference."""
    test = results[(results.split == "test") & (results.dataset == "ph2")]
    weak = test[test.n_clicks.notna()]
    full = test[(test.n_clicks.isna()) & (test.model == "unet") & (test.loss == "bce")]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for strategy, group in weak.groupby("click_strategy"):
        group = group.groupby("n_clicks").dice.mean()
        ax.plot(group.index, group.values, marker="o", label=strategy)
    if len(full):
        ax.axhline(full.dice.mean(), color="black", linestyle="--", label="fuld supervision")
    ax.set_xscale("log")
    ax.set_xlabel("klik per klasse per billede")
    ax.set_ylabel("test Dice")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
