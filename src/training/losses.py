"""Losses. Alle tager logits (B x H x W) og target (B x H x W) og returnerer en skalar."""
import torch
import torch.nn.functional as F


def bce_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)


def dice_loss(logits, target):
    probs = torch.sigmoid(logits)
    return 1 - (2 * (probs * target).mean() + 1) / ((probs + target).mean() + 1)


def focal_loss(logits, target, gamma=2.0):
    """Standard focal loss (Lin et al. 2017), gamma = 2.

    Begge led vægtes med (1 - p_t)^gamma, hvor p_t er sandsynligheden for den sande klasse,
    så lette pixels af begge klasser (fx al den nemme baggrund) nedvægtes.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = torch.exp(-bce)
    return ((1 - p_t) ** gamma * bce).mean()


class WeightedBCELoss:
    """BCE hvor forgrundspixels vægtes med pos_weight = #baggrund / #forgrund i træningssættet."""

    def __init__(self, pos_weight):
        self.pos_weight = torch.tensor(pos_weight)

    def __call__(self, logits, target):
        pos_weight = self.pos_weight.to(logits.device)
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def point_loss(logits, target, balanced=True):
    """BCE kun på klikkede pixels (target 1 eller 0); pixels med target -1 ignoreres.

    Med balanced=True vægtes positive og negative klik lige meget (gennemsnit af de to
    klassers middel-loss). Ellers normaliseres der med det samlede antal klik.
    """
    positive, negative = target == 1, target == 0
    bce = F.binary_cross_entropy_with_logits(logits, target.clamp(min=0), reduction="none")
    if balanced:
        terms = [bce[m].mean() for m in (positive, negative) if m.any()]
        return sum(terms) / len(terms) if terms else logits.sum() * 0
    clicked = positive | negative
    return bce[clicked].mean() if clicked.any() else logits.sum() * 0


LOSSES = ("bce", "dice", "focal", "weighted_bce", "point", "point_unbalanced")
POINT_LOSSES = ("point", "point_unbalanced")


def build_loss(name, pos_weight=None):
    if name == "bce":
        return bce_loss
    if name == "dice":
        return dice_loss
    if name == "focal":
        return focal_loss
    if name == "weighted_bce":
        return WeightedBCELoss(pos_weight)
    if name == "point":
        return point_loss
    if name == "point_unbalanced":
        return lambda logits, target: point_loss(logits, target, balanced=False)
    raise ValueError(f"Ukendt loss: {name}")
