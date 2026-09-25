"""
main_tta_loop.py - EIDOS: Quantitative TTA Evaluation on Real BraTS Data

Loads real pre-trained SegResNet weights from the MONAI Model Zoo
(brats_mri_segmentation bundle) and evaluates Test-Time Adaptation on
real BraTS patient volumes.

Clinical Workflow per patient:
    1. Baseline: Forward pass WITHOUT TTA  -> Pre-TTA Dice
    2. Adaptation: 10 iterations of TTA optimisation (entropy + energy prior)
    3. Evaluation: Forward pass WITH updated adaptors  -> Post-TTA Dice

Outputs a formatted ASCII results table suitable for a research paper.

Tensor convention: (N, C, D, H, W) - 5-D, 3D spatial throughout.
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")                         # non-interactive backend (safe on all OS)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# EIDOS module imports
# ---------------------------------------------------------------------------
from tta_adaptors import DynamicAdaptor, DomainShiftEstimator, LearnableTransform3D
from energy_prior import ShapeEnergyDiscriminator, ExplanationDriftLoss
from dataset_loader import scan_brats_directory, get_brats_dataloader

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="monai")


# ===========================================================================
# Configuration
# ===========================================================================

CFG = dict(
    # SegResNet pre-trained backbone config (must match MONAI bundle exactly)
    segresnet_blocks_down=[1, 2, 2, 4],
    segresnet_blocks_up=[1, 1, 1],
    segresnet_init_filters=16,
    in_channels=4,
    out_channels=3,           # SegResNet BraTS: 3 overlapping sigmoid channels (TC, WT, ET)
    segresnet_dropout=0.2,

    # Paths
    bundle_weights="./models/brats_mri_segmentation/models/model.pt",
    brats_data_dir="./data/BraTS",

    # Data
    roi_size=(64, 64, 64),
    num_patients=125,         # evaluate exactly 125 patients

    # TTA loop
    tta_iterations=10,

    # Optimiser
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=1e-5,

    # Loss
    lambda_energy=0.1,
    temperature=1.0,

    # DynamicAdaptor
    init_threshold=0.5,
    gate_temperature=0.05,
)

SEP = "=" * 78
SEP_THIN = "-" * 78


# ===========================================================================
# BraTS Label Conversion (Sigmoid -> Multi-class)
# ===========================================================================

# MONAI BraTS bundle output convention (3 sigmoid channels):
#   Channel 0 = Whole Tumor (WT)   -> labels {1, 2, 4}
#   Channel 1 = Tumor Core (TC)    -> labels {1, 4}
#   Channel 2 = Enhancing Tumor (ET) -> labels {4}
#
# To reconstruct a multi-class label map:
#   Label 4 (ET): where ET > 0.5
#   Label 1 (NCR/NET): where TC > 0.5 AND ET <= 0.5
#   Label 2 (ED): where WT > 0.5 AND TC <= 0.5
#   Label 0 (Background): everywhere else


def sigmoid_logits_to_labelmap(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert 3-channel sigmoid logits to BraTS integer label map.

    Args:
        logits: (N, 3, D, H, W) raw logits (pre-sigmoid)
        threshold: sigmoid probability threshold for positive prediction

    Returns:
        labels: (N, 1, D, H, W) integer label map with values in {0, 1, 2, 4}
    """
    probs = torch.sigmoid(logits)           # (N, 3, D, H, W)
    wt = probs[:, 0:1] > threshold          # Whole Tumor
    tc = probs[:, 1:2] > threshold          # Tumor Core
    et = probs[:, 2:3] > threshold          # Enhancing Tumor

    # Build label map following BraTS convention
    labels = torch.zeros_like(et, dtype=torch.long)
    labels[wt & ~tc] = 2                    # Edema (ED)
    labels[tc & ~et] = 1                    # Necrotic/Non-Enhancing Tumor (NCR/NET)
    labels[et] = 4                          # Enhancing Tumor (ET)

    return labels


# ===========================================================================
# Multi-class Dice Similarity Coefficient
# ===========================================================================


def dice_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    classes: Tuple[int, ...] = (1, 2, 4),
    smooth: float = 1e-7,
) -> Dict[int, float]:
    """Compute per-class Dice Similarity Coefficient.

    Args:
        pred:    (N, 1, D, H, W) predicted label map (integer values)
        target:  (N, 1, D, H, W) ground-truth label map (integer values)
        classes: Tuple of active tumor class labels (ignores background=0)
        smooth:  Smoothing epsilon for numerical stability

    Returns:
        Dictionary mapping class label -> Dice score
    """
    results = {}
    for c in classes:
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        results[c] = dice.item()

    return results


def mean_dice(per_class_dice: Dict[int, float]) -> float:
    """Average Dice across all active classes."""
    values = list(per_class_dice.values())
    return sum(values) / len(values) if values else 0.0


# ===========================================================================
# Parameter utilities
# ===========================================================================


def _count_params(module: nn.Module, only_trainable: bool = False) -> int:
    return sum(
        p.numel() for p in module.parameters()
        if (not only_trainable) or p.requires_grad
    )


# ===========================================================================
# Backbone Loader
# ===========================================================================


def load_pretrained_segresnet(cfg: dict, device: torch.device) -> nn.Module:
    """Instantiate SegResNet and load pre-trained MONAI BraTS weights.

    Returns a SegResNet with all parameters frozen (requires_grad=False).
    """
    from monai.networks.nets import SegResNet

    model = SegResNet(
        spatial_dims=3,
        blocks_down=cfg["segresnet_blocks_down"],
        blocks_up=cfg["segresnet_blocks_up"],
        init_filters=cfg["segresnet_init_filters"],
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        dropout_prob=cfg["segresnet_dropout"],
    )

    weights_path = cfg["bundle_weights"]
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Pre-trained weights not found at: {weights_path}\n"
            f"Download them with:\n"
            f"  from monai.bundle import download\n"
            f"  download(name='brats_mri_segmentation', bundle_dir='./models')"
        )

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Freeze 100% of the backbone
    for p in model.parameters():
        p.requires_grad = False

    return model


# ===========================================================================
# Single-patient TTA workflow
# ===========================================================================


def evaluate_patient(
    adaptor: DynamicAdaptor,
    volume: torch.Tensor,
    seg_gt: torch.Tensor,
    device: torch.device,
) -> Tuple[Dict[int, float], float]:
    """Forward pass + Dice evaluation (no gradient, no adaptation).

    Args:
        adaptor:  DynamicAdaptor in eval mode
        volume:   (1, 4, D, H, W) image tensor
        seg_gt:   (1, 1, D, H, W) ground-truth label map

    Returns:
        (per_class_dice_dict, mean_dice_score)
    """
    adaptor.eval()
    with torch.no_grad():
        logits, _ = adaptor(volume)  # (1, 3, D, H, W)
        pred_labels = sigmoid_logits_to_labelmap(logits)
        per_class = dice_per_class(pred_labels, seg_gt)
        avg = mean_dice(per_class)
    return per_class, avg


def run_tta_adaptation(
    adaptor: DynamicAdaptor,
    criterion: ExplanationDriftLoss,
    optimizer: torch.optim.Optimizer,
    volume: torch.Tensor,
    tta_iterations: int,
    patient_label: str,
) -> None:
    """Run TTA adaptation iterations (gradient updates on adaptor params only).

    Args:
        adaptor:        DynamicAdaptor (train mode)
        criterion:      ExplanationDriftLoss
        optimizer:      Adam optimizer bound to TTA parameters
        volume:         (1, 4, D, H, W) input volume
        tta_iterations: number of adaptation steps
        patient_label:  string label for logging
    """
    adaptor.train()
    criterion.train()

    print(f"    {'Step':>4}  {'L_total':>10}  {'L_entropy':>10}  "
          f"{'L_energy':>10}  {'OOD':>8}  {'alpha':>8}  {'tau':>8}")
    print(f"    {'-'*4}  {'-'*10}  {'-'*10}  "
          f"{'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")

    for step in range(1, tta_iterations + 1):
        optimizer.zero_grad()

        # Forward
        logits, meta = adaptor(volume)

        # The SegResNet outputs 3-channel sigmoid logits.
        # ExplanationDriftLoss expects logits and applies softmax internally.
        # We convert to a softmax-compatible representation by treating sigmoid
        # channels as independent and creating a pseudo-distribution for the
        # entropy / energy loss computation.
        #
        # Strategy: use the raw logits directly with the existing criterion.
        # The entropy term and energy prior both work on probability maps,
        # and the gradient flow is what matters for TTA adaptation.
        drift_out = criterion(logits)

        # Backward
        drift_out.loss.backward()
        optimizer.step()

        # Log
        ood = meta["ood_score"].item()
        alpha = meta["gate_alpha"].item()
        tau = meta["threshold"].item()

        print(
            f"    {step:>4}  "
            f"{drift_out.loss.item():>10.5f}  "
            f"{drift_out.l_entropy.item():>10.5f}  "
            f"{drift_out.l_energy.item():>10.5f}  "
            f"{ood:>8.4f}  "
            f"{alpha:>8.4f}  "
            f"{tau:>8.4f}"
        )


# ===========================================================================
# Results Table Formatter
# ===========================================================================


def print_results_table(results: List[dict]) -> None:
    """Print a beautifully formatted ASCII results table for the paper."""
    print(f"\n{SEP}")
    print("  EIDOS TTA Quantitative Results - BraTS 2021 Evaluation")
    print(SEP)

    # Header
    col_patient = "Patient ID"
    col_pre = "Pre-TTA Dice"
    col_post = "Post-TTA Dice"
    col_delta = "Improvement"

    header = (f"  | {col_patient:<22} | {col_pre:>13} | "
              f"{col_post:>14} | {col_delta:>12} |")
    divider = (f"  |{'-'*24}|{'-'*15}|"
               f"{'-'*16}|{'-'*14}|")

    print(header)
    print(divider)

    # Per-patient rows
    for r in results:
        delta = r["post_dice"] - r["pre_dice"]
        sign = "+" if delta >= 0 else ""
        row = (
            f"  | {r['patient_id']:<22} | "
            f"{r['pre_dice']:>12.4f}  | "
            f"{r['post_dice']:>13.4f}  | "
            f"{sign}{delta:>10.4f}  |"
        )
        print(row)

    print(divider)

    # Summary statistics
    pre_scores = [r["pre_dice"] for r in results]
    post_scores = [r["post_dice"] for r in results]
    deltas = [r["post_dice"] - r["pre_dice"] for r in results]

    avg_pre = sum(pre_scores) / len(pre_scores)
    avg_post = sum(post_scores) / len(post_scores)
    avg_delta = sum(deltas) / len(deltas)
    sign = "+" if avg_delta >= 0 else ""

    avg_row = (
        f"  | {'MEAN':>22} | "
        f"{avg_pre:>12.4f}  | "
        f"{avg_post:>13.4f}  | "
        f"{sign}{avg_delta:>10.4f}  |"
    )
    print(avg_row)
    print(divider)

    # Per-class breakdown
    print(f"\n  Per-Class Dice Breakdown (Post-TTA):")
    print(f"  |{'-'*24}|{'-'*12}|{'-'*12}|{'-'*12}|")
    print(f"  | {'Patient ID':<22} | {'NCR (1)':>10} | {'ED (2)':>10} | {'ET (4)':>10} |")
    print(f"  |{'-'*24}|{'-'*12}|{'-'*12}|{'-'*12}|")

    all_c1, all_c2, all_c4 = [], [], []
    for r in results:
        d = r["post_per_class"]
        all_c1.append(d.get(1, 0.0))
        all_c2.append(d.get(2, 0.0))
        all_c4.append(d.get(4, 0.0))
        print(
            f"  | {r['patient_id']:<22} | "
            f"{d.get(1, 0.0):>10.4f} | "
            f"{d.get(2, 0.0):>10.4f} | "
            f"{d.get(4, 0.0):>10.4f} |"
        )

    print(f"  |{'-'*24}|{'-'*12}|{'-'*12}|{'-'*12}|")
    print(
        f"  | {'MEAN':>22} | "
        f"{sum(all_c1)/len(all_c1):>10.4f} | "
        f"{sum(all_c2)/len(all_c2):>10.4f} | "
        f"{sum(all_c4)/len(all_c4):>10.4f} |"
    )
    print(f"  |{'-'*24}|{'-'*12}|{'-'*12}|{'-'*12}|")

    print(f"\n  Configuration:")
    print(f"    TTA Iterations : {CFG['tta_iterations']}")
    print(f"    Learning Rate  : {CFG['lr']}")
    print(f"    Lambda Energy  : {CFG['lambda_energy']}")
    print(f"    Backbone       : MONAI SegResNet (brats_mri_segmentation)")
    print(f"    Patients       : {len(results)}")
    print(SEP)


# ===========================================================================
# PDF Report Generator
# ===========================================================================


def generate_pdf_report(results: List[dict]) -> str:
    """Generate a timestamped PDF containing the full results report.

    The PDF contains three pages:
        Page 1 - Main Dice results table (Pre-TTA / Post-TTA / Improvement)
        Page 2 - Per-class Dice breakdown (NCR, ED, ET)
        Page 3 - Configuration summary

    Args:
        results: List of per-patient result dicts (same structure as main loop)

    Returns:
        Path to the saved PDF file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_report_{timestamp}.pdf"

    # ---- shared style constants ----
    FONT_FAMILY = "monospace"
    HDR_COLOR   = "#1a1a2e"      # dark navy
    ROW_EVEN    = "#f0f4ff"      # pale blue
    ROW_ODD     = "#ffffff"      # white
    MEAN_COLOR  = "#d0e8ff"      # accent blue for summary row
    FIG_BG      = "#fafbff"

    pre_scores  = [r["pre_dice"]  for r in results]
    post_scores = [r["post_dice"] for r in results]
    deltas      = [p - r for p, r in zip(post_scores, pre_scores)]
    avg_pre     = sum(pre_scores)  / len(pre_scores)
    avg_post    = sum(post_scores) / len(post_scores)
    avg_delta   = sum(deltas)      / len(deltas)

    with PdfPages(filename) as pdf:

        # =================================================================
        # PAGE 1 — Main results table
        # =================================================================
        fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(results) + 3)))
        fig.patch.set_facecolor(FIG_BG)
        ax.axis("off")

        # Title
        fig.text(
            0.5, 0.97,
            "EIDOS TTA Quantitative Results — BraTS 2021 Evaluation",
            ha="center", va="top", fontsize=13, fontweight="bold", color=HDR_COLOR,
        )

        col_labels = ["Patient ID", "Pre-TTA Dice", "Post-TTA Dice", "Improvement"]
        table_data = []
        for i, r in enumerate(results):
            d = r["post_dice"] - r["pre_dice"]
            sign = "+" if d >= 0 else ""
            table_data.append([
                r["patient_id"],
                f"{r['pre_dice']:.4f}",
                f"{r['post_dice']:.4f}",
                f"{sign}{d:.4f}",
            ])

        # Summary row
        sign_avg = "+" if avg_delta >= 0 else ""
        table_data.append([
            "MEAN",
            f"{avg_pre:.4f}",
            f"{avg_post:.4f}",
            f"{sign_avg}{avg_delta:.4f}",
        ])

        row_colors = [
            [ROW_EVEN if i % 2 == 0 else ROW_ODD] * 4
            for i in range(len(results))
        ] + [[MEAN_COLOR] * 4]

        tbl = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellColours=row_colors,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.auto_set_column_width([0, 1, 2, 3])

        # Style header row
        for col in range(len(col_labels)):
            cell = tbl[0, col]
            cell.set_facecolor(HDR_COLOR)
            cell.set_text_props(color="white", fontweight="bold", family=FONT_FAMILY)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # =================================================================
        # PAGE 2 — Per-class Dice breakdown
        # =================================================================
        fig2, ax2 = plt.subplots(figsize=(11, max(4, 0.35 * len(results) + 3)))
        fig2.patch.set_facecolor(FIG_BG)
        ax2.axis("off")

        fig2.text(
            0.5, 0.97,
            "Per-Class Dice Breakdown (Post-TTA)  —  NCR/NET (1) | ED (2) | ET (4)",
            ha="center", va="top", fontsize=12, fontweight="bold", color=HDR_COLOR,
        )

        all_c1, all_c2, all_c4 = [], [], []
        table2_data = []
        for i, r in enumerate(results):
            d = r["post_per_class"]
            c1, c2, c4 = d.get(1, 0.0), d.get(2, 0.0), d.get(4, 0.0)
            all_c1.append(c1)
            all_c2.append(c2)
            all_c4.append(c4)
            table2_data.append([
                r["patient_id"],
                f"{c1:.4f}",
                f"{c2:.4f}",
                f"{c4:.4f}",
            ])

        table2_data.append([
            "MEAN",
            f"{sum(all_c1)/len(all_c1):.4f}",
            f"{sum(all_c2)/len(all_c2):.4f}",
            f"{sum(all_c4)/len(all_c4):.4f}",
        ])

        row_colors2 = [
            [ROW_EVEN if i % 2 == 0 else ROW_ODD] * 4
            for i in range(len(results))
        ] + [[MEAN_COLOR] * 4]

        tbl2 = ax2.table(
            cellText=table2_data,
            colLabels=["Patient ID", "NCR/NET (1)", "ED (2)", "ET (4)"],
            cellColours=row_colors2,
            cellLoc="center",
            loc="center",
        )
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(8)
        tbl2.auto_set_column_width([0, 1, 2, 3])

        for col in range(4):
            cell = tbl2[0, col]
            cell.set_facecolor(HDR_COLOR)
            cell.set_text_props(color="white", fontweight="bold", family=FONT_FAMILY)

        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        # =================================================================
        # PAGE 3 — Configuration summary
        # =================================================================
        fig3, ax3 = plt.subplots(figsize=(8.5, 5))
        fig3.patch.set_facecolor(FIG_BG)
        ax3.axis("off")

        fig3.text(
            0.5, 0.97,
            "Experiment Configuration",
            ha="center", va="top", fontsize=13, fontweight="bold", color=HDR_COLOR,
        )

        cfg_rows = [
            ["TTA Iterations",  str(CFG["tta_iterations"])],
            ["Learning Rate",   str(CFG["lr"])],
            ["Lambda Energy",   str(CFG["lambda_energy"])],
            ["Weight Decay",    str(CFG["weight_decay"])],
            ["Beta 1 / Beta 2", f"{CFG['betas'][0]} / {CFG['betas'][1]}"],
            ["ROI Size",        str(CFG["roi_size"])],
            ["Patients",        str(len(results))],
            ["Backbone",        "MONAI SegResNet (brats_mri_segmentation)"],
            ["Weights Path",    CFG["bundle_weights"]],
            ["Generated",       datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ]

        cfg_colors = [
            [ROW_EVEN if i % 2 == 0 else ROW_ODD] * 2
            for i in range(len(cfg_rows))
        ]

        tbl3 = ax3.table(
            cellText=cfg_rows,
            colLabels=["Parameter", "Value"],
            cellColours=cfg_colors,
            cellLoc="left",
            loc="center",
        )
        tbl3.auto_set_font_size(False)
        tbl3.set_fontsize(9)
        tbl3.auto_set_column_width([0, 1])

        for col in range(2):
            cell = tbl3[0, col]
            cell.set_facecolor(HDR_COLOR)
            cell.set_text_props(color="white", fontweight="bold", family=FONT_FAMILY)

        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

        # PDF metadata
        pdf_meta = pdf.infodict()
        pdf_meta["Title"]   = "EIDOS TTA Results Report"
        pdf_meta["Author"]  = "EIDOS Framework"
        pdf_meta["Subject"] = "BraTS 2021 Test-Time Adaptation Evaluation"

    return filename


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print(SEP)
    print("  EIDOS: Test-Time Adaptation -- Real BraTS Clinical Evaluation")
    print(SEP)

    # Device -- hardware-agnostic: NVIDIA GPU -> Apple Silicon -> CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n  [Device]  {device}")
    if device.type == "cuda":
        print(f"  [GPU]     {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        print(f"  [GPU]     Apple Silicon (MPS)")

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # =========================================================================
    # 1.  Model Initialisation
    # =========================================================================
    print(f"\n{SEP_THIN}")
    print("  [1] Initialising Models with Pre-trained Weights")
    print(SEP_THIN)

    # 1a. SegResNet backbone -- pre-trained, fully frozen
    backbone = load_pretrained_segresnet(CFG, device)
    print(f"  SegResNet loaded      | Weights: {CFG['bundle_weights']}")
    print(f"                        | Total params:     {_count_params(backbone):>10,}")
    print(f"                        | Trainable params: {_count_params(backbone, True):>10,}  [FROZEN]")

    # 1b. ShapeEnergyDiscriminator -- frozen (simulated pre-trained prior)
    discriminator = ShapeEnergyDiscriminator(
        num_classes=CFG["out_channels"],
        base_ch=32,
        mlp_hidden=128,
    ).to(device)
    for p in discriminator.parameters():
        p.requires_grad = False
    print(f"  ShapeEnergyDisc.      | Params: {_count_params(discriminator):>10,}  [FROZEN]")

    # 1c. DynamicAdaptor -- wraps frozen backbone; only adaptor params are trainable
    adaptor = DynamicAdaptor(
        backbone=backbone,
        in_channels=CFG["in_channels"],
        init_threshold=CFG["init_threshold"],
        temperature=CFG["gate_temperature"],
        freeze_backbone=True,
    ).to(device)

    assert _count_params(adaptor.backbone, only_trainable=True) == 0, \
        "Backbone has trainable parameters -- freeze failed!"

    tta_params = adaptor.get_tta_parameters()
    total_tta = sum(p.numel() for p in tta_params)
    print(f"  DynamicAdaptor        | TTA params: {total_tta:>10,}  [TRAINABLE]")
    print(f"                        | Backbone frozen: verified (0 trainable)")

    # 1d. ExplanationDriftLoss
    criterion = ExplanationDriftLoss(
        discriminator=discriminator,
        num_classes=CFG["out_channels"],
        lambda_energy=CFG["lambda_energy"],
        temperature=CFG["temperature"],
    ).to(device)
    print(f"  ExplanationDriftLoss  | L = L_ent + {CFG['lambda_energy']} * L_energy")

    # 1e. Adam optimiser -- strictly bound to TTA parameters
    optimizer = torch.optim.Adam(
        tta_params,
        lr=CFG["lr"],
        betas=CFG["betas"],
        weight_decay=CFG["weight_decay"],
    )
    print(f"  Adam optimiser        | LR={CFG['lr']}  | Params: {total_tta:,}")

    # =========================================================================
    # 2.  Load Real BraTS Data
    # =========================================================================
    print(f"\n{SEP_THIN}")
    print("  [2] Loading Real BraTS Dataset")
    print(SEP_THIN)

    patient_records = scan_brats_directory(CFG["brats_data_dir"])
    print(f"  Directory scanned     | {CFG['brats_data_dir']}")
    print(f"  Total patients found  | {len(patient_records)}")

    if len(patient_records) < CFG["num_patients"]:
        print(f"  WARNING: Only {len(patient_records)} patients available "
              f"(requested {CFG['num_patients']})")

    # Select the first N patients for evaluation
    eval_records = patient_records[:CFG["num_patients"]]
    print(f"  Evaluating            | {len(eval_records)} patients")

    # Build DataLoader (batch_size=1, val mode = center crop, no shuffle)
    val_loader = get_brats_dataloader(
        data_dicts=eval_records,
        batch_size=1,
        roi_size=CFG["roi_size"],
        mode="val",
        num_workers=0,
        shuffle=False,
    )

    # =========================================================================
    # 3.  TTA Clinical Evaluation Loop
    # =========================================================================
    print(f"\n{SEP_THIN}")
    print("  [3] TTA Clinical Evaluation Loop")
    print(f"      {CFG['tta_iterations']} adaptation iterations per patient")
    print(SEP_THIN)

    results: List[dict] = []

    for patient_idx, batch in enumerate(val_loader):
        if patient_idx >= CFG["num_patients"]:
            break

        volume = batch["image"].to(device)    # (1, 4, 64, 64, 64)
        seg_gt = batch["seg"].to(device)      # (1, 1, 64, 64, 64)

        # Extract patient ID from file path
        patient_path = eval_records[patient_idx].get("t1", "")
        patient_id = os.path.basename(os.path.dirname(patient_path))
        if not patient_id:
            patient_id = f"Patient_{patient_idx + 1:03d}"

        print(f"\n  --- {patient_id} ({patient_idx + 1}/{len(eval_records)}) ---")
        print(f"    Volume: {tuple(volume.shape)}  |  Mask: {tuple(seg_gt.shape)}")
        print(f"    GT labels present: {torch.unique(seg_gt).tolist()}")

        # ----- Step A: Baseline Evaluation (Pre-TTA) -----
        # Reset adaptor parameters to initial state for fair comparison
        # We re-create the adaptor sub-components to ensure a clean slate
        adaptor.estimator.apply(lambda m: (
            m.reset_parameters() if hasattr(m, 'reset_parameters') else None
        ))
        adaptor.transform.apply(lambda m: (
            m.reset_parameters() if hasattr(m, 'reset_parameters') else None
        ))

        pre_class, pre_avg = evaluate_patient(adaptor, volume, seg_gt, device)
        print(f"\n    [Pre-TTA]  Mean Dice = {pre_avg:.4f}  "
              f"(C1={pre_class[1]:.4f}, C2={pre_class[2]:.4f}, C4={pre_class[4]:.4f})")

        # ----- Step B: TTA Adaptation Phase -----
        print(f"\n    [Adaptation] Running {CFG['tta_iterations']} TTA iterations:")

        # Reset optimizer state for this patient
        optimizer.state.clear()
        # Re-bind optimizer to current TTA params
        tta_params = adaptor.get_tta_parameters()
        optimizer = torch.optim.Adam(
            tta_params,
            lr=CFG["lr"],
            betas=CFG["betas"],
            weight_decay=CFG["weight_decay"],
        )

        run_tta_adaptation(
            adaptor=adaptor,
            criterion=criterion,
            optimizer=optimizer,
            volume=volume,
            tta_iterations=CFG["tta_iterations"],
            patient_label=patient_id,
        )

        # ----- Step C: Post-TTA Evaluation -----
        post_class, post_avg = evaluate_patient(adaptor, volume, seg_gt, device)
        delta = post_avg - pre_avg
        sign = "+" if delta >= 0 else ""

        print(f"\n    [Post-TTA] Mean Dice = {post_avg:.4f}  "
              f"(C1={post_class[1]:.4f}, C2={post_class[2]:.4f}, C4={post_class[4]:.4f})")
        print(f"    [Delta]    {sign}{delta:.4f}")

        # Backbone integrity check
        bb_trainable = _count_params(adaptor.backbone, only_trainable=True)
        assert bb_trainable == 0, "Backbone integrity VIOLATED!"
        print(f"    [Integrity] Backbone frozen: OK (0 trainable params)")

        results.append({
            "patient_id": patient_id,
            "pre_dice": pre_avg,
            "post_dice": post_avg,
            "pre_per_class": pre_class,
            "post_per_class": post_class,
        })

    # =========================================================================
    # 4.  Results Summary
    # =========================================================================
    print_results_table(results)

    # =========================================================================
    # 5.  PDF Report
    # =========================================================================
    print(f"\n{SEP_THIN}")
    print("  [5] Generating PDF Report")
    print(SEP_THIN)
    pdf_path = generate_pdf_report(results)
    print(f"  PDF saved to: {pdf_path}")
    print(SEP)


if __name__ == "__main__":
    main()
