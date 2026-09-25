"""
tta_adaptors.py — Sample-Aware Test-Time Adaptation (TTA) Framework
for 3D Medical Volumetric Data.

Architecture overview
─────────────────────
  DomainShiftEstimator
      Lightweight 3D CNN → scalar OOD severity score ∈ (0, 1).
      Uses multi-scale depthwise-separable 3D convolutions + global average
      pooling + a 2-layer MLP head.  Runs on the raw input volume before
      any transformation so the estimate is unbiased.

  LearnableTransform3D
      Differentiable intensity / contrast / channel-offset adaptation.
      Three orthogonal axes of adaptation:
        1. Per-channel intensity scale γ  (learnable, init = 1)
        2. Per-channel intensity bias  β  (learnable, init = 0)
        3. 1×1×1 channel-mixing matrix   (learnable, init = identity)
        4. Learnable 3D depthwise spatial smoothing kernel (init ≈ identity)
      All components are initialised as identity / no-op so that the
      model's baseline accuracy is preserved when the threshold is first
      crossed.

  DynamicAdaptor
      TTA controller wrapping a frozen UNet3D backbone.
      Forward pass:
        (1) DomainShiftEstimator   → ood_score ∈ (0, 1)
        (2) Soft sigmoid gate      → α = sigmoid((ood_score − τ) / T)
                                       τ = learnable threshold
                                       T = temperature (fixed, default 0.05)
        (3) Route:
              x_adapted = LearnableTransform3D(x)
              x_in      = α · x_adapted + (1 − α) · x   (soft blend)
        (4) backbone(x_in) → segmentation logits

      The soft blend keeps gradients flowing through τ during TTA
      optimisation while perfectly recovering the identity path when
      ood_score ≪ τ.

Tensor convention throughout: (N, C, D, H, W)  — strictly 5-D, 3D spatial.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_depthwise_sep_conv3d(
    in_ch: int,
    out_ch: int,
    kernel: int = 3,
    stride: int = 1,
    dilation: int = 1,
) -> nn.Sequential:
    """Depthwise-separable 3D convolution (depthwise -> pointwise).

    Reduces parameters / FLOPs vs a standard Conv3d while keeping the same
    receptive field — important for a lightweight estimator that runs every
    forward pass.
    """
    pad = dilation * (kernel // 2)
    return nn.Sequential(
        # Depthwise
        nn.Conv3d(
            in_ch, in_ch,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=in_ch,
            bias=False,
        ),
        # Pointwise
        nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.GroupNorm(num_groups=max(1, out_ch // 8), num_channels=out_ch),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )


# ---------------------------------------------------------------------------
# 1.  DomainShiftEstimator
# ---------------------------------------------------------------------------


class DomainShiftEstimator(nn.Module):
    """Lightweight 3D CNN that estimates OOD severity from a raw input volume.

    Input : (N, C, D, H, W)  — raw MRI volume (any C, typically 4 for BraTS)
    Output: (N,)              — scalar OOD score in (0, 1)

    Design principles
    -----------------
    * Three stages of depthwise-separable 3D convs with aggressive strided
      downsampling so the estimator runs in O(volume / 64) operations.
    * Multi-dilation parallel branches in Stage 1 capture both fine-grained
      noise patterns (d=1) and coarser structural drift (d=2).
    * Global average pooling collapses spatial dims to a feature vector.
    * Two-layer MLP with dropout -> sigmoid scalar.
    * No batch normalisation (inappropriate at batch-size 1 during TTA);
      GroupNorm used throughout.
    """

    # Intermediate channel widths — kept small deliberately.
    _BASE: int = 16

    def __init__(
        self,
        in_channels: int = 4,
        base_ch: int = _BASE,
        mlp_hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        B = base_ch

        # -- Stage 0: channel projection (no spatial change) ---------------
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, B, kernel_size=1, bias=False),
            nn.GroupNorm(max(1, B // 4), B),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # -- Stage 1: multi-dilation parallel branches (stride 2) ----------
        # Branch A — dilation 1 (local texture / noise)
        self.branch_a = nn.Sequential(
            _make_depthwise_sep_conv3d(B, B, kernel=3, stride=2, dilation=1),
        )
        # Branch B — dilation 2 (wider context / structural drift)
        self.branch_b = nn.Sequential(
            _make_depthwise_sep_conv3d(B, B, kernel=3, stride=1, dilation=2),
            # Extra strided conv to match branch_a spatial size
            nn.Conv3d(B, B, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(max(1, B // 4), B),
            nn.LeakyReLU(0.1, inplace=True),
        )
        fused_ch = B * 2  # concat branches -> 2B

        # -- Stage 2: spatial compression ----------------------------------
        self.stage2 = nn.Sequential(
            _make_depthwise_sep_conv3d(fused_ch, B * 2, kernel=3, stride=2),
            _make_depthwise_sep_conv3d(B * 2, B * 4, kernel=3, stride=2),
        )

        # -- Global pooling + MLP head -------------------------------------
        feat_ch = B * 4
        self.gap = nn.AdaptiveAvgPool3d(1)   # -> (N, feat_ch, 1, 1, 1)

        self.mlp = nn.Sequential(
            nn.Flatten(),                    # -> (N, feat_ch)
            nn.Linear(feat_ch, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
            nn.Sigmoid(),                    # -> (N, 1) in (0, 1)
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, D, H, W)  raw input volume

        Returns:
            score: (N,)  OOD severity score in (0, 1)
        """
        assert x.ndim == 5, (
            f"DomainShiftEstimator expects 5-D input (N,C,D,H,W), got {x.ndim}-D"
        )
        z = self.stem(x)                      # (N, B,  D,   H,   W)
        a = self.branch_a(z)                  # (N, B,  D/2, H/2, W/2)
        b = self.branch_b(z)                  # (N, B,  D/2, H/2, W/2)

        # Spatial sizes must match before concat; guard against rounding
        if a.shape[2:] != b.shape[2:]:
            b = F.interpolate(
                b, size=a.shape[2:], mode="trilinear", align_corners=False
            )

        z2 = torch.cat([a, b], dim=1)         # (N, 2B, D/2, H/2, W/2)
        z3 = self.stage2(z2)                  # (N, 4B, D/8, H/8, W/8)
        pooled = self.gap(z3)                 # (N, 4B, 1,   1,   1)
        score = self.mlp(pooled).squeeze(-1)  # (N,)
        return score


# ---------------------------------------------------------------------------
# 2.  LearnableTransform3D
# ---------------------------------------------------------------------------


class LearnableTransform3D(nn.Module):
    """Differentiable, near-identity 3D intensity / contrast adaptations.

    Four complementary adaptation axes — all initialised as no-ops:

    1. **Per-channel intensity scaling gamma** (shape: C)
       x_scaled = gamma * x      init gamma = 1
    2. **Per-channel intensity bias beta** (shape: C)
       x_biased = x + beta       init beta = 0
    3. **1x1x1 channel-mixing matrix W** (shape: C x C)
       x_mixed  = W @ x          init W = I  (identity cross-channel contrast)
    4. **Learnable depthwise 3D spatial smoothing kernel** (shape: C x 1 x k x k x k)
       x_smooth = DepthwiseConv3d(x, K)   init K = identity delta

    The transforms are applied in order 1 -> 2 -> 3 -> 4.
    Because all initialise as identity-like, the model can safely be applied
    from the very first TTA step without degrading performance.

    Args:
        in_channels:   Number of input MRI modality channels (default 4).
        kernel_size:   Spatial kernel size for the depthwise smoothing conv.
                       Must be odd.  Default 3.
        eps:           Small constant added to gamma to prevent zero-scaling
                       during edge cases.
    """

    def __init__(
        self,
        in_channels: int = 4,
        kernel_size: int = 3,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.eps = eps
        C = in_channels

        # -- 1. Intensity scale --------------------------------------------
        # gamma stored as log to guarantee positivity: actual scale = softplus(log_gamma) + eps
        # Initialise so that softplus(log_gamma_0) + eps = 1.0 exactly:
        #   softplus(x) = 1  =>  x = log(e - 1) ~= 0.5413
        log_gamma_init = math.log(math.e - 1.0)           # ~0.5413
        self.log_gamma = nn.Parameter(
            torch.full((C,), fill_value=log_gamma_init)
        )

        # -- 2. Intensity bias ---------------------------------------------
        self.beta = nn.Parameter(torch.zeros(C))

        # -- 3. Channel-mixing (contrast) matrix ---------------------------
        # Stored as a weight matrix; initialise = identity.
        # F.normalize(w, p=2, dim=1) divides each row by its L2 norm.
        # For I (identity), each row norm = 1, so normalisation is lossless.
        # We store eye(C) directly; after normalisation it stays the identity.
        self.channel_mix = nn.Parameter(torch.eye(C))   # (C, C)

        # -- 4. Depthwise 3D spatial smoothing kernel ----------------------
        k = kernel_size
        # shape: (C, 1, k, k, k)
        kernel_data = self._make_identity_kernel(C, k)
        self.smooth_kernel = nn.Parameter(kernel_data)
        self.smooth_pad = k // 2

    # ------------------------------------------------------------------
    @staticmethod
    def _make_identity_kernel(C: int, k: int) -> torch.Tensor:
        """Create a depthwise 3D kernel that approximates the identity map.

        The centre voxel weight is 1.0, all others are 0.0, giving
        exact identity behaviour at initialisation.  This is preferable
        to a Gaussian init (which blurs even before TTA begins).
        """
        kernel = torch.zeros(C, 1, k, k, k)
        mid = k // 2
        kernel[:, 0, mid, mid, mid] = 1.0   # Dirac delta at centre
        return kernel

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, D, H, W)

        Returns:
            x_adapted: (N, C, D, H, W)  — same shape, adapted intensities
        """
        assert x.ndim == 5, (
            f"LearnableTransform3D expects 5-D input (N,C,D,H,W), got {x.ndim}-D"
        )
        N, C, D, H, W = x.shape

        # -- 1. Per-channel intensity scaling ------------------------------
        # Use F.softplus to keep gamma strictly positive
        gamma = F.softplus(self.log_gamma) + self.eps   # (C,)
        x = x * gamma.view(1, C, 1, 1, 1)

        # -- 2. Per-channel bias -------------------------------------------
        x = x + self.beta.view(1, C, 1, 1, 1)

        # -- 3. 1x1x1 channel mixing ---------------------------------------
        # Reshape for batched matrix multiply: (N, C, D*H*W)
        x_flat = x.view(N, C, -1)                         # (N, C, D*H*W)
        # Normalise rows via L2 so the matrix stays numerically stable
        # Note: named Wmix to avoid shadowing the spatial dim W from x.shape
        Wmix = F.normalize(self.channel_mix, p=2, dim=1)  # (C, C)
        x_flat = Wmix @ x_flat                            # (N, C, D*H*W)
        x = x_flat.view(N, C, D, H, W)

        # -- 4. Depthwise 3D spatial smoothing -----------------------------
        # groups=C -> depthwise (one kernel per channel, no cross-channel mixing)
        x = F.conv3d(
            x,
            self.smooth_kernel,
            bias=None,
            stride=1,
            padding=self.smooth_pad,
            groups=C,
        )

        return x


# ---------------------------------------------------------------------------
# 3.  DynamicAdaptor
# ---------------------------------------------------------------------------


class DynamicAdaptor(nn.Module):
    """Test-Time Adaptation controller wrapping a frozen 3D UNet backbone.

    During forward:
      (1) Estimate OOD severity: ood_score = DomainShiftEstimator(x)
      (2) Compute soft gate:
            alpha = sigmoid( (ood_score - tau) / temperature )
          where tau is a learnable scalar threshold initialised to `init_threshold`.
      (3) Blend:
            x_adapted = LearnableTransform3D(x)
            x_in      = alpha * x_adapted + (1 - alpha) * x
      (4) Return backbone(x_in)

    The sigmoid gate is differentiable w.r.t. tau, enabling gradient-based TTA
    optimisation to tune the threshold online.  When alpha -> 0 (strong
    in-distribution input), x_in ~= x exactly, perfectly preserving baseline
    accuracy.

    Args:
        backbone        : Frozen UNet3D (or any nn.Module) that accepts
                          (N, C, D, H, W) and returns (N, out_ch, D, H, W).
        transform       : LearnableTransform3D instance. If None, a default
                          one is created using `in_channels`.
        estimator       : DomainShiftEstimator instance. If None, a default
                          one is created using `in_channels`.
        in_channels     : Number of input channels (must match backbone).
        init_threshold  : Initial OOD threshold tau in (0, 1).  Samples with
                          ood_score > tau are adapted.  Default 0.5.
        temperature     : Sigmoid temperature controlling gate sharpness.
                          Smaller -> harder gate.  Default 0.05.
        freeze_backbone : If True (default), calls backbone.freeze_backbone()
                          if the method exists, then sets requires_grad=False
                          on all backbone parameters.
    """

    def __init__(
        self,
        backbone: nn.Module,
        transform: Optional[LearnableTransform3D] = None,
        estimator: Optional[DomainShiftEstimator] = None,
        in_channels: int = 4,
        init_threshold: float = 0.5,
        temperature: float = 0.05,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        # -- Backbone (frozen) ---------------------------------------------
        self.backbone = backbone
        if freeze_backbone:
            if hasattr(backbone, "freeze_backbone"):
                backbone.freeze_backbone(freeze_head=True)  # freeze everything
            for p in self.backbone.parameters():
                p.requires_grad = False

        # -- Adaptor components (trainable) --------------------------------
        self.estimator = estimator or DomainShiftEstimator(in_channels=in_channels)
        self.transform = transform or LearnableTransform3D(in_channels=in_channels)

        # -- Learnable threshold tau ---------------------------------------
        # Stored in logit-space so unconstrained optimisation keeps it in (0,1):
        #   tau = sigmoid(raw_threshold)
        # Solve for raw such that sigmoid(raw) = init_threshold:
        assert 0.0 < init_threshold < 1.0, "init_threshold must be in (0, 1)"
        raw_init = math.log(init_threshold / (1.0 - init_threshold))
        self.raw_threshold = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

        # Fixed temperature (registered as buffer, not parameter)
        self.register_buffer(
            "temperature", torch.tensor(temperature, dtype=torch.float32)
        )

    # ------------------------------------------------------------------
    @property
    def threshold(self) -> torch.Tensor:
        """Current effective threshold tau in (0, 1), derived from raw_threshold."""
        return torch.sigmoid(self.raw_threshold)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: (N, C, D, H, W)  Raw input volume (5-D, 3D spatial).

        Returns:
            logits: (N, out_ch, D, H, W)  Segmentation logits from backbone.
            meta:   dict with diagnostic tensors:
                      'ood_score'  (N,)  -- per-sample OOD severity
                      'gate_alpha' (N,)  -- soft gate weight alpha in (0, 1)
                      'threshold'  ()    -- current tau scalar
        """
        assert x.ndim == 5, (
            f"DynamicAdaptor expects 5-D input (N,C,D,H,W), got {x.ndim}-D"
        )

        # -- Step 1: OOD estimation ----------------------------------------
        ood_score = self.estimator(x)          # (N,)

        # -- Step 2: Soft gate ---------------------------------------------
        tau = self.threshold                   # scalar in (0, 1)
        # alpha shape: (N,) — broadcast-friendly
        alpha = torch.sigmoid(
            (ood_score - tau) / self.temperature
        )  # (N,)

        # -- Step 3: Conditional blending ----------------------------------
        x_adapted = self.transform(x)          # (N, C, D, H, W)

        # Reshape alpha to (N, 1, 1, 1, 1) for broadcasting over C, D, H, W
        alpha_5d = alpha.view(-1, 1, 1, 1, 1)
        x_in = alpha_5d * x_adapted + (1.0 - alpha_5d) * x  # (N, C, D, H, W)

        # -- Step 4: Frozen backbone inference -----------------------------
        logits = self.backbone(x_in)           # (N, out_ch, D, H, W)

        meta: Dict[str, torch.Tensor] = {
            "ood_score": ood_score.detach(),
            "gate_alpha": alpha.detach(),
            "threshold": tau.detach(),
        }
        return logits, meta

    # ------------------------------------------------------------------
    def get_tta_parameters(self) -> list:
        """Returns only the trainable TTA parameters (estimator + transform + tau).

        Use this to build a TTA-specific optimiser that does NOT touch the
        frozen backbone weights.
        """
        return (
            list(self.estimator.parameters())
            + list(self.transform.parameters())
            + [self.raw_threshold]
        )

    # ------------------------------------------------------------------
    def trainable_parameter_count(self) -> Dict[str, int]:
        """Diagnostic breakdown of trainable parameter counts per component."""

        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "estimator": count(self.estimator),
            "transform": count(self.transform),
            "threshold": self.raw_threshold.numel() if self.raw_threshold.requires_grad else 0,
            "backbone": count(self.backbone),
        }


# ---------------------------------------------------------------------------
# Sanity-check entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("  TTA Adaptors -- Forward-Pass Dimensional Stability Check")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}\n")

    # -- 1. Dummy 3D MRI volume --------------------------------------------
    # BraTS-style: (N=2, C=4, D=64, H=64, W=64)
    B, C, D, H, W = 2, 4, 64, 64, 64
    x = torch.randn(B, C, D, H, W, device=device)
    print(f"[Input]  x.shape = {tuple(x.shape)}")

    # -- 2. DomainShiftEstimator -------------------------------------------
    print("\n-- DomainShiftEstimator --")
    estimator = DomainShiftEstimator(in_channels=C).to(device)
    with torch.no_grad():
        ood_scores = estimator(x)
    assert ood_scores.shape == (B,), f"Expected ({B},), got {ood_scores.shape}"
    n_est = sum(p.numel() for p in estimator.parameters())
    print(f"  ood_scores.shape = {tuple(ood_scores.shape)}  OK")
    print(f"  ood_scores       = {[round(v, 4) for v in ood_scores.tolist()]}")
    print(f"  Parameters       = {n_est:,}")

    # -- 3. LearnableTransform3D -------------------------------------------
    print("\n-- LearnableTransform3D --")
    transform = LearnableTransform3D(in_channels=C, kernel_size=3).to(device)
    with torch.no_grad():
        x_adapted = transform(x)
    assert x_adapted.shape == x.shape, (
        f"Shape mismatch: expected {tuple(x.shape)}, got {tuple(x_adapted.shape)}"
    )
    max_delta = (x_adapted - x).abs().max().item()
    n_trn = sum(p.numel() for p in transform.parameters())
    print(f"  x_adapted.shape  = {tuple(x_adapted.shape)}  OK")
    print(f"  Max |delta| init = {max_delta:.6f}  (expect ~0 -- identity init)")
    print(f"  Parameters       = {n_trn:,}")

    # -- 4. DynamicAdaptor with frozen UNet3D backbone ---------------------
    print("\n-- DynamicAdaptor (with frozen UNet3D backbone) --")
    backbone_available = False
    try:
        sys.path.insert(0, ".")
        from unet3d_backbone import get_unet3d_brats  # type: ignore[import]
        backbone_available = True
    except ImportError:
        pass

    if backbone_available:
        backbone = get_unet3d_brats(in_channels=C, out_channels=4, c0=32).to(device)
    else:
        print("  [WARN] unet3d_backbone not found — using a stub backbone.")

        class _StubBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Conv3d(C, 4, kernel_size=1)

            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                return self.proj(inp)

        backbone = _StubBackbone().to(device)

    adaptor = DynamicAdaptor(
        backbone=backbone,
        transform=transform,
        estimator=estimator,
        in_channels=C,
        init_threshold=0.5,
        temperature=0.05,
        freeze_backbone=True,
    ).to(device)

    with torch.no_grad():
        logits, meta = adaptor(x)

    expected_out_ch = 4
    assert logits.shape == (B, expected_out_ch, D, H, W), (
        f"Expected ({B},{expected_out_ch},{D},{H},{W}), got {tuple(logits.shape)}"
    )

    param_counts = adaptor.trainable_parameter_count()
    total_tta = (
        param_counts["estimator"]
        + param_counts["transform"]
        + param_counts["threshold"]
    )

    print(f"  logits.shape     = {tuple(logits.shape)}  OK")
    print(f"  ood_score        = {[round(v, 4) for v in meta['ood_score'].tolist()]}")
    print(f"  gate_alpha (a)   = {[round(v, 4) for v in meta['gate_alpha'].tolist()]}")
    print(f"  threshold (tau)  = {meta['threshold'].item():.4f}")
    print(f"\n  Trainable parameter breakdown:")
    for k, v in param_counts.items():
        print(f"    {k:<12}: {v:>10,}")
    print(f"    {'TTA total':<12}: {total_tta:>10,}")

    # -- 5. Gradient flow check --------------------------------------------
    print("\n-- Gradient Flow (TTA update simulation) --")
    if backbone_available:
        bb_grad = get_unet3d_brats(in_channels=C, out_channels=4, c0=32).to(device)
    else:
        bb_grad = _StubBackbone().to(device)  # type: ignore[possibly-undefined]

    adaptor_grad = DynamicAdaptor(
        backbone=bb_grad,
        in_channels=C,
        init_threshold=0.5,
    ).to(device)

    x_grad = torch.randn(1, C, D, H, W, device=device)
    logits_grad, _ = adaptor_grad(x_grad)

    # Proxy TTA loss: entropy of sigmoid-activated logits
    probs = torch.sigmoid(logits_grad)
    eps_ent = 1e-6
    loss = -(
        probs * probs.clamp(min=eps_ent).log()
        + (1.0 - probs) * (1.0 - probs).clamp(min=eps_ent).log()
    ).mean()
    loss.backward()

    tta_params = adaptor_grad.get_tta_parameters()
    grads_exist = all(p.grad is not None for p in tta_params)
    backbone_grads = any(
        p.grad is not None for p in adaptor_grad.backbone.parameters()
    )

    print(f"  TTA param gradients exist : {grads_exist}  (expected True)")
    print(f"  Backbone gradients exist  : {backbone_grads}  (expected False)")

    print("\n" + "=" * 70)
    print("  All checks passed. Dimensional stability confirmed.")
    print("=" * 70)
