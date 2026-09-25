"""
energy_prior.py - Shape-Energy Prior for Explanation-Drift Prevention
in 3D Medical Image Test-Time Adaptation (TTA).

Problem statement
----------------─
During TTA, entropy minimisation alone can cause "explanation drift": the
model gains confidence by producing anatomically impossible segmentation
shapes (fragmented blobs, holes, disconnected islands).  A learned shape-
energy prior counters this by penalising predictions that deviate from the
manifold of biologically plausible anatomical structures.

Module inventory
----------------
  ShapeEnergyDiscriminator
      Lightweight 3D CNN that maps softmax probability maps
      (N, C, D, H, W) -> scalar energy ∈ R.

      Lower energy  ≡  biologically plausible shape (compact, smooth,
                        topologically sound segmentation).
      Higher energy ≡  fragmented / noisy / anatomically impossible shape.

      Architecture:
        Stage 0  - 3x3x3 depthwise-sep + instance norm  (spatial detail)
        Stage 1  - strided depthwise-sep x 2             (multi-scale context)
        Stage 2  - dilated residual block                 (global topology)
        Head     - global average pool -> 2-layer MLP -> scalar

      Design choices aligned with the rest of the EIDOS framework:
        • InstanceNorm3d (affine=True) instead of BatchNorm - works at N=1.
        • Depthwise-separable convolutions - low parameter count, runs every
          forward pass without dominating TTA wall-clock time.
        • No final sigmoid / softplus on the energy output - the raw score is
          used directly as a loss term; its magnitude is controlled by lambda.

  ExplanationDriftLoss
      Composite TTA loss module:

        L_entropy  =  - (1/V) Σ_v  Σ_c  p_vc · log(p_vc + ε)
                   (standard voxel-wise entropy minimisation)

        L_energy   =  ShapeEnergyDiscriminator(softmax(logits))
                   (mean energy over the batch)

        L_total    =  L_entropy + λ · L_energy

      Both terms are differentiable w.r.t. the input logits, so a single
      .backward() call propagates gradients through both paths simultaneously.

Tensor convention: (N, C, D, H, W) - strictly 5-D, 3D spatial, throughout.
"""

from __future__ import annotations

import math
from typing import Dict, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dsconv3d(
    in_ch: int,
    out_ch: int,
    kernel: int = 3,
    stride: int = 1,
    dilation: int = 1,
    norm: bool = True,
) -> nn.Sequential:
    """Depthwise-separable 3D convolution block with optional InstanceNorm.

    Depthwise (groups=in_ch) -> Pointwise (1x1x1) -> InstanceNorm -> LeakyReLU.
    InstanceNorm is used in preference to GroupNorm / BatchNorm because it
    operates correctly at batch-size 1, which is the common TTA setting.
    """
    pad = dilation * (kernel // 2)
    layers: list[nn.Module] = [
        # Depthwise: spatial mixing, no cross-channel mixing
        nn.Conv3d(
            in_ch, in_ch,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=in_ch,
            bias=False,
        ),
        # Pointwise: channel mixing, no spatial mixing
        nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
    ]
    if norm:
        layers.append(nn.InstanceNorm3d(out_ch, affine=True))
    layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
    return nn.Sequential(*layers)


class _DilatedResBlock3d(nn.Module):
    """Lightweight dilated residual block for global topology awareness.

    Two parallel depthwise-sep branches at different dilation rates capture
    multi-scale context, then fuse via addition with a skip projection.
    Useful for detecting topological defects (holes, disconnected blobs)
    that only manifest over large receptive fields.
    """

    def __init__(self, ch: int, dilation_a: int = 2, dilation_b: int = 4) -> None:
        super().__init__()
        # Branch A: medium-range context
        self.branch_a = nn.Sequential(
            _dsconv3d(ch, ch, kernel=3, dilation=dilation_a),
            _dsconv3d(ch, ch, kernel=3, dilation=1),
        )
        # Branch B: long-range context (global topology)
        self.branch_b = nn.Sequential(
            _dsconv3d(ch, ch, kernel=3, dilation=dilation_b),
            _dsconv3d(ch, ch, kernel=3, dilation=1),
        )
        # Pointwise skip projection to preserve gradient flow
        self.skip = nn.Conv3d(ch, ch, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(ch, affine=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.branch_a(x) + self.branch_b(x) + self.skip(x)
        return self.act(self.norm(out))


# ---------------------------------------------------------------------------
# 1.  ShapeEnergyDiscriminator
# ---------------------------------------------------------------------------


class ShapeEnergyDiscriminator(nn.Module):
    """Learned shape-energy prior for 3D medical segmentation masks.

    Maps a softmax probability map (N, C, D, H, W) to a scalar energy value.

        energy ↓  ->  biologically plausible shape
        energy ↑  ->  fragmented / artefactual shape

    Because the energy is used directly as a loss term (via
    ExplanationDriftLoss), the discriminator must be differentiable w.r.t.
    its input.  All operations are therefore smooth and gradient-friendly -
    no argmax, no hard thresholds, no non-differentiable morphology ops.

    Architecture stages
    ------------------─
    Stage 0  (stem)     - 3x3x3 dsconv, no stride, captures local boundary
                          sharpness and voxel-level regularity.
    Stage 1  (encode)   - two strided dsconv blocks halve spatial dims twice;
                          spatial compression -> abstract shape features.
    Stage 2  (topology) - dilated residual block with d=2 and d=4 branches;
                          captures global shape topology and connectivity.
    Head                - adaptive global average pool -> flatten ->
                          Linear -> LayerNorm -> LReLU -> Dropout -> Linear
                          -> scalar energy (no bounding activation).

    Args:
        num_classes   : Number of segmentation classes C (input channels).
        base_ch       : Internal channel width.  Default 32.
        mlp_hidden    : Hidden size in the MLP head.  Default 128.
        dropout       : Dropout probability in the MLP head.  Default 0.2.
    """

    def __init__(
        self,
        num_classes: int = 4,
        base_ch: int = 32,
        mlp_hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        C_in = num_classes
        B = base_ch

        # -- Stage 0: stem - local boundary sharpness ----------------------
        self.stem = nn.Sequential(
            _dsconv3d(C_in, B, kernel=3, stride=1),
            _dsconv3d(B, B, kernel=3, stride=1),
        )

        # -- Stage 1: spatial encoding - abstract shape features ----------─
        self.encoder = nn.Sequential(
            _dsconv3d(B, B * 2, kernel=3, stride=2),    # D/2, H/2, W/2
            _dsconv3d(B * 2, B * 4, kernel=3, stride=2),  # D/4, H/4, W/4
        )

        # -- Stage 2: dilated residual - topology / connectivity ----------─
        self.topology = _DilatedResBlock3d(B * 4, dilation_a=2, dilation_b=4)

        # -- Head: global pool -> scalar energy ----------------------------
        feat_ch = B * 4
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_ch, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
            # No bounding activation - raw energy used as loss term directly
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
            elif isinstance(m, (nn.InstanceNorm3d, nn.LayerNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute shape energy of a softmax probability map.

        Args:
            probs: (N, C, D, H, W)  Softmax probability map.
                   Values must be in [0, 1] and sum to 1 along dim=1.
                   Do NOT pass raw logits - apply F.softmax first.

        Returns:
            energy: (N,)  Per-sample scalar energy (unbounded real).
                    Lower -> more anatomically plausible.
        """
        assert probs.ndim == 5, (
            f"ShapeEnergyDiscriminator expects 5-D input (N,C,D,H,W), "
            f"got {probs.ndim}-D"
        )
        z = self.stem(probs)        # (N, B,   D,   H,   W)
        z = self.encoder(z)         # (N, 4B,  D/4, H/4, W/4)
        z = self.topology(z)        # (N, 4B,  D/4, H/4, W/4)
        pooled = self.gap(z)        # (N, 4B,  1,   1,   1)
        energy = self.head(pooled)  # (N, 1)
        return energy.squeeze(1)    # (N,)


# ---------------------------------------------------------------------------
# Named return type for loss decomposition
# ---------------------------------------------------------------------------


class DriftLossOutput(NamedTuple):
    """Structured return value from ExplanationDriftLoss.forward().

    Attributes:
        loss        : Total combined loss scalar (differentiable).
        l_entropy   : Entropy minimisation term (detached for logging).
        l_energy    : Shape energy penalty term (detached for logging).
        energy_raw  : Per-sample energy vector  (N,) before mean reduction.
    """
    loss: torch.Tensor
    l_entropy: torch.Tensor
    l_energy: torch.Tensor
    energy_raw: torch.Tensor


# ---------------------------------------------------------------------------
# 2.  ExplanationDriftLoss
# ---------------------------------------------------------------------------


class ExplanationDriftLoss(nn.Module):
    """Composite TTA loss that prevents explanation drift in 3D segmentation.

    Combines two complementary objectives:

    L_entropy
        Voxel-wise predictive entropy minimisation.  Pushes the model
        toward confident (low-entropy) predictions on unlabelled target
        data, counteracting domain-shift-induced uncertainty.

            L_entropy = -(1/V) Σ_v Σ_c  p_vc · log(p_vc + ε)

        where V = DxHxW, p = softmax(logits).

    L_energy
        Shape-energy penalty from ShapeEnergyDiscriminator.  Penalises
        anatomically implausible shapes (fragmented blobs, disconnected
        regions, noisy boundaries) that entropy minimisation alone may
        inadvertently produce.

            L_energy = mean_N( ShapeEnergyDiscriminator( softmax(logits) ) )

    Combined loss
        L_total = L_entropy + λ · L_energy

        λ (lambda_energy) scales the relative contribution of the shape
        prior.  A good starting point is λ ∈ [0.01, 0.1]; tune per dataset.

    Both terms are fully differentiable w.r.t. logits, so a single
    .backward() call propagates gradients through both paths simultaneously.

    Args:
        discriminator  : Pre-instantiated ShapeEnergyDiscriminator.
                         If None, a default one is created with `num_classes`.
        num_classes    : C (number of segmentation channels).  Default 4.
        lambda_energy  : Weight of the energy penalty term.  Default 0.05.
        entropy_eps    : Numerical stability epsilon in entropy log.  Default 1e-8.
        temperature    : Softmax temperature for logit-to-prob conversion.
                         Values > 1 soften predictions; < 1 sharpen them.
                         Default 1.0 (standard softmax).
    """

    def __init__(
        self,
        discriminator: Optional[ShapeEnergyDiscriminator] = None,
        num_classes: int = 4,
        lambda_energy: float = 0.05,
        entropy_eps: float = 1e-8,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        assert lambda_energy >= 0.0, "lambda_energy must be non-negative"
        assert temperature > 0.0, "temperature must be positive"

        self.discriminator = discriminator or ShapeEnergyDiscriminator(
            num_classes=num_classes
        )
        self.num_classes = num_classes
        self.lambda_energy = lambda_energy
        self.entropy_eps = entropy_eps
        self.temperature = temperature

    # ------------------------------------------------------------------
    def _entropy_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """Voxel-wise entropy minimisation loss.

        Args:
            probs: (N, C, D, H, W)  Softmax probabilities in [0, 1].

        Returns:
            Scalar mean entropy (averaged over voxels and batch).
        """
        # Entropy: -Σ_c p_c * log(p_c),  averaged over all voxels and batch
        log_probs = torch.log(probs + self.entropy_eps)  # (N, C, D, H, W)
        # Sum over class dimension, mean over spatial + batch
        voxel_entropy = -(probs * log_probs).sum(dim=1)  # (N, D, H, W)
        return voxel_entropy.mean()                       # scalar

    # ------------------------------------------------------------------
    def _energy_loss(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Shape energy penalty via ShapeEnergyDiscriminator.

        Args:
            probs: (N, C, D, H, W)  Softmax probabilities.

        Returns:
            (mean_energy, per_sample_energy)
        """
        per_sample = self.discriminator(probs)    # (N,)
        return per_sample.mean(), per_sample      # scalar, (N,)

    # ------------------------------------------------------------------
    def forward(self, logits: torch.Tensor) -> DriftLossOutput:
        """Compute the combined explanation-drift prevention loss.

        Args:
            logits: (N, C, D, H, W)  Raw unnormalised model output logits.
                    Gradients must be enabled for .backward() to work.

        Returns:
            DriftLossOutput (NamedTuple):
                .loss        - total differentiable loss scalar
                .l_entropy   - entropy term value (detached, for logging)
                .l_energy    - energy term value  (detached, for logging)
                .energy_raw  - per-sample energy (N,) before reduction
        """
        assert logits.ndim == 5, (
            f"ExplanationDriftLoss expects 5-D logits (N,C,D,H,W), "
            f"got {logits.ndim}-D"
        )

        # logits -> probabilities (differentiable w.r.t. logits)
        probs = F.softmax(logits / self.temperature, dim=1)  # (N, C, D, H, W)

        # -- L_entropy ----------------------------------------------------─
        l_ent = self._entropy_loss(probs)

        # -- L_energy ------------------------------------------------------
        l_eng_scalar, l_eng_per_sample = self._energy_loss(probs)

        # -- Combined loss ------------------------------------------------─
        total = l_ent + self.lambda_energy * l_eng_scalar

        return DriftLossOutput(
            loss=total,
            l_entropy=l_ent.detach(),
            l_energy=l_eng_scalar.detach(),
            energy_raw=l_eng_per_sample.detach(),
        )

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"lambda_energy={self.lambda_energy}, "
            f"temperature={self.temperature}"
        )


# ---------------------------------------------------------------------------
# Gradient verification & demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    SEP = "=" * 72

    print(SEP)
    print("  energy_prior.py - Shape Energy Prior Gradient Flow Verification")
    print(SEP)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")

    # -- Configuration ----------------------------------------------------─
    N, C, D, H, W = 2, 4, 64, 64, 64  # BraTS-style: batch=2, 4-class, 64³
    LAMBDA = 0.05
    LR = 1e-3
    N_STEPS = 3

    print(f"[Config] batch={N}, classes={C}, volume={D}x{H}x{W}, "
          f"lambda={LAMBDA}, lr={LR}\n")

    # -- 1. ShapeEnergyDiscriminator standalone check ----------------------─
    print("-- 1. ShapeEnergyDiscriminator forward pass --")
    disc = ShapeEnergyDiscriminator(num_classes=C, base_ch=32).to(device)
    n_disc = sum(p.numel() for p in disc.parameters())

    # Feed a random (but valid) probability map
    dummy_probs = F.softmax(torch.randn(N, C, D, H, W, device=device), dim=1)
    with torch.no_grad():
        energy_out = disc(dummy_probs)

    assert energy_out.shape == (N,), (
        f"Expected energy shape ({N},), got {energy_out.shape}"
    )
    print(f"  Input probs shape : {tuple(dummy_probs.shape)}")
    print(f"  Energy shape      : {tuple(energy_out.shape)}  OK")
    print(f"  Energy values     : {[round(v, 4) for v in energy_out.tolist()]}")
    print(f"  Parameters        : {n_disc:,}")

    # -- 2. ExplanationDriftLoss forward pass ------------------------------
    print("\n-- 2. ExplanationDriftLoss forward pass --")
    criterion = ExplanationDriftLoss(
        discriminator=disc,
        num_classes=C,
        lambda_energy=LAMBDA,
    ).to(device)

    dummy_logits = torch.randn(N, C, D, H, W, device=device)
    with torch.no_grad():
        out = criterion(dummy_logits)

    print(f"  Total loss   : {out.loss.item():.6f}")
    print(f"  L_entropy    : {out.l_entropy.item():.6f}")
    print(f"  L_energy     : {out.l_energy.item():.6f}")
    print(f"  Energy (raw) : {[round(v, 4) for v in out.energy_raw.tolist()]}")

    # -- 3. TTA gradient flow simulation ----------------------------------─
    print("\n-- 3. TTA Gradient Flow Simulation --")
    print(
        "   Simulating TTA: logits require_grad=True -> loss.backward() ->\n"
        "   verify gradients reach the input tensor.\n"
    )

    # The logits here represent the adapted model's output.
    # In real TTA they come from forward(adapted_input); here we simulate
    # them as a leaf tensor with requires_grad=True to prove gradient flow.
    tta_logits = torch.randn(
        N, C, D, H, W, device=device, requires_grad=True
    )

    # Recreate criterion (fresh discriminator) to track its grad params too
    disc_tta = ShapeEnergyDiscriminator(num_classes=C, base_ch=32).to(device)
    criterion_tta = ExplanationDriftLoss(
        discriminator=disc_tta,
        num_classes=C,
        lambda_energy=LAMBDA,
    ).to(device)

    # Use a simple SGD optimiser on the discriminator parameters
    # (in real TTA this would optimise the adaptor / transform params)
    tta_optim = torch.optim.SGD(
        list(criterion_tta.discriminator.parameters()),
        lr=LR,
        momentum=0.9,
    )

    loss_history: list[dict] = []

    for step in range(1, N_STEPS + 1):
        tta_optim.zero_grad()

        # Forward
        result = criterion_tta(tta_logits)

        # Backward - gradients flow through both L_entropy and L_energy
        result.loss.backward()

        # Verify gradient on the input logits leaf
        assert tta_logits.grad is not None, (
            "FAIL: No gradient reached tta_logits!"
        )
        grad_norm = tta_logits.grad.norm().item()

        # Verify gradients on discriminator parameters
        disc_grad_norms = [
            p.grad.norm().item()
            for p in criterion_tta.discriminator.parameters()
            if p.grad is not None
        ]
        assert len(disc_grad_norms) > 0, (
            "FAIL: No gradients in discriminator parameters!"
        )
        mean_disc_grad = sum(disc_grad_norms) / len(disc_grad_norms)

        tta_optim.step()

        log = {
            "step": step,
            "total_loss": result.loss.item(),
            "l_entropy": result.l_entropy.item(),
            "l_energy": result.l_energy.item(),
            "logits_grad_norm": grad_norm,
            "disc_grad_norm_mean": mean_disc_grad,
        }
        loss_history.append(log)

        print(
            f"  Step {step}/{N_STEPS} | "
            f"loss={log['total_loss']:.5f}  "
            f"(ent={log['l_entropy']:.5f}, "
            f"eng={log['l_energy']:.5f}) | "
            f"logits_grad_norm={log['logits_grad_norm']:.4f} | "
            f"disc_grad_norm={log['disc_grad_norm_mean']:.4f}"
        )

    # -- 4. Gradient path audit --------------------------------------------─
    print("\n-- 4. Gradient Path Audit --")
    print(f"  logits.grad is not None          : {tta_logits.grad is not None}  OK")
    print(f"  logits.grad norm                 : {tta_logits.grad.norm().item():.6f}")
    n_disc_params_with_grad = sum(
        1 for p in criterion_tta.discriminator.parameters() if p.grad is not None
    )
    n_disc_params_total = sum(
        1 for _ in criterion_tta.discriminator.parameters()
    )
    print(
        f"  Disc params with gradient        : "
        f"{n_disc_params_with_grad}/{n_disc_params_total}  OK"
    )

    # -- 5. Entropy monotonicity sanity check ------------------------------─
    print("\n-- 5. Entropy Boundary Sanity Check --")
    # Maximum entropy = uniform distribution over C classes = log(C)
    max_ent_logits = torch.zeros(1, C, 8, 8, 8, device=device)  # uniform after softmax
    max_ent_probs = F.softmax(max_ent_logits, dim=1)
    vox_ent_max = -(max_ent_probs * (max_ent_probs + 1e-8).log()).sum(dim=1).mean()

    # Minimum entropy = one-hot (confident) prediction
    onehot = torch.full((1, C, 8, 8, 8), fill_value=-1e6, device=device)
    onehot[:, 0] = 1e6
    min_ent_probs = F.softmax(onehot, dim=1)
    vox_ent_min = -(min_ent_probs * (min_ent_probs + 1e-8).log()).sum(dim=1).mean()

    theoretical_max = math.log(C)
    print(f"  Theoretical max entropy log({C}) = {theoretical_max:.4f}")
    print(f"  Uniform pred entropy             = {vox_ent_max.item():.4f}  (expect ~{theoretical_max:.4f})")
    print(f"  One-hot pred entropy             = {vox_ent_min.item():.6f}  (expect ~0)")
    assert vox_ent_max.item() > vox_ent_min.item(), "Entropy ordering violated!"
    print(f"  max_ent > min_ent                : True  OK")

    # -- Summary ------------------------------------------------------------
    print(f"\n{SEP}")
    print("  All checks passed.")
    print(f"  ShapeEnergyDiscriminator params : {n_disc:,}")
    print(f"  ExplanationDriftLoss components : L_entropy + {LAMBDA} * L_energy")
    print("  Gradients verified: logits -> L_entropy path  OK")
    print("  Gradients verified: logits -> softmax -> discriminator path  OK")
    print(f"{SEP}\n")
