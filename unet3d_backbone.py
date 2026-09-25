"""
3D U-Net Backbone for BraTS Medical Image Segmentation.

Configured according to the MedSeg-TTA benchmark standards:
- 4 input channels (e.g., T1, T1ce, T2, FLAIR)
- 4 output channels (e.g., Background, Necrotic/Non-enhancing tumor core, Peritumoral edema, GD-enhancing tumor)
- 5 encoder stages / levels with channel depth scaling based on basic width C0 = 32:
  Level 1: C0 * 1 = 32  (or base scaling factor)
  Progression across the 5 encoder stages:
    - Block 1: C0 * 1  = 32
    - Block 2: C0 * 2  = 64
    - Block 3: C0 * 4  = 128
    - Block 4: C0 * 8  = 256
    - Block 5: C0 * 16 = 512 (Bottleneck)
- Highly modular building blocks (ConvBlock, EncoderBlock, DecoderBlock, UpSample) to support
  weight freezing and downstream adaptor injection (e.g., LoRA, Bottleneck Adapters, Prompt Tuning, Side-Tuning).
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3d(nn.Module):
    """
    Standard 3D convolutional block consisting of:
    [Conv3d -> GroupNorm/BatchNorm3d -> Activation (LeakyReLU)] x 2 (or configurable count).
    Designed to be easily wrapped or extended with custom adaptor modules.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 8,
        norm_type: str = "group",  # 'group', 'batch', or 'instance'
        act_fn: str = "leaky_relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        def get_norm(channels: int) -> nn.Module:
            if norm_type == "group":
                # Ensure num_groups divides channels
                groups = num_groups
                while channels % groups != 0 and groups > 1:
                    groups -= 1
                return nn.GroupNorm(num_groups=groups, num_channels=channels)
            elif norm_type == "instance":
                return nn.InstanceNorm3d(channels, affine=True)
            elif norm_type == "batch":
                return nn.BatchNorm3d(channels)
            else:
                return nn.Identity()

        def get_act() -> nn.Module:
            if act_fn == "leaky_relu":
                return nn.LeakyReLU(negative_slope=0.01, inplace=True)
            elif act_fn == "relu":
                return nn.ReLU(inplace=True)
            elif act_fn == "gelu":
                return nn.GELU()
            else:
                return nn.Identity()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm1 = get_norm(out_channels)
        self.act1 = get_act()

        self.dropout = nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity()

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = get_norm(out_channels)
        self.act2 = get_act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class EncoderBlock3d(nn.Module):
    """
    Encoder Stage combining an optional downsampling operation (pooling or strided conv)
    followed by a standard ConvBlock3d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = True,
        downsample_mode: str = "maxpool",  # 'maxpool' or 'strided_conv'
        norm_type: str = "group",
        act_fn: str = "leaky_relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.downsample = downsample

        if self.downsample:
            if downsample_mode == "maxpool":
                self.down_op = nn.MaxPool3d(kernel_size=2, stride=2)
            elif downsample_mode == "strided_conv":
                self.down_op = nn.Conv3d(
                    in_channels,
                    in_channels,
                    kernel_size=2,
                    stride=2,
                    bias=False,
                )
            else:
                raise ValueError(f"Unknown downsample_mode: {downsample_mode}")
        else:
            self.down_op = nn.Identity()

        self.conv_block = ConvBlock3d(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_type=norm_type,
            act_fn=act_fn,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            downsampled_feat: feature map after downsampling and conv block (passed to next stage)
            skip_feat: feature map before downsampling (if downsample=False, same as output for level 1)
        """
        x_down = self.down_op(x)
        out = self.conv_block(x_down)
        return out


class DecoderBlock3d(nn.Module):
    """
    Decoder Stage performing upsampling (transposed conv or trilinear interpolation),
    channel concatenation with skip connections, and feature aggregation via ConvBlock3d.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        upsample_mode: str = "trilinear",  # 'trilinear' or 'transpose'
        norm_type: str = "group",
        act_fn: str = "leaky_relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode

        if upsample_mode == "transpose":
            self.up = nn.ConvTranspose3d(
                in_channels,
                in_channels,
                kernel_size=2,
                stride=2,
            )
        else:
            # trilinear + 1x1 conv to match channels if needed
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False),
            )

        # Conv block processes concatenated (upsampled features + skip connection features)
        self.conv_block = ConvBlock3d(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            norm_type=norm_type,
            act_fn=act_fn,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x)

        # Handle potential shape mismatches due to odd input spatial dimensions
        if x_up.shape[2:] != skip.shape[2:]:
            diff_d = skip.size(2) - x_up.size(2)
            diff_h = skip.size(3) - x_up.size(3)
            diff_w = skip.size(4) - x_up.size(4)
            x_up = F.pad(
                x_up,
                [
                    diff_w // 2,
                    diff_w - diff_w // 2,
                    diff_h // 2,
                    diff_h - diff_h // 2,
                    diff_d // 2,
                    diff_d - diff_d // 2,
                ],
            )

        cat_feat = torch.cat([x_up, skip], dim=1)
        out = self.conv_block(cat_feat)
        return out


class UNet3D(nn.Module):
    """
    3D U-Net Architecture configured for BraTS and MedSeg-TTA Benchmarks.

    Specifications:
        - In Channels: 4 (Multi-modal MRI: T1, T1ce, T2, FLAIR)
        - Out Channels: 4 (Segmentation Classes)
        - Basic Width (C0): 32
        - 5 Encoder stages with channel depths:
            Level 1: C0 * 1  = 32
            Level 2: C0 * 2  = 64
            Level 3: C0 * 4  = 128
            Level 4: C0 * 8  = 256
            Level 5: C0 * 16 = 512 (Bottleneck)
        - 4 Decoder stages mapping 512 -> 256 -> 128 -> 64 -> 32
        - Final 1x1x1 segmentation projection head -> 4 channels

    Designed for modularity:
        - Encoder/Decoder blocks and layers are individually accessible.
        - Supports selective weight freezing via `freeze_backbone()`.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        c0: int = 32,
        norm_type: str = "group",
        act_fn: str = "leaky_relu",
        dropout: float = 0.0,
        upsample_mode: str = "trilinear",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.c0 = c0

        # Encoder channel depth progression: [32, 64, 128, 256, 512]
        self.enc_channels: List[int] = [
            c0,       # C0 * 1  = 32
            c0 * 2,   # C0 * 2  = 64
            c0 * 4,   # C0 * 4  = 128
            c0 * 8,   # C0 * 8  = 256
            c0 * 16,  # C0 * 16 = 512 (Bottleneck)
        ]

        # --- Encoder Stage (5 blocks) ---
        self.enc1 = EncoderBlock3d(
            in_channels=in_channels,
            out_channels=self.enc_channels[0],
            downsample=False,  # Level 1 keeps initial spatial resolution
            norm_type=norm_type,
            act_fn=act_fn,
            dropout=dropout,
        )
        self.enc2 = EncoderBlock3d(
            in_channels=self.enc_channels[0],
            out_channels=self.enc_channels[1],
            downsample=True,
            norm_type=norm_type,
            act_fn=act_fn,
            dropout=dropout,
        )
        self.enc3 = EncoderBlock3d(
            in_channels=self.enc_channels[1],
            out_channels=self.enc_channels[2],
            downsample=True,
            norm_type=norm_type,
            act_fn=act_fn,
            dropout=dropout,
        )
        self.enc4 = EncoderBlock3d(
            in_channels=self.enc_channels[2],
            out_channels=self.enc_channels[3],
            downsample=True,
            norm_type=norm_type,
            act_fn=act_fn,
            dropout=dropout,
        )
        self.enc5 = EncoderBlock3d(
            in_channels=self.enc_channels[3],
            out_channels=self.enc_channels[4],
            downsample=True,
            norm_type=norm_type,
            act_fn=act_fn,
            dropout=dropout,
        )

        # --- Decoder Stage (4 blocks) ---
        self.dec4 = DecoderBlock3d(
            in_channels=self.enc_channels[4],
            skip_channels=self.enc_channels[3],
            out_channels=self.enc_channels[3],
            upsample_mode=upsample_mode,
            norm_type=norm_type,
            act_fn=act_fn,
            dropout=dropout,
        )
        self.dec3 = DecoderBlock3d(
            in_channels=self.enc_channels[3],
            skip_channels=self.enc_channels[2],
            out_channels=self.enc_channels[2],
            upsample_mode=upsample_mode,
            norm_type=norm_type,
            act_fn=act_fn,
            dropout=dropout,
        )
        self.dec2 = DecoderBlock3d(
            in_channels=self.enc_channels[2],
            skip_channels=self.enc_channels[1],
            out_channels=self.enc_channels[1],
            upsample_mode=upsample_mode,
            norm_type=norm_type,
            act_fn=act_fn,
            dropout=dropout,
        )
        self.dec1 = DecoderBlock3d(
            in_channels=self.enc_channels[1],
            skip_channels=self.enc_channels[0],
            out_channels=self.enc_channels[0],
            upsample_mode=upsample_mode,
            norm_type=norm_type,
            act_fn=act_fn,
            dropout=dropout,
        )

        # --- Segmentation Head ---
        self.final_conv = nn.Conv3d(
            in_channels=self.enc_channels[0],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming (He) normal initialization standard for medical segmentation."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self, freeze_head: bool = False) -> None:
        """
        Utility method to freeze all backbone weights for parameter-efficient fine-tuning / TTA adaptor training.
        """
        for param in self.parameters():
            param.requires_grad = False
        if not freeze_head:
            for param in self.final_conv.parameters():
                param.requires_grad = True

    def forward_encoder(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extracts multi-scale encoder features. Useful for downstream feature extraction or side-adaptors.
        """
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        s5 = self.enc5(s4)
        return [s1, s2, s3, s4, s5]

    def forward_decoder(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """Decodes multi-scale features back to original resolution."""
        s1, s2, s3, s4, s5 = encoder_features
        d4 = self.dec4(s5, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return d1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of UNet3D.

        Args:
            x: Input 5D Tensor of shape (Batch, 4, Depth, Height, Width)

        Returns:
            Logits 5D Tensor of shape (Batch, 4, Depth, Height, Width)
        """
        encoder_features = self.forward_encoder(x)
        dec_features = self.forward_decoder(encoder_features)
        logits = self.final_conv(dec_features)
        return logits


def get_unet3d_brats(
    in_channels: int = 4,
    out_channels: int = 4,
    c0: int = 32,
    **kwargs,
) -> UNet3D:
    """Helper factory function to instantiate the MedSeg-TTA compliant UNet3D model."""
    return UNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        c0=c0,
        **kwargs,
    )


if __name__ == "__main__":
    # Sanity check forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_unet3d_brats(in_channels=4, out_channels=4, c0=32).to(device)
    
    # BraTS input tensor: [B, C, D, H, W]
    sample_input = torch.randn(1, 4, 64, 64, 64, device=device)
    
    print("Running UNet3D BraTS forward pass check...")
    with torch.no_grad():
        output = model(sample_input)
    
    print(f"Input shape:  {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test freezing utility
    model.freeze_backbone(freeze_head=False)
    trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after freezing backbone (head unfrozen): {trainable_params_after:,}")
    print("Sanity check completed successfully.")
