"""
dataset_loader.py - 3D Multimodal MRI DataLoader for BraTS Medical Image Segmentation.

Features:
- Dynamically scans patient directories from local storage (e.g. ./data/BraTS/).
- Reads NIfTI (.nii.gz) files for 4 modalities: _t1, _t1ce, _t2, _flair, and segmentation mask _seg.
- Concatenates modalities into a single 4-channel 3D volume of shape (4, D, H, W).
- MONAI dictionary transforms:
    * LoadImaged
    * EnsureChannelFirstd
    * Orientationd (RAS orientation)
    * ConcatItemsd (combining 4 modalities into 'image')
    * NormalizeIntensityd (per-channel non-zero voxel normalization for 'image')
    * SpatialPadd (padding 'image' and 'seg' if smaller than roi_size)
    * RandSpatialCropSamplesd / CenterSpatialCropd for patch extraction (e.g. 64, 64, 64)
    * EnsureTyped (torch.float32 for 'image', torch.long/int for 'seg')
- Returns batches containing 'image': (B, 4, 64, 64, 64) and 'seg': (B, 1, 64, 64, 64).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

import monai
from monai.data import CacheDataset, Dataset, list_data_collate
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandSpatialCropSamplesd,
    SpatialPadd,
)


def scan_brats_directory(root_dir: str = "./data/BraTS") -> List[Dict[str, str]]:
    """
    Scans root_dir for patient subdirectories and maps the 4 MRI modalities and segmentation mask.

    Expected file suffixes:
        - _t1.nii.gz
        - _t1ce.nii.gz
        - _t2.nii.gz
        - _flair.nii.gz
        - _seg.nii.gz

    Ignores hidden files and hidden directories (such as .DS_Store).

    Args:
        root_dir: Path to directory containing patient subfolders.

    Returns:
        List of dictionaries with keys: 't1', 't1ce', 't2', 'flair', 'seg'.
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"BraTS directory not found: {root_dir}")

    patient_records: List[Dict[str, str]] = []
    required_modalities = {
        "t1": "_t1.nii.gz",
        "t1ce": "_t1ce.nii.gz",
        "t2": "_t2.nii.gz",
        "flair": "_flair.nii.gz",
    }
    seg_suffix = "_seg.nii.gz"

    for entry in sorted(os.scandir(root_dir), key=lambda e: e.name):
        # Ignore hidden items and non-directories
        if entry.name.startswith(".") or not entry.is_dir():
            continue

        patient_dir = entry.path
        patient_files = [f for f in os.listdir(patient_dir) if not f.startswith(".")]

        patient_dict: Dict[str, str] = {}
        missing = False

        for mod_key, suffix in required_modalities.items():
            matched = [f for f in patient_files if f.endswith(suffix)]
            if matched:
                patient_dict[mod_key] = os.path.join(patient_dir, matched[0])
            else:
                missing = True
                break

        if missing:
            continue

        matched_seg = [f for f in patient_files if f.endswith(seg_suffix)]
        if matched_seg:
            patient_dict["seg"] = os.path.join(patient_dir, matched_seg[0])
            patient_records.append(patient_dict)

    return patient_records


def get_brats_transforms(
    roi_size: Sequence[int] = (64, 64, 64),
    num_samples_per_volume: int = 2,
    mode: str = "train",
    image_keys: Sequence[str] = ("t1", "t1ce", "t2", "flair"),
    seg_key: Optional[str] = "seg",
) -> Compose:
    """
    Constructs a MONAI transform pipeline for 4-modality BraTS 3D MRI and segmentation mask.

    Args:
        roi_size: Desired patch size, default (64, 64, 64).
        num_samples_per_volume: Number of cropped patches per volume when mode=='train'.
        mode: 'train' (random spatial crops) or 'val'/'test' (center crop).
        image_keys: Modality keys in dictionary items to load and concatenate.
        seg_key: Segmentation mask key in dictionary items.

    Returns:
        MONAI Compose pipeline.
    """
    all_load_keys = list(image_keys)
    if seg_key is not None:
        all_load_keys.append(seg_key)

    transform_list = [
        LoadImaged(keys=all_load_keys, image_only=True),
        EnsureChannelFirstd(keys=all_load_keys),
        Orientationd(keys=all_load_keys, axcodes="RAS"),
        # Concatenate 4 modalities along channel dim -> single tensor under 'image' of shape (4, D, H, W)
        ConcatItemsd(keys=image_keys, name="image", dim=0),
        # Normalize each channel independently over non-zero voxels for image
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]

    crop_keys = ["image"]
    if seg_key is not None:
        crop_keys.append(seg_key)

    # Pad if volume is smaller than roi_size
    transform_list.append(SpatialPadd(keys=crop_keys, spatial_size=roi_size))

    if mode == "train":
        transform_list.append(
            RandSpatialCropSamplesd(
                keys=crop_keys,
                roi_size=roi_size,
                num_samples=num_samples_per_volume,
                random_size=False,
            )
        )
    else:
        transform_list.append(
            CenterSpatialCropd(
                keys=crop_keys,
                roi_size=roi_size,
            )
        )

    transform_list.append(EnsureTyped(keys="image", dtype=torch.float32))
    if seg_key is not None:
        transform_list.append(EnsureTyped(keys=seg_key, dtype=torch.long))

    return Compose(transform_list)


class BraTSDataset(Dataset):
    """
    Dataset wrapper for BraTS multi-sequence volumes.
    Expects data_dicts with modality keys and segmentation mask path.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        transform: Optional[Compose] = None,
    ) -> None:
        super().__init__(data=data, transform=transform)


def get_brats_dataloader(
    data_dicts: List[Dict[str, str]],
    batch_size: int = 1,
    roi_size: Sequence[int] = (64, 64, 64),
    mode: str = "train",
    num_samples_per_volume: int = 2,
    num_workers: int = 0,
    use_cache: bool = False,
    cache_rate: float = 1.0,
    shuffle: Optional[bool] = None,
    image_keys: Optional[Sequence[str]] = None,
    seg_key: Optional[str] = "seg",
) -> DataLoader:
    """
    Creates an optimized PyTorch DataLoader yielding (Batch, 4, 64, 64, 64) image tensors
    and (Batch, 1, 64, 64, 64) segmentation masks.

    Args:
        data_dicts: List of dicts mapping modality keys to file paths.
        batch_size: Mini-batch size.
        roi_size: Spatial patch size, default (64, 64, 64).
        mode: 'train' or 'val'.
        num_samples_per_volume: Number of crops per volume if mode == 'train'.
        num_workers: DataLoader multiprocessing worker count.
        use_cache: If True, uses MONAI CacheDataset.
        cache_rate: Fraction of dataset to cache if use_cache is True.
        shuffle: Whether to shuffle batches. Defaults to True for train, False otherwise.
        image_keys: Tuple/list of modality keys. Defaults to ("t1", "t1ce", "t2", "flair")
                    or adapts if data_dicts contain "t1c".
        seg_key: Segmentation key name in dicts. If not present in dicts, set to None.

    Returns:
        DataLoader returning dict batches with 'image' and 'seg' tensors.
    """
    if not data_dicts:
        raise ValueError("data_dicts list is empty. No BraTS cases provided.")

    # Auto-detect modality keys if not specified
    if image_keys is None:
        first_item = data_dicts[0]
        if "t1ce" in first_item:
            image_keys = ("t1", "t1ce", "t2", "flair")
        elif "t1c" in first_item:
            image_keys = ("t1", "t1c", "t2", "flair")
        else:
            image_keys = ("t1", "t1ce", "t2", "flair")

    # Check if seg key is actually present in data_dicts
    if seg_key is not None and seg_key not in data_dicts[0]:
        seg_key = None

    transforms = get_brats_transforms(
        roi_size=roi_size,
        num_samples_per_volume=num_samples_per_volume,
        mode=mode,
        image_keys=image_keys,
        seg_key=seg_key,
    )

    if shuffle is None:
        shuffle = (mode == "train")

    if use_cache:
        ds = CacheDataset(
            data=data_dicts,
            transform=transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
    else:
        ds = BraTSDataset(data=data_dicts, transform=transforms)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


# ---------------------------------------------------------------------------
# Verification on Real BraTS Dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_dir = "./data/BraTS"
    print("=" * 70)
    print("  BraTS 3D Multimodal MRI DataLoader - Real Data Verification")
    print(f"  Scanning directory: {data_dir}")
    print("=" * 70)

    patient_records = scan_brats_directory(data_dir)
    print(f"Discovered {len(patient_records)} valid patient directories.")

    if not patient_records:
        print(f"No valid patient folders with all 4 modalities and mask found in {data_dir}.")
    else:
        print(f"Sample patient paths: {patient_records[0]}")

        # Build DataLoader for a single batch
        batch_size = 1
        roi_size = (64, 64, 64)
        loader = get_brats_dataloader(
            data_dicts=patient_records,
            batch_size=batch_size,
            roi_size=roi_size,
            mode="val",
            num_workers=0,
            shuffle=False,
        )

        print("\nLoading single batch from real dataset...")
        for batch in loader:
            image_batch = batch["image"]
            seg_batch = batch["seg"]

            print(f"Batch 'image' shape: {tuple(image_batch.shape)} (dtype: {image_batch.dtype})")
            print(f"Batch 'seg' shape:   {tuple(seg_batch.shape)} (dtype: {seg_batch.dtype})")
            print(f"Image intensity range: [{image_batch.min().item():.3f}, {image_batch.max().item():.3f}]")
            print(f"Unique mask labels:    {torch.unique(seg_batch).tolist()}")
            break

    print("\n" + "=" * 70)
    print("  DataLoader verification completed successfully.")
    print("=" * 70)
