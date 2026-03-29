"""BraTS dataset class for loading and serving data."""

from pathlib import Path
from typing import Dict, List, Optional

from monai.data import CacheDataset, DataLoader


def build_file_list(
    case_dirs: List[Path],
    modalities: List[str],
    include_label: bool = True,
) -> List[Dict[str, str]]:
    """Build MONAI-compatible file list from case directories.

    Args:
        case_dirs: List of case directory paths.
        modalities: List of modality suffixes (e.g., ["t1c", "t1n", "t2f", "t2w"]).
        include_label: Whether to include segmentation label files.

    Returns:
        List of dicts with keys like "image_t1c", "image_t1n", ..., "label".
    """
    file_list = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        entry = {"case_id": case_id}

        # Find modality files
        all_found = True
        for mod in modalities:
            candidates = list(case_dir.glob(f"*-{mod}.nii.gz"))
            if not candidates:
                all_found = False
                break
            entry[f"image_{mod}"] = str(candidates[0])

        if not all_found:
            continue

        # Find segmentation
        if include_label:
            seg_candidates = list(case_dir.glob("*-seg.nii.gz"))
            if not seg_candidates:
                continue
            entry["label"] = str(seg_candidates[0])

        file_list.append(entry)

    return file_list


def get_dataloaders(
    train_cases: List[Path],
    val_cases: List[Path],
    test_cases: List[Path],
    modalities: List[str],
    train_transform,
    val_transform,
    batch_size: int = 2,
    num_workers: int = 4,
    cache_rate: float = 0.5,
) -> Dict[str, DataLoader]:
    """Create train/val/test DataLoaders with caching.

    Returns:
        Dict with keys "train", "val", "test" -> DataLoader
    """
    train_files = build_file_list(train_cases, modalities, include_label=True)
    val_files = build_file_list(val_cases, modalities, include_label=True)
    test_files = build_file_list(test_cases, modalities, include_label=True)

    train_ds = CacheDataset(train_files, transform=train_transform, cache_rate=cache_rate, num_workers=num_workers)
    val_ds = CacheDataset(val_files, transform=val_transform, cache_rate=1.0, num_workers=num_workers)
    test_ds = CacheDataset(test_files, transform=val_transform, cache_rate=1.0, num_workers=num_workers)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return {"train": train_loader, "val": val_loader, "test": test_loader}
