"""BraTS dataset class for loading and serving data."""

from pathlib import Path
from typing import Dict, List, Optional

from monai.data import CacheDataset, PersistentDataset, Dataset, DataLoader


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
    cache_dir: str = None,
) -> Dict[str, DataLoader]:
    """Create train/val/test DataLoaders.

    Uses PersistentDataset (disk cache) when cache_dir is set, otherwise
    plain Dataset with no RAM caching. This avoids OOM on large datasets
    like BraTS with 1000+ cases.

    Args:
        cache_dir: If provided, use PersistentDataset to cache transformed
                   data to disk. First epoch is slow, subsequent epochs are fast.
                   If None, no caching (each sample is loaded + transformed on the fly).

    Returns:
        Dict with keys "train", "val", "test" -> DataLoader
    """
    train_files = build_file_list(train_cases, modalities, include_label=True)
    val_files = build_file_list(val_cases, modalities, include_label=True)
    test_files = build_file_list(test_cases, modalities, include_label=True)

    if cache_dir:
        cache_path = Path(cache_dir).expanduser()
        train_cache = cache_path / "train"
        val_cache = cache_path / "val"
        test_cache = cache_path / "test"
        for d in [train_cache, val_cache, test_cache]:
            d.mkdir(parents=True, exist_ok=True)

        train_ds = PersistentDataset(train_files, transform=train_transform, cache_dir=str(train_cache))
        val_ds = PersistentDataset(val_files, transform=val_transform, cache_dir=str(val_cache))
        test_ds = PersistentDataset(test_files, transform=val_transform, cache_dir=str(test_cache))
    else:
        train_ds = Dataset(train_files, transform=train_transform)
        val_ds = Dataset(val_files, transform=val_transform)
        test_ds = Dataset(test_files, transform=val_transform)

    # persistent_workers keeps worker processes alive between epochs,
    # preventing fork deadlocks in Docker containers
    use_persistent = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=use_persistent)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=use_persistent)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=use_persistent)

    return {"train": train_loader, "val": val_loader, "test": test_loader}
