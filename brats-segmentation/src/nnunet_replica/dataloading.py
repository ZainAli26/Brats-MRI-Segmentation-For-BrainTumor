"""Patch sampling identical to ``nnUNetDataLoader3D``, served through torch DataLoader.

Two things here matter for replication and are easy to get wrong:

**Forced-foreground oversampling.** nnU-Net does not use a pos/neg ratio like MONAI's
``RandCropByPosNegLabeld``. It builds the batch positionally: the *last*
``round(batch_size * oversample_foreground_percent)`` samples are guaranteed to be
centred on a randomly chosen foreground *class* (uniform over the classes present in
that case, not over voxels — which is what keeps tiny classes like NETC alive), and the
rest are uniformly random locations. With batch 2 / 0.33 that is exactly "sample 0
random, sample 1 forced foreground".

**Over-the-edge patches.** ``get_bbox`` deliberately lets the patch hang off the volume
(``need_to_pad``) and pads with 0 for data and ``-1`` for seg. That ``-1`` is later
turned back into 0 by ``RemoveLabelTransform``, but only *after* the spatial transform,
so border voxels are handled the same way nnU-Net handles them.

The dataloader yields one *whole batch* per item (``batch_size=None`` on the torch
DataLoader), because the oversampling pattern is defined over batch positions.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


class PreprocessedDataset:
    """Read-only view over a preprocessed-case folder (nnU-Net unpacked layout)."""

    def __init__(self, folder: str | Path, case_ids: Sequence[str]):
        self.folder = Path(folder)
        self.case_ids = list(case_ids)
        missing = [c for c in self.case_ids
                   if not (self.folder / f"{c}.npy").is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} case(s) missing from {self.folder} "
                f"(e.g. {missing[:3]}). Run preprocess_replica.py first."
            )
        self._props_cache: Dict[str, Dict] = {}

    def __len__(self) -> int:
        return len(self.case_ids)

    def load_properties(self, case_id: str) -> Dict:
        if case_id not in self._props_cache:
            with open(self.folder / f"{case_id}.pkl", "rb") as f:
                self._props_cache[case_id] = pickle.load(f)
        return self._props_cache[case_id]

    def load_case(self, case_id: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Memory-map the arrays — only the sampled patch is faulted in from disk."""
        data = np.load(self.folder / f"{case_id}.npy", mmap_mode="r")
        seg = np.load(self.folder / f"{case_id}_seg.npy", mmap_mode="r")
        return data, seg, self.load_properties(case_id)


class PatchSampler3D:
    """Port of ``nnUNetDataLoader3D.generate_train_batch`` + ``nnUNetDataLoaderBase.get_bbox``."""

    def __init__(
        self,
        dataset: PreprocessedDataset,
        batch_size: int,
        patch_size: Sequence[int],
        final_patch_size: Sequence[int],
        foreground_labels: Sequence[int],
        oversample_foreground_percent: float = 0.33,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.patch_size = np.array(patch_size, dtype=int)          # sampled (possibly inflated)
        self.final_patch_size = np.array(final_patch_size, dtype=int)  # what the network sees
        self.foreground_labels = list(foreground_labels)
        self.oversample_foreground_percent = float(oversample_foreground_percent)
        self.need_to_pad = (self.patch_size - self.final_patch_size).astype(int)
        self.rng = rng if rng is not None else np.random.RandomState()

    def _do_oversample(self, sample_idx: int) -> bool:
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def get_bbox(self, data_shape: Sequence[int], force_fg: bool,
                 class_locations: Optional[Dict]) -> Tuple[List[int], List[int]]:
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i]
               for i in range(dim)]

        if not force_fg:
            bbox_lbs = [self.rng.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
        else:
            assert class_locations is not None, "force_fg needs class_locations"
            eligible = [c for c in class_locations if len(class_locations[c]) > 0]
            if len(eligible) == 0:
                # Case has no foreground at all — fall back to a random location.
                bbox_lbs = [self.rng.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            else:
                selected_class = eligible[self.rng.choice(len(eligible))]
                voxels = class_locations[selected_class]
                voxel = voxels[self.rng.choice(len(voxels))]
                # voxel carries the leading channel axis, hence voxel[i + 1].
                bbox_lbs = [max(lbs[i], int(voxel[i + 1]) - self.patch_size[i] // 2)
                            for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs

    def generate_batch(self) -> Dict[str, np.ndarray]:
        selected = [self.dataset.case_ids[self.rng.randint(len(self.dataset))]
                    for _ in range(self.batch_size)]

        data_all: Optional[np.ndarray] = None
        seg_all: Optional[np.ndarray] = None

        for j, case_id in enumerate(selected):
            force_fg = self._do_oversample(j)
            data, seg, props = self.dataset.load_case(case_id)

            if data_all is None:
                data_all = np.zeros((self.batch_size, data.shape[0], *self.patch_size), np.float32)
                seg_all = np.zeros((self.batch_size, seg.shape[0], *self.patch_size), np.int16)

            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, props.get("class_locations"))

            # Crop only the in-bounds part (cheap on a memmap) and write it straight into
            # the pre-zeroed batch slot. nnU-Net np.pad()s the crop first; writing into a
            # slice gives the identical array without materialising a second ~215 MB
            # temporary, which matters because the sampled patch here is larger than the
            # volume on every axis, so most of it is padding.
            valid_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]
            src = tuple(slice(a, b) for a, b in zip(valid_lbs, valid_ubs))
            dst = tuple(slice(a - b, a - b + (u - a))
                        for a, b, u in zip(valid_lbs, bbox_lbs, valid_ubs))

            seg_all[j].fill(-1)   # outside the sampled volume is "unlabelled", as in nnU-Net
            data_all[j][(slice(None), *dst)] = data[(slice(None), *src)]
            seg_all[j][(slice(None), *dst)] = seg[(slice(None), *src)]

        return {"data": data_all, "seg": seg_all, "keys": selected}


class _BatchStream(IterableDataset):
    """Infinite stream of augmented batches; each worker owns an independent RNG."""

    def __init__(self, sampler: PatchSampler3D, transform, base_seed: int):
        self.sampler = sampler
        self.transform = transform
        self.base_seed = base_seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        if info is not None:
            # One thread per worker. Augmentation is single-threaded numpy/scipy work, so
            # letting each of ~12 workers spawn its own thread pool oversubscribes the CPU
            # and costs more than it buys (nnU-Net pins its DA processes the same way).
            torch.set_num_threads(1)
        seed = (self.base_seed + 1_000_003 * (worker_id + 1)) % (2 ** 31 - 1)
        self.sampler.rng = np.random.RandomState(seed)
        np.random.seed(seed)          # batchgenerators transforms use the global numpy RNG
        torch.manual_seed(seed)
        while True:
            batch = self.sampler.generate_batch()
            yield self.transform(**batch) if self.transform is not None else batch


def build_loader(
    dataset: PreprocessedDataset,
    batch_size: int,
    sampled_patch_size: Sequence[int],
    final_patch_size: Sequence[int],
    foreground_labels: Sequence[int],
    transform,
    num_workers: int = 6,
    oversample_foreground_percent: float = 0.33,
    seed: int = 42,
    prefetch_factor: int = 1,
) -> DataLoader:
    """Wire sampler + transforms into a DataLoader that yields whole batches.

    ``prefetch_factor=1`` is deliberate: a worker's working set while augmenting the
    rotation-inflated [243, 270, 205] patch is ~1 GB, so 12 workers x 2 prefetched
    batches would put ~25 GB of transient arrays in flight. One buffered batch per worker
    is already enough to hide the ~11 s augmentation behind the ~1 s GPU step.
    """
    sampler = PatchSampler3D(
        dataset, batch_size, sampled_patch_size, final_patch_size,
        foreground_labels, oversample_foreground_percent,
    )
    stream = _BatchStream(sampler, transform, seed)
    kwargs = {}
    if num_workers > 0:
        kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=True)
    # batch_size=None: the stream already emits assembled batches.
    return DataLoader(stream, batch_size=None, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), **kwargs)
