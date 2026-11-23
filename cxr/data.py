"""Manifests, preprocessed image caches and PyTorch datasets.

A manifest is a CSV with at least: image_id, image_path, label, source, split, patient_id.
`label` is 1 = pneumonia / lung opacity, 0 = negative, empty = not part of the binary task.
image_path is relative to the project root.

Each manifest gets a uint8 cache (N x S x S) of preprocessed images, so real and synthetic
images go through identical decoding/resizing and training never re-decodes JPEG/DICOM.
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from . import PROJECT_ROOT
from .imaging import load_gray, to_square

CACHE_DIR = PROJECT_ROOT / "artifacts" / "cache"


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(resolve(path), dtype={"image_id": str, "patient_id": str})


_OPS = {"lt": "__lt__", "le": "__le__", "gt": "__gt__", "ge": "__ge__", "ne": "__ne__"}


def select(df: pd.DataFrame, where: dict | None) -> pd.DataFrame:
    """Filter rows by {column: value | [values] | {"lt"|"le"|"gt"|"ge"|"ne": value}}."""
    for col, val in (where or {}).items():
        if isinstance(val, dict):
            for op, v in val.items():
                df = df[getattr(df[col], _OPS[op])(v)]
        else:
            df = df[df[col].isin(val if isinstance(val, list) else [val])]
    return df


def _prep_one(args):
    path, size, mode = args
    return to_square(load_gray(resolve(path)), size, mode)


def ensure_cache(manifest_path: str | Path, size: int, mode: str, workers: int = 8):
    """Return (memmap array, {image_id: row}) for a manifest, building the cache if stale."""
    df = read_manifest(manifest_path)
    stem = Path(manifest_path).stem
    arr_path = CACHE_DIR / f"{stem}_{size}_{mode}.npy"
    ids_path = arr_path.with_suffix(".ids")
    ids = df["image_id"].tolist()

    fresh = arr_path.exists() and ids_path.exists() and ids_path.read_text().split("\n") == ids
    if not fresh:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = arr_path.with_suffix(".tmp.npy")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(df), size, size))
        jobs = [(p, size, mode) for p in df["image_path"]]
        with Pool(workers) as pool:
            it = pool.imap(_prep_one, jobs, chunksize=16)
            for i, img in enumerate(tqdm(it, total=len(jobs), desc=f"cache {stem}", mininterval=10)):
                out[i] = img
        out.flush()
        del out
        tmp.rename(arr_path)
        ids_path.write_text("\n".join(ids))

    arr = np.load(arr_path, mmap_mode="r")
    return arr, {iid: i for i, iid in enumerate(ids)}


class TensorSet:
    """A manifest selection materialised as tensors (uint8 images [N,1,S,S], float labels [N]).

    Training/validation sets are small (<=~5k images at 224px = ~250 MB as uint8), so they are kept
    on the GPU and batches are formed by indexing: no DataLoader/worker overhead, no host RAM copies.
    Large evaluation sets stay on the memmap and are streamed with `batches()`.
    """

    def __init__(self, images, labels, df, on_device: bool):
        self.images, self.labels, self.df, self.on_device = images, labels, df, on_device

    def __len__(self):
        return len(self.df)

    def batches(self, batch_size: int, device):
        for i in range(0, len(self), batch_size):
            x = self.images[i:i + batch_size]
            if not self.on_device:
                x = torch.from_numpy(np.ascontiguousarray(x))
            yield x.to(device, non_blocking=True).float().div_(255)


def load_set(manifest_path, where, size, mode, device=None, require_label=True, label_col="label"):
    """Materialise rows of `manifest_path` matching `where`.

    device=None keeps images on the host memmap (streamed later); otherwise they are copied to `device`.
    """
    images, row_of = ensure_cache(manifest_path, size, mode)
    df = select(read_manifest(manifest_path), where)
    if require_label:
        df = df[df[label_col].notna()]
    df = df.reset_index(drop=True)
    rows = np.array([row_of[i] for i in df["image_id"]], dtype=np.int64)
    labels = torch.tensor(df[label_col].fillna(-1).to_numpy(), dtype=torch.float32)
    if device is None:
        # sorted contiguous access for the memmap; keep df aligned with the row order
        order = np.argsort(rows, kind="stable")
        df, rows, labels = df.iloc[order].reset_index(drop=True), rows[order], labels[order]
        return TensorSet(_RowView(images, rows), labels, df, on_device=False)
    x = torch.from_numpy(np.ascontiguousarray(images[np.sort(rows)]))
    x = x[torch.from_numpy(np.argsort(np.argsort(rows)))]  # back to df order
    return TensorSet(x.unsqueeze(1).to(device), labels.to(device), df, on_device=True)


class _RowView:
    """Lazy row-subset view over a memmap; slicing returns [n,1,S,S] uint8 numpy."""

    def __init__(self, arr, rows):
        self.arr, self.rows = arr, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, sl):
        return self.arr[self.rows[sl]][:, None]
