"""Read ordering metadata from Scan Context NPZ without reading its descriptors."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

ALLOWED_KEYS = ("frame_ids", "source_filenames", "scan_timestamps", "sequence")
FORBIDDEN_INPUT_KEYS = ("scan_contexts", "ring_keys")

@dataclass(frozen=True)
class Manifest:
    sequence: str
    frame_ids: np.ndarray
    source_filenames: np.ndarray
    scan_timestamps: np.ndarray

    def __len__(self): return len(self.frame_ids)


def validate_manifest_arrays(frame_ids, source_filenames, scan_timestamps, sequence, expected_count=None):
    frame_ids = np.asarray(frame_ids)
    filenames = np.asarray(source_filenames).astype(str)
    timestamps = np.asarray(scan_timestamps)
    seq = str(np.asarray(sequence).item())
    lengths = {len(frame_ids), len(filenames), len(timestamps)}
    if len(lengths) != 1:
        raise ValueError("frame_ids, source_filenames and scan_timestamps lengths differ")
    if expected_count is not None and len(frame_ids) != expected_count:
        raise RuntimeError(f"{seq}: expected {expected_count} manifest frames, found {len(frame_ids)}")
    if frame_ids.ndim != 1 or timestamps.ndim != 1 or filenames.ndim != 1:
        raise ValueError("Manifest fields must be one-dimensional")
    if not all(name.endswith(".bin") for name in filenames):
        raise ValueError("Every source_filename must already include the .bin suffix")
    return Manifest(seq, frame_ids.astype(np.int64), filenames, timestamps.astype(np.int64))


def load_manifest(manifest_root, sequence, expected_count=None):
    path = Path(manifest_root) / f"{sequence}.npz"
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in ALLOWED_KEYS if key not in archive]
        if missing: raise ValueError(f"{path}: missing manifest keys {missing}; available={archive.files}")
        # Deliberately access metadata only; scan_contexts/ring_keys are prohibited model inputs.
        manifest = validate_manifest_arrays(*(archive[key] for key in ALLOWED_KEYS), expected_count)
    if manifest.sequence != sequence:
        raise ValueError(f"Manifest sequence {manifest.sequence!r} does not match {sequence!r}")
    return manifest


def scan_path(raw_root, sequence, source_filename):
    return Path(raw_root) / f"{sequence}_vel" / "velodyne_sync" / str(source_filename)
