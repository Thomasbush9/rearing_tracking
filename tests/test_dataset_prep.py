"""Tests for batched dataset preparation and column selection."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from behavex.models.train_transformer import (
    prepare_masked_transformer_data,
    prepare_and_save_dataset_chunked,
    load_preprocessed_dataset,
)


def _synthetic_df(n_rows=200, cols=None):
    if cols is None:
        cols = ["timestamp", "height", "height_scaled", "dist_head", "x", "y"]
    n = len(cols)
    data = np.random.randn(n_rows, n).astype(np.float32)
    return pd.DataFrame(data, columns=cols)


def test_column_selection_keep_cols_only():
    df = _synthetic_df(cols=["a", "b", "c", "d"])
    w, names, _ = prepare_masked_transformer_data(
        df, window_size=10, stride=5, keep_cols=["a", "c"]
    )
    assert names == ["a", "c"]
    assert w.shape[-1] == 2


def test_column_selection_drop_cols_only():
    df = _synthetic_df(cols=["a", "b", "c", "d"])
    w, names, _ = prepare_masked_transformer_data(
        df, window_size=10, stride=5, drop_cols=["b", "d"]
    )
    assert set(names) == {"a", "c"}
    assert w.shape[-1] == 2


def test_column_selection_keep_overrides_drop():
    df = _synthetic_df(cols=["a", "b", "c", "d"])
    w, names, _ = prepare_masked_transformer_data(
        df, window_size=10, stride=5, keep_cols=["b"], drop_cols=["a", "c"]
    )
    assert names == ["b"]


def test_column_selection_all_columns():
    df = _synthetic_df(cols=["a", "b", "c"])
    w, names, _ = prepare_masked_transformer_data(df, window_size=10, stride=5)
    assert len(names) == 3
    assert w.shape[-1] == 3


def test_chunked_roundtrip_and_load():
    """Run chunked build on a tiny dir and load .h5; check feature_names and shapes."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not installed")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Two minimal session-like files (pattern m*_s*_cricket|object.xlsx)
        for name in ["m001_s001_cricket.xlsx", "m001_s002_object.xlsx"]:
            df = pd.DataFrame(
                {
                    "timestamp": np.arange(200, dtype=float),
                    "height": np.random.rand(200),
                    "height_scaled": np.random.rand(200),
                    "dist_head": np.full(200, np.nan),
                }
            )
            df.loc[50:, "dist_head"] = np.random.rand(150)
            df.to_excel(root / name, index=False)
        out = root / "out.h5"
        prepare_and_save_dataset_chunked(
            str(root),
            str(out),
            val_ratio=0.2,
            test_ratio=0.2,
            window_size=32,
            stride=8,
            use_dist_head_event=True,
            trim_before_dist_head=False,
        )
        assert out.exists()
        train_w, val_w, test_w, feature_names, te, ve, ee = load_preprocessed_dataset(
            out
        )
        assert feature_names is not None
        assert len(feature_names) == train_w.shape[-1]
        assert train_w.ndim == 3 and train_w.shape[1] == 32
        assert val_w.shape[1] == 32 and test_w.shape[1] == 32
