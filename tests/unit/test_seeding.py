"""Determinism.

The point of `SeedReport` is that a run whose determinism was silently downgraded must
not masquerade as a deterministic one. `fully_deterministic` is the flag that goes into
the run record, and it must be False whenever anything was skipped.
"""

from __future__ import annotations

import os
import random

import numpy as np

from rdl.seeding import DEFAULT_SEED, SeedReport, seed_worker, set_all_seeds


def test_seeding_is_reproducible_across_libraries():
    set_all_seeds(123)
    a = (random.random(), float(np.random.rand()))
    set_all_seeds(123)
    b = (random.random(), float(np.random.rand()))
    assert a == b


def test_different_seeds_give_different_draws():
    set_all_seeds(1)
    a = random.random()
    set_all_seeds(2)
    b = random.random()
    assert a != b


def test_report_records_which_rngs_were_seeded():
    report = set_all_seeds(DEFAULT_SEED)
    assert report.seed == DEFAULT_SEED
    assert report.python_seeded
    assert report.numpy_seeded


def test_cublas_workspace_is_set_before_any_cuda_context():
    """Must be in the environment, or use_deterministic_algorithms raises at the first matmul."""
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    report = set_all_seeds(7, deterministic=True)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert report.cublas_workspace_config == ":4096:8"


def test_non_deterministic_mode_skips_the_cublas_setting():
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    report = set_all_seeds(7, deterministic=False)
    assert report.cublas_workspace_config is None
    assert not report.deterministic_algorithms


def test_pythonhashseed_is_exported_for_subprocesses():
    """The open-unlearning bridge spawns a subprocess; it must inherit the seed."""
    set_all_seeds(99)
    assert "PYTHONHASHSEED" in os.environ


def test_fully_deterministic_is_false_when_anything_was_skipped():
    degraded = SeedReport(
        seed=42,
        python_seeded=True,
        numpy_seeded=True,
        torch_seeded=True,
        deterministic_algorithms=True,
        warnings=["cudnn determinism unavailable"],
    )
    assert not degraded.fully_deterministic

    clean = SeedReport(
        seed=42,
        python_seeded=True,
        numpy_seeded=True,
        torch_seeded=True,
        deterministic_algorithms=True,
    )
    assert clean.fully_deterministic


def test_report_is_json_safe():
    import json

    d = set_all_seeds(5).to_dict()
    json.dumps(d)
    assert "fully_deterministic" in d


def test_seed_worker_derives_distinct_seeds():
    seed_worker(0, base_seed=100)
    a = random.random()
    seed_worker(1, base_seed=100)
    b = random.random()
    assert a != b

    seed_worker(0, base_seed=100)
    assert random.random() == a
