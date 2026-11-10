"""Two queue entries of the same code share one DEM pickle; the file must outlive the
first entry to finish (2026-10-13 b144 Phase B final: reference row + the LLM's copy)."""
from __future__ import annotations

from omniqec_agentic.evaluation.pool import release_shared_pickle


def test_shared_pickle_survives_until_the_last_owner_releases(tmp_path):
    path = tmp_path / "dem_bb_abc_noise_0.pkl"
    path.write_bytes(b"dem")
    owners = {str(path): {0, 1}}

    assert release_shared_pickle(str(path), 0, owners) is False
    assert path.exists(), "the second entry still needs the pickle"
    assert owners[str(path)] == {1}

    assert release_shared_pickle(str(path), 1, owners) is True
    assert not path.exists()


def test_unregistered_or_missing_paths_are_harmless(tmp_path):
    path = tmp_path / "dem_single.pkl"
    path.write_bytes(b"dem")
    # a path nobody registered (older call sites) is released unconditionally
    assert release_shared_pickle(str(path), 3, {}) is True
    assert not path.exists()
    # releasing again, or releasing None, never raises
    assert release_shared_pickle(str(path), 3, {}) is False
    assert release_shared_pickle(None, 0, {}) is False


def test_release_twice_by_the_same_owner_is_idempotent(tmp_path):
    path = tmp_path / "dem_twice.pkl"
    path.write_bytes(b"dem")
    owners = {str(path): {0, 1}}
    assert release_shared_pickle(str(path), 0, owners) is False
    assert release_shared_pickle(str(path), 0, owners) is False
    assert path.exists()
