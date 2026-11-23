"""Tests: no session_id appears in more than one split; the phone channel is deterministic."""
import json, os, collections
import numpy as np
import pytest

CONV = "data/processed/conversations.json"


@pytest.mark.skipif(not os.path.exists(CONV), reason="run `python -m src.data.icfd` first")
def test_no_session_overlap_between_splits():
    convs = json.load(open(CONV, encoding="utf8"))
    where = collections.defaultdict(set)
    for c in convs:
        where[c["session_id"]].add(c["split"])
    bad = {s: v for s, v in where.items() if len(v) > 1}
    assert not bad, f"session_ids in several splits: {list(bad)[:5]}"


@pytest.mark.skipif(not os.path.exists("data/manifest/manifest.jsonl"), reason="manifest not built yet")
def test_manifest_no_session_overlap():
    where = collections.defaultdict(set)
    for line in open("data/manifest/manifest.jsonl", encoding="utf8"):
        r = json.loads(line); where[r["session_id"]].add(r["split"])
    assert all(len(v) == 1 for v in where.values())


def test_split_labels_are_balanced():
    if not os.path.exists(CONV):
        pytest.skip("sample not built")
    convs = json.load(open(CONV, encoding="utf8"))
    for sp in ("train", "val", "test"):
        l = [c["label"] for c in convs if c["split"] == sp]
        assert 0.4 < np.mean(l) < 0.6, (sp, np.mean(l))


@pytest.mark.parametrize("name", ["bw8k", "mulaw", "gsm", "opus8", "snr10", "loss10", "hard"])
def test_channel_deterministic(name):
    from src.channel.channel import apply_channel, TEST_GRID
    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal(16000 * 2)).astype(np.float32)
    a = apply_channel(x, TEST_GRID[name], 5); b = apply_channel(x, TEST_GRID[name], 5)
    assert np.array_equal(a, b)
    assert len(a) == len(x) // 2 and a.dtype == np.float32


def test_channel_seed_changes_noise_and_snr_is_right():
    from src.channel.channel import apply_channel, add_noise
    rng = np.random.default_rng(1)
    x = (0.1 * rng.standard_normal(32000)).astype(np.float32)
    cfg = {"codec": "none", "snr_db": 10, "noise": "white"}
    assert not np.array_equal(apply_channel(x, cfg, 1), apply_channel(x, cfg, 2))
    n = add_noise(x, 10, "white", np.random.default_rng(3)) - x
    snr = 10 * np.log10(np.mean(x ** 2) / np.mean(n ** 2))
    assert abs(snr - 10) < 0.5


def test_packet_loss_fraction():
    from src.channel.channel import drop_packets
    x = np.ones(160 * 100, np.float32)
    y = drop_packets(x, 10, np.random.default_rng(0))
    assert abs((y.reshape(100, 160).sum(1) == 0).mean() - 0.10) < 1e-6
