import numpy as np
import soundfile as sf

import hum_remover as hr

SR = 48000
HUM = [1610.0, 1875.5, 2133.0, 2390.0, 2711.0, 2905.0, 2990.0 - 40]


def make(path, loud_noise_from=None, seconds=60, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(SR * seconds) / SR
    sig = 0.002 * rng.standard_normal(t.size)
    env = np.ones_like(t)
    music = np.zeros_like(t)
    # loud "music" in 0-20 s and 40-60 s, quiet gap in between
    loud = (t < 20) | (t > 40)
    music[loud] = 0.3 * rng.standard_normal(loud.sum())
    hum = sum(0.0015 * np.sin(2 * np.pi * f * t + i) for i, f in enumerate(HUM))
    x = sig + music + hum
    sf.write(path, np.stack([x, x * 0.9], axis=1).astype(np.float32), SR, subtype="FLOAT")
    return hum


def test_analyze_finds_hum(tmp_path):
    p = str(tmp_path / "a.wav")
    make(p)
    res = hr.analyze(p)
    found = sorted(q["freq_hz"] for q in res["peaks"])
    assert len(found) == len(HUM)
    for f, g in zip(sorted(HUM), found):
        assert abs(f - g) < 6


def test_remove_quiet_part_and_no_overwrite(tmp_path):
    p = str(tmp_path / "a.wav")
    make(p)
    before = open(p, "rb").read()
    path, fh = hr.open_new_output(p)
    with fh:
        hr.remove_hum(p, fh, hr.analyze(p), mask_db=6.0, log=lambda *a, **k: None)
    assert path != p and "-nohum-" in path
    assert open(p, "rb").read() == before
    x, _ = sf.read(p)
    y, _ = sf.read(path)
    assert x.shape == y.shape
    gap = slice(SR * 25, SR * 35)
    assert np.std(y[gap, 0]) < 0.6 * np.std(x[gap, 0])
    loud = slice(SR * 5, SR * 15)  # hum masked by music: nearly untouched
    assert np.std(y[loud, 0] - x[loud, 0]) < 0.0005
