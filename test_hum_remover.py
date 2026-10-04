import argparse
import base64
import struct
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pytest
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


def test_analyze_time_range_detects_only_selected_segment(tmp_path):
    p = str(tmp_path / "segments.wav")
    t = np.arange(SR * 4) / SR
    signal = np.zeros_like(t)
    signal[:SR * 2] = 0.01 * np.sin(2 * np.pi * 1800 * t[:SR * 2])
    signal[SR * 2:] = 0.01 * np.sin(2 * np.pi * 2400 * t[SR * 2:])
    sf.write(p, signal, SR, subtype="FLOAT")

    res = hr.analyze(p, start=2, end=4)
    assert len(res["peaks"]) == 1
    assert abs(res["peaks"][0]["freq_hz"] - 2400) < 3
    assert res["analysis_start_s"] == 2
    assert res["analysis_end_s"] == 4
    assert res["analysis_duration_s"] == 2
    assert res["duration_s"] == 4
    assert res["quiet_first_s"] >= 2


def test_hum_only_uses_all_non_silent_frames(tmp_path):
    p = str(tmp_path / "varying-hum.wav")
    t = np.arange(SR * 4) / SR
    amplitude = np.where(t < 2, 0.001, 0.02)
    sf.write(p, amplitude * np.sin(2 * np.pi * 1800 * t), SR, subtype="FLOAT")

    quiet = hr.analyze(p, start=0, end=4)
    hum_only = hr.analyze(p, start=0, end=4, hum_only=True)
    assert quiet["reference_mode"] == "quietest"
    assert hum_only["reference_mode"] == "hum-only"
    assert hum_only["quiet_frames"] > quiet["quiet_frames"]


def test_cli_analyze_defaults_to_all_channels_and_hum_only(tmp_path, capsys):
    p = str(tmp_path / "a.wav")
    make(p, seconds=2)
    assert hr.main(["analyze", p, "--hum-only"]) == 0
    output = capsys.readouterr().out
    for channel in ("mix", "left", "right"):
        assert f"analysed channel: {channel}" in output
        assert "hum-only reference:" in output


def test_analyze_channels_shares_file_reads(tmp_path, monkeypatch):
    p = str(tmp_path / "a.wav")
    make(p, seconds=2)
    original_read_chunks = hr._read_chunks
    read_count = 0

    def count_reads(*args, **kwargs):
        nonlocal read_count
        read_count += 1
        yield from original_read_chunks(*args, **kwargs)

    monkeypatch.setattr(hr, "_read_chunks", count_reads)
    results = hr.analyze_channels(p, progress=lambda _: None)
    assert set(results) == {"mix", "left", "right"}
    assert read_count == 3


def test_analyze_channels_reports_silent_channel_without_aborting(tmp_path):
    p = str(tmp_path / "one-silent-channel.wav")
    t = np.arange(SR * 2) / SR
    samples = np.column_stack((0.01 * np.sin(2 * np.pi * 1800 * t), np.zeros_like(t)))
    sf.write(p, samples, SR, subtype="FLOAT")
    results = hr.analyze_channels(p, progress=lambda _: None)
    assert results["left"]["peaks"]
    assert results["mix"]["peaks"]
    assert results["right"]["peaks"] == []
    assert results["right"]["quiet_frames"] == 0
    assert results["right"]["quiet_first_s"] is None
    assert results["right"]["quiet_last_s"] is None


def test_channel_results_are_deterministic_independent_of_batch(tmp_path, monkeypatch):
    p = str(tmp_path / "a.wav")
    make(p, seconds=10)
    monkeypatch.setattr(hr, "PROFILE_SAMPLE_FRAMES", 16)
    all_channels = hr.analyze_channels(p, progress=lambda _: None)
    right_only = hr.analyze(p, channel="right", progress=lambda _: None)
    assert all_channels["right"] == right_only


def test_all_channel_progress_tracks_three_analysis_phases(tmp_path):
    p = str(tmp_path / "a.wav")
    make(p, seconds=2)
    updates = []
    hr.analyze_channels(p, progress=updates.append)
    assert updates == sorted(updates)
    assert any(np.isclose(value, 1 / 3) for value in updates)
    assert any(np.isclose(value, 2 / 3) for value in updates)
    assert updates[-1] == 1


def test_analyze_progress_reaches_completion(tmp_path):
    p = str(tmp_path / "a.wav")
    make(p, seconds=2)
    updates = []
    hr.analyze(p, progress=updates.append)
    assert updates[0] == 0
    assert updates[-1] == 1
    assert updates == sorted(updates)


def test_analyze_log_receives_status_text(tmp_path):
    p = str(tmp_path / "a.wav")
    make(p, seconds=2)
    messages = []
    hr.analyze(p, log=messages.append, progress=lambda _: None)
    assert messages == ["Analyzing mix...", "Analysis complete: mix"]


def test_spectrogram_respects_time_frequency_bounds_and_resolution(tmp_path):
    p = str(tmp_path / "a.wav")
    out = str(tmp_path / "chart.svg")
    make(p, seconds=4)
    hr.write_spectrogram(p, out, fmin=1500, fmax=2000, start=1, end=3,
                         x_resolution=2, y_resolution=1)
    root = ET.parse(out).getroot()
    assert root.attrib["width"] == "130"
    assert root.attrib["height"] == "575"
    text = "".join(root.itertext())
    assert "1500-2000" in text
    assert "1.0" in text and "3.0" in text
    image = root.find(".//{http://www.w3.org/2000/svg}image")
    assert image is not None
    png = base64.b64decode(image.attrib["href"].split(",", 1)[1])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", png[16:24]) == (40, 500)


def test_spectrogram_accepts_falsy_progress_callback(tmp_path):
    class FalsyProgress:
        def __init__(self):
            self.values = []

        def __bool__(self):
            return False

        def __call__(self, value):
            self.values.append(value)

    p = str(tmp_path / "a.wav")
    out = str(tmp_path / "chart.svg")
    make(p, seconds=2)
    progress = FalsyProgress()
    hr.write_spectrogram(p, out, progress=progress)
    assert progress.values[0] == 0
    assert progress.values[-1] == 1


def test_spectrogram_forward_fills_columns_without_frames(tmp_path):
    p = str(tmp_path / "tone.wav")
    out = str(tmp_path / "chart.svg")
    t = np.arange(SR * 2) / SR
    sf.write(p, 0.1 * np.sin(2 * np.pi * 1800 * t), SR, subtype="FLOAT")
    hr.write_spectrogram(p, out, fmin=1500, fmax=2100, x_resolution=20)
    root = ET.parse(out).getroot()
    png = base64.b64decode(root.find(".//{http://www.w3.org/2000/svg}image").attrib["href"].split(",", 1)[1])
    width, height = struct.unpack(">II", png[16:24])
    offset, compressed = 8, bytearray()
    while offset < len(png):
        size = struct.unpack(">I", png[offset:offset + 4])[0]
        chunk_type = png[offset + 4:offset + 8]
        if chunk_type == b"IDAT":
            compressed.extend(png[offset + 8:offset + 8 + size])
        offset += size + 12
    rows = np.frombuffer(zlib.decompress(compressed), np.uint8).reshape(height, width + 1)
    pixels = rows[:, 1:]
    nframes = (SR * 2 - hr.N_FFT) // hr.HOP + 1
    frame_columns = ((np.arange(nframes) * hr.HOP + hr.N_FFT / 2) / SR * 200).astype(int)
    gap = next(right - 1 for left, right in zip(frame_columns, frame_columns[1:]) if right > left + 1)
    assert np.array_equal(pixels[:, gap], pixels[:, gap - 1])


@pytest.mark.parametrize("x_resolution,y_resolution", [
    (0, 1), (1, -1), (float("inf"), 1), (1e9, 1),
])
def test_spectrogram_rejects_invalid_resolution(tmp_path, x_resolution, y_resolution):
    p = str(tmp_path / "a.wav")
    out = str(tmp_path / "chart.svg")
    make(p, seconds=2)
    with pytest.raises(ValueError, match="resolution"):
        hr.write_spectrogram(p, out, x_resolution=x_resolution, y_resolution=y_resolution)
    assert not (tmp_path / "chart.svg").exists()


def test_spectrogram_escapes_svg_text(tmp_path):
    p = str(tmp_path / "a.wav")
    out = str(tmp_path / "chart.svg")
    make(p, seconds=2)
    hr.write_spectrogram(p, out, channel="mix & <test>")
    text = "".join(ET.parse(out).getroot().itertext())
    assert "mix & <test>" in text


def test_spectrogram_removes_partial_file_after_read_failure(tmp_path, monkeypatch):
    p = str(tmp_path / "a.wav")
    out = tmp_path / "chart.svg"
    make(p, seconds=2)

    def broken_spectra(*args, **kwargs):
        yield np.ones((1, 257), np.float32)
        raise RuntimeError("read failed")

    monkeypatch.setattr(hr, "_iter_band_mags", broken_spectra)
    with pytest.raises(RuntimeError, match="read failed"):
        hr.write_spectrogram(p, str(out))
    assert not out.exists()


def test_spectrogram_output_path_is_unique_and_protects_input(tmp_path):
    source = str(tmp_path / "a.wav")
    assert hr._spectrogram_output_path(source, "chart.svg", "left", True) == "chart-left.svg"
    assert hr._spectrogram_output_path(source, "", "mix", False).startswith(
        str(tmp_path / "a-spectrogram-mix-"))
    with pytest.raises(ValueError, match="input"):
        hr._spectrogram_output_path(source, source, "mix", False)


def test_cli_generates_spectrogram_for_all_channels(tmp_path, capsys):
    p = str(tmp_path / "a.wav")
    make(p, seconds=2)
    assert hr.main(["analyze", p, "--spectrogram"]) == 0
    output = capsys.readouterr().out
    for channel in ("mix", "left", "right"):
        assert f"analysed channel: {channel}" in output
        assert list(tmp_path.glob(f"a-spectrogram-{channel}-*.svg"))


def test_cli_validates_spectrogram_before_analysis(tmp_path, monkeypatch):
    p = str(tmp_path / "a.wav")
    make(p, seconds=2)

    def unexpected_analysis(*args, **kwargs):
        raise AssertionError("analysis must not run for invalid chart options")

    monkeypatch.setattr(hr, "analyze_channels", unexpected_analysis)
    with pytest.raises(SystemExit):
        hr.main(["analyze", p, "--spectrogram", "--x-resolution", "0"])


def test_cli_rejects_profile_spectrogram_path_collision_before_analysis(tmp_path, monkeypatch):
    p = str(tmp_path / "a.wav")
    chart = str(tmp_path / "chart.svg")
    make(p, seconds=2)

    def unexpected_analysis(*args, **kwargs):
        raise AssertionError("analysis must not run for colliding output paths")

    monkeypatch.setattr(hr, "analyze_channels", unexpected_analysis)
    with pytest.raises(SystemExit):
        hr.main(["analyze", p, "--channel", "mix", "--spectrogram", chart,
                 "--save-profile", chart])


def test_cli_continues_other_spectrograms_after_one_fails(tmp_path, monkeypatch, capsys):
    p = str(tmp_path / "a.wav")
    make(p, seconds=2)
    channels = []

    def fail_chart(path, output, channel, *args, **kwargs):
        channels.append(channel)
        raise RuntimeError("chart read failed")

    monkeypatch.setattr(hr, "write_spectrogram", fail_chart)
    assert hr.main(["analyze", p, "--spectrogram"]) == 1
    assert channels == ["mix", "left", "right"]
    assert capsys.readouterr().err.count("Spectrogram for") == 3


@pytest.mark.parametrize("value,expected", [
    ("35:10", 2110),
    ("1:02:03.5", 3723.5),
    ("12.5", 12.5),
])
def test_parse_time(value, expected):
    assert hr.parse_time(value) == expected


@pytest.mark.parametrize("value", ["-1", "1:60", "1:2:60", "1:2:3:4", "nope"])
def test_parse_time_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        hr.parse_time(value)


def test_analyze_rejects_invalid_time_range(tmp_path):
    p = str(tmp_path / "a.wav")
    make(p, seconds=2)
    with pytest.raises(ValueError, match="analysis range"):
        hr.analyze(p, start=1.5, end=1)
    with pytest.raises(ValueError, match="analysis range"):
        hr.analyze(p, start=0, end=3)


def test_remove_quiet_part_and_no_overwrite(tmp_path):
    p = str(tmp_path / "a.wav")
    make(p)
    before = open(p, "rb").read()
    res = hr.analyze(p)
    path, fh = hr.open_new_output(p)
    messages, progress = [], []
    with fh:
        hr.remove_hum(p, fh, res, mask_db=6.0, log=messages.append, progress=progress.append)
    assert messages == ["Processing audio...", "Audio processing complete."]
    assert progress[0] == 0
    assert progress[-1] == 1
    assert progress == sorted(progress)
    assert path != p and "-nohum-" in path
    assert open(p, "rb").read() == before
    x, _ = sf.read(p)
    y, _ = sf.read(path)
    assert x.shape == y.shape
    gap = slice(SR * 25, SR * 35)
    assert np.std(y[gap, 0]) < 0.6 * np.std(x[gap, 0])
    loud = slice(SR * 5, SR * 15)  # hum masked by music: nearly untouched
    assert np.std(y[loud, 0] - x[loud, 0]) < 0.0005
    always_path, always_fh = hr.open_new_output(p)
    with always_fh:
        hr.remove_hum(p, always_fh, res, mask_db=0, log=lambda *a, **k: None)
    always, _ = sf.read(always_path)
    assert np.std(always[loud, 0] - x[loud, 0]) > np.std(y[loud, 0] - x[loud, 0])


def test_channel_selector(tmp_path):
    p = str(tmp_path / "a.wav")
    make(p)
    out = {c: hr.analyze(p, channel=c) for c in ("left", "right", "mix")}
    assert out["left"]["channel"] == "left"
    for c in out:
        assert len(out[c]["peaks"]) == len(HUM)
    # right channel is the left scaled by 0.9 -> about 0.9 dB lower level
    d = out["left"]["peaks"][0]["level_db"] - out["right"]["peaks"][0]["level_db"]
    assert 0.5 < d < 1.5


@pytest.mark.parametrize("fmin,fmax", [
    (-1, 3000), (2000, 1000), (24000, 25000), (1000, 24001), (float("nan"), 2000),
])
def test_analyze_rejects_invalid_frequency_band(tmp_path, fmin, fmax):
    p = str(tmp_path / "a.wav")
    make(p, seconds=2)
    with pytest.raises(ValueError, match="frequency band"):
        hr.analyze(p, fmin=fmin, fmax=fmax)


def test_save_profile_cannot_overwrite_input(tmp_path):
    p = str(tmp_path / "a.wav")
    make(p, seconds=2)
    before = open(p, "rb").read()
    with pytest.raises(SystemExit):
        hr.main(["analyze", p, "--save-profile", p])
    assert open(p, "rb").read() == before


def test_profile_must_match_input_metadata(tmp_path):
    p = str(tmp_path / "a.wav")
    profile = str(tmp_path / "profile.json")
    make(p, seconds=2)
    with open(profile, "w") as f:
        f.write('{"samplerate": 44100, "n_fft": 8192, "peaks": []}')
    with pytest.raises(SystemExit):
        hr.main(["remove", p, "--profile", profile])


def test_reject_negative_reduction_settings():
    with pytest.raises(ValueError, match="max_reduction_db"):
        hr.remove_hum("unused.wav", None, {}, max_reduction_db=-1)
    with pytest.raises(ValueError, match="mask_db"):
        hr.remove_hum("unused.wav", None, {}, mask_db=-1)


def test_analysis_profile_sampling_is_bounded(tmp_path, monkeypatch):
    p = str(tmp_path / "a.wav")
    make(p, seconds=60)
    monkeypatch.setattr(hr, "PROFILE_SAMPLE_FRAMES", 32)
    res = hr.analyze(p)
    assert len(res["peaks"]) == len(HUM)
    assert res["quiet_frames"] > 32


def test_24bit_output_mirrors_input(tmp_path):
    p = str(tmp_path / "a.wav")
    x = np.random.default_rng(0).standard_normal((44100 * 5, 2)) * 0.05
    sf.write(p, x, 44100, subtype="PCM_24")
    path, fh = hr.open_new_output(p)
    res = {"peaks": [], "channel": "left"}
    with fh:
        hr.remove_hum(p, fh, res, log=lambda *a, **k: None)
    i, o = sf.info(p), sf.info(path)
    assert (o.samplerate, o.channels, o.subtype, o.frames) == (44100, 2, "PCM_24", i.frames)


def test_8bit_pcm_output_preserves_subtype(tmp_path):
    p = str(tmp_path / "a.wav")
    sf.write(p, np.zeros((SR, 2), np.float32), SR, subtype="PCM_U8")
    path, fh = hr.open_new_output(p)
    with fh:
        hr.remove_hum(p, fh, {"peaks": [], "channel": "mix"}, log=lambda *a, **k: None)
    assert sf.info(path).subtype == "PCM_U8"
