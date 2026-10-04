#!/usr/bin/env python3
"""Identify and remove a tonal hum (a few isolated peaks, default 1.5-3 kHz) from a WAV file.

Modes:
  analyze  Print statistics about the identified hum.
  remove   Write a new file with the hum removed ("<name>-nohum-<random>.wav"); the input is never touched.

Algorithm (all streaming, so a 1 h / 700 MB file needs little RAM):
  * STFT (Hann, 8192 pt, 75 % overlap) of the mono mix, restricted to the hum band.
  * "Quiet" frames = the frames with the lowest median in-band level (hum is most audible there).
  * Median spectrum of the quiet frames -> peaks that stick out of a running-median baseline = hum.
  * Removal: STFT magnitude subtraction at the hum bins, with the phase kept. For every frame the amount
    is scaled by how audible the hum is, i.e. by the level of the surrounding spectrum (masking):
    background below hum -> 100 % removed, background >= hum + mask_db -> 0 % removed, linear in dB between.
"""
import argparse
import json
import os
import secrets
import sys

import numpy as np
import scipy.fft
import scipy.ndimage
import scipy.signal
import soundfile as sf

N_FFT = 8192
HOP = N_FFT // 4
CHUNK = 1 << 20
PEAK_HALF_WIDTH = 3  # bins on each side of a peak that belong to the hum
NEIGHBOUR_BINS = 40  # bins on each side used to estimate the masking background
PROFILE_SAMPLE_FRAMES = 8192
LEVEL_HISTOGRAM_BINS = 4096
LEVEL_MIN_DB, LEVEL_MAX_DB = -180.0, 60.0
WINDOW = np.sqrt(scipy.signal.get_window("hann", N_FFT, fftbins=True))
WIN_NORM = float(np.sum(WINDOW**2) / HOP)  # sqrt-hann analysis+synthesis OLA gain compensation
WIN_SUM = float(np.sum(WINDOW)) / 2  # amplitude normalisation so a unit sine gives magnitude ~1


def _frames(data):
    """data: (ch, n). Returns windowed frames (nfr, ch, N_FFT) and number of samples consumed."""
    n = data.shape[1]
    nfr = (n - N_FFT) // HOP + 1 if n >= N_FFT else 0
    if nfr <= 0:
        return np.empty((0, data.shape[0], N_FFT), np.float32), 0
    view = np.lib.stride_tricks.sliding_window_view(data, N_FFT, axis=1)[:, ::HOP][:, :nfr]
    return (view.transpose(1, 0, 2) * WINDOW).astype(np.float32), nfr * HOP


def _read_chunks(path, start_frame=0, end_frame=None):
    with sf.SoundFile(path) as f:
        f.seek(start_frame)
        remaining = None if end_frame is None else end_frame - start_frame
        while True:
            size = CHUNK if remaining is None else min(CHUNK, remaining)
            if size <= 0:
                return
            blk = f.read(size, dtype="float32", always_2d=True)
            if len(blk) == 0:
                return
            if remaining is not None:
                remaining -= len(blk)
            yield blk.T


def select_channel(x, channel, axis=1):
    """Reduce the channel axis of x to one signal: 'left', 'right' or 'mix' (mean of all channels)."""
    n = x.shape[axis]
    if channel == "mix" or n == 1:
        return x.mean(axis=axis)
    return np.take(x, 0 if channel == "left" else 1, axis=axis)


def band_bins(sr, fmin, fmax):
    lo = int(np.floor(fmin * N_FFT / sr))
    hi = int(np.ceil(fmax * N_FFT / sr)) + 1
    return lo, min(hi, N_FFT // 2 + 1)


def _validate_band(sr, fmin, fmax):
    nyquist = sr / 2
    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmin < 0 or fmax <= fmin or fmax > nyquist:
        raise ValueError(f"frequency band must satisfy 0 <= fmin < fmax <= Nyquist ({nyquist:g} Hz)")


def _iter_band_mags(path, lo, hi, channel, start_frame=0, end_frame=None):
    info = sf.info(path)
    buf = np.zeros((info.channels, 0), np.float32)
    for blk in _read_chunks(path, start_frame, end_frame):
        buf = np.concatenate([buf, blk], axis=1)
        fr, used = _frames(buf)
        if used:
            mono = select_channel(fr, channel)
            yield (np.abs(scipy.fft.rfft(mono, axis=-1)[:, lo:hi]) / WIN_SUM).astype(np.float32)
            buf = buf[:, used:]


def _level_bins(level):
    level_db = 20 * np.log10(level)
    return np.clip(
        ((level_db - LEVEL_MIN_DB) * LEVEL_HISTOGRAM_BINS / (LEVEL_MAX_DB - LEVEL_MIN_DB)).astype(int),
        0, LEVEL_HISTOGRAM_BINS - 1)


def analyze(path, fmin=1500.0, fmax=3000.0, quiet_percent=10.0, min_prominence_db=6.0, max_peaks=12, channel="mix", log=print, start=0.0, end=None, hum_only=False):
    info = sf.info(path)
    sr = info.samplerate
    _validate_band(sr, fmin, fmax)
    if not 0 < quiet_percent <= 100:
        raise ValueError("quiet_percent must be greater than 0 and at most 100")
    duration = info.frames / sr
    if start is None:
        start = 0.0
    if end is None:
        end = duration
    if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start or end > duration:
        raise ValueError(f"analysis range must satisfy 0 <= start < end <= file duration ({duration:.3f} s)")
    start_frame, end_frame = round(start * sr), round(end * sr)
    if end_frame - start_frame < N_FFT:
        raise ValueError(f"analysis range must contain at least {N_FFT / sr:.3f} seconds of audio")
    lo, hi = band_bins(sr, fmin, fmax)
    level_histogram = np.zeros(LEVEL_HISTOGRAM_BINS, np.int64)
    valid_count = 0
    for mag in _iter_band_mags(path, lo, hi, channel, start_frame, end_frame):
        level = np.median(mag, axis=1)
        valid = level > 1e-9  # skip digital silence
        valid_count += int(valid.sum())
        if valid.any() and not hum_only:
            level_histogram += np.bincount(_level_bins(level[valid]), minlength=LEVEL_HISTOGRAM_BINS)
    if valid_count == 0:
        raise ValueError("file too short to analyze")
    if valid_count < 4:
        raise ValueError("file is (nearly) digital silence")
    if hum_only:
        threshold_bin = LEVEL_HISTOGRAM_BINS - 1
    else:
        n_quiet = max(3, int(valid_count * quiet_percent / 100))
        threshold_bin = np.searchsorted(np.cumsum(level_histogram), n_quiet)
    rng = np.random.default_rng(0)
    quiet_sample = []
    quiet_count = 0
    quiet_first = quiet_last = None
    frame_offset = 0
    for mag in _iter_band_mags(path, lo, hi, channel, start_frame, end_frame):
        level = np.median(mag, axis=1)
        valid = level > 1e-9
        valid_indices = np.flatnonzero(valid)
        quiet = valid_indices if hum_only else valid_indices[_level_bins(level[valid]) <= threshold_bin]
        for i in quiet:
            frame_time = (start_frame + (frame_offset + int(i)) * HOP) / sr
            if quiet_first is None:
                quiet_first = frame_time
            quiet_last = frame_time
            quiet_count += 1
            if len(quiet_sample) < PROFILE_SAMPLE_FRAMES:
                quiet_sample.append(mag[i].copy())
            else:
                replacement = int(rng.integers(quiet_count))
                if replacement < PROFILE_SAMPLE_FRAMES:
                    quiet_sample[replacement] = mag[i].copy()
        frame_offset += len(mag)
    if not quiet_sample:
        raise ValueError("file is (nearly) digital silence")
    profile = np.median(np.asarray(quiet_sample), axis=0)
    baseline = scipy.ndimage.median_filter(profile, size=NEIGHBOUR_BINS + 1, mode="nearest")
    ratio_db = 20 * np.log10((profile + 1e-12) / (baseline + 1e-12))
    peaks, _ = scipy.signal.find_peaks(ratio_db, height=min_prominence_db, distance=2 * PEAK_HALF_WIDTH)
    peaks = peaks[np.argsort(ratio_db[peaks])[::-1][:max_peaks]]
    peaks = np.sort(peaks)
    audible_counts = np.zeros(len(peaks), np.int64)
    total_frames = 0
    for mag in _iter_band_mags(path, lo, hi, channel, start_frame, end_frame):
        total_frames += len(mag)
        if len(peaks):
            audible_counts += np.sum(mag[:, peaks] > 2 * baseline[peaks], axis=0)
    result_peaks = []
    for j, p in enumerate(peaks):
        # parabolic interpolation on log magnitude for sub-bin frequency
        a, b, c = (np.log(profile[p + d] + 1e-12) if 0 <= p + d < len(profile) else np.log(profile[p] + 1e-12) for d in (-1, 0, 1))
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        present = float(audible_counts[j] / total_frames) if total_frames else 0.0
        result_peaks.append({
            "bin": int(p + lo),
            "freq_hz": float((p + lo + delta) * sr / N_FFT),
            "level_db": float(20 * np.log10(profile[p] + 1e-12)),
            "prominence_db": float(ratio_db[p]),
            "amplitude": float(profile[p]),
            "baseline": float(baseline[p]),
            "frames_audible_pct": 100 * present,
            "half_width_bins": PEAK_HALF_WIDTH,
        })
    res = {
        "file": os.path.basename(path), "samplerate": sr, "channels": info.channels, "channel": channel,
        "duration_s": duration, "analysis_duration_s": (end_frame - start_frame) / sr,
        "analysis_start_s": start_frame / sr, "analysis_end_s": end_frame / sr,
        "n_fft": N_FFT, "band_hz": [fmin, fmax],
        "reference_mode": "hum-only" if hum_only else "quietest",
        "quiet_frames": quiet_count, "quiet_total_s": float(quiet_count * HOP / sr),
        "quiet_first_s": float(quiet_first), "quiet_last_s": float(quiet_last),
        "peaks": result_peaks,
    }
    return res


def print_stats(res, out=print):
    out(f"File: {res['file']}  {res['samplerate']} Hz, {res['channels']} ch, {res['duration_s'] / 60:.1f} min"
        f"  [analysed channel: {res.get('channel', 'mix')}]")
    if "analysis_start_s" in res and (res["analysis_start_s"] != 0 or res["analysis_end_s"] != res["duration_s"]):
        out(f"Analyzed interval: {res['analysis_start_s']:.2f}-{res['analysis_end_s']:.2f} s")
    band = f"Band searched: {res['band_hz'][0]:.0f}-{res['band_hz'][1]:.0f} Hz"
    if res.get("reference_mode") == "hum-only":
        out(f"{band}, hum-only reference: {res['quiet_frames']} frames")
    else:
        out(f"{band}, quiet reference: {res['quiet_frames']} frames (~{res['quiet_total_s']:.0f} s, "
            f"between {res['quiet_first_s']:.0f}s and {res['quiet_last_s']:.0f}s)")
    if not res["peaks"]:
        out("No hum peaks found. Try lowering --min-prominence-db or widening --fmin/--fmax.")
        return
    out(f"Identified {len(res['peaks'])} hum peaks:")
    out("   #   freq (Hz)   level (dBFS)   prominence (dB)   audible in frames (%)")
    for i, p in enumerate(res["peaks"], 1):
        out(f"  {i:2d}  {p['freq_hz']:9.1f}   {p['level_db']:11.1f}   {p['prominence_db']:15.1f}   {p['frames_audible_pct']:19.1f}")


def remove_hum(src, dst_file, res, mask_db=12.0, max_reduction_db=30.0, log=print):
    """Stream src through the STFT hum canceller, writing to the open binary file object dst_file."""
    if mask_db < 0:
        raise ValueError("mask_db must be nonnegative")
    if max_reduction_db < 0:
        raise ValueError("max_reduction_db must be nonnegative")
    info = sf.info(src)
    sr, nch = info.samplerate, info.channels
    if info.subtype not in sf.available_subtypes("WAV"):
        raise ValueError(f"unsupported WAV subtype: {info.subtype}")
    peaks = res["peaks"]
    nb = N_FFT // 2 + 1
    amp = np.zeros(nb, np.float32)  # hum amplitude per bin (excess over baseline)
    regions = []
    for p in peaks:
        k, w = p["bin"], p["half_width_bins"]
        a, b = max(k - w, 0), min(k + w + 1, nb)
        amp[k] = max(amp[k], p["amplitude"] - p["baseline"])
        for j in range(a, b):  # approximate Hann main-lobe shape
            amp[j] = max(amp[j], (p["amplitude"] - p["baseline"]) * (0.5 + 0.5 * np.cos(np.pi * (j - k) / (w + 1))))
        n_lo, n_hi = max(k - NEIGHBOUR_BINS, 0), min(k + NEIGHBOUR_BINS + 1, nb)
        keep = np.ones(n_hi - n_lo, bool)
        keep[max(a - 1 - n_lo, 0):max(b + 1 - n_lo, 0)] = False
        regions.append((a, b, n_lo, n_hi, keep, p["amplitude"]))
    floor = 10 ** (-max_reduction_db / 20)

    def process(fr):
        spec = scipy.fft.rfft(fr, axis=-1)  # (nfr, ch, nb)
        mag = np.abs(spec)
        monomag = np.abs(select_channel(spec, res.get("channel", "mix"))) / WIN_SUM
        gain = np.ones_like(mag)
        for a, b, n_lo, n_hi, keep, hum_ref in regions:
            bg = np.median(monomag[:, n_lo:n_hi][:, keep], axis=1)
            d = 20 * np.log10((bg + 1e-12) / hum_ref)
            s = np.ones_like(d) if mask_db == 0 else np.clip(1 - d / mask_db, 0, 1)  # (nfr,)
            target = np.maximum(mag[:, :, a:b] - s[:, None, None] * amp[a:b] * WIN_SUM, floor * mag[:, :, a:b])
            gain[:, :, a:b] = target / np.maximum(mag[:, :, a:b], 1e-20)
        out = scipy.fft.irfft(spec * gain, n=N_FFT, axis=-1) * WINDOW
        return out.astype(np.float32)

    pad = N_FFT - HOP
    total = info.frames
    buf = np.zeros((nch, pad), np.float32)
    carry = np.zeros((nch, pad), np.float32)
    skip, remaining = pad, total
    subtype = info.subtype
    done = 0

    def run(buf):
        nonlocal carry, skip, remaining
        fr, used = _frames(buf)
        if not used:
            return buf, None
        y = process(fr)
        nfr = len(y)
        acc = np.zeros((nch, (nfr - 1) * HOP + N_FFT), np.float32)
        for i in range(nfr):
            acc[:, i * HOP:i * HOP + N_FFT] += y[i]
        acc[:, :pad] += carry
        emit = acc[:, :nfr * HOP] / WIN_NORM
        carry = acc[:, nfr * HOP:]
        if skip:
            cut = min(skip, emit.shape[1])
            emit, skip = emit[:, cut:], skip - cut
        emit = emit[:, :remaining]
        remaining -= emit.shape[1]
        return buf[:, used:], emit

    with sf.SoundFile(dst_file, "w", samplerate=sr, channels=nch, format="WAV", subtype=subtype) as out:
        def write(e):
            nonlocal done
            if e is not None and e.shape[1]:
                out.write(e.T if subtype in ("FLOAT", "DOUBLE") else np.clip(e.T, -1.0, 1.0))
                done += e.shape[1]

        for blk in _read_chunks(src):
            buf = np.concatenate([buf, blk], axis=1)
            buf, e = run(buf)
            write(e)
            log(f"\r  {100 * done / max(total, 1):5.1f} %", end="", flush=True)
        buf = np.concatenate([buf, np.zeros((nch, N_FFT + HOP), np.float32)], axis=1)
        buf, e = run(buf)
        write(e)
        log("\r  100.0 %")


def make_output_path(src):
    stem, ext = os.path.splitext(src)
    return f"{stem}-nohum-{secrets.token_hex(4)}{ext or '.wav'}"


def open_new_output(src):
    """Open an unused output file exclusively; never overwrites an existing file (incl. the input)."""
    while True:
        path = make_output_path(src)
        try:
            return path, open(path, "xb")
        except FileExistsError:
            continue


def validate_profile(res, input_path):
    info = sf.info(input_path)
    if res.get("samplerate") != info.samplerate or res.get("n_fft") != N_FFT:
        raise ValueError("profile sample rate or FFT size does not match the input")


def parse_time(value):
    """Parse seconds, MM:SS, or HH:MM:SS into seconds."""
    try:
        parts = value.split(":")
        if len(parts) == 1:
            seconds = float(parts[0])
        elif len(parts) == 2:
            minutes = int(parts[0])
            seconds_part = float(parts[1])
            if minutes < 0 or not 0 <= seconds_part < 60:
                raise ValueError
            seconds = minutes * 60 + seconds_part
        elif len(parts) == 3:
            hours, minutes = int(parts[0]), int(parts[1])
            seconds_part = float(parts[2])
            if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds_part < 60:
                raise ValueError
            seconds = hours * 3600 + minutes * 60 + seconds_part
        else:
            raise ValueError
    except ValueError as e:
        raise argparse.ArgumentTypeError("time must be seconds, MM:SS, or HH:MM:SS") from e
    if not np.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("time must be a finite nonnegative value")
    return seconds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["analyze", "remove"])
    ap.add_argument("input", help="input WAV file (never modified)")
    ap.add_argument("--fmin", type=float, default=1500.0, help="lower edge of hum search band in Hz")
    ap.add_argument("--fmax", type=float, default=3000.0, help="upper edge of hum search band in Hz")
    ap.add_argument("--quiet-percent", type=float, default=10.0, help="%% of quietest frames used as hum reference")
    ap.add_argument("--min-prominence-db", type=float, default=6.0, help="min. peak height over local baseline")
    ap.add_argument("--channel", choices=["left", "right", "mix"],
                    help="channel to analyze; analyze defaults to all channels, remove defaults to mix")
    ap.add_argument("--max-peaks", type=int, default=12)
    ap.add_argument("--start", type=parse_time, help="analyze from this time (seconds, MM:SS, or HH:MM:SS)")
    ap.add_argument("--end", type=parse_time, help="analyze until this time (seconds, MM:SS, or HH:MM:SS)")
    ap.add_argument("--hum-only", action="store_true",
                    help="use all non-silent frames in the selected interval as the hum reference")
    ap.add_argument("--save-profile", help="analyze: write identified hum profile to this JSON file")
    ap.add_argument("--profile", help="remove: use this JSON profile instead of analysing the input again")
    ap.add_argument("--mask-db", type=float, default=12.0,
                    help="remove: if the surrounding audio is this many dB above the hum, the hum is considered "
                         "masked and left alone; at/below hum level it is removed fully; linear in between "
                         "(0 = always remove fully)")
    ap.add_argument("--max-reduction-db", type=float, default=30.0, help="remove: max attenuation of the hum")
    a = ap.parse_args(argv)

    if not os.path.isfile(a.input):
        ap.error(f"input not found: {a.input}")
    if a.save_profile and (
        os.path.realpath(a.save_profile) == os.path.realpath(a.input)
        or (os.path.exists(a.save_profile) and os.path.samefile(a.save_profile, a.input))
    ):
        ap.error("--save-profile must not resolve to the input file")
    if a.mask_db < 0:
        ap.error("--mask-db must be nonnegative")
    if a.max_reduction_db < 0:
        ap.error("--max-reduction-db must be nonnegative")
    if a.mode == "remove" and a.profile:
        with open(a.profile) as f:
            res = json.load(f)
        try:
            validate_profile(res, a.input)
        except ValueError as e:
            ap.error(str(e))
        print_stats(res)
    else:
        print("Analyzing...")
        channels = ([a.channel] if a.channel else
                    ["mix", "left", "right"] if a.mode == "analyze" else ["mix"])
        results = {}
        try:
            for channel in channels:
                result = analyze(a.input, a.fmin, a.fmax, a.quiet_percent, a.min_prominence_db,
                                 a.max_peaks, channel, start=a.start or 0.0, end=a.end,
                                 hum_only=a.hum_only)
                results[channel] = result
                print_stats(result)
        except ValueError as e:
            ap.error(str(e))
        res = results.get("mix", next(iter(results.values())))
    if a.save_profile:
        with open(a.save_profile, "w") as f:
            json.dump(res, f, indent=1)
    if a.mode == "analyze":
        return 0
    if not res["peaks"]:
        print("Nothing to remove.")
        return 1
    path, fh = open_new_output(a.input)
    try:
        with fh:
            print(f"Writing {path}")
            remove_hum(a.input, fh, res, a.mask_db, a.max_reduction_db, log=lambda *x, **k: print(*x, **k))
    except BaseException:
        if os.path.exists(path):
            os.remove(path)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
