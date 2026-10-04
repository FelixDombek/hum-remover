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
import base64
import json
import os
import secrets
import struct
import sys
from contextlib import contextmanager
from tempfile import SpooledTemporaryFile
from xml.sax.saxutils import escape
import zlib

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
PROGRESS_WIDTH = 28
SPECTROGRAM_PALETTE_SIZE = 64
SPECTROGRAM_DB_FLOOR = -100.0
SPECTROGRAM_DB_CEILING = 0.0
SPECTROGRAM_AMP_FLOOR = 1e-12
SPECTROGRAM_MAX_PIXELS = 64_000_000
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
    for mags in _iter_band_mags_multi(path, lo, hi, [channel], start_frame, end_frame):
        yield mags[:, 0, :]


def _iter_band_mags_multi(path, lo, hi, channels, start_frame=0, end_frame=None):
    info = sf.info(path)
    buf = np.zeros((info.channels, 0), np.float32)
    for blk in _read_chunks(path, start_frame, end_frame):
        buf = np.concatenate([buf, blk], axis=1)
        fr, used = _frames(buf)
        if used:
            mono = np.stack([select_channel(fr, channel) for channel in channels], axis=1)
            yield (np.abs(scipy.fft.rfft(mono, axis=-1)[..., lo:hi]) / WIN_SUM).astype(np.float32)
            buf = buf[:, used:]


def _progress_line(label, fraction):
    fraction = float(np.clip(fraction, 0.0, 1.0))
    filled = int(PROGRESS_WIDTH * fraction)
    return f"{label} [{'#' * filled}{'-' * (PROGRESS_WIDTH - filled)}] {fraction * 100:5.1f}%"


def _show_progress(label, fraction, stream=sys.stderr):
    stream.write("\r" + _progress_line(label, fraction))
    if fraction >= 1:
        stream.write("\n")
    stream.flush()


def _progress_reporter(callback):
    last = -1.0

    def report(fraction):
        nonlocal last
        fraction = float(np.clip(fraction, 0.0, 1.0))
        if fraction > last:
            callback(fraction)
            last = fraction

    return report


def _spectrogram_palette():
    anchors = np.asarray([[8, 20, 38], [18, 91, 145], [20, 184, 166], [255, 230, 109]], np.float32)
    palette = []
    for index in range(SPECTROGRAM_PALETTE_SIZE):
        position = index / (SPECTROGRAM_PALETTE_SIZE - 1) * (len(anchors) - 1)
        lower = int(position)
        upper = min(lower + 1, len(anchors) - 1)
        rgb = np.rint(anchors[lower] + (anchors[upper] - anchors[lower]) * (position - lower)).astype(int)
        palette.append("#" + "".join(f"{value:02x}" for value in rgb))
    return palette


@contextmanager
def _exclusive_text_output(path):
    output = open(path, "x", encoding="utf-8")
    failed = False
    try:
        yield output
    except BaseException:
        failed = True
        raise
    finally:
        try:
            output.close()
        except BaseException:
            failed = True
            raise
        finally:
            if failed:
                try:
                    os.remove(path)
                except OSError:
                    pass


def _write_png_chunk(output, chunk_type, data):
    output.write(struct.pack(">I", len(data)))
    output.write(chunk_type)
    output.write(data)
    crc = zlib.crc32(data, zlib.crc32(chunk_type))
    output.write(struct.pack(">I", crc & 0xffffffff))


def _write_embedded_png(svg, pixels, palette):
    height, width = pixels.shape
    png = SpooledTemporaryFile(max_size=1 << 20)
    compressed = SpooledTemporaryFile(max_size=1 << 20)
    try:
        compressor = zlib.compressobj()
        for row in pixels:
            compressed.write(compressor.compress(b"\0" + row.tobytes()))
        compressed.write(compressor.flush())
        compressed_length = compressed.tell()
        png.write(b"\x89PNG\r\n\x1a\n")
        _write_png_chunk(png, b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0))
        color_table = bytes(int(color[index:index + 2], 16)
                            for color in palette for index in (1, 3, 5))
        _write_png_chunk(png, b"PLTE", color_table)
        png.write(struct.pack(">I", compressed_length))
        png.write(b"IDAT")
        crc = zlib.crc32(b"IDAT")
        compressed.seek(0)
        while data := compressed.read(1 << 16):
            png.write(data)
            crc = zlib.crc32(data, crc)
        png.write(struct.pack(">I", crc & 0xffffffff))
        _write_png_chunk(png, b"IEND", b"")
        png.seek(0)
        svg.write(f'<image width="{width}" height="{height}" '
                  f'href="data:image/png;base64,')
        remainder = b""
        while data := png.read(1 << 16):
            data = remainder + data
            encoded_size = len(data) - len(data) % 3
            svg.write(base64.b64encode(data[:encoded_size]).decode("ascii"))
            remainder = data[encoded_size:]
        if remainder:
            svg.write(base64.b64encode(remainder).decode("ascii"))
        svg.write('"/>\n')
    finally:
        compressed.close()
        png.close()


def _level_bins(level):
    level_db = 20 * np.log10(level)
    return np.clip(
        ((level_db - LEVEL_MIN_DB) * LEVEL_HISTOGRAM_BINS / (LEVEL_MAX_DB - LEVEL_MIN_DB)).astype(int),
        0, LEVEL_HISTOGRAM_BINS - 1)


def analyze(path, fmin=1500.0, fmax=3000.0, quiet_percent=10.0, min_prominence_db=6.0, max_peaks=12, channel="mix", log=print, start=0.0, end=None, hum_only=False, progress=None):
    """Analyze one channel; `log` receives status text and `progress` receives fractions."""
    log(f"Analyzing {channel}...")
    result = analyze_channels(
        path, fmin, fmax, quiet_percent, min_prominence_db, max_peaks, [channel],
        start=start, end=end, hum_only=hum_only, progress=progress)[channel]
    log(f"Analysis complete: {channel}")
    return result


def analyze_channels(path, fmin=1500.0, fmax=3000.0, quiet_percent=10.0,
                     min_prominence_db=6.0, max_peaks=12,
                     channels=("mix", "left", "right"), start=0.0, end=None,
                     hum_only=False, progress=None):
    """Analyze channels together in three streaming passes and return results keyed by channel.

    Progress fractions advance through level histograms, quiet-frame profiles, and audibility counts.
    """
    channels = tuple(dict.fromkeys(channels))
    if not channels or any(channel not in ("left", "right", "mix") for channel in channels):
        raise ValueError("channels must be selected from left, right, and mix")
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
    expected_frames = max(1, (end_frame - start_frame - N_FFT) // HOP + 1)
    channel_label = "/".join(channels)
    report_progress = _progress_reporter(
        progress if progress is not None else
        lambda fraction: _show_progress(f"Analyze {channel_label}", fraction))
    level_histograms = [np.zeros(LEVEL_HISTOGRAM_BINS, np.int64) for _ in channels]
    valid_counts = np.zeros(len(channels), np.int64)
    frame_count = 0
    report_progress(0.0)
    for mags in _iter_band_mags_multi(path, lo, hi, channels, start_frame, end_frame):
        frame_count += len(mags)
        report_progress(min(frame_count / (expected_frames * 3), 1 / 3))
        levels = np.median(mags, axis=2)
        for index in range(len(channels)):
            valid = levels[:, index] > 1e-9
            valid_counts[index] += int(valid.sum())
            if valid.any() and not hum_only:
                level_histograms[index] += np.bincount(
                    _level_bins(levels[valid, index]), minlength=LEVEL_HISTOGRAM_BINS)
    report_progress(1 / 3)
    active_channels = valid_counts >= 4
    if not np.any(active_channels):
        silent = ", ".join(channels)
        raise ValueError(f"file is (nearly) digital silence on selected channels: {silent}")
    threshold_bins = []
    for index, active in enumerate(active_channels):
        if not active:
            threshold_bins.append(0)
            continue
        if hum_only:
            threshold_bins.append(LEVEL_HISTOGRAM_BINS - 1)
        else:
            n_quiet = max(3, int(valid_counts[index] * quiet_percent / 100))
            threshold_bins.append(np.searchsorted(np.cumsum(level_histograms[index]), n_quiet))

    channel_seeds = {"mix": 0, "left": 1, "right": 2}
    rngs = [np.random.default_rng(channel_seeds[channel]) for channel in channels]
    quiet_samples = [[] for _ in channels]
    quiet_counts = np.zeros(len(channels), np.int64)
    quiet_first = [None for _ in channels]
    quiet_last = [None for _ in channels]
    frame_offset = 0
    for mags in _iter_band_mags_multi(path, lo, hi, channels, start_frame, end_frame):
        report_progress(min(1 / 3 + (frame_offset + len(mags)) / (expected_frames * 3), 2 / 3))
        levels = np.median(mags, axis=2)
        for index in range(len(channels)):
            if not active_channels[index]:
                continue
            valid_indices = np.flatnonzero(levels[:, index] > 1e-9)
            quiet = (valid_indices if hum_only else
                     valid_indices[_level_bins(levels[valid_indices, index]) <= threshold_bins[index]])
            for i in quiet:
                frame_time = (start_frame + (frame_offset + int(i)) * HOP) / sr
                if quiet_first[index] is None:
                    quiet_first[index] = frame_time
                quiet_last[index] = frame_time
                quiet_counts[index] += 1
                sample = quiet_samples[index]
                if len(sample) < PROFILE_SAMPLE_FRAMES:
                    sample.append(mags[i, index].copy())
                else:
                    replacement = int(rngs[index].integers(quiet_counts[index]))
                    if replacement < PROFILE_SAMPLE_FRAMES:
                        sample[replacement] = mags[i, index].copy()
        frame_offset += len(mags)
    report_progress(2 / 3)
    profiles, baselines, ratios, peaks_by_channel = [], [], [], []
    for index, sample in enumerate(quiet_samples):
        if not active_channels[index]:
            profile = np.zeros(hi - lo, np.float32)
            baseline = profile.copy()
            ratio_db = profile.copy()
            peaks = np.empty(0, np.int64)
            profiles.append(profile)
            baselines.append(baseline)
            ratios.append(ratio_db)
            peaks_by_channel.append(peaks)
            continue
        profile = np.median(np.asarray(sample), axis=0)
        baseline = scipy.ndimage.median_filter(profile, size=NEIGHBOUR_BINS + 1, mode="nearest")
        ratio_db = 20 * np.log10((profile + 1e-12) / (baseline + 1e-12))
        peaks, _ = scipy.signal.find_peaks(
            ratio_db, height=min_prominence_db, distance=2 * PEAK_HALF_WIDTH)
        peaks = peaks[np.argsort(ratio_db[peaks])[::-1][:max_peaks]]
        profiles.append(profile)
        baselines.append(baseline)
        ratios.append(ratio_db)
        peaks_by_channel.append(np.sort(peaks))
    audible_counts = [np.zeros(len(peaks), np.int64) for peaks in peaks_by_channel]
    total_frames = 0
    for mags in _iter_band_mags_multi(path, lo, hi, channels, start_frame, end_frame):
        total_frames += len(mags)
        report_progress(min(2 / 3 + total_frames / (expected_frames * 3), 1.0))
        for index, peaks in enumerate(peaks_by_channel):
            if len(peaks):
                audible_counts[index] += np.sum(
                    mags[:, index, peaks] > 2 * baselines[index][peaks], axis=0)
    results = {}
    for index, channel in enumerate(channels):
        profile, baseline, ratio_db = profiles[index], baselines[index], ratios[index]
        result_peaks = []
        for j, p in enumerate(peaks_by_channel[index]):
            a, b, c = (np.log(profile[p + d] + 1e-12) if 0 <= p + d < len(profile)
                       else np.log(profile[p] + 1e-12) for d in (-1, 0, 1))
            denom = a - 2 * b + c
            delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
            delta = float(np.clip(delta, -0.5, 0.5))
            present = float(audible_counts[index][j] / total_frames) if total_frames else 0.0
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
        results[channel] = {
            "file": os.path.basename(path), "samplerate": sr, "channels": info.channels, "channel": channel,
            "duration_s": duration, "analysis_duration_s": (end_frame - start_frame) / sr,
            "analysis_start_s": start_frame / sr, "analysis_end_s": end_frame / sr,
            "n_fft": N_FFT, "band_hz": [fmin, fmax],
            "reference_mode": "hum-only" if hum_only else "quietest",
            "quiet_frames": int(quiet_counts[index]),
            "quiet_total_s": float(quiet_counts[index] * HOP / sr),
            "quiet_first_s": float(quiet_first[index]) if quiet_first[index] is not None else None,
            "quiet_last_s": float(quiet_last[index]) if quiet_last[index] is not None else None,
            "peaks": result_peaks,
        }
    report_progress(1.0)
    return results


def _spectrogram_geometry(info, fmin, fmax, start, end, x_resolution, y_resolution):
    sr = info.samplerate
    _validate_band(sr, fmin, fmax)
    duration = info.frames / sr
    start = 0.0 if start is None else start
    end = duration if end is None else end
    if (not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start
            or end > duration):
        raise ValueError("spectrogram range must satisfy 0 <= start < end <= file duration")
    if (not np.isfinite(x_resolution) or x_resolution <= 0
            or not np.isfinite(y_resolution) or y_resolution <= 0):
        raise ValueError("spectrogram resolutions must be positive finite numbers")
    start_frame, end_frame = round(start * sr), round(end * sr)
    if end_frame - start_frame < N_FFT:
        raise ValueError(f"spectrogram range must contain at least {N_FFT / sr:.3f} seconds of audio")
    plot_duration = (end_frame - start_frame) / sr
    raw_width = plot_duration * 10 * x_resolution
    raw_height = (fmax - fmin) * y_resolution
    if not np.isfinite(raw_width) or not np.isfinite(raw_height):
        raise ValueError("spectrogram resolution must produce finite dimensions")
    width = max(1, int(np.ceil(raw_width)))
    height = max(1, int(np.ceil(raw_height)))
    if width * height > SPECTROGRAM_MAX_PIXELS:
        raise ValueError(
            f"spectrogram resolution exceeds the {SPECTROGRAM_MAX_PIXELS:,}-pixel limit")
    return start_frame, end_frame, plot_duration, width, height


def write_spectrogram(path, output_path, channel="mix", fmin=1500.0, fmax=3000.0,
                      start=0.0, end=None, x_resolution=1.0, y_resolution=1.0,
                      progress=None):
    """Write an exclusive-create SVG spectrogram for the selected audio range.

    x_resolution is pixels per 0.1 seconds; y_resolution is pixels per Hz. The raster plot
    is capped by SPECTROGRAM_MAX_PIXELS. progress, when supplied, receives fractions from 0 to 1.
    Raises FileExistsError if output_path already exists.
    """
    info = sf.info(path)
    sr = info.samplerate
    start_frame, end_frame, plot_duration, width, height = _spectrogram_geometry(
        info, fmin, fmax, start, end, x_resolution, y_resolution)
    lo, hi = band_bins(sr, fmin, fmax)
    bin_freqs = np.arange(lo, hi) * sr / N_FFT
    plot_freqs = fmin + (np.arange(height) + 0.5) * (fmax - fmin) / height
    right_bins = np.clip(np.searchsorted(bin_freqs, plot_freqs), 1, len(bin_freqs) - 1)
    left_bins = right_bins - 1
    interpolation = ((plot_freqs - bin_freqs[left_bins])
                     / (bin_freqs[right_bins] - bin_freqs[left_bins]))
    in_band = (plot_freqs >= bin_freqs[0]) & (plot_freqs <= bin_freqs[-1])
    expected_frames = max(1, (end_frame - start_frame - N_FFT) // HOP + 1)
    report_progress = _progress_reporter(
        progress if progress is not None else
        lambda fraction: _show_progress(f"Spectrogram {channel}", fraction))
    palette = _spectrogram_palette()

    margin_left, margin_top, margin_right, margin_bottom = 70, 20, 20, 55
    svg_width = width + margin_left + margin_right
    svg_height = height + margin_top + margin_bottom
    pixels = np.zeros((height, width), np.uint8)
    with _exclusive_text_output(output_path) as svg:
        svg.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" '
                  f'viewBox="0 0 {svg_width} {svg_height}">\n')
        svg.write(f'<rect width="{svg_width}" height="{svg_height}" fill="{palette[0]}"/>\n')
        svg.write(f'<text x="{margin_left}" y="14" fill="white">Spectrogram: {escape(channel)} '
                  f'(magnitude {SPECTROGRAM_DB_FLOOR:g} to {SPECTROGRAM_DB_CEILING:g} dBFS, '
                  f'brighter is stronger)</text>\n')
        svg.write(f'<g transform="translate({margin_left},{margin_top})">\n')
        current_column = None
        column_values = np.zeros(height, np.float32)
        frame_offset = 0

        def flush_column():
            if current_column is None:
                return
            colors = np.clip(
                ((20 * np.log10(column_values + SPECTROGRAM_AMP_FLOOR) - SPECTROGRAM_DB_FLOOR)
                 * (SPECTROGRAM_PALETTE_SIZE - 1)
                 / (SPECTROGRAM_DB_CEILING - SPECTROGRAM_DB_FLOOR)).astype(int),
                0, SPECTROGRAM_PALETTE_SIZE - 1)
            pixels[:, current_column] = colors[::-1].astype(np.uint8)

        total_frames = 0
        report_progress(0.0)
        for mags in _iter_band_mags(path, lo, hi, channel, start_frame, end_frame):
            frames = frame_offset + np.arange(len(mags))
            # Frame order guarantees pixel columns never decrease.
            columns = np.minimum(
                width - 1,
                ((frames * HOP + N_FFT / 2) / sr * 10 * x_resolution).astype(int))
            group_starts = np.r_[0, np.flatnonzero(columns[1:] != columns[:-1]) + 1]
            grouped_mags = np.maximum.reduceat(mags, group_starts, axis=0)
            for x, spectrum in zip(columns[group_starts], grouped_mags):
                if current_column is None:
                    current_column = x
                elif x != current_column:
                    flush_column()
                    # Missing time columns repeat the prior rendered column; silence is palette index zero.
                    if x > current_column + 1:
                        pixels[:, current_column + 1:x] = pixels[:, current_column, None]
                    column_values.fill(0)
                    current_column = x
                values = spectrum[left_bins] * (1 - interpolation) + spectrum[right_bins] * interpolation
                values[~in_band] = 0
                np.maximum(column_values, values, out=column_values)
            frame_offset += len(mags)
            total_frames += len(mags)
            report_progress(min(total_frames / expected_frames, 1.0))
        flush_column()
        _write_embedded_png(svg, pixels, palette)
        svg.write("</g>\n")
        start_s, end_s = start_frame / sr, end_frame / sr
        svg.write(f'<text x="{margin_left + width / 2}" y="{svg_height - 10}" fill="white" '
                  f'text-anchor="middle">Time (s), {start_s:.3f}-{end_s:.3f}</text>\n')
        svg.write(f'<text x="14" y="{margin_top + height / 2}" fill="white" '
                  f'transform="rotate(-90 14 {margin_top + height / 2})" text-anchor="middle">'
                  f'Frequency (Hz), {fmin:g}-{fmax:g}</text>\n')
        svg.write('<g fill="white" font-size="10">')
        for tick in range(6):
            x = margin_left + width * tick / 5
            seconds = start_s + plot_duration * tick / 5
            svg.write(f'<text x="{x:.1f}" y="{svg_height-30}" text-anchor="middle">{seconds:.1f}</text>')
            y = margin_top + height * tick / 5
            frequency = fmax - (fmax - fmin) * tick / 5
            svg.write(f'<text x="{margin_left-5}" y="{y:.1f}" text-anchor="end">{frequency:.0f}</text>')
        svg.write("</g>\n</svg>\n")
    report_progress(1.0)
    return output_path


def _spectrogram_output_path(input_path, requested, channel, multiple_channels):
    if requested:
        stem, ext = os.path.splitext(requested)
        channel_suffix = f"-{channel}" if multiple_channels else ""
        output_path = f"{stem}{channel_suffix}{ext or '.svg'}"
    else:
        stem, _ = os.path.splitext(input_path)
        output_path = f"{stem}-spectrogram-{channel}-{secrets.token_hex(4)}.svg"
    if os.path.realpath(output_path) == os.path.realpath(input_path):
        raise ValueError("spectrogram output must not resolve to the input file")
    return output_path


def print_stats(res, out=print):
    out(f"File: {res['file']}  {res['samplerate']} Hz, {res['channels']} ch, {res['duration_s'] / 60:.1f} min"
        f"  [analysed channel: {res.get('channel', 'mix')}]")
    if "analysis_start_s" in res and (res["analysis_start_s"] != 0 or res["analysis_end_s"] != res["duration_s"]):
        out(f"Analyzed interval: {res['analysis_start_s']:.2f}-{res['analysis_end_s']:.2f} s")
    band = f"Band searched: {res['band_hz'][0]:.0f}-{res['band_hz'][1]:.0f} Hz"
    if res.get("reference_mode") == "hum-only":
        out(f"{band}, hum-only reference: {res['quiet_frames']} frames")
    elif not res["quiet_frames"]:
        out(f"{band}, no non-silent reference frames")
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


def remove_hum(src, dst_file, res, mask_db=12.0, max_reduction_db=30.0, log=print, progress=None):
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
    report_progress = _progress_reporter(
        progress if progress is not None else lambda fraction: _show_progress("Remove", fraction))
    log("Processing audio...")

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
                report_progress(done / max(total, 1))

        report_progress(0.0)
        for blk in _read_chunks(src):
            buf = np.concatenate([buf, blk], axis=1)
            buf, e = run(buf)
            write(e)
        buf = np.concatenate([buf, np.zeros((nch, N_FFT + HOP), np.float32)], axis=1)
        buf, e = run(buf)
        write(e)
        report_progress(1.0)
    log("Audio processing complete.")


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
    ap.add_argument("--spectrogram", nargs="?", const="", metavar="SVG",
                    help="write a frequency-over-time SVG (default filename generated automatically)")
    ap.add_argument("--x-resolution", type=float, default=1.0,
                    help="spectrogram pixels per 0.1 seconds (default: 1)")
    ap.add_argument("--y-resolution", type=float, default=1.0,
                    help="spectrogram pixels per Hz (default: 1)")
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
    if a.spectrogram is not None and a.mode != "analyze":
        ap.error("--spectrogram is only available in analyze mode")
    channels = None
    spectrogram_paths = {}
    if a.spectrogram is not None:
        channels = ([a.channel] if a.channel else ["mix", "left", "right"])
        try:
            _spectrogram_geometry(
                sf.info(a.input), a.fmin, a.fmax, a.start or 0.0, a.end,
                a.x_resolution, a.y_resolution)
            for channel in channels:
                output_path = _spectrogram_output_path(
                    a.input, a.spectrogram, channel, len(channels) > 1)
                if os.path.lexists(output_path):
                    raise FileExistsError(f"spectrogram output already exists: {output_path}")
                if a.save_profile and os.path.realpath(a.save_profile) == os.path.realpath(output_path):
                    raise ValueError("profile and spectrogram outputs must use different paths")
                spectrogram_paths[channel] = output_path
        except (ValueError, FileExistsError) as e:
            ap.error(str(e))
    spectrogram_failed = False
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
        if channels is None:
            channels = ([a.channel] if a.channel else
                        ["mix", "left", "right"] if a.mode == "analyze" else ["mix"])
        try:
            results = analyze_channels(
                a.input, a.fmin, a.fmax, a.quiet_percent, a.min_prominence_db,
                a.max_peaks, channels, start=a.start or 0.0, end=a.end,
                hum_only=a.hum_only)
            for channel in channels:
                result = results[channel]
                print_stats(result)
                if a.spectrogram is not None:
                    try:
                        output_path = spectrogram_paths[channel]
                        write_spectrogram(a.input, output_path, channel, a.fmin, a.fmax,
                                          a.start or 0.0, a.end, a.x_resolution, a.y_resolution)
                    except (OSError, ValueError, RuntimeError) as e:
                        print(f"Spectrogram for {channel} failed ({type(e).__name__}): {e}",
                              file=sys.stderr)
                        spectrogram_failed = True
                        continue
                    print(f"Spectrogram written to {output_path}")
        except ValueError as e:
            ap.error(str(e))
        res = results.get("mix", next(iter(results.values())))
    if a.save_profile:
        with open(a.save_profile, "w") as f:
            json.dump(res, f, indent=1)
    if a.mode == "analyze":
        return 1 if spectrogram_failed else 0
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
