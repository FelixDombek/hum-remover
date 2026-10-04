# hum-remover
Identifies and removes hum (a few isolated tonal peaks, by default searched between 1.5 and 3 kHz) in WAV files.

## Setup (Windows 11 / any OS)
```
pip install -r requirements.txt
```
Libraries: `numpy`/`scipy` (FFT, peak finding) and `soundfile` (streaming WAV I/O, so a 1 h / 700 MB file is
processed in chunks without loading it into RAM).

## Usage
```
python hum_remover.py analyze concert.wav            # print statistics about the hum
python hum_remover.py analyze concert.wav --save-profile hum.json
python hum_remover.py analyze concert.wav --start "35:10" --end "35:12"
python hum_remover.py analyze concert.wav --start "35:10" --end "35:12" --hum-only
python hum_remover.py analyze concert.wav --spectrogram [chart.svg] [--x-resolution 1] [--y-resolution 1]
python hum_remover.py remove  concert.wav [--profile hum.json] [--mask-db 12] [--max-reduction-db 30]
```
`analyze` accepts optional `--start` and `--end` bounds in seconds, `MM:SS`, or `HH:MM:SS` format. Omit either
bound to analyze from the beginning or through the end of the file, respectively. By default, `analyze` reports
mix, left, and right results; `--channel` selects a single channel instead. Use `--hum-only` when the selected
interval contains only hum to use all its non-silent frames as the reference rather than selecting the quietest
frames. When analyzing all channels, `--save-profile` saves the mix result; use `--channel` to save another
channel's profile. All three channel analyses share each streaming pass; three passes are used to find quiet-frame
thresholds, build profiles, and measure audibility without retaining the recording in memory.
Both modes show a progress bar. `--spectrogram` writes a frequency-over-time SVG with a compressed raster plot for
each analyzed channel, limited to the selected time and frequency ranges. Its defaults are 1 horizontal pixel per
0.1 seconds and 1 vertical pixel per Hz; adjust them with `--x-resolution` and `--y-resolution`. Images are capped
at 100 million pixels. If no SVG filename is given, unique files are created next to the input. Creating each
channel's spectrogram requires an additional streaming FFT pass over the selected audio range. Color indicates
magnitude from -100 to 0 dBFS; brighter colors indicate stronger frequencies.
`remove` writes `concert-nohum-<random>.wav` next to the input; existing files (including the input) are never
overwritten.

`--channel left|right|mix` (default `mix`) selects the channel used for detection and the masking decision, so you
can compare analyze results per channel. Output mirrors the input's sample rate, channel count and bit depth.

## How it works
1. **Detection**: an STFT (8192 pt, 75 % overlap) of the mono mix is computed over the hum band. The quietest
   frames (`--quiet-percent`, default 10 %) are used as reference, since the hum is most audible there. Their median
   spectrum is compared with a running-median baseline; peaks above `--min-prominence-db` (default 6 dB) are the hum.
2. **Removal**: in the STFT domain, the hum amplitude at the detected bins is subtracted (phase kept), so music at the
   same frequencies is preserved. The amount is scaled per frame by how audible the hum is, based on the level of
   the surrounding spectrum (masking): background at/below the hum level → 100 % removed; background
   `--mask-db` dB above the hum → 0 % removed; linear in dB in between. `--max-reduction-db` caps the attenuation.

## Tests
```
pip install pytest && python -m pytest
```
