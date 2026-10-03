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
python hum_remover.py remove  concert.wav [--profile hum.json] [--mask-db 12] [--max-reduction-db 30]
```
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
