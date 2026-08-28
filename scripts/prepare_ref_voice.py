#!/usr/bin/env python3
"""prepare_ref_voice.py — turn a raw recording into a VibeVoice-ready reference.

Steps:
  1. trim silence at the start/end (so the ref starts at speech onset),
  2. convert to mono 24 kHz WAV,
  3. write the exact transcript next to it.

Usage:
    python prepare_ref_voice.py input.mp3 "The exact spoken transcript..."
        [--out ref_voice.wav]

The transcript MUST match the recording word-for-word — this is the #1 driver of
clone quality. Record ~12 s of clean speech (no background music/noise).
"""
import argparse, subprocess, sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="source recording (mp3/wav/m4a…)")
    ap.add_argument("transcript", help="exact transcript of the spoken words")
    ap.add_argument("--out", default="ref_voice.wav", help="output wav path")
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.out)
    if not src.exists():
        print(f"ERROR: {src} not found", file=sys.stderr)
        return 1

    # trim leading/trailing silence (0.5 s guard), convert to mono 24 kHz s16
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-af", "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.3,"
                "areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.3,areverse",
         "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", str(out)],
        check=True)

    # write the transcript beside the wav (same stem, .txt)
    txt = out.with_suffix(".txt")
    txt.write_text(args.transcript.strip() + "\n", encoding="utf-8")

    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(out)],
        capture_output=True, text=True).stdout.strip()
    print(f"OK: {out}  ({float(dur):.1f} s, mono 24k)")
    print(f"    transcript -> {txt} ({len(args.transcript)} chars)")
    print("    drop into demo/voices/<Name>.wav to register it on the server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())