# VibeVoice reliability patches (apply on top of upstream demo/gradio_demo.py)

These are the minimal diffs that turn the stock demo into a long-form-safe
server. Apply with `git apply patches/*.patch` after cloning VibeVoice.

## 01_max_length_times.patch — no tail truncation
Raises the audio-token length cap from 2× to 8× the base length so long
narration (100+ word chapters) never stops early.

## 02_sampling_no_loops.patch — no repetition loops
Switches the decoder from greedy to sampling (temperature/top-p) so long
generations don't get stuck repeating one sentence.

---
> These patches are written against the upstream community fork
> (`vibevoice-community/VibeVoice`, MIT). The actual hunk context may shift
> slightly between versions — if `git apply` fails, locate the same lines
> manually (the changes are described in README.md §6).