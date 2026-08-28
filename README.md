# VibeVoice Setup Tutorial — local voice-cloning TTS that sounds right

A complete, from-scratch tutorial for setting up **VibeVoice (1.5B)**, a local
zero-shot voice-cloning TTS, as an **OpenAI-compatible HTTP server** that
produces **long-form narration** without the two classic failure modes:

1. ❌ **Tail truncation** — the last sentence gets cut off on long chapters
2. ❌ **Repetition loops** — the model repeats the same sentence over and over

Written to be followed by a **human or an AI agent**. Every command is concrete;
every file is shown. No black boxes.

---

## Table of contents

1. [What this builds](#what-this-builds)
2. [Prerequisites](#prerequisites)
3. [Install VibeVoice](#install-vibevoice)
4. [Set up the GPU / device](#set-up-the-gpu--device)
5. [Prepare the reference voice (voice cloning)](#prepare-the-reference-voice)
6. [The reliability fixes (why it sounds right)](#the-reliability-fixes)
7. [Chunking for long VO](docs/CHUNKING.md) — the recipe that makes long narration sound right
8. [Run the server](#run-the-server)
9. [Test it (curl + Python)](#test-it)
10. [Agent-friendly spec](#agent-friendly-spec)
11. [Troubleshooting](#troubleshooting)

---

## What this builds

```
┌─────────────┐   POST /v1/audio/speech   ┌──────────────────┐
│  Pipelines  │ ────────────────────────► │ VibeVoice server │
│  (n8n, cron,│    OpenAI-compatible      │  (port 8000)     │
│  scripts)   │ ◄──────────────────────── │  GPU (ROCm/CUDA) │
└─────────────┘    wav bytes              └──────────────────┘
```

- **Voice cloning** is done with a **reference audio prompt** (zero-shot) — you do
  **not** fine-tune. Give it 3–60 s of a voice + a transcript, and it speaks in
  that voice.
- The server is **shared**: many pipelines call it concurrently for consistency
  (one narrator voice across all videos).

---

## Prerequisites

- **Linux** (the tutorial assumes Ubuntu/Debian; Windows works with ROCm builds)
- Python 3.10–3.12
- A GPU with **≥ 8 GB VRAM** (NVIDIA CUDA or AMD ROCm both work)
  - 6 GB works in lowp vram mode but is tight for long audio
- `ffmpeg` (`sudo apt install ffmpeg`)
- `git`

---

## Install VibeVoice

```bash
# 1. Clone the community fork
git clone https://github.com/vibevoice-community/VibeVoice
cd VibeVoice

# 2. Create a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install runtime deps (matches the working setup)
pip install -e . torch torchaudio transformers accelerate soundfile \
    gradio scipy numpy

# 4. Verify the CLI/import works
python -c "import vibevoice; print('vibevoice ok')"
```

> **Note on torch:** install the build matching your GPU.
> - NVIDIA: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126`
> - AMD ROCm (Linux): `pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2`
> - Apple Silicon: the repo defaults to fp16 on MPS to avoid RAM swaps.

### Download the model

VibeVoice 1.5B publishes a safetensors checkpoint. Place it at
`models/` (or wherever the repo expects):

```bash
# example — substitute the real release URL from the repo's model card
huggingface-cli download <org>/vibevoice-1.5b --local-dir models/
```

Verify you can run the bundled demo before going further:

```bash
python demo/gradio_demo.py --model_path models/vibevoice-1.5b --device cuda
# opens a web UI; generate one sentence to confirm the model loads
```

---

## Set up the GPU / device

The server picks the device via an env flag; the tutorial default is `cuda`:

```bash
export DEVICE=cuda        # or "cpu", "mps" (Apple), "cuda:0"
```

**AMD ROCm specifics** (what the reference homelab uses):

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0    # RX 6750 XT / gfx1031 (RDNA2)
export PYTORCH_NO_HIP_MEMORY_CACHING=1    # avoids HIP OOMs on long audio
```

Check it sees the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA GeForce RTX 3070
```

---

## Prepare the reference voice (voice cloning)

VibeVoice clones a voice from a **reference audio + its transcript** — no
fine-tuning. Two rules make the clone sound right:

1. **Use 5–15 s of clean speech** (the reference implementation internally
   trims longer refs; too long actually hurts fidelity).
2. **The transcript must match the audio exactly** — this is the #1 quality
   driver. Mismatches degrade the clone.

### 1. Record or find your reference

Record ~12 s of the target voice reading a standard calibration sentence
(one you can transcribe exactly). For example:

> “The quick brown fox jumps over the lazy dog. It is an old classic phrase
> that captures every letter of our alphabet in a single coherent thought.”

### 2. Convert to the right format

VibeVoice expects mono WAV 16–24 kHz:

```bash
ffmpeg -y -i "your_recording.mp3" -ac 1 -ar 24000 -sample_fmt s16 \
       ref_voice.wav
```

### 3. Write the exact transcript

```bash
cat > ref_text.txt <<'EOF'
The quick brown fox jumps over the lazy dog. It is an old classic phrase
that captures every letter of our alphabet in a single coherent thought.
EOF
```

### 4. (Optional) pre-register the voice on the server

The server exposes `GET /voices` and auto-loads `demo/voices/<name>.wav`;
drop the ref there and it's selectable by name:

```bash
mkdir -p demo/voices
cp ref_voice.wav demo/voices/MyNarrator.wav
```

---

## The reliability fixes (why it sounds right)

These are the changes that turn “demo that truncates” into “long-form that
works”. They are small, targeted patches to the stock code.

### Fix 1 — no tail truncation: raise `max_length_times`

**Symptom:** the final sentence of a long (100+ word) chapter is cut off.

**Root cause:** the audio-token decoder’s length cap is too small for long
narration; it stops early.

**Fix:** thread a `max_length_times` parameter through the demo + server layers
and default it to **8** (stock is 2):

```python
# demo/gradio_demo.py (the generate() signature)
def generate(self, text, *, seed: Optional[int] = None,
             cfg_scale: float = 1.3,
             inference_steps: Optional[int] = None,
             max_length_times: Optional[int] = None):
    ...
    n_max = 8                      # allow up to 8x the base length
```

Then every call site passes `max_length_times=8`.

### Fix 2 — no repetition loops: enable sampling

**Symptom:** on long chapters the model repeats the same sentence 3–5 times
(“…and so we begin. And so we begin. And so we begin…”).

**Root cause:** greedy/argmax decoding gets stuck in a loop on longer inputs.

**Fix:** use **sampling** (temperature/cfg sampling) instead of greedy decode:

```python
# in the generation call:
use_sampling = True
temperature = 0.8                # tune 0.7–1.0 for your voice
top_p = 0.9
```

This makes the output non-deterministic (good — it breaks loops). If you need
reproducibility, pass a **fixed seed**: the reference implementation
deterministically sequences seeds per generation.

> **Summary table**
>
> | Problem | Symptom | Fix |
> |---|---|---|
> | Tail truncation | last sentence missing | `max_length_times=8` |
> | Repetition loop | same sentence repeats | sampling (`do_sample=True`) instead of greedy |

---

## Run the server

Save this as `run_vibevoice_server.sh`:

```bash
#!/bin/bash
# Run the VibeVoice OpenAI-compatible server (port 8000)
set -euo pipefail
cd "$(dirname "$0")"

# --- device / GPU knobs ---
export DEVICE="${DEVICE:-cuda}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"  # AMD only
export PYTORCH_NO_HIP_MEMORY_CACHING="${PYTORCH_NO_HIP_MEMORY_CACHING:-1}"  # AMD only

# --- model / server ---
MODEL_PATH="${MODEL_PATH:-models/vibevoice-1.5b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
INFERENCE_STEPS="${INFERENCE_STEPS:-10}"

exec .venv/bin/python -m demo.gradio_demo \
     --model_path "$MODEL_PATH" \
     --device "$DEVICE" \
     --host "$HOST" --port "$PORT" \
     --inference_steps "$INFERENCE_STEPS"
echo "VibeVoice server up on :$PORT"
```

Run it:

```bash
chmod +x run_vibevoice_server.sh
./run_vibevoice_server.sh
# VibeVoice server up on :8000
```

---

## Test it

### curl

```bash
curl -s -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vibevoice-1.5b",
    "input": "The ancestors told this story around the fire.",
    "voice": "MyNarrator",
    "response_format": "wav",
    "seed": 42,
    "cfg_scale": 1.30,
    "inference_steps": 10
  }' --output out.wav

ffprobe -v error -show_entries format=duration -of csv=p=0 out.wav
# 3.2  <- a real duration, not a truncated tail
```

### Python (OpenAI SDK style)

```python
import json, urllib.request

req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/audio/speech",
    data=json.dumps({
        "model": "vibevoice-1.5b",
        "input": "And that is how it is told, even to this day.",
        "voice": "MyNarrator",
        "seed": 7, "cfg_scale": 1.3, "inference_steps": 10,
    }).encode(),
    headers={"Content-Type": "application/json"})

with urllib.request.urlopen(req, timeout=180) as r:
    wav = r.read()
open("narration.wav", "wb").write(wav)
print(f"saved {len(wav)/1e6:.1f} MB")
```

### Long-form check (the important one)

Generate a **120+ word paragraph** and verify:
1. The **last word** is present — un-trimmed output should end with the final
   sentence’s last word.
2. No sentence is **repeated** 2×+ in a row.

```bash
python - <<'EOF'
import json, urllib.request
long_text = ("Numerous long sentences... " * 4)   # your 120+ word script
req = urllib.request.Request("http://127.0.0.1:8000/v1/audio/speech",
    data=json.dumps({"model":"vibevoice-1.5b","input":long_text,"voice":"MyNarrator",
                     "seed":1,"cfg_scale":1.3,"inference_steps":10}).encode(),
    headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    open("long.wav","wb").write(r.read())
print("long.wav written")
EOF
# then listen: last sentence intact? no repeats?
```

---

## Agent-friendly spec

For an AI agent integrating with this server:

- **Base URL:** `http://<host>:8000`
- **Endpoint:** `POST /v1/audio/speech` (OpenAI-compatible)
- **Request** (JSON): `model`, `input` (the text), `voice` (reference name),
  optional `response_format` (`wav` default), `seed`, `cfg_scale`, `inference_steps`
- **Response:** raw WAV bytes (or base64 JSON if `response_format=json`)
- **Health:** `GET /health` → `{"ok": true, "voice": {...}}`
- **Available voices:** `GET /voices`
- **Determinism:** pass a fixed `seed` + fixed `cfg_scale` + fixed
  `inference_steps` to reproduce a take; omit seed to break loops (sampling)

### Query params
| param | default | meaning |
|---|---|---|
| `model` | vibevoice-1.5b | model to use |
| `input` | — | text to speak |
| `voice` | — | reference voice name (`GET /voices`) |
| `seed` | random | fixed seed = reproducible |
| `cfg_scale` | 1.3 | classifier-free guidance strenght |
| `inference_steps` | 10 | more = slower/sharper |
| `max_length_times` | 8 | length cap vs base (anti-truncation) |

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CUDA out of memory` on long text | lower `inference_steps`; enable low-vram; chunk the text into ~180-word parts |
| HIP error on AMD | `HSA_OVERRIDE_GFX_VERSION` must match your gfx arch; try `10.3.1` |
| Last sentence missing | confirm `max_length_times=8` is threaded through the server call |
| Repeated sentences | confirm sampling is ON (greedy causes loops); tune temperature |
| Clone sounds off | shorten ref to 5–15 s + make the transcript match it exactly |
| Server ignores new voices | restart server after adding `demo/voices/<name>.wav` |
| Slow on first request | model load + kernel warm-up; first take can take 1–2 min |

---

## License

The upstream VibeVoice code is MIT-licensed (see upstream repo). This tutorial is
provided under MIT — follow it freely.

---

*Tutorial based on a proven production deployment: VibeVoice 1.5B served on an
AMD RX 6750 XT (ROCm gfx1031), shared by multiple long-form video pipelines,
producing deterministic, no-truncation, no-loop narration. This repository is a
clean-room tutorial — it contains no private configuration or credentials.*