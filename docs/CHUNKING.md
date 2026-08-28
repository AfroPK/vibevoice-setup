# Chunking for long VO — the recipe that makes narration sound right

VibeVoice (like most local TTS) is byte-for-byte reliable on **short inputs**
(~30–100 words) and degrades on one giant paragraph: tail truncation and
repetition loops. The proven solution is **chunked continuous narration**:
split the script into WPM-calibrated chunks, generate each chunk separately,
then stitch with a natural seam.

This is the exact recipe used in production (156 WPM narrator, chapter-per-chunk
VO). Implementations below are complete and copy-pasteable.

---

## The two chunking functions

### 1. `_chunk_text` — small natural chunks (~70 words, ~2 sentences)

Use for **short/medium** narration (a single ~30–60 s chapter paragraph).

```python
import re as _re

def _chunk_text(text: str, max_words: int = 70) -> list:
    """Split narration into small NATURAL chunks (~2 sentences / ~70 words)
    that TTS reliably completes without truncating the tail. Each chunk ends on
    a sentence boundary so it still reads like flowing narration."""
    sentences = _re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if not s.strip():
            continue
        if len(cur.split()) + len(s.split()) <= max_words:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s.strip()
    if cur:
        chunks.append(cur)
    return chunks
```

### 2. `chunk_by_wpm` — uniform WPM-calibrated chunks (the long-form one)

Use for **full episodes** (5–10 min scripts). Every chunk is sized to fill the
**same duration** at the narrator's speaking rate, so:

- no speed/voice **drift** across chunks (each TTS call spans the same time),
- predictable production time,
- sentence boundaries respected; over-long sentences hard-split at the cap.

```python
import re as _re

def chunk_by_wpm(script: str, target_sec: float = 90.0,
                 wpm: float = 156.0, max_words: int = 250) -> list:
    """Split a narration script into uniform, WPM-calibrated chunks.
    Each chunk ~fills target_sec of speech at `wpm`."""
    sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", script) if s.strip()]
    per_chunk = int(max(30, min(max_words, round(wpm * target_sec / 60))))
    chunks, cur, n = [], "", 0
    for sen in sentences:
        w = len(sen.split())
        if w > per_chunk:
            if cur:
                chunks.append(cur); cur, n = "", 0
            words = sen.split()
            for i in range(0, len(words), per_chunk):
                chunks.append(" ".join(words[i:i + per_chunk]))
            continue
        if n + w > per_chunk and cur:
            chunks.append(cur); cur, n = sen, w
        else:
            cur = (cur + " " + sen).strip(); n += w
    if cur:
        chunks.append(cur)
    return chunks
```

### Parameters that work (the production values)

| param | value | why |
|---|---|---|
| `wpm` | **156** | measured pace of the reference narrator (33.2 s speech = ~156 WPM) |
| `target_sec` | **90** (or chapter length) | each chunk ≈ one chapter segment of VO |
| `max_words` | **180–250** | hard ceiling well under the truncation threshold |
| script source | sentence-aligned chapters | the script is already split at sentence boundaries upstream |

> **Rule of thumb:** keep every TTS call under **~180–200 words**. At 156 WPM
> that's ≈ 75–85 s of audio per chunk — reliably complete, still a natural
> paragraph.

---

## The stitch: concat FILTER (not demuxer)

After generating each chunk WAV, join them with **ffmpeg's `concat` filter**
(re-encodes to consistent PCM) rather than the concat **demuxer** (`-c copy`),
which is fragile with differently-sized chunks.

```python
import subprocess as _s
from pathlib import Path

def _concat_wavs(parts: list, target: Path) -> None:
    if not parts:
        return
    n = len(parts)
    inputs = []
    for p in parts:
        inputs += ["-i", str(p)]
    fc = "".join(f"[{i}:a]" for i in range(n)) + f"concat=n={n}:v=0:a=1[aout]"
    cmd = ["ffmpeg", "-y", "-v", "error", *inputs,
           "-filter_complex", fc,
           "-map", "[aout]", "-ar", "44100", "-ac", "2",
           "-c:a", "pcm_s16le", str(target)]
    _s.run(cmd, check=True)
```

**Seam pauses:** do **not** pass raw chunked audio with zero gap — it sounds
choppy. Insert a **0.45–1.0 s silence** between chunks (adelay/apad), or (better)
generate each chunk with a trailing pause by ending the input text at a
sentence you'd naturally pause after. The concat filter keeps the format
consistent so the seam is a clean, breathing beat — not a click.

---

## Full long-form pipeline (chunk → TTS → stitch)

```python
def vo_for_script(script: str, tts_url: str = "http://127.0.0.1:8000/v1/audio/speech",
                  voice: str = "MyNarrator", out: str = "vo.wav") -> None:
    import json, urllib.request, tempfile
    from pathlib import Path

    chunks = chunk_by_wpm(script, target_sec=90, wpm=156, max_words=200)
    parts = []
    for i, chunk in enumerate(chunks):
        req = urllib.request.Request(
            tts_url,
            data=json.dumps({"model": "vibevoice-1.5b", "input": chunk,
                             "voice": voice, "seed": i,          # fixed seed per chunk
                             "cfg_scale": 1.3, "inference_steps": 10}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            wav = r.read()
        p = Path(f"/tmp/chunk_{i:02d}.wav")
        p.write_bytes(wav)
        parts.append(p)
        print(f"chunk {i+1}/{len(chunks)}: {len(chunk.split())} words")

    _concat_wavs(parts, Path(out))
    print(f"DONE -> {out}")
```

**Determinism:** pass a fixed `seed = chunk_index` so re-runs are identical and
each chunk is independently reproducible.

---

## Verification checklist (long VO)

After stitching, check:
1. **Tail intact** — the final chunk's last word is audible (no truncation).
2. **No repeats** — no sentence appears 2×+ in a row (sampling not greedy).
3. **Seams natural** — 0.45–1.0 s breathes between chunks, no clicks/clips.
4. **Duration matches** — total ≈ `word_count / wpm * 60 + n_chunks * seam`.
5. **Voice stable** — same reference voice + fixed seed per chunk = no drift.