# Staged pipeline resume and laptop-friendly execution

**Read this document before implementing pipeline/orchestration changes.** It avoids re-discovering the whole repo.

## Implementation status (done)

Orchestration lives in `youtube_publish_scheduler.py` (`run_phased_pipeline`, `run_manifest.json` per run under `OUTPUT_ROOT/run_<slug>/`). CLI examples:

```bash
# Full run (new folder under output/)
python youtube_publish_scheduler.py

# Resume after failure — only encode + upload, reuse existing run folder
python youtube_publish_scheduler.py --run-dir output/run_20260112_153045 --from render --to upload

# Skip re-scraping LLM if graph already finished (same folder)
python youtube_publish_scheduler.py --run-dir output/run_20260112_153045 --from graph --to upload --resume
```

**Env:** `OUTPUT_ROOT` (default `output`), `SKIP_WHISPER`, `ENABLE_SUBTITLES`, `WHISPER_MODEL`, `FFMPEG_PRESET`, `ENCODE_THREADS` — see `video_stitcher.create_video_from_audio_images`.

## Goal

Allow reruns like: *script already generated → only TTS*, *voiceover ready → only images + encode*, *MP4 exists → only upload*, using a **run manifest** and **explicit phases** instead of always running the full stack.

## Is “script done, only do the rest” possible?

**Yes.** Work is already a **linear sequence** in `youtube_publish_scheduler.py` → `run_entire_pipeline()`:

1. Ingest + `AIChainPipeline.run` in `pipeline.py` (LangGraph: long script, five shorts, YouTube metadata, thumbnail).
2. **TTS** → `longform_voiceover_{run_slug}.mp3` + five short MP3s (`text_to_speech_edge` in `youtube_automation_utils.py`).
3. **Images** → HF / placeholders under `./output/visuals_*_{run_slug}`.
4. **Encode** → `create_video_from_audio_images` in `video_stitcher.py` (Ken Burns + Whisper + `libx264`).
5. **Upload** → YouTube in `youtube_publish_scheduler.py`.

There is **no** orchestration layer yet; every scheduler run effectively starts the chain from step 1 (except what you manually skip by editing). **Artifact + manifest resume** is the intended fix.

## Recommended design

### 1. Run directory + manifest

- Use `./output/run_{run_slug}/` (reuse existing `run_slug` pattern: `YYYYMMDD_HHMMSS`).
- After each successful sub-phase, write **`run_manifest.json`** containing at least:
  - Paths (or relative paths) to: long script file or `scripts.json`, five short scripts, each MP3, each image dir, each MP4, optional YouTube video IDs after upload.
  - Phase flags: `graph_done`, `audio_done`, `visuals_done`, `render_done`, `upload_done` (or per-asset booleans for shorts).

- On `--resume` / `--run-dir`, **skip** steps when output files exist (e.g. MP3 size &gt; 10 KB, MP4 exists and non-zero).

### 2. CLI phases

Implement via `argparse` on `youtube_publish_scheduler.py` or a small `run_pipeline_stages.py`:

| Phase | Does | Skips if |
|--------|------|----------|
| `graph` | RSS + `pipeline.run` + persist state/scripts to disk | `graph_done` in manifest |
| `audio` | TTS long + five shorts | corresponding `.mp3` exist |
| `visuals` | HF image generation only | dirs populated + `--reuse-images` |
| `render` | `create_video_from_audio_images` only | `.mp4` exists per target |
| `upload` | YouTube only | manifest marks uploaded (optional) |

**Resume:** run `graph` once; later `audio` → `visuals` → `render`; if encode crashes, rerun **`render`** only when inputs still on disk.

### 3. LangGraph checkpoints (`checkpoints.db`)

`pipeline.py` uses `SqliteSaver` + fixed `thread_id`. That does **not** replace user-visible **phase resume** for TTS/render/upload. Prefer **manifest + files** for clarity; keep LangGraph for script generation only unless you add explicit LangGraph resume APIs.

## Laptop safety (RTX 3050 6 GB, ~121 GB free on C:, Ryzen 5)

- **Safe** in the normal sense (no special hardware risk); expect **long high CPU** use and possible **GPU** use if Whisper runs on CUDA.
- **Bottlenecks in this codebase:**
  1. **Whisper** (`video_stitcher.py` → `transcribe_audio_to_subtitles`): `small` model, loaded per render; dominates wall time on long audio.
  2. **H.264 encode** (`write_videofile`): CPU-heavy at 1080p / vertical 1080×1920.

**Quality-preserving knobs:**

- Keep **1080p / 24 fps** for final output.
- Optional env **`FFMPEG_PRESET`** (e.g. `medium` default, `faster` if user accepts slightly less compression efficiency—not lower resolution).
- Optional **`ENCODE_THREADS`** passed to `write_videofile(..., threads=...)`.

**Strong time saver (optional tradeoff):**

- **`SKIP_WHISPER=1`** or `--no-subtitles`: skip Whisper → much faster; **no burned-in captions**; motion + audio unchanged.

**Disk:**

- Prefer **`OUTPUT_ROOT`** on the drive with more free space if split across volumes.
- Keep **~40–50 GB** free on the volume that holds temp + outputs before a full 6-video run.

## Implementation checklist (execute in order)

1. **Manifest schema** – define `run_manifest.json` + read/write helpers under `output/run_{slug}/`.
2. **Refactor** `run_entire_pipeline` into `run_graph_phase`, `run_audio_phase`, `run_visual_phase`, `run_render_phase`, `run_upload_phase` with skip-if-artifact-exists.
3. **CLI** – `--run-dir`, `--from {graph|audio|visuals|render|upload}`, `--resume`; document `OUTPUT_ROOT`, `SKIP_WHISPER`, `FFMPEG_PRESET`, `ENCODE_THREADS`.
4. **`video_stitcher.py`** – optional `enable_subtitles`, `subtitle_model`, `ffmpeg_preset`, `threads` (from env or kwargs) without changing default visual quality when unset.

## Key files (minimal set to touch)

- `youtube_publish_scheduler.py` – orchestration, CLI, manifest.
- `video_stitcher.py` – encode/subtitle/ffmpeg tuning.
- `pipeline.py` – only if persisting/reloading graph outputs to disk for `graph` phase.

## Related implemented behavior (context only)

Long-form writer, 48h RSS ingest, packaging metadata, six MP4 outputs, Ken Burns + centered captions, cinematic HF prompt wrap—these are **already implemented** elsewhere in the repo; this plan is **only** staged resume + laptop tuning.
