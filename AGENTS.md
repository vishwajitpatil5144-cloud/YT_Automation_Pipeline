# Agent instructions (read first)

1. **Before large refactors or pipeline work**, read **[`docs/STAGED_RESUME_AND_LAPTOP_PLAN.md`](docs/STAGED_RESUME_AND_LAPTOP_PLAN.md)**. It is the canonical plan for staged resume (script → audio → visuals → render → upload), run manifests, and laptop-safe encode options. Implement that plan instead of re-exploring the whole tree.

2. **Core modules** (only open if your task needs them):
   - `youtube_publish_scheduler.py` – full run, uploads
   - `pipeline.py` – LangGraph, writer, packaging
   - `youtube_automation_utils.py` – RSS/TTS/HF images
   - `video_stitcher.py` – MoviePy encode, Whisper, Ken Burns

3. **Do not** duplicate long architecture investigation if the doc above already answers it.

4. **Phased runs** are implemented: `python youtube_publish_scheduler.py --help` and `docs/STAGED_RESUME_AND_LAPTOP_PLAN.md` (Implementation status section).
