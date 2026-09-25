import argparse
import json
import os
import re
import time
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRETS_FILE = os.getenv("YOUTUBE_CLIENT_SECRETS_FILE", "client_secrets.json")
TOKEN_FILE = os.getenv("YOUTUBE_TOKEN_FILE", "youtube_token.json")
DEFAULT_TTS_VOICE = os.getenv("TTS_VOICE", "en-US-GuyNeural")

PHASE_ORDER = ["graph", "audio", "visuals", "render", "upload"]
MANIFEST_NAME = "run_manifest.json"
SCRIPTS_NAME = "scripts.json"
STATE_NAME = "pipeline_state.json"
MIN_AUDIO_BYTES = 1024
MIN_VIDEO_BYTES = 50_000


def _mask_secret(value: Optional[str]) -> str:
    if not value:
        return "missing"
    if len(value) <= 8:
        return "set"
    return f"{value[:4]}...{value[-4:]}"


def preflight_check() -> None:
    """Validate local environment and dependencies before running the full pipeline."""
    optional_env = [
        "TELEGRAM_TOKEN",
        "TELEGRAM_CHAT_ID",
        "HF_TOKEN",
        "HUGGINGFACE_API_TOKEN",
        "SERPAPI_API_KEY",
        "RESEARCH_SEARCH_PROVIDER",
        "THUMBNAIL_IMAGE_MODEL",
    ]

    missing: List[str] = []

    required_files = [
        CLIENT_SECRETS_FILE,
        "pipeline.py",
        "youtube_automation_utils.py",
        "video_stitcher.py",
    ]
    for file_name in required_files:
        if not Path(file_name).exists():
            missing.append(file_name)

    required_modules = [
        "moviepy",
        "requests",
        "bs4",
        "googleapiclient",
        "pydantic",
        "langgraph",
        "edge_tts",
        "telegram",
    ]
    for module_name in required_modules:
        if importlib.util.find_spec(module_name) is None:
            missing.append(f"python module: {module_name}")

    if missing:
        raise RuntimeError(
            "Preflight failed. Missing requirements:\n- " + "\n- ".join(missing)
        )

    print("Preflight OK: required files, env vars, and core modules are present.")
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GROQ_API_KEY")):
        print("Info: No LLM API key is configured; the pipeline will use a local fallback generator.")
    print("Configured keys:")
    print(f"- GOOGLE_API_KEY: {_mask_secret(os.getenv('GOOGLE_API_KEY'))}")
    print(f"- GROQ_API_KEY: {_mask_secret(os.getenv('GROQ_API_KEY'))}")
    print(f"- TELEGRAM_TOKEN: {_mask_secret(os.getenv('TELEGRAM_TOKEN'))}")
    print(f"- TELEGRAM_CHAT_ID: {_mask_secret(os.getenv('TELEGRAM_CHAT_ID'))}")

    if not (os.getenv("TELEGRAM_TOKEN") and os.getenv("TELEGRAM_CHAT_ID")):
        print("Info: Telegram approval is skipped by default unless SKIP_TELEGRAM_APPROVAL is disabled.")

    if not (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")):
        print("Warning: HF_TOKEN/HUGGINGFACE_API_TOKEN not set. Thumbnail/image generation may fail.")
    if not os.getenv("SERPAPI_API_KEY"):
        print("Warning: SERPAPI_API_KEY not set. Researcher will rely on DuckDuckGo fallback.")

    provider = os.getenv("RESEARCH_SEARCH_PROVIDER", "auto")
    print(f"- RESEARCH_SEARCH_PROVIDER: {provider}")
    print(f"- THUMBNAIL_IMAGE_MODEL: {os.getenv('THUMBNAIL_IMAGE_MODEL', 'stabilityai/stable-diffusion-2')}")


def get_authenticated_service(
    client_secrets_file: str = CLIENT_SECRETS_FILE,
    token_file: str = TOKEN_FILE,
    scopes: List[str] = SCOPES,
):
    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
                creds = flow.run_local_server(port=8080)
            except Exception as exc:
                message = str(exc)
                if "org_internal" in message:
                    raise RuntimeError(
                        "Google blocked this OAuth client with org_internal. "
                        "Use an OAuth client that is allowed for external users, "
                        "or sign in with a Google Workspace account in the same organization as the client."
                    ) from exc
                raise

        with open(token_file, "w", encoding="utf-8") as token:
            token.write(creds.to_json())

    return build("youtube", "v3", credentials=creds)


def upload_video_to_youtube(
    youtube_service,
    video_file_path: str,
    title: str,
    description: str,
    tags: List[str],
    thumbnail_path: Optional[str] = None,
    privacy_status: str = "public",
    category_id: str = "22",
) -> dict:
    """Upload a video file to YouTube and return the API response."""
    media_body = MediaFileUpload(video_file_path, chunksize=-1, resumable=True)
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id,
        },
        "status": {"privacyStatus": privacy_status},
    }

    request = youtube_service.videos().insert(part="snippet,status", body=body, media_body=media_body)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Upload progress: {int(status.progress() * 100)}%")

    print(f"Upload complete. Video ID: {response['id']}")

    if not thumbnail_path:
        print("Info: no thumbnail_path provided; skipping custom thumbnail upload.")
    elif not Path(thumbnail_path).exists():
        print(f"Warning: thumbnail file missing at {thumbnail_path}; skipping thumbnails.set.")

    if thumbnail_path and Path(thumbnail_path).exists():
        try:
            thumb_media = MediaFileUpload(thumbnail_path)
            youtube_service.thumbnails().set(videoId=response["id"], media_body=thumb_media).execute()
            print(f"Thumbnail uploaded: {thumbnail_path}")
        except HttpError as exc:
            print(f"Thumbnail upload failed: {exc}")

    return response


def _output_root() -> Path:
    return Path(os.getenv("OUTPUT_ROOT", "output")).resolve()


def _slug_from_run_dir(run_dir: Path) -> str:
    m = re.match(r"^run_(.+)$", run_dir.name)
    if m:
        return m.group(1)
    raise ValueError(
        f"Run directory name must be run_<slug>, got {run_dir.name!r}. Example: output/run_20260112_153045"
    )


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / MANIFEST_NAME
    if not path.is_file():
        return {}
    return _load_json(path)


def save_manifest(run_dir: Path, manifest: Dict[str, Any]) -> None:
    _save_json(run_dir / MANIFEST_NAME, manifest)


def _file_ready(path: Path, min_bytes: int) -> bool:
    return path.is_file() and path.stat().st_size >= min_bytes


def _collect_png_paths(directory: Path) -> List[str]:
    if not directory.is_dir():
        return []
    paths = sorted(directory.glob("*.png"))
    return [str(p.resolve()) for p in paths]


def _phase_slice(from_phase: str, to_phase: str) -> List[str]:
    if from_phase not in PHASE_ORDER or to_phase not in PHASE_ORDER:
        raise ValueError(f"Phases must be one of {PHASE_ORDER}")
    i0, i1 = PHASE_ORDER.index(from_phase), PHASE_ORDER.index(to_phase)
    if i0 > i1:
        raise ValueError("--from must not be after --to")
    return PHASE_ORDER[i0 : i1 + 1]


def _ensure_manifest_image_lists(manifest: Dict[str, Any], run_dir: Path) -> None:
    paths = manifest.setdefault("paths", {})
    long_dir = run_dir / paths.get("main_visuals_dir", "visuals_long")
    if not paths.get("main_image_paths") and long_dir.is_dir():
        paths["main_image_paths"] = [
            p.relative_to(run_dir).as_posix() for p in sorted(long_dir.glob("*.png"))
        ]
    shorts = paths.setdefault("short_image_paths", {})
    for i in range(1, 6):
        if shorts.get(str(i)):
            continue
        sd = run_dir / f"visuals_short_{i}"
        if sd.is_dir():
            shorts[str(i)] = [
                p.relative_to(run_dir).as_posix() for p in sorted(sd.glob("*.png"))
            ]


def run_phased_pipeline(
    from_phase: str = "graph",
    to_phase: str = "upload",
    run_dir: Optional[Path] = None,
    resume: bool = False,
    reuse_images: bool = False,
    no_prune: bool = False,
) -> Path:
    """Run pipeline phases; returns the run directory used."""
    from pipeline import AINewsState, AIChainPipeline
    from youtube_automation_utils import (
        generate_background_images_hf,
        scrape_trending_ai_news_last_48h,
        text_to_speech_edge,
    )
    from video_stitcher import create_video_from_audio_images

    preflight_check()
    if resume and run_dir is None:
        raise ValueError("--resume requires --run-dir")
    if run_dir is None and from_phase != "graph":
        raise ValueError(
            f"--from {from_phase!r} requires --run-dir pointing at an existing output/run_<slug> folder."
        )

    phases = _phase_slice(from_phase, to_phase)
    print(f"Phases to run: {', '.join(phases)}")

    out_root = _output_root()
    out_root.mkdir(parents=True, exist_ok=True)

    if run_dir is None:
        slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = out_root / f"run_{slug}"
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest: Dict[str, Any] = {
            "run_slug": slug,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "graph_done": False,
            "audio_done": False,
            "visuals_done": False,
            "render_done": False,
            "upload_done": False,
            "paths": {},
            "youtube": {"main_video_id": None, "short_video_ids": []},
        }
        save_manifest(run_dir, manifest)
    else:
        run_dir = run_dir.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        manifest = load_manifest(run_dir)
        if not manifest:
            slug = _slug_from_run_dir(run_dir)
            manifest = {
                "run_slug": slug,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "graph_done": False,
                "audio_done": False,
                "visuals_done": False,
                "render_done": False,
                "upload_done": False,
                "paths": {},
                "youtube": {"main_video_id": None, "short_video_ids": []},
            }
            save_manifest(run_dir, manifest)
        elif "run_slug" not in manifest:
            manifest["run_slug"] = _slug_from_run_dir(run_dir)
            save_manifest(run_dir, manifest)

    slug = str(manifest["run_slug"])
    print(f"Run directory: {run_dir}")

    long_prompts = [
        "Neon megacity skyline at night with holographic data streams and AI circuitry",
        "Cyberpunk command center with panoramic glass overlooking a storm of digital particles",
        "Macro close-up of a futuristic AI processor die with cyan and magenta light trails",
    ]

    pipeline_state: Optional[AINewsState] = None
    scripts: Dict[str, Any] = {}

    for phase in phases:
        print(f"\n--- Phase: {phase} ---")

        if phase == "graph":
            if (
                manifest.get("graph_done")
                and resume
                and (run_dir / SCRIPTS_NAME).is_file()
                and (run_dir / STATE_NAME).is_file()
            ):
                print("Skipping graph (resume: scripts and state already present).")
                scripts = _load_json(run_dir / SCRIPTS_NAME)
                pipeline_state = AINewsState.model_validate(_load_json(run_dir / STATE_NAME))
            else:
                raw_news = scrape_trending_ai_news_last_48h(count=7)
                news_items = [item["title"] for item in raw_news[:7] if item.get("title")]
                if not news_items:
                    raise RuntimeError("No news headlines returned from ingestion.")
                pipeline = AIChainPipeline()
                pipeline.run(news_items)
                pipeline_state = pipeline.state
                script_text = (pipeline_state.long_form_script or "").strip()
                if not script_text:
                    raise RuntimeError("Pipeline did not produce a long_form_script.")
                if len(pipeline_state.short_scripts) != 5:
                    raise RuntimeError(
                        f"Expected 5 short_scripts from pipeline, got {len(pipeline_state.short_scripts)}."
                    )
                scripts = {
                    "long_form_script": pipeline_state.long_form_script,
                    "short_scripts": list(pipeline_state.short_scripts),
                    "news_items": news_items,
                }
                _save_json(run_dir / SCRIPTS_NAME, scripts)
                _save_json(run_dir / STATE_NAME, pipeline_state.model_dump(mode="json"))
                manifest["graph_done"] = True
                manifest["paths"]["scripts_json"] = SCRIPTS_NAME
                manifest["paths"]["pipeline_state_json"] = STATE_NAME
                save_manifest(run_dir, manifest)

        elif phase == "audio":
            if not (run_dir / SCRIPTS_NAME).is_file():
                raise FileNotFoundError(f"Missing {SCRIPTS_NAME}; run --from graph first.")
            scripts = _load_json(run_dir / SCRIPTS_NAME)
            main_voice = run_dir / "longform_voiceover.mp3"
            if _file_ready(main_voice, MIN_AUDIO_BYTES):
                print(f"Skip TTS (exists): {main_voice.name}")
            else:
                text_to_speech_edge(
                    scripts["long_form_script"], str(main_voice), voice=DEFAULT_TTS_VOICE
                )
            for i in range(1, 6):
                vp = run_dir / f"short_voiceover_{i}.mp3"
                if _file_ready(vp, MIN_AUDIO_BYTES):
                    print(f"Skip TTS (exists): {vp.name}")
                else:
                    text_to_speech_edge(
                        scripts["short_scripts"][i - 1], str(vp), voice=DEFAULT_TTS_VOICE
                    )
            manifest["audio_done"] = True
            manifest["paths"]["main_voice"] = "longform_voiceover.mp3"
            manifest["paths"]["short_voices"] = [f"short_voiceover_{i}.mp3" for i in range(1, 6)]
            save_manifest(run_dir, manifest)

        elif phase == "visuals":
            if not (run_dir / SCRIPTS_NAME).is_file():
                raise FileNotFoundError(f"Missing {SCRIPTS_NAME}; run graph first.")
            long_dir = run_dir / "visuals_long"
            has_long = long_dir.is_dir() and any(long_dir.glob("*.png"))
            if reuse_images and has_long:
                print("Reuse long-form images (directory already has PNGs).")
            else:
                long_dir.mkdir(parents=True, exist_ok=True)
                generate_background_images_hf(long_prompts, output_dir=str(long_dir))
            for i in range(1, 6):
                sd = run_dir / f"visuals_short_{i}"
                has_s = sd.is_dir() and any(sd.glob("*.png"))
                if reuse_images and has_s:
                    print(f"Reuse short {i} images.")
                    continue
                sd.mkdir(parents=True, exist_ok=True)
                if i % 2 == 1:
                    sprompts = [
                        "Vertical cyberpunk alley with rain reflections and towering holographic AI billboards",
                    ]
                else:
                    sprompts = [
                        "Futuristic vertical lab with glowing server racks and particle energy burst",
                        "Macro glowing neural pathways in vertical cinematic composition",
                    ]
                generate_background_images_hf(sprompts, output_dir=str(sd))
            manifest["visuals_done"] = True
            manifest["paths"]["main_visuals_dir"] = "visuals_long"
            manifest["paths"]["short_visuals_dirs"] = [f"visuals_short_{i}" for i in range(1, 6)]
            paths = _collect_png_paths(long_dir)
            manifest["paths"]["main_image_paths"] = [
                Path(p).relative_to(run_dir).as_posix() for p in paths
            ]
            short_map: Dict[str, List[str]] = {}
            for i in range(1, 6):
                sd = run_dir / f"visuals_short_{i}"
                short_map[str(i)] = [
                    p.relative_to(run_dir).as_posix() for p in sorted(sd.glob("*.png"))
                ]
            manifest["paths"]["short_image_paths"] = short_map
            save_manifest(run_dir, manifest)

        elif phase == "render":
            if not (run_dir / SCRIPTS_NAME).is_file():
                raise FileNotFoundError(f"Missing {SCRIPTS_NAME}; run graph first.")
            scripts = _load_json(run_dir / SCRIPTS_NAME)
            _ensure_manifest_image_lists(manifest, run_dir)
            main_voice = run_dir / "longform_voiceover.mp3"
            if not _file_ready(main_voice, MIN_AUDIO_BYTES):
                raise FileNotFoundError("Long voiceover missing; run audio phase.")
            main_imgs = [str(run_dir / p) for p in manifest["paths"].get("main_image_paths", [])]
            if not main_imgs:
                main_imgs = _collect_png_paths(run_dir / "visuals_long")
            if not main_imgs:
                raise FileNotFoundError("No long-form PNGs; run visuals phase.")
            long_mp4 = run_dir / "longform.mp4"
            if _file_ready(long_mp4, MIN_VIDEO_BYTES):
                print(f"Skip render (exists): {long_mp4.name}")
            else:
                create_video_from_audio_images(
                    audio_path=str(main_voice),
                    image_paths=main_imgs,
                    output_path=str(long_mp4),
                    resolution="16:9",
                )
            for i in range(1, 6):
                vp = run_dir / f"short_voiceover_{i}.mp3"
                if not _file_ready(vp, MIN_AUDIO_BYTES):
                    raise FileNotFoundError(f"Short {i} voiceover missing; run audio phase.")
                rels = manifest["paths"].get("short_image_paths", {}).get(str(i), [])
                simgs = [str(run_dir / p) for p in rels] if rels else _collect_png_paths(
                    run_dir / f"visuals_short_{i}"
                )
                if not simgs:
                    raise FileNotFoundError(f"No PNGs for short {i}; run visuals phase.")
                short_mp4 = run_dir / f"short_{i}.mp4"
                if _file_ready(short_mp4, MIN_VIDEO_BYTES):
                    print(f"Skip render (exists): {short_mp4.name}")
                else:
                    create_video_from_audio_images(
                        audio_path=str(vp),
                        image_paths=simgs,
                        output_path=str(short_mp4),
                        resolution="9:16",
                    )
            manifest["render_done"] = True
            manifest["paths"]["main_mp4"] = "longform.mp4"
            manifest["paths"]["short_mp4s"] = [f"short_{i}.mp4" for i in range(1, 6)]
            save_manifest(run_dir, manifest)

        elif phase == "upload":
            state_path = run_dir / STATE_NAME
            if not state_path.is_file():
                raise FileNotFoundError(f"Missing {STATE_NAME}; run graph phase.")
            pipeline_state = AINewsState.model_validate(_load_json(state_path))
            scripts = _load_json(run_dir / SCRIPTS_NAME)
            news_items = scripts.get("news_items") or []
            long_mp4 = run_dir / "longform.mp4"
            if not _file_ready(long_mp4, MIN_VIDEO_BYTES):
                raise FileNotFoundError("longform.mp4 missing; run render phase.")

            youtube = get_authenticated_service()
            main_title = pipeline_state.youtube_title or f"AI News Update | {news_items[0] if news_items else 'AI'}"
            main_description = pipeline_state.youtube_description or (
                "AI news automation.\n\nSources:\n" + "\n".join(f"- {t}" for t in news_items)
            )
            main_tags = pipeline_state.youtube_tags or [
                "AI news",
                "technology",
                "automation",
                "machine learning",
            ]
            main_thumb = pipeline_state.thumbnail_image_path
            resp = upload_video_to_youtube(
                youtube_service=youtube,
                video_file_path=str(long_mp4),
                title=main_title[:100],
                description=main_description,
                tags=list(main_tags)[:30],
                thumbnail_path=main_thumb,
            )
            manifest["youtube"]["main_video_id"] = resp.get("id")
            short_tags_base = (pipeline_state.youtube_tags or [])[:8] or [
                "AI news",
                "shorts",
                "technology",
                "automation",
            ]
            short_ids: List[str] = []
            for i in range(1, 6):
                short_mp4 = run_dir / f"short_{i}.mp4"
                if not _file_ready(short_mp4, MIN_VIDEO_BYTES):
                    raise FileNotFoundError(f"short_{i}.mp4 missing; run render phase.")
                stitle = f"AI Short {i} | {news_items[0] if news_items else 'AI'}"[:100]
                sdesc = (
                    f"AI news short {i}.\n\n{(pipeline_state.youtube_description or '')}"[:4900]
                )
                stags: List[str] = []
                seen_tag: set = set()
                for t in short_tags_base + [f"short{i}", "vertical", "AI"]:
                    if t and t not in seen_tag:
                        seen_tag.add(t)
                        stags.append(t)
                    if len(stags) >= 15:
                        break
                sresp = upload_video_to_youtube(
                    youtube_service=youtube,
                    video_file_path=str(short_mp4),
                    title=stitle,
                    description=sdesc,
                    tags=stags,
                    thumbnail_path=None,
                )
                short_ids.append(str(sresp.get("id", "")))
                print(f"Short {i} uploaded.")
            manifest["youtube"]["short_video_ids"] = short_ids
            manifest["upload_done"] = True
            save_manifest(run_dir, manifest)

    if "upload" in phases and manifest.get("upload_done") and not no_prune:
        prune_old_assets()

    return run_dir


def run_entire_pipeline() -> None:
    """Backward-compatible: full pipeline with a fresh run directory under OUTPUT_ROOT."""
    run_phased_pipeline(
        from_phase="graph",
        to_phase="upload",
        run_dir=None,
        resume=False,
        reuse_images=False,
        no_prune=False,
    )


def prune_old_assets(hours_old: int = 48) -> None:
    """Delete .mp3, .png, and .mp4 files older than the specified hours from ./assets and ./output folders."""
    cutoff_time = time.time() - (hours_old * 3600)
    folders = ["./assets", "./output"]
    extensions = [".mp3", ".png", ".mp4"]
    
    for folder in folders:
        if not os.path.exists(folder):
            continue
        for root, dirs, files in os.walk(folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    try:
                        if os.path.getmtime(file_path) < cutoff_time:
                            os.remove(file_path)
                            print(f"Pruned old file: {file_path}")
                    except OSError as e:
                        print(f"Error pruning {file_path}: {e}")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YouTube automation: phased pipeline (see docs/STAGED_RESUME_AND_LAPTOP_PLAN.md)."
    )
    parser.add_argument(
        "--from",
        dest="from_phase",
        default="graph",
        choices=PHASE_ORDER,
        help="First phase to execute (inclusive).",
    )
    parser.add_argument(
        "--to",
        dest="to_phase",
        default="upload",
        choices=PHASE_ORDER,
        help="Last phase to execute (inclusive).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Existing run folder, e.g. output/run_20260112_153045 (required if --from is not graph).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip graph when scripts/state already exist in --run-dir.",
    )
    parser.add_argument(
        "--reuse-images",
        action="store_true",
        help="Skip HF image generation when PNGs already exist in each visuals_* folder.",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Do not delete old assets from ./output and ./assets after a successful upload phase.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    rd = Path(args.run_dir) if args.run_dir else None
    run_phased_pipeline(
        from_phase=args.from_phase,
        to_phase=args.to_phase,
        run_dir=rd,
        resume=args.resume,
        reuse_images=args.reuse_images,
        no_prune=args.no_prune,
    )
