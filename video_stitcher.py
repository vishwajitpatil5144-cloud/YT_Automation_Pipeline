import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from imageio.v2 import imread
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    TextClip,
    VideoClip,
    concatenate_videoclips,
)
from PIL import Image

try:
    import whisper
except Exception:
    whisper = None


def transcribe_audio_to_subtitles(
    audio_path: str,
    model_name: str = "small",
    language: Optional[str] = None,
) -> List[dict]:
    """Transcribe audio locally into subtitle blocks using Whisper."""
    if whisper is None:
        return []

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, language=language, verbose=False)
    segments = []

    for segment in result.get("segments", []):
        segments.append(
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": segment["text"].strip(),
            }
        )

    return segments


def _punchy_caption_chunks(text: str, max_chars: int = 48) -> List[str]:
    """Split transcript text into short, punchy overlay lines."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text.upper()]
    parts: List[str] = []
    words = text.split()
    cur: List[str] = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur_len + add > max_chars and cur:
            parts.append(" ".join(cur).upper())
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        parts.append(" ".join(cur).upper())
    return parts[:4]


def _segments_to_punchy_captions(segments: List[dict]) -> List[dict]:
    """Expand Whisper segments into shorter on-screen phrases."""
    out: List[dict] = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        start = float(seg["start"])
        end = float(seg["end"])
        chunks = _punchy_caption_chunks(text)
        if not chunks:
            continue
        span = max(0.35, (end - start) / len(chunks))
        t0 = start
        for chunk in chunks:
            out.append({"start": t0, "end": min(end, t0 + span), "text": chunk})
            t0 += span
    return out


def _resolve_image_paths(image_paths: List[str]) -> List[str]:
    resolved: List[str] = []
    for raw in image_paths:
        p = Path(raw)
        if p.is_file():
            resolved.append(str(p.resolve()))
    return resolved


def _ken_burns_clip(
    image_path: str,
    duration: float,
    target_size: Tuple[int, int],
    pan_mode: int = 0,
    fps: int = 24,
) -> VideoClip:
    """Time-varying slow zoom + pan (Ken Burns) across one still image."""
    img = imread(image_path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    ih, iw = img.shape[:2]
    tw, th = target_size

    def make_frame(t: float):
        progress = min(1.0, max(0.0, t / max(duration, 1e-6)))
        ease = progress * progress * (3.0 - 2.0 * progress)
        base = max(tw / iw, th / ih)
        zoom_s, zoom_e = 1.04, 1.12
        z = zoom_s + (zoom_e - zoom_s) * ease
        scale = base * z
        new_w = max(1, int(round(iw * scale)))
        new_h = max(1, int(round(ih * scale)))
        pil = Image.fromarray(img).resize((new_w, new_h), Image.Resampling.LANCZOS)
        scaled = np.asarray(pil)
        sh, sw = scaled.shape[:2]
        dx = max(sw - tw, 0)
        dy = max(sh - th, 0)
        if pan_mode % 4 == 0:
            x1 = int(dx * ease)
            y1 = int(dy * ease * 0.65)
        elif pan_mode % 4 == 1:
            x1 = int(dx * (1.0 - ease))
            y1 = int(dy / 2)
        elif pan_mode % 4 == 2:
            x1 = int(dx / 2)
            y1 = int(dy * ease)
        else:
            x1 = int(dx * (0.35 + 0.45 * ease))
            y1 = int(dy * (1.0 - ease))
        x1 = min(max(x1, 0), max(sw - tw, 0))
        y1 = min(max(y1, 0), max(sh - th, 0))
        return scaled[y1 : y1 + th, x1 : x1 + tw]

    return VideoClip(frame_function=make_frame, duration=duration).with_fps(fps)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


def create_video_from_audio_images(
    audio_path: str,
    image_paths: List[str],
    output_path: str,
    resolution: str = "16:9",
    fps: int = 24,
    subtitle_model: Optional[str] = None,
    subtitle_font: str = "Arial-Bold",
    subtitle_font_size: int = 56,
    subtitle_color: str = "white",
    subtitle_bg_color: str = "black",
    subtitle_bg_opacity: float = 0.55,
    audio_language: Optional[str] = None,
    enable_subtitles: Optional[bool] = None,
    ffmpeg_preset: Optional[str] = None,
    encode_threads: Optional[int] = None,
) -> str:
    """Render a video from an MP3 voiceover and image list with Ken Burns motion and subtitles.

    Env (used when kwargs are None): SKIP_WHISPER / ENABLE_SUBTITLES, WHISPER_MODEL,
    FFMPEG_PRESET, ENCODE_THREADS.
    """
    if resolution == "16:9":
        target_size = (1920, 1080)
    elif resolution == "9:16":
        target_size = (1080, 1920)
    else:
        raise ValueError("resolution must be either '16:9' or '9:16'")

    valid_paths = _resolve_image_paths(image_paths)
    if not valid_paths:
        raise ValueError("No valid image files found; check image_paths.")

    if subtitle_model is None:
        subtitle_model = os.getenv("WHISPER_MODEL", "small")

    if enable_subtitles is None:
        if _env_bool("SKIP_WHISPER", False):
            enable_subtitles = False
        elif os.getenv("ENABLE_SUBTITLES", "").strip() != "":
            enable_subtitles = _env_bool("ENABLE_SUBTITLES", True)
        else:
            enable_subtitles = True

    preset = ffmpeg_preset or os.getenv("FFMPEG_PRESET", "medium")
    threads_raw = encode_threads
    if threads_raw is None and os.getenv("ENCODE_THREADS"):
        try:
            threads_raw = int(os.getenv("ENCODE_THREADS", ""))
        except ValueError:
            threads_raw = None

    output_file = Path(output_path)
    audio_clip = AudioFileClip(audio_path)
    try:
        audio_duration = float(audio_clip.duration or 0.0)
        if audio_duration <= 0:
            raise ValueError("Audio duration is zero or unknown.")

        per_image = audio_duration / len(valid_paths)

        if enable_subtitles:
            raw_segments = transcribe_audio_to_subtitles(
                audio_path, model_name=subtitle_model, language=audio_language
            )
            subtitle_segments = _segments_to_punchy_captions(raw_segments)
        else:
            subtitle_segments = []

        image_clips = []
        for index, image_path in enumerate(valid_paths):
            clip = _ken_burns_clip(
                image_path,
                duration=per_image,
                target_size=target_size,
                pan_mode=index,
                fps=fps,
            )
            image_clips.append(clip)

        video_sequence = concatenate_videoclips(image_clips, method="compose")
        video_sequence = video_sequence.with_audio(audio_clip)

        subtitle_clips = []
        tw, th = target_size
        for segment in subtitle_segments:
            text = segment["text"]
            if not text:
                continue

            subtitle = TextClip(
                text,
                fontsize=subtitle_font_size,
                font=subtitle_font,
                color=subtitle_color,
                method="caption",
                size=(int(tw * 0.85), None),
                align="center",
            )

            background = subtitle.on_color(
                size=(subtitle.w + 48, subtitle.h + 28),
                color=subtitle_bg_color,
                pos=("center", "center"),
                col_opacity=subtitle_bg_opacity,
            )

            subtitle_clip = background.with_position(("center", "center"))
            subtitle_clip = subtitle_clip.with_start(segment["start"]).with_end(segment["end"])
            subtitle_clips.append(subtitle_clip)

        final = CompositeVideoClip([video_sequence, *subtitle_clips], size=target_size)
        final = final.with_duration(audio_duration)
        final = final.with_fps(fps)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        write_kw = {
            "filename": str(output_file),
            "codec": "libx264",
            "audio_codec": "aac",
            "fps": fps,
            "preset": preset,
        }
        if threads_raw is not None:
            write_kw["threads"] = threads_raw
        final.write_videofile(**write_kw)
    finally:
        audio_clip.close()

    return str(output_file)


if __name__ == "__main__":
    sample_audio = "./output/demo_voiceover.mp3"
    sample_images = [
        "./assets/visuals/background_1.png",
        "./assets/visuals/background_2.png",
        "./assets/visuals/background_3.png",
    ]
    output_file = "./output/final_video_16_9.mp4"
    print(f"Rendering video to {output_file}...")
    create_video_from_audio_images(
        audio_path=sample_audio,
        image_paths=sample_images,
        output_path=output_file,
        resolution="16:9",
    )
    print("Render complete.")
