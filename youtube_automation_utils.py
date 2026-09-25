import asyncio
import base64
import os
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:
    retry = None
    stop_after_attempt = None
    wait_exponential = None

try:
    import edge_tts
except ImportError:  # pragma: no cover
    edge_tts = None

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


_PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j+P0AAAAASUVORK5CYII="
)

_GOOGLE_NEWS_AI_RSS = (
    "https://news.google.com/rss/search?"
    "q=artificial+intelligence+OR+machine+learning+OR+LLM+OR+GPT+OR+OpenAI+OR+Anthropic"
    "&hl=en-US&gl=US&ceid=US:en"
)

_AI_TOPIC_RE = re.compile(
    r"\b(ai|ml|llm|gpt|openai|anthropic|gemini|claude|nvidia|cuda|neural|"
    r"transformer|inference|model|benchmark|chip|tensor|agent|sora|"
    r"diffusion|embedding|alignment|safety)\b",
    re.I,
)

_SPICE_RE = re.compile(
    r"\b(ban|banned|lawsuit|leak|leaked|fired|regulation|breakthrough|sota|"
    r"surpass|vulnerability|exploit|copyright|controvers|scandal|warning|"
    r"trillion|billion|parameters|open weights|closed source|benchmark)\b",
    re.I,
)


def _wrap_cinematic_tech_prompt(user_prompt: str) -> str:
    """Prefix style tokens for HF image models: cinematic cyberpunk tech backgrounds."""
    style = (
        "Ultra-detailed cinematic wide shot, volumetric lighting and subtle lens bloom, "
        "high-end cyberpunk and futuristic technology aesthetic, neon accents on dark surfaces, "
        "depth of field, dramatic contrast, 8k render quality, film grain, "
        "no readable text, no logos, no watermarks. Subject: "
    )
    suffix = " — professional broadcast motion-graphics background, not a UI screenshot."
    return f"{style}{user_prompt.strip()}{suffix}"


def _parse_rss_datetime(pub_raw: str) -> Optional[datetime]:
    if not pub_raw:
        return None
    try:
        dt = parsedate_to_datetime(pub_raw.strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def _ai_headline_relevant(title: str, summary: str) -> bool:
    blob = f"{title} {summary}"
    return bool(_AI_TOPIC_RE.search(blob))


def _score_spice_and_impact(title: str, summary: str, published: Optional[datetime]) -> float:
    blob = f"{title} {summary}"
    score = 0.0
    for m in _SPICE_RE.finditer(blob):
        score += 1.2
    if re.search(r"\b\d+%|\b\d+\s*(billion|million|trillion|B|M|T)\b", blob, re.I):
        score += 1.0
    if published:
        age_h = (datetime.now(timezone.utc) - published).total_seconds() / 3600.0
        if age_h < 6:
            score += 2.0
        elif age_h < 24:
            score += 1.0
        elif age_h < 48:
            score += 0.3
    return score


def _fetch_rss_items_with_pubdate(rss_url: str, max_items: int) -> List[Dict[str, str]]:
    """Return items with title, url, summary, published_raw (RFC822 string)."""
    response = requests.get(
        rss_url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (compatible; yt-news-ingest/1.0)"},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "xml")
    items: List[Dict[str, str]] = []
    for item in soup.find_all("item")[:max_items]:
        title = item.title.get_text(strip=True) if item.title else ""
        link = item.link.get_text(strip=True) if item.link else ""
        summary = item.description.get_text(strip=True) if item.description else ""
        pub_el = item.find("pubDate")
        published_raw = pub_el.get_text(strip=True) if pub_el else ""
        items.append(
            {
                "title": title,
                "url": link,
                "summary": summary,
                "published_raw": published_raw,
            }
        )
    return items


def scrape_trending_ai_news_last_48h(
    count: int = 7,
    rss_url: str = _GOOGLE_NEWS_AI_RSS,
    candidate_pool: int = 60,
) -> List[Dict[str, str]]:
    """Top AI-trending stories from RSS with pubDate within the last 48 hours, ranked by spice/impact."""
    if retry is not None:
        return _scrape_trending_ai_news_last_48h_with_retry(count, rss_url, candidate_pool)
    return _scrape_trending_ai_news_last_48h_no_retry(count, rss_url, candidate_pool)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
def _scrape_trending_ai_news_last_48h_with_retry(
    count: int, rss_url: str, candidate_pool: int
) -> List[Dict[str, str]]:
    return _scrape_trending_ai_news_last_48h_no_retry(count, rss_url, candidate_pool)


def _scrape_trending_ai_news_last_48h_no_retry(
    count: int, rss_url: str, candidate_pool: int,
) -> List[Dict[str, str]]:
    raw_items = _fetch_rss_items_with_pubdate(rss_url, max_items=candidate_pool)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
    scored: List[Tuple[float, Dict[str, str]]] = []

    for row in raw_items:
        pub = _parse_rss_datetime(row.get("published_raw", ""))
        if pub is None or pub < cutoff:
            continue
        title, summary = row.get("title", ""), row.get("summary", "")
        if not _ai_headline_relevant(title, summary):
            continue
        spice = _score_spice_and_impact(title, summary, pub)
        out = {"title": title, "url": row.get("url", ""), "summary": summary}
        scored.append((spice, out))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        for row in raw_items:
            title, summary = row.get("title", ""), row.get("summary", "")
            if not title:
                continue
            pub = _parse_rss_datetime(row.get("published_raw", ""))
            if _ai_headline_relevant(title, summary):
                scored.append(
                    (
                        _score_spice_and_impact(title, summary, pub),
                        {"title": title, "url": row.get("url", ""), "summary": summary},
                    )
                )
    if not scored:
        for row in raw_items[:count]:
            scored.append(
                (
                    0.0,
                    {
                        "title": row.get("title", "AI news") or "AI news",
                        "url": row.get("url", ""),
                        "summary": row.get("summary", ""),
                    },
                )
            )
    return [item for _, item in scored[:count]]


def _write_placeholder_images(prompts: List[str], output_dir: str, prefix: str = "placeholder") -> List[str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    is_short_canvas = "short" in output_path.name.lower() or "short" in output_path.as_posix().lower()
    canvas_size = (1080, 1920) if is_short_canvas else (1920, 1080)

    saved_files: List[str] = []
    for index, prompt in enumerate(prompts, start=1):
        filename = output_path / f"{prefix}_{index}.png"

        if Image is not None and ImageDraw is not None:
            width, height = canvas_size
            seed = sum(ord(ch) for ch in prompt) + (index * 37)
            c1 = ((seed * 3) % 256, (seed * 5) % 256, (seed * 7) % 256)
            c2 = (((seed + 97) * 11) % 256, ((seed + 53) * 13) % 256, ((seed + 19) * 17) % 256)

            image = Image.new("RGB", (width, height), c1)
            draw = ImageDraw.Draw(image)

            for y in range(height):
                blend = y / max(height - 1, 1)
                r = int(c1[0] * (1 - blend) + c2[0] * blend)
                g = int(c1[1] * (1 - blend) + c2[1] * blend)
                b = int(c1[2] * (1 - blend) + c2[2] * blend)
                draw.line((0, y, width, y), fill=(r, g, b))

            caption = (prompt[:80] + "...") if len(prompt) > 80 else prompt
            caption_bar_h = int(height * 0.12)
            draw.rectangle((0, height - caption_bar_h, width, height), fill=(0, 0, 0))
            draw.text((24, height - caption_bar_h + 18), caption, fill=(255, 255, 255))
            image.save(filename)
        else:
            with open(filename, "wb") as image_file:
                image_file.write(_PLACEHOLDER_PNG)

        saved_files.append(str(filename))

    return saved_files


def scrape_top_hacker_news(count: int = 5) -> List[Dict[str, str]]:
    """Scrape the top Hacker News stories and return up to `count` items."""
    if retry is not None:
        return _scrape_top_hacker_news(count)
    else:
        return _scrape_top_hacker_news_no_retry(count)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
def _scrape_top_hacker_news(count: int = 5) -> List[Dict[str, str]]:
    url = "https://news.ycombinator.com/"
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    items = []

    for row in soup.select("tr.athing")[:count]:
        title = row.select_one("span.titleline > a") or row.select_one("a.storylink")
        link = title["href"] if title else None
        subtitle = row.find_next_sibling("tr")
        site = subtitle.select_one("span.sitestr")
        items.append(
            {
                "title": title.get_text(strip=True) if title else "",
                "url": link or "",
                "source": site.get_text(strip=True) if site else "Hacker News",
            }
        )

    return items


def _scrape_top_hacker_news_no_retry(count: int = 5) -> List[Dict[str, str]]:
    url = "https://news.ycombinator.com/"
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    items = []

    for row in soup.select("tr.athing")[:count]:
        title = row.select_one("span.titleline > a") or row.select_one("a.storylink")
        link = title["href"] if title else None
        subtitle = row.find_next_sibling("tr")
        site = subtitle.select_one("span.sitestr")
        items.append(
            {
                "title": title.get_text(strip=True) if title else "",
                "url": link or "",
                "source": site.get_text(strip=True) if site else "Hacker News",
            }
        )

    return items


def scrape_ai_news_rss(rss_url: str, count: int = 5) -> List[Dict[str, str]]:
    """Fetch a generic RSS feed and return the top items."""
    if retry is not None:
        return _scrape_ai_news_rss(rss_url, count)
    else:
        return _scrape_ai_news_rss_no_retry(rss_url, count)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
def _scrape_ai_news_rss(rss_url: str, count: int = 5) -> List[Dict[str, str]]:
    response = requests.get(rss_url, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "xml")
    items = []
    for item in soup.find_all("item")[:count]:
        title = item.title.get_text(strip=True) if item.title else ""
        link = item.link.get_text(strip=True) if item.link else ""
        summary = item.description.get_text(strip=True) if item.description else ""
        items.append({"title": title, "url": link, "summary": summary})

    return items


def _scrape_ai_news_rss_no_retry(rss_url: str, count: int = 5) -> List[Dict[str, str]]:
    response = requests.get(rss_url, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "xml")
    items = []
    for item in soup.find_all("item")[:count]:
        title = item.title.get_text(strip=True) if item.title else ""
        link = item.link.get_text(strip=True) if item.link else ""
        summary = item.description.get_text(strip=True) if item.description else ""
        items.append({"title": title, "url": link, "summary": summary})

    return items


def scrape_trending_ai_news(
    source: str = "hackernews",
    rss_url: Optional[str] = None,
    count: int = 5,
) -> List[Dict[str, str]]:
    """Choose Hacker News, Google News AI (last 48h), or a custom RSS feed."""
    lowered = source.lower()
    if lowered in {"google_ai_48h", "ai_48h", "trending_ai_48h"}:
        return scrape_trending_ai_news_last_48h(count=count, rss_url=rss_url or _GOOGLE_NEWS_AI_RSS)
    if lowered == "hackernews":
        return scrape_top_hacker_news(count=count)

    if not rss_url:
        raise ValueError("rss_url must be provided when source is not hackernews or google_ai_48h")

    return scrape_ai_news_rss(rss_url=rss_url, count=count)


def web_search_research(
    query: str,
    max_results: int = 5,
    provider: str = "auto",
    serpapi_key: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Search the web for technical context using SerpApi or DuckDuckGo.

    Provider priority:
    1) SerpApi when provider is "serpapi" or when provider is "auto" and key is available.
    2) DuckDuckGo when provider is "duckduckgo" or as fallback for "auto".
    """
    normalized_provider = provider.lower().strip()
    if normalized_provider not in {"auto", "serpapi", "duckduckgo"}:
        raise ValueError("provider must be one of: auto, serpapi, duckduckgo")

    key = serpapi_key or os.getenv("SERPAPI_API_KEY")

    if normalized_provider == "serpapi" or (normalized_provider == "auto" and key):
        try:
            return _serpapi_search(query=query, max_results=max_results, serpapi_key=key)
        except Exception:
            if normalized_provider == "serpapi":
                raise

    if normalized_provider in {"auto", "duckduckgo"}:
        return _duckduckgo_search(query=query, max_results=max_results)

    return []


def format_search_results_for_prompt(results: List[Dict[str, str]]) -> str:
    """Render search results into a prompt-friendly source list."""
    if not results:
        return "No external web search results were found."

    formatted: List[str] = []
    for index, item in enumerate(results, start=1):
        title = item.get("title", "Untitled result")
        snippet = item.get("snippet", "")
        url = item.get("url", "")
        source = item.get("source", "web")
        formatted.append(
            f"{index}. {title}\n"
            f"   Source: {source}\n"
            f"   URL: {url}\n"
            f"   Snippet: {snippet}"
        )
    return "\n".join(formatted)


def _serpapi_search(
    query: str,
    max_results: int = 5,
    serpapi_key: Optional[str] = None,
) -> List[Dict[str, str]]:
    if retry is not None:
        return _serpapi_search_with_retry(query, max_results, serpapi_key)
    return _serpapi_search_no_retry(query, max_results, serpapi_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
def _serpapi_search_with_retry(
    query: str,
    max_results: int = 5,
    serpapi_key: Optional[str] = None,
) -> List[Dict[str, str]]:
    return _serpapi_search_no_retry(query, max_results, serpapi_key)


def _serpapi_search_no_retry(
    query: str,
    max_results: int = 5,
    serpapi_key: Optional[str] = None,
) -> List[Dict[str, str]]:
    key = serpapi_key or os.getenv("SERPAPI_API_KEY")
    if not key:
        raise RuntimeError("SERPAPI_API_KEY is required for SerpApi search.")

    params = {
        "engine": "google",
        "q": query,
        "api_key": key,
        "num": max_results,
        "hl": "en",
        "gl": "us",
    }
    response = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()

    organic_results = payload.get("organic_results", [])
    results: List[Dict[str, str]] = []
    for item in organic_results[:max_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": "SerpApi",
            }
        )
    return results


def _duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    if DDGS is None:
        raise ImportError(
            "duckduckgo-search is not installed. Install it with `pip install duckduckgo-search`."
        )

    if retry is not None:
        return _duckduckgo_search_with_retry(query, max_results)
    return _duckduckgo_search_no_retry(query, max_results)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
def _duckduckgo_search_with_retry(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    return _duckduckgo_search_no_retry(query, max_results)


def _duckduckgo_search_no_retry(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                    "source": "DuckDuckGo",
                }
            )
    return results[:max_results]


async def _edge_tts_save(text: str, voice: str, output_path: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def text_to_speech_edge(
    script_text: str,
    output_path: str,
    voice: str = "en-US-GuyNeural",
) -> str:
    """Convert text to an energetic, clear MP3 file using edge-tts."""
    if edge_tts is None:
        print("Warning: edge-tts is not installed, using a silent fallback track.")
        return _write_silent_mp3(script_text, output_path)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(_edge_tts_save(script_text, voice, str(output_file)))
        return str(output_file)
    except Exception as exc:
        print(f"Warning: edge-tts failed, using a silent fallback track: {exc}")
        return _write_silent_mp3(script_text, output_path)


def _write_silent_mp3(script_text: str, output_path: str) -> str:
    from moviepy import AudioClip

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    words = max(1, len(script_text.split()))
    duration = float(min(420.0, max(6.0, words * 0.42)))
    audio_clip = AudioClip(lambda t: 0 * t, duration=duration, fps=44100)
    audio_clip.write_audiofile(str(output_file), fps=44100, codec="libmp3lame", verbose=False, logger=None)
    return str(output_file)


def generate_background_images_hf(
    prompts: List[str],
    output_dir: str = "./assets/visuals",
    model: str = "black-forest-labs/FLUX.1-schnell",
    hf_token: Optional[str] = None,
) -> List[str]:
    """Generate background images from prompts using the Hugging Face inference API."""
    if retry is not None:
        try:
            return _generate_background_images_hf(prompts, output_dir, model, hf_token)
        except Exception as exc:
            print(f"Warning: Hugging Face image generation failed, using local placeholders: {exc}")
            return _write_placeholder_images(prompts, output_dir)
    else:
        try:
            return _generate_background_images_hf_no_retry(prompts, output_dir, model, hf_token)
        except Exception as exc:
            print(f"Warning: Hugging Face image generation failed, using local placeholders: {exc}")
            return _write_placeholder_images(prompts, output_dir)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=120))
def _generate_background_images_hf(
    prompts: List[str],
    output_dir: str = "./assets/visuals",
    model: str = "black-forest-labs/FLUX.1-schnell",
    hf_token: Optional[str] = None,
) -> List[str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
    saved_files: List[str] = []

    if InferenceClient is not None:
        client = InferenceClient(api_key=token)
        for index, prompt in enumerate(prompts, start=1):
            wrapped = _wrap_cinematic_tech_prompt(prompt)
            image = client.text_to_image(prompt=wrapped, model=model)
            filename = output_path / f"background_{index}.png"
            image.save(filename)
            saved_files.append(str(filename))

        return saved_files

    headers = {"Accept": "image/png"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://router.huggingface.co/hf-inference/models/{model}"

    for index, prompt in enumerate(prompts, start=1):
        wrapped = _wrap_cinematic_tech_prompt(prompt)
        payload = {
            "inputs": wrapped,
            "options": {"wait_for_model": True},
        }
        response = requests.post(url, headers=headers, json=payload, timeout=120)

        if response.status_code != 200:
            raise RuntimeError(
                f"Hugging Face API failed ({response.status_code}): {response.text}"
            )

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            raise RuntimeError(
                f"Hugging Face returned an error payload: {response.text}"
            )

        filename = output_path / f"background_{index}.png"
        with open(filename, "wb") as image_file:
            image_file.write(response.content)

        saved_files.append(str(filename))

    return saved_files


def _generate_background_images_hf_no_retry(
    prompts: List[str],
    output_dir: str = "./assets/visuals",
    model: str = "black-forest-labs/FLUX.1-schnell",
    hf_token: Optional[str] = None,
) -> List[str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
    headers = {"Accept": "image/png"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://router.huggingface.co/hf-inference/models/{model}"
    saved_files: List[str] = []

    for index, prompt in enumerate(prompts, start=1):
        wrapped = _wrap_cinematic_tech_prompt(prompt)
        payload = {
            "inputs": wrapped,
            "options": {"wait_for_model": True},
        }
        response = requests.post(url, headers=headers, json=payload, timeout=120)

        if response.status_code != 200:
            raise RuntimeError(
                f"Hugging Face API failed ({response.status_code}): {response.text}"
            )

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            raise RuntimeError(
                f"Hugging Face returned an error payload: {response.text}"
            )

        filename = output_path / f"background_{index}.png"
        with open(filename, "wb") as image_file:
            image_file.write(response.content)

        saved_files.append(str(filename))

    return saved_files


if __name__ == "__main__":
    print("Scraping top Hacker News items...")
    items = scrape_top_hacker_news()
    for item in items:
        print(item)

    print("\nConverting a short demo script to MP3...")
    demo_script = "This is a fast, energetic trailer voiceover test for the next AI news video."
    mp3_path = text_to_speech_edge(demo_script, "./output/demo_voiceover.mp3")
    print(f"Saved voiceover to {mp3_path}")

    print("\nGenerating demo background images...")
    prompts = [
        "Futuristic AI news studio with neon circuit boards and glowing text",
        "A fast-moving tech city skyline with AI holograms and digital charts",
        "A clean YouTube thumbnail background for AI news, dark theme with vibrant accents",
    ]
    images = generate_background_images_hf(prompts)
    print("Saved images:")
    for path in images:
        print(path)
