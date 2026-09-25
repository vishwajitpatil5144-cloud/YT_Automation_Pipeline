import os
import re
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from youtube_automation_utils import (
    format_search_results_for_prompt,
    generate_background_images_hf,
    web_search_research,
)

try:
    import google.generativeai as google_genai
except ImportError:
    google_genai = None

try:
    from langchain.llms.groq import Groq
except ImportError:
    Groq = None

try:
    from groq import Groq as GroqClient
except ImportError:
    GroqClient = None

try:
    from langchain.llms.googlegemini import GoogleGemini
except ImportError:
    GoogleGemini = None

try:
    from langgraph import StateGraph, START, END
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    StateGraph = None
    START = None
    END = None
    SqliteSaver = None

try:
    from telegram import Bot
    from telegram.error import TelegramError
except ImportError:
    Bot = None
    TelegramError = None


def _parse_writer_llm_output(raw: str) -> Tuple[str, List[str]]:
    """Extract long-form script and five short scripts from delimited LLM output."""
    long_text = ""
    parts = re.split(r"###\s*SHORT_SCRIPTS", raw, maxsplit=1, flags=re.IGNORECASE)
    head = parts[0]
    rest = parts[1] if len(parts) > 1 else ""

    lm = re.search(r"###\s*LONG_FORM_SCRIPT\s*(.*)", head, re.DOTALL | re.IGNORECASE)
    if lm:
        long_text = lm.group(1).strip()

    shorts: List[str] = []
    for i in range(1, 6):
        if i < 5:
            pat = rf"####\s*SHORT_{i}\s*\n(.*?)(?=\n####\s*SHORT_{i + 1}\b)"
        else:
            pat = r"####\s*SHORT_5\s*\n(.*)"
        m = re.search(pat, rest, re.DOTALL | re.IGNORECASE)
        if m:
            shorts.append(m.group(1).strip())
        else:
            break

    if len(shorts) < 5:
        shorts = []
        search_space = rest if rest.strip() else raw
        for i in range(1, 6):
            if i < 5:
                pat = rf"^Short\s*{i}\s*:\s*(.*?)(?=^Short\s*{i + 1}\s*:|$)"
            else:
                pat = r"^Short\s*5\s*:\s*(.*)"
            m = re.search(pat, search_space, re.DOTALL | re.IGNORECASE | re.MULTILINE)
            if m:
                shorts.append(m.group(1).strip())
            else:
                shorts = []
                break

    return long_text, shorts[:5]


def _normalize_youtube_tags(tags: List[str], target: int = 15) -> List[str]:
    cleaned: List[str] = []
    for t in tags:
        s = t.strip()
        if not s or len(s) > 30:
            s = s[:30].strip()
        if s and s not in cleaned:
            cleaned.append(s)
        if len(cleaned) >= target:
            break
    suffix = 0
    while len(cleaned) < target:
        candidate = f"AI news {suffix}".strip()[:30]
        suffix += 1
        if candidate not in cleaned:
            cleaned.append(candidate)
    return cleaned[:target]


def _parse_packaging_response(text: str) -> Dict[str, Any]:
    """Parse TITLE / DESCRIPTION / TAGS / THUMBNAIL_PROMPT blocks."""
    title = ""
    description = ""
    tags: List[str] = []
    thumbnail_prompt = ""

    current: Optional[str] = None
    buf: List[str] = []

    def flush() -> None:
        nonlocal title, description, tags, thumbnail_prompt, current, buf
        if not current:
            return
        body = "\n".join(buf).strip()
        if current == "TITLE":
            title = body.splitlines()[0].strip() if body else ""
        elif current == "DESCRIPTION":
            description = body
        elif current == "TAGS":
            tags = [p.strip() for p in re.split(r"[,;]", body) if p.strip()]
        elif current == "THUMBNAIL_PROMPT":
            thumbnail_prompt = body
        current = None
        buf = []

    for line in text.splitlines():
        m = re.match(r"^(TITLE|DESCRIPTION|TAGS|THUMBNAIL_PROMPT):\s*(.*)$", line, re.I)
        if m:
            flush()
            current = m.group(1).upper()
            rest = m.group(2).strip()
            if rest:
                buf = [rest]
            else:
                buf = []
        elif current:
            buf.append(line)
    flush()

    return {
        "title": title,
        "description": description,
        "tags": _normalize_youtube_tags(tags, 15),
        "thumbnail_prompt": thumbnail_prompt
        or "Cinematic 16:9 YouTube thumbnail, cyberpunk tech news, high contrast, no text",
    }


class AINewsState(BaseModel):
    news_items: List[str] = Field(default_factory=list)
    supervisor_plan: Optional[str] = None
    research_notes: List[str] = Field(default_factory=list)
    long_form_script: Optional[str] = None
    short_scripts: List[str] = Field(default_factory=list)
    quality_report: Optional[str] = None
    approval_status: str = "pending"
    youtube_title: Optional[str] = None
    youtube_description: Optional[str] = None
    youtube_tags: List[str] = Field(default_factory=list)
    youtube_thumbnail_prompt: Optional[str] = None
    thumbnail_image_path: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class LangGraphStateSchema:
    """A simple LangGraph-compatible schema wrapper for our AI pipeline."""

    def __init__(self, state: AINewsState):
        self.state = state

    def serialize(self) -> Dict:
        return self.state.model_dump()

    def update(self, values: Dict) -> None:
        self.state = self.state.model_copy(update=values)


class AIChainPipeline:
    def __init__(self):
        self.state = AINewsState()
        self.graph_state = LangGraphStateSchema(self.state)
        self.llms = self._select_llm()
        self.graph = None
        self.checkpointer = None
        if StateGraph is not None and SqliteSaver is not None:
            self.checkpointer = SqliteSaver.from_conn_string("sqlite:///checkpoints.db")
            self._build_graph()

    def _approval_is_skipped(self) -> bool:
        value = os.getenv("SKIP_TELEGRAM_APPROVAL", "1").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _build_graph(self):
        if StateGraph is None:
            return
        
        graph = StateGraph(AINewsState)
        
        # Add nodes
        graph.add_node("supervisor", self.supervisor_node)
        graph.add_node("researcher", self.researcher_node)
        graph.add_node("writer", self.writer_node)
        graph.add_node("approval", self.approval_node)
        graph.add_node("quality", self.quality_node)
        graph.add_node("packaging", self.packaging_node)
        graph.add_node("finalize", self.finalize_node)

        # Add edges
        graph.add_edge(START, "supervisor")
        graph.add_edge("supervisor", "researcher")
        graph.add_edge("researcher", "writer")
        graph.add_edge("writer", "approval")
        graph.add_edge("approval", "quality")
        graph.add_edge("quality", "packaging")
        graph.add_edge("packaging", "finalize")
        graph.add_edge("finalize", END)
        
        self.graph = graph.compile(checkpointer=self.checkpointer)

    def _select_llm(self):
        llms = []
        if os.getenv("GROQ_API_KEY") and GroqClient is not None:
            model_env = os.getenv(
                "GROQ_MODEL",
                "llama-3.3-70b-versatile,llama-3.1-70b-versatile,llama3-70b-8192",
            )
            model_names = [name.strip() for name in model_env.split(",") if name.strip()]
            llms.append(
                {
                    "provider": "groq_sdk",
                    "client": GroqClient(api_key=os.getenv("GROQ_API_KEY")),
                    "model_names": model_names,
                }
            )
        if os.getenv("GROQ_API_KEY") and Groq is not None:
            llms.append(Groq())
        if os.getenv("GOOGLE_API_KEY") and GoogleGemini is not None:
            llms.append(GoogleGemini())
        if os.getenv("GOOGLE_API_KEY") and google_genai is not None:
            google_genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model_env = os.getenv(
                "GEMINI_MODEL",
                "models/gemini-2.5-flash,models/gemini-2.0-flash,gemini-2.0-flash",
            )
            model_names = [name.strip() for name in model_env.split(",") if name.strip()]
            llms.append({"provider": "gemini_sdk", "model_names": model_names})
        if not llms:
            llms.append({"provider": "local_stub"})
        return llms

    def _local_llm_response(self, prompt: str) -> str:
        lowered_prompt = prompt.lower()

        if "supervisor agent" in lowered_prompt:
            return (
                "Plan: Focus on the most newsworthy AI release, verify claims with current sources, "
                "and present the story with a fast-paced technical angle."
            )
        if "researcher agent" in lowered_prompt:
            return (
                "- Verify the announcement details\n"
                "- Capture the key technical change\n"
                "- Note any product name or metric mentioned\n"
                "- Keep uncertainty explicit when source quality is weak"
            )
        if "writer agent" in lowered_prompt:
            filler = (
                "What if your stack just became obsolete overnight? "
                "That is the question nobody wants to answer in public, yet every hyperscaler is racing to ship the fix. "
            ) * 45
            return (
                "### LONG_FORM_SCRIPT\n"
                + filler
                + "\n\nWe are living through a compressed hype cycle where benchmarks become memes before the PDF is cold. "
                "The technical reality is messier: memory bandwidth still matters, eval contamination is still boring, "
                "and your favorite demo is still one bad prompt away from embarrassment. "
                "So here is the adult version of the story in three acts: what shipped, what it actually changes, "
                "and why your roadmap just got forked without a changelog entry.\n\n"
                "### SHORT_SCRIPTS\n"
                "#### SHORT_1\n"
                "Stop scrolling. This AI drop is the kind that quietly rewrites pricing pages before the press release finishes loading.\n\n"
                "#### SHORT_2\n"
                "Faster inference is cute. The real flex is what downstream products can suddenly assume is free.\n\n"
                "#### SHORT_3\n"
                "One benchmark, two vendors, three angry quote tweets. Welcome to enterprise AI theater.\n\n"
                "#### SHORT_4\n"
                "If your security model assumed humans were the weakest link, congrats: you were right, just for the wrong decade.\n\n"
                "#### SHORT_5\n"
                "The headline is calm. The implications are not. Here is the one detail that actually matters.\n"
            )
        if "quality-check agent" in lowered_prompt:
            return "Quality check passed. The script is punchy, technical, and ready for publication."
        if "packaging agent" in lowered_prompt:
            tags = ", ".join(
                [
                    "AI news",
                    "machine learning",
                    "LLM",
                    "OpenAI",
                    "tech explainer",
                    "developer news",
                    "cyberpunk aesthetic",
                    "future tech",
                    "startup drama",
                    "GPU",
                    "benchmarks",
                    "AI safety",
                    "coding",
                    "automation",
                    "deep dive",
                ]
            )
            return (
                "TITLE: This AI Update Just Rewired The Whole Market (And Nobody Is Ready)\n"
                "DESCRIPTION: A fast, sarcastic, deeply technical breakdown of the last 48 hours of AI news. "
                "Timestamps are vibes-only because the story moves faster than your backlog.\n\n"
                "Chapters:\n"
                "0:00 — The hook\n"
                "0:45 — What actually changed\n"
                "3:00 — Why competitors should be nervous\n\n"
                f"TAGS: {tags}\n"
                "THUMBNAIL_PROMPT: Ultra sharp 16:9 YouTube thumbnail frame, cinematic cyberpunk server room, "
                "massive holographic neural graph exploding toward camera, neon magenta and electric cyan rim light, "
                "dramatic low angle, volumetric fog, rain on glass foreground, terrified engineer silhouette tiny in frame, "
                "no text, no logos, photoreal materials, IMAX grade contrast\n"
            )
        return "Approved."

    def run(self, news_items: List[str]) -> None:
        if self.graph is not None:
            # Use LangGraph with checkpointing
            config = {"configurable": {"thread_id": "ai_news_pipeline"}}
            result = self.graph.invoke({"news_items": news_items}, config)
            if isinstance(result, dict):
                self.state = AINewsState.model_validate(result)
            else:
                self.state = result
        else:
            # Fallback to sequential node execution when LangGraph is unavailable.
            state = self.state.model_copy(update={"news_items": news_items})
            state = self.supervisor_node(state)
            state = self.researcher_node(state)
            state = self.writer_node(state)
            state = self.approval_node(state)
            state = self.quality_node(state)
            state = self.packaging_node(state)
            state = self.finalize_node(state)
            self.state = state

    def supervisor_node(self, state: AINewsState) -> AINewsState:
        prompt = (
            "You are a Supervisor Agent for an AI news video pipeline. "
            "News bullets are curated from roughly the last 48 hours and skew toward AI/ML. "
            "Pick the highest-impact thread for a long-form technical explainer, note what must be verified, "
            "and set the narrative style. Return a concise plan in plain text."
        )
        response = self._llm_prompt(prompt + "\n\nNews:\n" + "\n".join(state.news_items))
        return state.model_copy(update={"supervisor_plan": response.strip()})

    def researcher_node(self, state: AINewsState) -> AINewsState:
        research_query = state.supervisor_plan or " ".join(state.news_items)
        search_provider = os.getenv("RESEARCH_SEARCH_PROVIDER", "auto")
        search_results: List[Dict[str, str]] = []
        search_status = "not-run"

        try:
            search_results = web_search_research(
                query=research_query,
                max_results=6,
                provider=search_provider,
            )
            search_status = f"ok:{len(search_results)}"
        except Exception as exc:
            search_status = f"failed:{exc}"

        external_sources = format_search_results_for_prompt(search_results)
        prompt = (
            "You are a Researcher Agent extracting dense, technical facts from an AI news story. "
            "The story should reflect developments from roughly the last 48 hours when sources allow. "
            "Produce a list of concise, technical facts and call out any important numbers, acronyms, or product names. "
            "Use the external sources to resolve vague claims and add concrete context. "
            "If sources conflict, note the uncertainty briefly."
        )
        response = self._llm_prompt(
            prompt
            + "\n\nPlan:\n"
            + (state.supervisor_plan or "")
            + "\n\nExternal Web Sources:\n"
            + external_sources
        )
        research_notes = [line.strip() for line in response.splitlines() if line.strip()]
        metadata = dict(state.metadata)
        metadata["research_query"] = research_query
        metadata["search_provider"] = search_provider
        metadata["search_status"] = search_status
        return state.model_copy(update={"research_notes": research_notes, "metadata": metadata})

    def writer_node(self, state: AINewsState) -> AINewsState:
        news_block = "\n".join(f"- {n}" for n in state.news_items)
        prompt = (
            "You are a Writer Agent for an AI news channel. "
            "Write in the fast-paced, deeply technical, slightly sarcastic tone of Fireship. "
            "The first lines must be an explosive ~5 second spoken hook using the Aprilynne Alter pattern interrupt method "
            "(immediate stakes, curiosity gap, zero throat-clearing). "
            "Then deliver an 800 to 1000 word main script that could realistically fill a 5 to 6 minute voiceover: "
            "dense specifics, named systems, numbers where grounded in the research, and connect the last ~48 hours of "
            "headlines into one coherent arc—do not skim; explain why it matters and who loses if they ignore it.\n\n"
            "Also write five DISTINCT YouTube Shorts scripts, each punchy and about 45 seconds when read aloud (~110 to 140 words each), "
            "each with its own hook and angle.\n\n"
            "Output format (exact headings):\n"
            "### LONG_FORM_SCRIPT\n"
            "<single continuous script text>\n\n"
            "### SHORT_SCRIPTS\n"
            "#### SHORT_1\n"
            "<script>\n"
            "#### SHORT_2\n"
            "<script>\n"
            "#### SHORT_3\n"
            "<script>\n"
            "#### SHORT_4\n"
            "<script>\n"
            "#### SHORT_5\n"
            "<script>\n"
        )
        facts = "\n".join(f"- {fact}" for fact in state.research_notes)
        response = self._llm_prompt(
            prompt
            + "\n\nRecent news context (last ~48h):\n"
            + news_block
            + "\n\nSupervisor plan:\n"
            + (state.supervisor_plan or "")
            + "\n\nResearch Notes:\n"
            + facts
        )
        long_form, shorts = _parse_writer_llm_output(response)
        if len(shorts) < 5:
            fix = self._llm_prompt(
                "You failed the output format. Return ONLY the ### SHORT_SCRIPTS section with #### SHORT_1 through "
                "#### SHORT_5, each containing a complete ~45 second script. No other text."
                + "\n\nOriginal draft for reference:\n"
                + response[:12000]
            )
            block = fix if "### SHORT_SCRIPTS" in fix else "### SHORT_SCRIPTS\n" + fix
            merged = (
                "### LONG_FORM_SCRIPT\n"
                + (long_form.strip() or "See previous response body.\n")
                + "\n\n"
                + block
            )
            long_form2, shorts2 = _parse_writer_llm_output(merged)
            if long_form2.strip():
                long_form = long_form2
            if len(shorts2) >= 5:
                shorts = shorts2[:5]
        if len(shorts) < 5:
            pool = list(state.news_items) or ["AI is moving fast this week."]
            while len(shorts) < 5:
                headline = pool[len(shorts) % len(pool)]
                shorts.append(
                    f"Forty five seconds on why this headline actually matters: {headline}. "
                    "Same chaos, new numbers, and the same people pretending they saw it coming."
                )
        if not long_form.strip():
            long_form = response.strip()
        return state.model_copy(update={"long_form_script": long_form.strip(), "short_scripts": shorts[:5]})

    def approval_node(self, state: AINewsState) -> AINewsState:
        if self._approval_is_skipped():
            print("\n===== HUMAN APPROVAL SKIPPED =====")
            return state.model_copy(update={"approval_status": "approved"})

        print("\n===== HUMAN APPROVAL BREAKPOINT =====")
        print("Sending draft script to Telegram for approval...")
        
        if Bot is None:
            raise RuntimeError("python-telegram-bot is not installed.")
        
        approved = asyncio.run(
            self._send_and_wait_approval(state.long_form_script or "[No draft script available]")
        )
        
        approval_status = "approved" if approved else "rejected"
        return state.model_copy(update={"approval_status": approval_status})

    def quality_node(self, state: AINewsState) -> AINewsState:
        prompt = (
            "You are a Quality-Check Agent. Verify that the tone is sarcastic, fast-paced, and Fireship-like. "
            "Ensure the script includes a strong 5-second hook, accurate technical detail, and no unsupported claims. "
            "Flag if the main script is materially under ~800 words or over ~1100 words. "
            "If there are problems, list them clearly; otherwise confirm that the script is ready."
        )
        response = self._llm_prompt(prompt + "\n\nScript:\n" + (state.long_form_script or ""))
        return state.model_copy(update={"quality_report": response.strip()})

    def finalize_node(self, state: AINewsState) -> AINewsState:
        if state.approval_status != "approved":
            raise RuntimeError("Finalization blocked: script not approved.")
        script = state.long_form_script
        print("\n===== SCRIPT FINALIZED =====")
        print(script or "[No final script]")
        print("\nQuality Report:\n" + (state.quality_report or "[No quality report]"))
        return state.model_copy(update={})

    def packaging_node(self, state: AINewsState) -> AINewsState:
        if state.approval_status != "approved":
            raise RuntimeError("Packaging blocked: script not approved.")
        script = state.long_form_script or ""
        lead_topic = state.news_items[0] if state.news_items else "AI breakthrough"

        prompt = (
            "You are a Packaging Agent for YouTube. Using the approved long-form script, produce metadata that maximizes "
            "clicks and search reach while staying faithful to the content.\n\n"
            "Return exactly these sections and labels (one per line start, DESCRIPTION and THUMBNAIL_PROMPT may span multiple lines):\n"
            "TITLE: <single line, irresistible, under ~95 characters>\n"
            "DESCRIPTION: <SEO-rich multi-line description with strong first two lines, optional light markdown, include a "
            "soft CTA to subscribe>\n"
            "TAGS: <exactly fifteen comma-separated tags, no hashtags, each tag short and strategic>\n"
            "THUMBNAIL_PROMPT: <one highly detailed English prompt for a 16:9 cinematic YouTube thumbnail image generator; "
            "specify lighting, composition, subject emotion, color palette, depth of field; forbid any readable text or logos>\n"
        )

        response = self._llm_prompt(
            prompt + "\n\nLead Topic:\n" + lead_topic + "\n\nApproved Long Script:\n" + script
        )
        parsed = _parse_packaging_response(response)
        title = parsed["title"] or f"AI just changed again — {lead_topic[:60]}"
        description = parsed["description"] or script[:4000]
        tags = parsed["tags"]
        thumb_prompt = parsed["thumbnail_prompt"]

        thumbnail_path: Optional[str] = None
        metadata = dict(state.metadata)
        metadata["packaging_status"] = "prompt-ready"

        assets_dir = Path("./assets")
        assets_dir.mkdir(parents=True, exist_ok=True)
        thumb_dir = assets_dir / "youtube_thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)

        try:
            thumbnail_model = os.getenv("THUMBNAIL_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
            rendered = generate_background_images_hf(
                [thumb_prompt],
                output_dir=str(thumb_dir),
                model=thumbnail_model,
            )
            if rendered:
                thumbnail_path = rendered[0]
                metadata["packaging_status"] = "thumbnail-rendered"
        except Exception as exc:
            metadata["packaging_status"] = f"thumbnail-render-failed:{exc}"

        return state.model_copy(
            update={
                "youtube_title": title,
                "youtube_description": description,
                "youtube_tags": tags,
                "youtube_thumbnail_prompt": thumb_prompt,
                "thumbnail_image_path": thumbnail_path,
                "metadata": metadata,
            }
        )

    def human_approval_breakpoint(self) -> None:
        if self._approval_is_skipped():
            print("\n===== HUMAN APPROVAL SKIPPED =====")
            self.state.approval_status = "approved"
            self.graph_state.update({"approval_status": self.state.approval_status})
            return

        print("\n===== HUMAN APPROVAL BREAKPOINT =====")
        print("Sending draft script to Telegram for approval...")
        
        if Bot is None:
            raise RuntimeError("python-telegram-bot is not installed.")
        
        approved = asyncio.run(self._send_and_wait_approval())
        
        if approved:
            self.state.approval_status = "approved"
        else:
            self.state.approval_status = "rejected"
        
        self.graph_state.update({"approval_status": self.state.approval_status})

        if self.state.approval_status != "approved":
            raise RuntimeError("Pipeline stopped: script rejected by human reviewer.")

    async def _send_and_wait_approval(self, script_text: Optional[str] = None) -> bool:
        bot_token = os.getenv("TELEGRAM_TOKEN")
        chat_id_str = os.getenv("TELEGRAM_CHAT_ID")
        
        if not bot_token or not chat_id_str:
            raise RuntimeError("TELEGRAM_TOKEN and TELEGRAM_CHAT_ID must be set in environment variables.")
        
        try:
            chat_id = int(chat_id_str)
        except ValueError:
            raise RuntimeError("TELEGRAM_CHAT_ID must be a valid integer.")
        
        bot = Bot(token=bot_token)
        
        script = script_text or self.state.long_form_script or "[No draft script available]"
        
        intro = "Please review and approve the script. Reply with 'APPROVED' to continue or 'REJECTED' to stop."
        try:
            await bot.send_message(chat_id=chat_id, text=intro)
            chunk_size = 3500
            for index in range(0, len(script), chunk_size):
                chunk = script[index:index + chunk_size]
                part = (index // chunk_size) + 1
                await bot.send_message(chat_id=chat_id, text=f"Script part {part}:\n{chunk}")
        except TelegramError as e:
            print(f"Failed to send message: {e}")
            return False
        
        # Poll for updates
        last_update_id = 0
        while True:
            try:
                updates = await bot.get_updates(offset=last_update_id + 1, timeout=30)
                for update in updates:
                    if update.message and update.message.chat.id == chat_id:
                        text = update.message.text.lower().strip()
                        if text == "approved":
                            return True
                        elif text == "rejected":
                            return False
                    if update.update_id > last_update_id:
                        last_update_id = update.update_id
            except TelegramError as e:
                print(f"Error polling updates: {e}")
            await asyncio.sleep(5)  # wait 5 seconds before polling again

    def quality_check_agent(self) -> None:
        prompt = (
            "You are a Quality-Check Agent. Verify that the tone is sarcastic, fast-paced, and Fireship-like. "
            "Ensure the script includes a strong 5-second hook, accurate technical detail, and no unsupported claims. "
            "If there are problems, list them clearly; otherwise confirm that the script is ready."
        )
        response = self._llm_prompt(prompt + "\n\nScript:\n" + (self.state.long_form_script or ""))
        self.state.quality_report = response.strip()
        self.graph_state.update({"quality_report": self.state.quality_report})

    def finalize_script(self) -> None:
        if self.state.approval_status != "approved":
            raise RuntimeError("Finalization blocked: script not approved.")
        self.graph_state.update({})
        print("\n===== SCRIPT FINALIZED =====")
        print(self.state.long_form_script or "[No final script]")
        print("\nQuality Report:\n" + (self.state.quality_report or "[No quality report]"))

    def _llm_prompt(self, prompt: str) -> str:
        if not self.llms:
            raise RuntimeError("LLM backend is not available. Install the correct provider library.")
        
        last_error = None
        for llm in self.llms:
            try:
                if isinstance(llm, dict) and llm.get("provider") == "local_stub":
                    return self._local_llm_response(prompt)
                if isinstance(llm, dict) and llm.get("provider") == "groq_sdk":
                    groq_error = None
                    for model_name in llm.get("model_names", []):
                        try:
                            response = llm["client"].chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": prompt}],
                            )
                            return response.choices[0].message.content
                        except Exception as exc:
                            groq_error = exc
                            continue
                    if groq_error is not None:
                        raise groq_error
                if isinstance(llm, dict) and llm.get("provider") == "gemini_sdk":
                    gemini_error = None
                    for model_name in llm.get("model_names", []):
                        try:
                            client = google_genai.GenerativeModel(model_name)
                            response = client.generate_content(prompt)
                            text = getattr(response, "text", None)
                            if text:
                                return text
                            return str(response)
                        except Exception as exc:
                            gemini_error = exc
                            continue
                    if gemini_error is not None:
                        raise gemini_error
                if hasattr(llm, "__call__"):
                    response = llm(prompt)
                    if hasattr(response, "content"):
                        return response.content
                    return str(response)
                else:
                    raise RuntimeError("Unsupported LLM client interface.")
            except Exception as e:
                last_error = e
                continue
        
        raise RuntimeError(f"All LLM backends failed. Last error: {last_error}")


def main() -> None:
    news_items = [
        "Google Gemini just launched a new multimodal model update with faster memory recall.",
        "A hot AI safety paper claims transformer scaling laws still apply beyond 1T parameters.",
        "New startup announces synthesis of full-stack developer code in under 10 seconds."
    ]

    pipeline = AIChainPipeline()
    try:
        pipeline.run(news_items)
    except Exception as exc:
        print(f"Pipeline stopped: {exc}")


if __name__ == "__main__":
    main()
