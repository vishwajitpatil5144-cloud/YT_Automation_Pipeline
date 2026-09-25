# AI News Video Pipeline

A Python scaffold for a LangGraph-driven, multi-agent YouTube news video pipeline.

## What this project includes

- `pipeline.py`: Defines a state schema and orchestration for:
  - Supervisor Agent
  - Researcher Agent
  - Writer Agent
  - Quality-Check Agent
   - Packaging Agent (thumbnail concept generation + render)
  - Human-in-the-loop approval breakpoint
- `requirements.txt`: Dependencies for `langchain`, `langgraph`, and `google-generativeai`

## Setup

1. Create a virtual environment and activate it on Windows PowerShell:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Upgrade pip and install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Set one of these environment variables:
   - `GOOGLE_API_KEY`
   - `GROQ_API_KEY`

4. **Optional: Active Web Search for the Researcher Agent**:
    - The Researcher now enriches story context with live web search before generating technical notes.
    - Supported providers:
       - `SerpApi` (Google Search API) via `SERPAPI_API_KEY`
       - `DuckDuckGo` via the installed `duckduckgo-search` package
    - Optional environment variables:
       - `SERPAPI_API_KEY=your_serpapi_key`
       - `RESEARCH_SEARCH_PROVIDER=auto` (options: `auto`, `serpapi`, `duckduckgo`)
    - Default behavior is `auto`: use SerpApi when key is set, otherwise fallback to DuckDuckGo.

5. **Telegram Bot Setup for Remote Approval**:
   - Create a new bot with [@BotFather](https://t.me/botfather) on Telegram.
   - Get the bot token from BotFather.
   - Start a chat with your bot and send a message.
   - Get your chat ID by visiting `https://api.telegram.org/bot<YourBOTToken>/getUpdates` and find the "chat":{"id":...}.
   - Set environment variables:
     - `TELEGRAM_TOKEN=your_bot_token`
     - `TELEGRAM_CHAT_ID=your_chat_id`

6. Run the pipeline:
   ```powershell
   python pipeline.py
   ```

8. For the full render-and-upload flow, run:
   ```powershell
   python youtube_publish_scheduler.py
   ```
   This command now executes a preflight check first and stops immediately if required files, environment variables, or core modules are missing.

   If Google blocks the browser login with `Error 403: org_internal`, the OAuth client in `client_secrets.json` is restricted to an internal Workspace organization. Use an external OAuth client instead, or point the script at a different secrets file with:
   - `YOUTUBE_CLIENT_SECRETS_FILE=path\to\external_client_secrets.json`
   - `YOUTUBE_TOKEN_FILE=path\to\youtube_token.json`

7. Optional packaging environment variables:
   - `THUMBNAIL_IMAGE_MODEL=stabilityai/stable-diffusion-2`

Alternatively, run the helper script:
```powershell
.\setup_env.ps1
```

## Notes

- The pipeline includes a remote human-in-the-loop approval via Telegram before finalizing the script. Reply "APPROVED" or "REJECTED" to the bot message.
- The writer voice is designed to be fast-paced, sarcastic, and include a strong 5-second YouTube hook.
- The pipeline generates 1 main 16:9 video and 5 Shorts (9:16) per run.
- State is checkpointed to a local SQLite database (`checkpoints.db`) after each step, allowing the pipeline to resume from interruptions without losing progress.
- Exponential backoff retries are implemented for network requests (scraping, image generation) to handle transient failures.
- LLM fallback: If the primary LLM (Groq) fails, the pipeline automatically switches to Google Gemini.
- Automatic asset pruning: Deletes .mp3, .png, and .mp4 files older than 48 hours from ./assets and ./output folders to manage disk space.
- The script is not published automatically until the human approves it.
- The Packaging Agent generates 3 thumbnail concepts, selects the best one, renders it, and stores the result in state as `thumbnail_image_path`.
- `run_entire_pipeline()` passes `thumbnail_image_path` into the YouTube upload function so the video is uploaded with a generated thumbnail.
- `youtube_publish_scheduler.py` includes a startup preflight validator for runtime readiness.
