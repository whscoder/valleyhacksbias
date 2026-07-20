# Fact GPT

Chrome extension plus FastAPI backend for article bias and reliability analysis.

The extension has two modes:

- **Article** extracts the current page and runs the existing fact/opinion,
  bias, and source-backed research pipeline.
- **Podcast** inspects the current open-web episode page, prefers a publisher
  transcript, and otherwise transcribes a public audio enclosure with safe
  speaker labels and timestamps before running episode-wide analysis.

Podcast mode does not capture tab audio or bypass authenticated, DRM, blob, or
protected iframe players. Long work is queued in the current backend process;
closing the popup is safe, but a backend restart requires starting the job again.

See [docs/architecture.md](docs/architecture.md) for the end-to-end flow and a purpose statement for every tracked file.

## Production Security Checklist

- Keep `OPENAI_API_KEY` only in the deployment secret store, such as Render environment variables. Do not commit or package `back-end/data/apikey.env`.
- Rotate the OpenAI key before launch if this workspace, a zip, or a screenshot containing `back-end/data/apikey.env` was ever shared.
- After publishing the Chrome extension, set `FACTGPT_ALLOWED_ORIGIN_REGEX` to the exact extension origin, for example `^chrome-extension://<extension-id>$`, and keep `FACTGPT_REQUIRE_ALLOWED_ORIGIN=true`.
- Do not put `FACTGPT_PUBLIC_API_TOKEN` in the public extension. It is only for private/internal clients because extension code is visible to users.
- Set OpenAI project budget limits and usage alerts. Server-side rate limiting reduces abuse but does not replace account-level spend caps.
- Keep the production extension host permission exact: `https://bias-article-detector.onrender.com/*`.
- Run the container as a non-root user and use a seccomp profile for Chromium if your host supports it.
- Remove old `.venv` objects from Git history before making the repository public.

## Required Backend Environment

Copy `.env.example` into your deployment provider and replace placeholder values:

```text
OPENAI_API_KEY
FACTGPT_CLASSIFICATION_SIGNING_KEY
FACTGPT_REQUIRE_ALLOWED_ORIGIN
FACTGPT_ALLOWED_ORIGIN_REGEX
FACTGPT_RATE_LIMIT_PER_MINUTE
```

The backend also supports request and fetch caps through `FACTGPT_MAX_REQUEST_BODY_BYTES`, `FACTGPT_MAX_REQUEST_TEXT_CHARS`, `FACTGPT_MAX_FETCH_BYTES`, and `FACTGPT_MAX_EXTRACTED_TEXT_CHARS`.

Podcast downloads and job retention are controlled separately through
`FACTGPT_MAX_PODCAST_AUDIO_BYTES`, `FACTGPT_MAX_PODCAST_DURATION_SECONDS`,
`FACTGPT_MAX_PODCAST_TRANSCRIPT_BYTES`, `FACTGPT_MAX_PODCAST_PAGE_BYTES`, and
`FACTGPT_PODCAST_JOB_TTL_SECONDS`. The production image installs FFmpeg.

## Validation

Run deterministic backend and presenter contracts without paid model calls:

```bash
cd back-end
../.venv/bin/python -m unittest -v test_fact_opinion_route.py test_home_quotes.py test_research_contract.py
../.venv/bin/python -m unittest -v test_podcast.py
cd ..
node front-end/test_presenters.mjs
node front-end/test_podcast_helpers.mjs
docker build -t factgpt-podcast .
docker run --rm --entrypoint sh factgpt-podcast -c 'ffmpeg -version && ffprobe -version'
```

Preview the fixed-seed set that the paid OpenAI fallback evaluation would review:

```bash
cd back-end
../.venv/bin/python evaluate_openai_fallback.py --dry-run
```

Omit `--dry-run` only when you intend to make paid API requests; the run writes a JSON report under `back-end/testingcode/reports/`.
