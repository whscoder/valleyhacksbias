# Fact GPT Architecture and File Guide

## Architecture definition

Fact GPT is a three-layer client/server system:

1. A Chrome Manifest V3 extension is the presentation and browser-integration layer. Its popup starts an analysis, displays durable progress/results, and can highlight flagged phrases in the active article.
2. A FastAPI service is the application and security layer. It validates callers and URLs, extracts article text, invokes OpenAI for structured bias and claim-research results, and validates the model output before returning it.
3. A Playwright/Common Crawl harness is the end-to-end evaluation layer. It discovers real news URLs, loads a temporary test version of the extension, drives analyses in Chromium, and writes JSON reports.

The harness replaces the production worker/popup orchestration with instrumented test overrides. It retains its detailed extraction telemetry while importing the untouched production result renderer, so it checks the shipping classifier/research presentation but not the shipping service-worker/storage lifecycle.

Per-page job state is stored in `chrome.storage.local`. The backend keeps process-local rate-limit/classification caches, but article classifications carry an exact-text HMAC token so a later research request can safely reuse a result across workers or restarts.

## End-to-end production flow

1. `extension.html` loads `popup.js`. The popup finds the active tab, asks `background.js` for the normalized per-URL storage key, and restores any saved state.
2. Clicking **Analyze Current Page** gives the run a unique request ID and sends `FACTGPT_START_ANALYSIS` to the MV3 background service worker. The worker reads visible DOM text when permitted and makes one idempotent `POST /article-jobs` request. It stores the returned job ID, responds to the popup, and retains no article timer, port, or detached job promise, so Chrome can terminate it after the event completes and the worker becomes idle.
3. The backend-owned article job uses supplied DOM text when it is sufficient. Otherwise it tries direct HTTPX/BeautifulSoup extraction and then headless-Chromium rendered extraction. Repeating an ambiguous create request with the same request ID returns the same job.
4. The backend segments the exact normalized article and runs the local fact/opinion classifier first. Low-confidence items, every exclusion-sensitive local opinion, and confident facts containing high-precision subjectivity cues go to OpenAI review. Final decisions are `fact`, `opinion`, `mixed`, or `unresolved` with an auditable basis.
5. Pure opinions and unresolved passages are excluded from downstream factual analysis. Mixed passages retain their exact opinion wording for bias analysis, while those excerpts are masked for factual research. The backend job preserves partial bias output if later research fails.
6. Research receives the validated bias result for context. The Responses API is forced to complete web search; returned citations must come from that tool output, and decisive verdicts require an official, primary, or reputable secondary source. Server-derived coverage states how many candidate segments were checked, left unchecked, or truncated.
7. While open, `popup.js` polls `GET /article-jobs/{job_id}` directly, with one request in flight at a time, and persists each snapshot. Closing the popup stops polling without stopping backend work. Reopening restores the job ID and resumes GET polling; it does not create another job. `parseScript.js` renders the completed result and page highlights.

## Podcast-mode flow

1. The popup's Article/Podcast selector keeps article state unchanged and uses a
   separate URL-keyed podcast state. On start, the worker collects best-effort
   RSS, transcript, direct-audio, and JSON-LD hints from the active tab.
2. `POST /podcast-jobs` creates or reuses an in-process job. The popup can close;
   the worker persists the job ID and resumes polling `GET /podcast-jobs/{id}`.
3. The backend fetches and validates every candidate and redirect. It prefers a
   matching RSS `podcast:transcript`, then structured page transcripts, then a
   matching RSS enclosure or direct public audio. Ambiguous episodes and
   authenticated/protected media fail closed.
4. Publisher VTT, SRT, JSON, HTML, and plain-text transcripts normalize into
   exact speaker turns. Audio is duration-checked, split near silence with
   FFmpeg, and sent to OpenAI's diarized transcription endpoint. Explicit
   publisher names are preserved; generated labels never claim real identity.
5. Speaker-aligned transcript windows run through the existing classifier.
   Podcast bias includes resolved opinion speech, while research receives only
   resolved factual content sampled across the whole episode. Window bias is
   aggregated to one 0-10 result with exact timestamped highlight locations.
6. The compact analysis is stored by the extension. Full speaker turns stay in
   the backend job and are loaded in pages from
   `GET /podcast-jobs/{id}/segments`; timestamp clicks seek only directly
   accessible page media.

## Backend safeguards

- CORS plus optional exact origin and API-token enforcement protects expensive routes.
- A per-client, in-memory sliding-window rate limit reduces repeated work.
- Request, text, response-body, and extracted-text caps bound resource usage.
- URL validation rejects non-HTTP schemes, local hostnames, and non-public DNS results.
- Redirects and Playwright subresources are validated separately to reduce SSRF and DNS-rebinding exposure.
- Pydantic models reject malformed model output before it reaches the extension.
- Bias highlights must be exact substrings of the analyzed text, and signed classification payloads must match the article's exact segment IDs, text, and offsets.
- Client-visible model failures contain a safe code and reference ID; provider exception details remain in server logs.

## Tracked file inventory

### Repository and deployment

- `.dockerignore` — keeps local environments, secrets, frontend packages, test output, and other irrelevant artifacts out of the backend container build context.
- `.env.example` — documents required production secrets/security settings and optional resource caps; it contains placeholders only.
- `.gitignore` — prevents secrets, virtual environments, reports, release archives, caches, backups, and downloaded crawl data from entering Git.
- `Dockerfile` — builds the FastAPI service on the official Playwright Python image, installs Python/Chromium dependencies, copies only the backend, drops to `pwuser`, and starts Uvicorn.
- `README.md` — short project description, production security checklist, required environment variables, and link to this guide.
- `requirements.txt` — pinned runtime dependencies for FastAPI, OpenAI, HTTP extraction, HTML parsing, Playwright, dotenv, and validation.

### Backend application

- `back-end/home.py` — the backend composition root: configuration, FastAPI middleware, request/response models, SSRF defenses, HTML extraction, OpenAI calls, response normalization, health routes, and analysis routes.
- `back-end/ai_prompts.py` — the bias/research instructions and strict JSON schemas passed to OpenAI; it defines the semantic contract that `home.py` validates.
- `back-end/podcast.py` — podcast page/RSS discovery, publisher transcript parsers, canonical speaker-turn contracts, bounded public downloads, FFmpeg chunking, and OpenAI diarized transcription helpers.
- `back-end/run.sh` — currently empty; reserved as a possible backend launch script. The active launch commands live in the Dockerfile and `home.py`.

### Production Chrome extension

- `front-end/manifest.json` — Manifest V3 metadata, popup/service-worker entrypoints, Chrome permissions, and the exact production backend host permission.
- `front-end/extension.html` — popup DOM skeleton: Article/Podcast selector, action button, status/progress area, analysis cards, and expandable transcript container.
- `front-end/extension.css` — popup design tokens, layout, progress animation, results, bias-chip tooltips, source cards, and responsive sizing.
- `front-end/config.js` — constructs the ordered, deduplicated list of backend base URLs from the runtime override and hosted production URL.
- `front-end/api.js` — low-level fetch adapter that builds endpoint URLs, tries configured backends, parses JSON/text errors, and throws consistent JavaScript errors.
- `front-end/backendClient.js` — small domain API over `api.js` for health, article extraction/analysis, podcast jobs, polling, and transcript pagination.
- `front-end/podcast.js` — pure podcast cache-key, compact-result, pagination, timestamp, and progress-label helpers shared by the worker, popup, and Node tests.
- `front-end/article.js` — pure article cache-key, resumability, result, and progress-label helpers.
- `front-end/background.js` — one-shot MV3 controller: captures optional page text, creates backend article jobs, stores their IDs, and releases its event so the worker can become idle.
- `front-end/popup.js` — popup lifecycle controller: restores the selected mode, polls backend-owned article jobs directly while open, watches saved state, pages transcript turns, and seeks accessible page media by timestamp.
- `front-end/parseScript.js` — UI/rendering and page-injection helpers for text normalization, bias phrase highlighting, result cards, tooltips, and citation links. Its exported `parseText` function is an older direct-from-popup pipeline; current production execution is owned by `background.js`.

### Test and evaluation harness

- `back-end/testingcode/crawlparse.py` — downloads a CC-NEWS WARC path index, streams one Common Crawl archive with `warcio`, deduplicates target article URLs, and prints a JSON URL sample.
- `back-end/testingcode/testing.py` — checks the local backend, builds a temporary extension, launches persistent Chromium, tests a random URL batch, validates production-renderer UI contracts, records extraction/analysis metadata and research coverage, prints aggregates, and writes a timestamped report. Set `FACTGPT_E2E_BIAS_ONLY=0` to include paid research checks.
- `back-end/testingcode/testing-requirements.txt` — extra packages used only by the crawl/evaluation harness (`warcio` and Playwright).
- `back-end/testingcode/url.txt` — currently empty; an unused placeholder for URL test data.
- `back-end/testingcode/test-extension-overrides/README.md` — explains why the temporary test extension differs from production and warns not to ship the overrides.
- `back-end/testingcode/test-extension-overrides/manifest.json` — test manifest with localhost/cloud-host permissions and a discoverable background worker.
- `back-end/testingcode/test-extension-overrides/background.js` — minimal test worker that lets Playwright discover the generated extension ID.
- `back-end/testingcode/test-extension-overrides/config.js` — replaces production backend configuration with a localhost-first test configuration.
- `back-end/testingcode/test-extension-overrides/parseScript.js` — test analysis pipeline with detailed extraction attempts, error stages, previews, and optional bias-only execution; it delegates result rendering to a preserved copy of the production module.
- `back-end/testingcode/test-extension-overrides/popup.js` — accepts test controls through query parameters, auto-runs analyses, applies fallbacks, and exposes `window.__FACTGPT_TEST_STATE__` for Playwright.

### Operations documentation and checks

- `docs/render-keepalive.md` — instructions for monitoring `/health` with UptimeRobot and an explanation of the cost/cold-start tradeoff.
- `docs/architecture.md` — this architecture definition, runtime walkthrough, and complete tracked-file inventory.
- `scripts/check-render-health.sh` — calls a deployed `/health`, requires HTTP 200 and JSON `status: ok`, and never invokes an OpenAI-backed route.

### Packaged release snapshot

`dist/chrome-extension-v1.0.1/` is a checked-in release snapshot. At the time of this guide it mirrors the ten runtime files in `front-end/` byte for byte, including a manifest version of `1.1.0`; the directory name is therefore historical and potentially confusing.

- `dist/chrome-extension-v1.0.1/manifest.json` — packaged copy of `front-end/manifest.json`.
- `dist/chrome-extension-v1.0.1/extension.html` — packaged copy of `front-end/extension.html`.
- `dist/chrome-extension-v1.0.1/extension.css` — packaged copy of `front-end/extension.css`.
- `dist/chrome-extension-v1.0.1/config.js` — packaged copy of `front-end/config.js`.
- `dist/chrome-extension-v1.0.1/api.js` — packaged copy of `front-end/api.js`.
- `dist/chrome-extension-v1.0.1/backendClient.js` — packaged copy of `front-end/backendClient.js`.
- `dist/chrome-extension-v1.0.1/background.js` — packaged copy of `front-end/background.js`.
- `dist/chrome-extension-v1.0.1/popup.js` — packaged copy of `front-end/popup.js`.
- `dist/chrome-extension-v1.0.1/parseScript.js` — packaged copy of `front-end/parseScript.js`.
- `dist/chrome-extension-v1.0.1/podcast.js` — packaged copy of `front-end/podcast.js`.

## Local and generated files outside Git

The workspace also contains ignored artifacts: `.venv/`, Python caches, `.DS_Store` files, `back-end/data/apikey.env`, timestamped Playwright reports, backup frontend snapshots, extension ZIP archives, and `warc.paths.gz`. They are local dependencies, secrets, generated evidence, backups, or release outputs rather than additional architecture components.

## Important maintenance notes

- Treat `front-end/` as the production extension source. If `dist/` remains checked in, regenerate or synchronize it deliberately so reviews do not compare stale copies.
- The `dist/chrome-extension-v1.0.1` directory name is historical and does not match its current `1.1.2` manifest.
- Production uses `background.js` for orchestration; the similarly named `parseText` pipeline in `parseScript.js` is not called by the shipping popup and can drift unless it is removed or explicitly retained for compatibility.
- The tracked Python suites cover classification routing, quote/speaker attribution, signed handoffs, research provenance, article/podcast job contracts, discovery/parsing, media bounds, and window routing. The frontend Node suites cover presenters, article helpers and worker lifecycle, and podcast storage/pagination helpers.
- The integration harness is still live and network-dependent. It now exercises the production result renderer, but a passing report is not a regression test of the shipping service-worker/storage lifecycle.
- Chrome exposes no supported API for an extension worker to terminate itself. Article analysis therefore uses one-shot worker events and backend-owned jobs. Once the create request completes, the worker has no article keepalive; Chrome may terminate it on its normal idle schedule. The current backend job registry is process-local, so a Render restart or multi-worker routing can lose an active job; the popup turns that 404 into a retryable error. Shared Redis/database queueing is required for persistence across backend restarts.
