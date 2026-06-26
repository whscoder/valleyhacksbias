# Fact GPT

Chrome extension plus FastAPI backend for article bias and reliability analysis.

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
FACTGPT_REQUIRE_ALLOWED_ORIGIN
FACTGPT_ALLOWED_ORIGIN_REGEX
FACTGPT_RATE_LIMIT_PER_MINUTE
```

The backend also supports request and fetch caps through `FACTGPT_MAX_REQUEST_BODY_BYTES`, `FACTGPT_MAX_REQUEST_TEXT_CHARS`, `FACTGPT_MAX_FETCH_BYTES`, and `FACTGPT_MAX_EXTRACTED_TEXT_CHARS`.
