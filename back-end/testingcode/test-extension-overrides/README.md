These files preserve the Playwright-specific extension hooks used during local testing.

How it works:
- `front-end/` is the clean production extension intended for shipping.
- `back-end/testingcode/testing.py` copies `front-end/` into a temporary bundle.
- The files in this folder are then copied on top of that bundle so Playwright can:
  - auto-run the popup
  - pass a test URL through query params
  - expose structured test state
  - load a background service worker for extension ID discovery

Do not ship these override files to the Chrome Web Store bundle.
