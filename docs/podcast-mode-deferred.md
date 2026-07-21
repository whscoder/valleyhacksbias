# Podcast mode deferred

Podcast mode is intentionally unavailable in the extension popup for the
article-only release. The implementation remains in the repository so it can
be restored after the article workflow ships.

## Why it was deferred

The podcast test page on Crooked.com returned `403 Forbidden` when the backend
attempted to fetch the public episode page. That prevented transcript discovery
before the audio parsing and analysis stages could run. Article analysis is not
affected.

## Preserved implementation

- `front-end/podcast.js` contains transcript paging and timestamp helpers.
- `front-end/popup.js` retains the podcast rendering and resume functions, but
  the release forces Article mode.
- `front-end/background.js` retains podcast job and transcript message handlers.
- `back-end/home.py` and `back-end/podcast.py` retain podcast endpoints,
  transcript discovery, parsing, and analysis.
- Existing podcast tests remain unchanged.

## Re-enable checklist

1. Make podcast page fetching resilient to providers that reject ordinary HTTP
   clients, or use a supported transcript/audio source that does not return 403.
2. Restore the Article/Podcast selector and podcast transcript section in
   `front-end/extension.html`.
3. Allow `switchMode` and popup initialization to restore Podcast mode again.
4. Run podcast helper, background lifecycle, popup, and backend podcast tests.
5. Test one episode end to end, including transcript paging and timestamp seek.
6. Synchronize the verified popup files into the current packaged extension.
