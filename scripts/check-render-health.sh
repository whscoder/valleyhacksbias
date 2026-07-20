#!/bin/sh
# Verify that a deployed backend's cheap liveness endpoint is ready for monitoring.
set -eu

BASE_URL="${1:-https://bias-article-detector.onrender.com}"
BASE_URL="${BASE_URL%/}"
HEALTH_URL="${BASE_URL}/health"

TMP_BODY="$(mktemp)"
cleanup() {
  rm -f "$TMP_BODY"
}
trap cleanup EXIT

STATUS_CODE="$(
  curl \
    --silent \
    --show-error \
    --location \
    --max-time 20 \
    --output "$TMP_BODY" \
    --write-out "%{http_code}" \
    "$HEALTH_URL"
)"

if [ "$STATUS_CODE" != "200" ]; then
  echo "Health check failed: HTTP ${STATUS_CODE}"
  echo "URL: ${HEALTH_URL}"
  echo "Response:"
  sed -n '1,40p' "$TMP_BODY"
  exit 1
fi

if ! grep -q '"status"[[:space:]]*:[[:space:]]*"ok"' "$TMP_BODY"; then
  echo "Health check returned HTTP 200 but did not include status ok."
  echo "URL: ${HEALTH_URL}"
  echo "Response:"
  sed -n '1,40p' "$TMP_BODY"
  exit 1
fi

echo "Render health endpoint is ready for UptimeRobot."
echo "URL: ${HEALTH_URL}"
echo "This endpoint does not call OpenAI APIs."
