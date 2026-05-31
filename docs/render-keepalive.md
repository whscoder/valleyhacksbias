# Render Keepalive With UptimeRobot

This project already exposes a cheap health endpoint:

```text
https://bias-article-detector.onrender.com/health
```

Use that URL for uptime monitoring. Do not monitor `/analyze`, `/analyze-bias`, `/research`, `/extract`, or `/extract-rendered`; those endpoints can do real work and may consume OpenAI tokens or browser resources.

## Recommended Setup

1. Create a free UptimeRobot account.
2. Add a new monitor.
3. Set monitor type to `HTTP(s)`.
4. Set URL to:

   ```text
   https://bias-article-detector.onrender.com/health
   ```

5. Set interval to the free/default interval, typically 5 minutes.
6. If UptimeRobot shows an HTTP method setting, use `GET`. The backend also accepts `HEAD` for `/health`, but `GET` is better if you want to inspect the JSON body.
7. Use keyword monitoring only if available on your plan and the monitor method is `GET`. If enabled, require this text in the response:

   ```text
   "status":"ok"
   ```

8. Save the monitor.

## Cost Behavior

- OpenAI API tokens: none. `/health` only returns local uptime and status data.
- GitHub Actions minutes: none.
- Render instance hours: yes. Keeping the service warm 24/7 uses about 720 to 744 hours per month.

That tradeoff is intentional: avoiding Render cold starts means preventing the service from sleeping. Since this is the only Render service planned for the account, it should fit within a single free-service monthly hour budget.

## Verify The URL Locally

Run:

```sh
sh scripts/check-render-health.sh
```

Or test a different backend URL:

```sh
sh scripts/check-render-health.sh https://your-render-service.onrender.com
```

The script calls `/health`, checks for HTTP 200, and confirms the response includes `"status":"ok"`.
