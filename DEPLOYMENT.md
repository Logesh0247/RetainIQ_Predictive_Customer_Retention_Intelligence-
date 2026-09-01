# Deploying RetainIQ

The app is a plain WSGI Flask app (`app:app`) served by gunicorn. Everything it
needs to boot is in the repository — no database, no external services.

## Start command

```
gunicorn app:app --config gunicorn.conf.py
```

`gunicorn.conf.py` is auto-loaded when gunicorn starts in the repo root, so the
same settings apply from the Procfile, the Dockerfile, or a platform start
command. Set it as the **Start Command** on Render / Railway / Fly, or keep the
Procfile on Heroku.

Key settings (all overridable with environment variables):

| Setting | Default | Why |
| --- | --- | --- |
| `bind` | `0.0.0.0:$PORT` (8000 fallback) | Hosted platforms inject `$PORT`; binding to a fixed port makes the platform report "no open ports". |
| `timeout` | `300` (`GUNICORN_TIMEOUT`) | Scoring a large CSV or training the universal churn model takes ~1–6 s on a laptop but far longer on a shared-CPU instance. gunicorn's 30 s default kills the worker mid-upload, which the browser shows as a 502 / "Application error". |
| `workers` | `1` (`WEB_CONCURRENCY`) | pandas + scikit-learn use ~180 MB per worker; one worker fits a 512 MB instance. |
| `threads` | `4` (`GUNICORN_THREADS`) | Keeps the UI responsive while a long scoring request runs. |
| `max_requests` | `300` | Recycles workers so slow memory growth cannot trigger an OOM kill. |
| `preload_app` | `True` | Model is loaded once before forking. |

## Environment variables

| Variable | Purpose |
| --- | --- |
| `RETAINIQ_SECRET_KEY` | Flask session signing key. **Set this in production**; otherwise sessions reset on every redeploy. |
| `RETAINIQ_DATA_DIR` | Optional writable directory for generated reports/uploads (use it when a persistent disk is mounted, e.g. `/var/data`). |
| `WEB_CONCURRENCY`, `GUNICORN_TIMEOUT`, `GUNICORN_THREADS` | gunicorn tuning, see above. |

## Runtime storage

Generated CSVs and cached scoring runs are written under `reports/runs/`.
Hosted instances usually have an **ephemeral** filesystem, so:

* if the application directory is read-only, the app falls back to
  `$RETAINIQ_DATA_DIR` and then to a temp directory instead of failing at import;
* the most recent scored runs are also held in memory, so the dashboard and the
  CSV download keep working even after the disk copy disappears;
* after an instance restart or a redeploy, older runs are gone by design — mount
  a persistent disk and point `RETAINIQ_DATA_DIR` at it if history must survive.

## Python version

Python 3.11 is pinned in three places, keep them in sync: `.python-version`
(Render/pyenv), `runtime.txt` (Heroku) and the `DockerFile` base image.
