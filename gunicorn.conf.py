"""
Gunicorn settings for RetainIQ in production.

Gunicorn auto-loads this file when it is present in the working directory, so
these settings apply whether the process is started from the Procfile, the
Dockerfile, or a platform start command such as `gunicorn app:app`.

Why this exists: the defaults (30 s request timeout, no keep-alive tuning) are
fine on a laptop but not on a small shared-CPU instance. Scoring a large CSV or
training the universal churn model takes a couple of seconds locally and can
take far longer on a 0.1 vCPU host — gunicorn then kills the worker mid-request
and the browser shows a 502 / "Application error" even though the same upload
works locally.
"""

import multiprocessing
import os


def _int_env(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Bind to the port the platform hands us (Render, Heroku, Fly, Cloud Run...).
bind = os.environ.get("GUNICORN_BIND") or f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# One worker by default: the app keeps the scored run in memory, and small
# instances (512 MB) cannot afford several copies of pandas/scikit-learn.
workers = _int_env("WEB_CONCURRENCY", 1)
threads = _int_env("GUNICORN_THREADS", 4)
worker_class = os.environ.get("GUNICORN_WORKER_CLASS", "gthread")

# Long-running uploads: model scoring on shared CPU is much slower than local.
timeout = _int_env("GUNICORN_TIMEOUT", 300)
graceful_timeout = _int_env("GUNICORN_GRACEFUL_TIMEOUT", 60)
keepalive = _int_env("GUNICORN_KEEPALIVE", 15)

# Recycle workers periodically so a long-lived instance cannot creep past the
# memory limit and get OOM-killed (which shows up as a random 502).
max_requests = _int_env("GUNICORN_MAX_REQUESTS", 300)
max_requests_jitter = _int_env("GUNICORN_MAX_REQUESTS_JITTER", 50)

# Load the app before forking: the model is read once, and workers share it.
preload_app = True

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")

# Trust the platform's proxy headers (Render/Heroku terminate TLS upstream).
forwarded_allow_ips = os.environ.get("FORWARDED_ALLOW_IPS", "*")

_ = multiprocessing  # kept for operators who want to scale workers by CPU count
