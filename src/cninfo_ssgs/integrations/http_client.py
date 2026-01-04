from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    base_sleep_s: float = 1.0
    max_sleep_s: float = 8.0


class RateLimiter:
    def __init__(self, min_interval_s: float = 0.4, jitter_s: float = 0.2) -> None:
        self._min_interval_s = max(0.0, float(min_interval_s))
        self._jitter_s = max(0.0, float(jitter_s))
        self._lock = threading.Lock()
        self._last_ts = 0.0

    def wait(self) -> None:
        if self._min_interval_s <= 0:
            return
        with self._lock:
            now = time.time()
            elapsed = now - self._last_ts
            need = self._min_interval_s - elapsed
            if need > 0:
                time.sleep(need + random.random() * self._jitter_s)
            self._last_ts = time.time()


class HttpClient:
    def __init__(
        self,
        session: requests.Session | None = None,
        retry: RetryConfig | None = None,
        rate_limiter: RateLimiter | None = None,
        default_timeout_s: float = 30.0,
    ) -> None:
        self.session = session or requests.Session()
        self.retry = retry or RetryConfig()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.default_timeout_s = default_timeout_s

    def get(self, url: str, *, headers: dict[str, str] | None = None, timeout_s: float | None = None, stream: bool = False) -> requests.Response:
        return self._request("GET", url, headers=headers, timeout_s=timeout_s, stream=stream)

    def post_form(
        self,
        url: str,
        *,
        data: dict[str, Any],
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> requests.Response:
        return self._request("POST", url, headers=headers, timeout_s=timeout_s, data=data)

    def _request(self, method: str, url: str, *, headers: dict[str, str] | None, timeout_s: float | None, **kwargs: Any) -> requests.Response:
        timeout = float(timeout_s or self.default_timeout_s)
        last_err: Exception | None = None

        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                self.rate_limiter.wait()
                resp = self.session.request(method, url, headers=headers, timeout=timeout, **kwargs)
                resp.raise_for_status()
                return resp
            except Exception as exc:
                last_err = exc
                if attempt >= self.retry.max_attempts:
                    break
                sleep_s = min(self.retry.max_sleep_s, self.retry.base_sleep_s * (2 ** (attempt - 1)))
                time.sleep(sleep_s + random.random() * 0.3)

        assert last_err is not None
        raise last_err
