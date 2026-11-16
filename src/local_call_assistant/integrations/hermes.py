"""Send a finished transcript to Hermes for analysis.

Uses the OpenAI-compatible ``POST /v1/chat/completions`` endpoint of the
Hermes API server. Only transcript text is sent, never audio. Disabled by
default: enabling it sends call content to Hermes and therefore to whichever
model provider Hermes is configured to use.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import os
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit

from local_call_assistant.config import HermesConfig
from local_call_assistant.integrations.analysis import (
    SYSTEM_PROMPT, transcript_text, user_message, write_analysis,
)
from local_call_assistant.processing import StageContext, StageSkipped

PROVIDER = "hermes"
COMPLETIONS_PATH = "/v1/chat/completions"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class HermesError(RuntimeError):
    pass


def check_endpoint(base_url: str) -> str:
    """Require HTTPS, except plain HTTP to a loopback address. Returns the URL."""
    parts = urlsplit(base_url)
    if parts.scheme == "https" and parts.hostname:
        return base_url.rstrip("/") + COMPLETIONS_PATH
    if parts.scheme == "http" and _is_loopback(parts.hostname):
        return base_url.rstrip("/") + COMPLETIONS_PATH
    raise HermesError("Hermes base_url must use https (plain http only to localhost)")


def build_payload(config: HermesConfig, job_id: str, text: str) -> dict[str, Any]:
    return {
        "model": config.model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(job_id, text)},
        ],
    }


def analyse(config: HermesConfig, job_id: str, text: str) -> str:
    """Error messages are fixed texts: nothing from the server, the key or the
    transcript is ever copied into them, because they are stored in the manifest."""
    url = check_endpoint(config.base_url)
    key = _api_key(config.api_key_env)
    body = json.dumps(build_payload(config, job_id, text)).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with _opener().open(request, timeout=config.timeout_seconds) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise HermesError(f"Hermes returned HTTP {int(error.code)}") from None
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        raise HermesError("Hermes request failed (connection or protocol error)") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise HermesError("Hermes response too large")
    return _content_of(raw)


def analysis_handler(config: HermesConfig):
    def handle(context: StageContext) -> dict[str, Any]:
        if not config.enabled:
            raise StageSkipped("Hermes analysis is disabled in the configuration")
        markdown = analyse(config, context.manifest.job_id, transcript_text(context))
        return write_analysis(context, markdown, PROVIDER, config.model)
    return handle


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    # A redirect could send the bearer key to another host or over plain HTTP.
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        raise HermesError("Hermes answered with a redirect; refusing to follow it")


def _opener() -> urllib.request.OpenerDirector:
    # No proxies: an HTTP(S)_PROXY setting must not route transcripts and the
    # key through a third party. Hermes is self-hosted and reached directly.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _RefuseRedirects)


def _api_key(variable: str) -> str:
    key = os.environ.get(variable, "")
    if not key:
        raise HermesError(f"environment variable {variable} is not set")
    if any(ch in key for ch in "\r\n") or not key.isprintable():
        raise HermesError(f"environment variable {variable} contains invalid characters")
    return key


def _content_of(raw: bytes) -> str:
    try:
        content = json.loads(raw)["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise HermesError("Hermes response was not a chat completion") from None
    if not isinstance(content, str) or not content.strip():
        raise HermesError("Hermes returned an empty analysis")
    return content


def _is_loopback(host: str | None) -> bool:
    if host is None:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
