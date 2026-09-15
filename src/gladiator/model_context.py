from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from gladiator import __version__
from gladiator.config import ProviderConfig

_MIN_CONTEXT_TOKENS = 1_024
_MAX_REASONABLE_CONTEXT_TOKENS = 100_000_000

_EFFECTIVE_CONTEXT_KEYS = (
    "context_window",
    "context_length",
    "max_context_length",
    "max_model_len",
    "max_sequence_length",
    "max_position_embeddings",
    "n_ctx",
)
_MAX_CONTEXT_KEYS = (
    "max_context_window",
    "maximum_context_window",
    "max_context_length",
    "max_model_len",
    "max_sequence_length",
    "max_position_embeddings",
    "n_ctx",
)


@dataclass(frozen=True, slots=True)
class ModelContextInfo:
    model: str
    context_window: int
    max_context_window: int
    source: str

    @property
    def has_distinct_maximum(self) -> bool:
        return self.max_context_window != self.context_window


def _token_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float) and value.is_integer():
        parsed = int(value)
    elif isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("_", "")
        if not cleaned.isdigit():
            return None
        parsed = int(cleaned)
    else:
        return None
    if _MIN_CONTEXT_TOKENS <= parsed <= _MAX_REASONABLE_CONTEXT_TOKENS:
        return parsed
    return None


def _nested_dicts(value: dict[str, Any], *, depth: int = 0) -> list[dict[str, Any]]:
    if depth >= 4:
        return []
    children: list[dict[str, Any]] = []
    for child in value.values():
        if isinstance(child, dict):
            children.append(child)
            children.extend(_nested_dicts(child, depth=depth + 1))
    return children


def _first_limit(record: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    scopes = [record, *_nested_dicts(record)]
    for key in keys:
        for scope in scopes:
            if key in scope:
                parsed = _token_count(scope.get(key))
                if parsed is not None:
                    return parsed
    return None


def extract_model_context(record: dict[str, Any], *, model: str, source: str) -> ModelContextInfo | None:
    effective = _first_limit(record, _EFFECTIVE_CONTEXT_KEYS)
    maximum = _first_limit(record, _MAX_CONTEXT_KEYS)
    if effective is None and maximum is None:
        return None
    if effective is None:
        effective = maximum
    if maximum is None:
        maximum = effective
    assert effective is not None and maximum is not None
    maximum = max(maximum, effective)
    return ModelContextInfo(
        model=model,
        context_window=effective,
        max_context_window=maximum,
        source=source,
    )


def _model_record(payload: Any, model: str) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        for collection_key in ("data", "models"):
            collection = payload.get(collection_key)
            if isinstance(collection, list):
                for item in collection:
                    if not isinstance(item, dict):
                        continue
                    identifiers = (item.get("id"), item.get("slug"), item.get("model"))
                    if any(str(identifier) == model for identifier in identifiers if identifier is not None):
                        return item
        identifiers = (payload.get("id"), payload.get("slug"), payload.get("model"))
        if any(str(identifier) == model for identifier in identifiers if identifier is not None):
            return payload
        nested = payload.get("data")
        if isinstance(nested, dict):
            return _model_record(nested, model)
    return None


def _normalized_client_version() -> str:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", __version__)
    return ".".join(match.groups()) if match else "0.1.0"


def _discover_openai_compatible(provider: ProviderConfig, *, timeout_seconds: float) -> ModelContextInfo | None:
    base_url = provider.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {provider.api_key.get_secret_value()}"}
    model_path = quote(provider.model, safe="")
    urls = (
        (f"{base_url}/models/{model_path}", "api:/models/{id}"),
        (f"{base_url}/models", "api:/models"),
    )
    timeout = httpx.Timeout(timeout_seconds, connect=min(5.0, timeout_seconds))
    with httpx.Client(timeout=timeout) as client:
        for url, source in urls:
            try:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, ValueError):
                continue
            record = _model_record(payload, provider.model)
            if record is None and isinstance(payload, dict) and source.endswith("{id}"):
                record = payload
            if record is None:
                continue
            info = extract_model_context(record, model=provider.model, source=source)
            if info is not None:
                return info
    return None


def _codex_models_url(provider: ProviderConfig) -> str:
    responses_url = provider.codex_responses_url.rstrip("/")
    base = responses_url.rsplit("/", 1)[0] if responses_url.endswith("/responses") else responses_url
    return f"{base}/models"


def _discover_codex(provider: ProviderConfig, *, timeout_seconds: float) -> ModelContextInfo | None:
    token = provider.codex_access_token.get_secret_value()
    account_id = provider.codex_account_id
    if not token or not account_id:
        return None
    headers = {
        "Authorization": f"Bearer {token}",
        "ChatGPT-Account-ID": account_id,
        "Accept": "application/json",
    }
    params = {"client_version": _normalized_client_version()}
    timeout = httpx.Timeout(timeout_seconds, connect=min(5.0, timeout_seconds))
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(_codex_models_url(provider), headers=headers, params=params)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    record = _model_record(payload, provider.model)
    if record is None:
        return None
    return extract_model_context(record, model=provider.model, source="api:/models")


def discover_model_context(provider: ProviderConfig, *, timeout_seconds: float = 10.0) -> ModelContextInfo | None:
    """Discover model context limits from the currently selected provider API.

    No model-name lookup table is used. If the endpoint does not advertise a usable
    limit, this returns None and Gladiator leaves the context window unknown.
    """
    if provider.mode == "codex_oauth":
        return _discover_codex(provider, timeout_seconds=timeout_seconds)
    return _discover_openai_compatible(provider, timeout_seconds=timeout_seconds)


def clear_model_context(provider: ProviderConfig) -> None:
    provider.context_window = None
    provider.max_context_window = None
    provider.context_window_source = None


def apply_model_context(provider: ProviderConfig, info: ModelContextInfo) -> None:
    if info.model != provider.model:
        raise ValueError("model metadata does not match the selected model")
    provider.context_window = info.context_window
    provider.max_context_window = info.max_context_window
    provider.context_window_source = info.source


def format_model_context(provider: ProviderConfig) -> str:
    effective = provider.context_window
    maximum = provider.max_context_window
    if effective is None and maximum is None:
        return "Context window: not reported by provider"
    if effective is None:
        return f"Maximum context: {maximum:,} tokens" if maximum is not None else "Context window: unknown"
    if maximum is not None and maximum != effective:
        return f"Context window: {effective:,} tokens · maximum advertised: {maximum:,} tokens"
    return f"Context window: {effective:,} tokens"
