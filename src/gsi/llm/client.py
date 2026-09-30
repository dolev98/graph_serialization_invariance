from __future__ import annotations

import json
import os
import random
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from gsi.prompts.modes import Prompt

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class ModelSpec:
    name: str
    provider: str = "openai"
    base_url: str | None = None
    api_key_env: str | None = None
    model: str | None = None
    max_concurrency: int = 4
    max_tokens: int = 4096
    timeout: float = 180.0
    max_retries: int = 5
    extra: dict = field(default_factory=dict)
    # Appended to the user message at call time, for model-specific control tokens such as
    # Qwen3's "/no_think". The Prompt object, its hash and the response log keep the shared,
    # model-independent text; only the bytes sent to this one model differ.
    user_suffix: str = ""
    stub: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "ModelSpec":
        return cls(name=name, **d)


@dataclass
class LLMResponse:
    prompt_id: str
    model: str
    prompt_hash: str
    raw_text: str
    code: str | None
    usage: dict
    latency_s: float
    cached: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# CodeGraph's exec_py: the code is whatever sits between the markers; no other fallback.
_MARKERS = re.compile(r"#\s*CODE\s+START\s*\n(.*?)#\s*CODE\s+END", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str | None:
    if not text:
        return None
    m = _MARKERS.findall(text)
    if not m:
        return None
    return max(m, key=len).strip("\n")


class LLMClient:
    def __init__(self, spec: ModelSpec, cache_dir: str | Path):
        self.spec = spec
        self.cache_dir = Path(cache_dir) / spec.name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._locks_guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def _openai(self):
        if self._client is None:
            from openai import OpenAI
            key = os.environ.get(self.spec.api_key_env or "", "")
            if not key:
                raise RuntimeError(f"environment variable {self.spec.api_key_env} is not set for model {self.spec.name}")
            self._client = OpenAI(base_url=self.spec.base_url, api_key=key, timeout=self.spec.timeout, max_retries=0)
        return self._client

    def _cache_path(self, prompt: Prompt) -> Path:
        # repeat>0 gets its own entry so an identical prompt can actually be re-asked; the hash,
        # and therefore the bytes sent to the model, are the same for every repeat.
        suffix = f".r{prompt.repeat}" if getattr(prompt, "repeat", 0) else ""
        return self.cache_dir / f"{prompt.prompt_hash}{suffix}.json"

    def _from_cache(self, prompt: Prompt) -> LLMResponse | None:
        p = self._cache_path(prompt)
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            return None
        if d.get("model") not in (None, self.spec.model):
            # the cache folder is named after the config entry, so an entry whose model id changed would
            # otherwise be answered by the old model's replies
            raise RuntimeError(f"{p} holds a reply from {d['model']}, but {self.spec.name} is now {self.spec.model}; "
                               f"move {self.cache_dir} aside or restore the old model id")
        return LLMResponse(prompt_id=prompt.prompt_id, model=self.spec.name, prompt_hash=prompt.prompt_hash,
                           raw_text=d["raw_text"], code=extract_code(d["raw_text"]), usage=d.get("usage", {}),
                           latency_s=d.get("latency_s", 0.0), cached=True)

    def _to_cache(self, prompt: Prompt, raw_text: str, usage: dict, latency_s: float) -> None:
        from gsi.exec.sandbox import atomic_write_text
        atomic_write_text(self._cache_path(prompt), json.dumps({
            "raw_text": raw_text, "usage": usage, "latency_s": latency_s,
            "model": self.spec.model, "ts": time.time(), "prompt_id": prompt.prompt_id,
        }))

    def _key_lock(self, prompt: Prompt) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(prompt.prompt_hash, threading.Lock())

    def complete(self, prompt: Prompt, *, context: dict | None = None) -> LLMResponse:
        # identical prompts in flight at the same time are billed once
        with self._key_lock(prompt):
            cached = self._from_cache(prompt)
            if cached is not None:
                return cached
            raw_text, usage, latency = self._call(prompt)
            self._to_cache(prompt, raw_text, usage, latency)
        return LLMResponse(prompt_id=prompt.prompt_id, model=self.spec.name, prompt_hash=prompt.prompt_hash,
                           raw_text=raw_text, code=extract_code(raw_text), usage=usage, latency_s=latency)

    def messages(self, prompt: Prompt) -> list[dict]:
        user = prompt.user + (" " + self.spec.user_suffix if self.spec.user_suffix else "")
        return [{"role": "system", "content": prompt.system}, {"role": "user", "content": user}]

    def _call(self, prompt: Prompt) -> tuple[str, dict, float]:
        import openai

        client = self._openai()
        messages = self.messages(prompt)
        last_err: Exception | None = None
        for attempt in range(self.spec.max_retries):
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=self.spec.model, messages=messages, temperature=0,
                    max_tokens=self.spec.max_tokens, **self.spec.extra,
                )
                choice = resp.choices[0]
                text = choice.message.content or ""
                usage = resp.usage.model_dump() if resp.usage else {}
                # Recorded so a reply cut off at max_tokens is visible as truncation, not as an
                # inexplicable `format` failure. A thinking model that spends the budget on
                # reasoning returns empty content with finish_reason="length".
                usage["finish_reason"] = choice.finish_reason
                reasoning = getattr(choice.message, "reasoning_content", None) or \
                    (choice.message.model_extra or {}).get("reasoning_content") or \
                    (choice.message.model_extra or {}).get("reasoning")
                if reasoning:
                    usage["reasoning_chars"] = len(reasoning)
                served = getattr(resp, "model", None)
                if served and served != self.spec.model:
                    usage["served_model"] = served
                return text, usage, time.time() - t0
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as e:
                last_err = e
            except openai.APIStatusError as e:
                last_err = e
                if e.status_code not in RETRYABLE_STATUS:
                    raise
            wait = min(60, 2 ** attempt) + random.random()
            # visible, because a run that is silently sleeping through 429s looks identical to a
            # slow provider from the outside
            print(f"[{self.spec.name}] retry {attempt + 1}/{self.spec.max_retries} in {wait:.0f}s: "
                  f"{type(last_err).__name__}: {str(last_err)[:120]}", file=sys.stderr, flush=True)
            time.sleep(wait)
        raise RuntimeError(f"{self.spec.name}: giving up after {self.spec.max_retries} attempts: {last_err}")
