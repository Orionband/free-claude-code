"""NVIDIA NIM provider implementation."""

import json
import random
from typing import Any

import httpx
import openai
from loguru import logger
from openai import AsyncOpenAI

from config.nim import NimSettings
from providers.base import ProviderConfig
from providers.defaults import NVIDIA_NIM_DEFAULT_BASE
from providers.openai_compat import OpenAIChatTransport

from .keys import parse_nvidia_nim_api_keys
from .request import (
    body_without_nim_tool_argument_aliases,
    build_request_body,
    clone_body_without_chat_template,
    clone_body_without_reasoning_budget,
    clone_body_without_reasoning_content,
    nim_tool_argument_aliases_from_body,
)


class NvidiaNimProvider(OpenAIChatTransport):
    """NVIDIA NIM provider using official OpenAI client."""

    def __init__(self, config: ProviderConfig, *, nim_settings: NimSettings):
        self._api_keys = parse_nvidia_nim_api_keys(config.api_key)
        primary_key = self._api_keys[0] if self._api_keys else config.api_key
        base_url = config.base_url or NVIDIA_NIM_DEFAULT_BASE
        shared_http_client = None
        if config.proxy:
            shared_http_client = httpx.AsyncClient(
                proxy=config.proxy,
                timeout=httpx.Timeout(
                    config.http_read_timeout,
                    connect=config.http_connect_timeout,
                    read=config.http_read_timeout,
                    write=config.http_write_timeout,
                ),
            )
        super().__init__(
            config,
            provider_name="NIM",
            base_url=base_url,
            api_key=primary_key,
            http_client=shared_http_client,
        )
        self._nim_settings = nim_settings
        if len(self._api_keys) > 1:
            extra_clients = tuple(
                self._make_openai_client(
                    api_key=api_key,
                    base_url=self._base_url,
                    config=config,
                    http_client=shared_http_client,
                )
                for api_key in self._api_keys[1:]
            )
            self._clients: tuple[AsyncOpenAI, ...] = (self._client, *extra_clients)
        else:
            self._clients = (self._client,)

    def _openai_client(self) -> AsyncOpenAI:
        return random.choice(self._clients)

    async def cleanup(self) -> None:
        seen: set[int] = set()
        for client in self._clients:
            client_id = id(client)
            if client_id in seen:
                continue
            seen.add(client_id)
            await client.close()

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(
            request,
            self._nim_settings,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    def _prepare_create_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Strip private request metadata before calling NVIDIA NIM."""
        return body_without_nim_tool_argument_aliases(body)

    def _tool_argument_aliases(self, body: dict[str, Any]) -> dict[str, dict[str, str]]:
        """Return NIM tool argument aliases captured while building this request."""
        return nim_tool_argument_aliases_from_body(body)

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Retry once with a downgraded body when NIM rejects a known field."""
        status_code = getattr(error, "status_code", None)
        if not isinstance(error, openai.BadRequestError) and status_code != 400:
            return None

        error_text = str(error)
        error_body = getattr(error, "body", None)
        if error_body is not None:
            error_text = f"{error_text} {json.dumps(error_body, default=str)}"
        error_text = error_text.lower()

        if "reasoning_budget" in error_text:
            retry_body = clone_body_without_reasoning_budget(body)
            if retry_body is None:
                return None
            logger.warning(
                "NIM_STREAM: retrying without reasoning_budget after 400 error"
            )
            return retry_body

        if "chat_template" in error_text:
            retry_body = clone_body_without_chat_template(body)
            if retry_body is None:
                return None
            logger.warning("NIM_STREAM: retrying without chat_template after 400 error")
            return retry_body

        if "reasoning_content" in error_text:
            retry_body = clone_body_without_reasoning_content(body)
            if retry_body is None:
                return None
            logger.warning(
                "NIM_STREAM: retrying without reasoning_content after 400 error"
            )
            return retry_body

        return None
