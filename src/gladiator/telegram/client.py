from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx


class TelegramAPIError(RuntimeError):
    pass


class TelegramClient:
    def __init__(self, token: str, *, timeout_seconds: float = 60.0):
        self.token = token
        self.api_base = f"https://api.telegram.org/bot{token}"
        self.file_base = f"https://api.telegram.org/file/bot{token}"
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=20.0))

    async def close(self) -> None:
        await self.client.aclose()

    async def _call(self, method: str, payload: dict[str, Any] | None = None) -> Any:
        response = await self.client.post(f"{self.api_base}/{method}", json=payload or {})
        response.raise_for_status()
        body = response.json()
        if not body.get("ok"):
            raise TelegramAPIError(f"{method} failed: {body}")
        return body.get("result")

    async def get_updates(self, *, offset: int | None = None, timeout: int = 30) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "timeout": timeout,
            "allowed_updates": ["message", "callback_query", "stopped_message_generation"],
        }
        if offset is not None:
            payload["offset"] = offset
        result = await self._call("getUpdates", payload)
        return result if isinstance(result, list) else []

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        parse_mode: str | None = "HTML",
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        return await self._call("sendMessage", payload)

    async def edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        *,
        parse_mode: str | None = "HTML",
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any] | bool:
        payload: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        return await self._call("editMessageText", payload)

    async def edit_message_reply_markup(
        self,
        chat_id: int,
        message_id: int,
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any] | bool:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "reply_markup": reply_markup or {"inline_keyboard": []},
        }
        return await self._call("editMessageReplyMarkup", payload)

    async def answer_callback_query(self, callback_query_id: str, text: str | None = None) -> bool:
        payload: dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        return bool(await self._call("answerCallbackQuery", payload))

    async def send_message_draft(
        self,
        chat_id: int,
        draft_id: int,
        text: str = "",
        *,
        parse_mode: str | None = "HTML",
        can_stop: bool = True,
    ) -> bool:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "draft_id": draft_id,
            "text": text,
            "can_stop": can_stop,
        }
        if parse_mode and text:
            payload["parse_mode"] = parse_mode
        return bool(await self._call("sendMessageDraft", payload))

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> bool:
        return bool(await self._call("sendChatAction", {"chat_id": chat_id, "action": action}))

    async def delete_message(self, chat_id: int, message_id: int) -> bool:
        return bool(await self._call("deleteMessage", {"chat_id": chat_id, "message_id": message_id}))

    async def get_file_path(self, file_id: str) -> str:
        result = await self._call("getFile", {"file_id": file_id})
        path = result.get("file_path") if isinstance(result, dict) else None
        if not path:
            raise TelegramAPIError("Telegram getFile returned no file_path")
        return str(path)

    async def download_file(self, file_id: str, destination: Path) -> Path:
        remote_path = await self.get_file_path(file_id)
        response = await self.client.get(f"{self.file_base}/{remote_path}")
        response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        return destination

    async def send_document(self, chat_id: int, path: Path, *, caption: str | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption
        with path.open("rb") as handle:
            response = await self.client.post(
                f"{self.api_base}/sendDocument",
                data=data,
                files={"document": (path.name, handle, "application/octet-stream")},
            )
        response.raise_for_status()
        body = response.json()
        if not body.get("ok"):
            raise TelegramAPIError(f"sendDocument failed: {body}")
        return body["result"]

    async def send_photo(self, chat_id: int, path: Path, *, caption: str | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption
        with path.open("rb") as handle:
            response = await self.client.post(
                f"{self.api_base}/sendPhoto",
                data=data,
                files={"photo": (path.name, handle, "application/octet-stream")},
            )
        response.raise_for_status()
        body = response.json()
        if not body.get("ok"):
            raise TelegramAPIError(f"sendPhoto failed: {body}")
        return body["result"]
