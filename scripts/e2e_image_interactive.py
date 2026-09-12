from __future__ import annotations

import asyncio
import base64
import io
import os
import tempfile
import time
from pathlib import Path

import httpx
from PIL import Image
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

from gladiator.config import (
    GladiatorConfig,
    ProviderConfig,
    RuntimeConfig,
    SearchConfig,
    TelegramConfig,
    save_config,
)
from gladiator.service_ext import ExtendedGladiatorService
from gladiator.telegram.bot import IncomingTask

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "dots-studio/dots-3-note-preview:free"
FINAL_PREFIX = "IMAGE_DESCRIPTION:"
OUTPUT_NAME = "gladiator_square.png"


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required E2E environment variable is missing: {name}")
    return value


def _synthetic_image_data_url() -> str:
    image = Image.new("RGB", (64, 48), (32, 96, 160))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _preflight_request(
    *,
    api_key: str,
    base_url: str,
    model: str,
    include_image: bool,
    app_attribution: bool,
) -> tuple[int, str]:
    content: str | list[dict]
    if include_image:
        content = [
            {"type": "text", "text": "Synthetic multimodal harness preflight. Inspect the attached color block and call the bash tool."},
            {"type": "image_url", "image_url": {"url": _synthetic_image_data_url()}},
        ]
    else:
        content = "Agentic harness preflight. Call the bash tool with command: printf preflight-ok\\n"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "tools": [BASH_TOOL],
        "tool_choice": "auto",
        "stream": False,
        "reasoning_effort": "low",
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if app_attribution:
        headers["HTTP-Referer"] = "https://github.com/invisiblestrangler/GladiatorAgent"
        headers["X-Title"] = "GladiatorAgent"

    with httpx.Client(timeout=httpx.Timeout(90.0, connect=20.0)) as client:
        response = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
    return response.status_code, response.text[:3000]


async def run_provider_preflight(*, api_key: str, base_url: str, model: str) -> None:
    probes = [
        ("text_tools_plain", False, False),
        ("text_tools_attributed", False, True),
        ("image_tools_attributed", True, True),
    ]
    failures: list[str] = []
    for name, include_image, app_attribution in probes:
        status, body = await asyncio.to_thread(
            _preflight_request,
            api_key=api_key,
            base_url=base_url,
            model=model,
            include_image=include_image,
            app_attribution=app_attribution,
        )
        print(f"MODEL_PREFLIGHT {name} status={status} body={body}")
        if status >= 400:
            failures.append(f"{name}: HTTP {status}: {body[:1200]}")

    if any(item.startswith("image_tools_attributed:") for item in failures):
        raise RuntimeError("Multimodal provider preflight failed: " + " | ".join(failures))


async def drain_pending_updates(service: ExtendedGladiatorService) -> int | None:
    """Acknowledge old Telegram updates so the test only accepts a newly sent image."""
    offset: int | None = None
    for _ in range(20):
        updates = await service.bot.client.get_updates(offset=offset, timeout=0)
        if not updates:
            return offset
        offset = max(int(update["update_id"]) for update in updates) + 1
    return offset


async def wait_for_new_image(
    service: ExtendedGladiatorService,
    *,
    user_id: int,
    offset: int | None,
    timeout_seconds: int = 600,
) -> IncomingTask:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        remaining = max(1, int(deadline - time.monotonic()))
        updates = await service.bot.client.get_updates(offset=offset, timeout=min(30, remaining))
        for update in updates:
            offset = int(update["update_id"]) + 1
            message = update.get("message")
            if not isinstance(message, dict):
                continue
            chat = message.get("chat") or {}
            if chat.get("id") != user_id or chat.get("type") != "private":
                continue
            document = message.get("document")
            is_image_document = isinstance(document, dict) and str(document.get("mime_type", "")).startswith("image/")
            if not message.get("photo") and not is_image_document:
                await service.bot.client.send_message(
                    user_id,
                    "Image E2E is waiting for a photo or image file. Please send an image attachment.",
                )
                continue
            incoming = await service.bot._build_task(message, "")
            if incoming.image_paths:
                return incoming
    raise TimeoutError("No new Telegram image was received within the interactive E2E window")


async def main() -> None:
    api_key = required_env("E2E_API_KEY")
    bot_token = required_env("GLADIATOR_TG_BOT")
    user_id = int(required_env("TG_USER_ID"))
    base_url = os.environ.get("E2E_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("E2E_MODEL", DEFAULT_MODEL)

    await run_provider_preflight(api_key=api_key, base_url=base_url, model=model)

    workspace = Path(tempfile.mkdtemp(prefix="gladiator-image-e2e-"))
    config_path = workspace / ".gladiator" / "e2e-image-config.json"
    config = GladiatorConfig(
        provider=ProviderConfig(
            name="image-e2e",
            base_url=base_url,
            api_key=api_key,
            model=model,
            reasoning_effort="low",
            context_window=1_048_576,
        ),
        telegram=TelegramConfig(bot_token=bot_token, allowed_user_ids=[user_id]),
        search=SearchConfig(mode="none"),
        runtime=RuntimeConfig(trace_mode="milestones", compact_threshold_tokens=300_000),
    )
    save_config(config, config_path)

    service = ExtendedGladiatorService(config=config, config_path=config_path, workspace=workspace)
    service._loop = asyncio.get_running_loop()
    service.agent.config.step_limit = 12
    service.agent.config.wall_time_limit_seconds = 240

    final_outputs: list[str] = []
    sent_artifacts: list[Path] = []
    original_send_markdown = service._send_markdown
    original_send_artifact = service._send_artifact

    async def capture_markdown(chat_id: int, text: str) -> None:
        final_outputs.append(text)
        await original_send_markdown(chat_id, text)

    async def capture_artifact(chat_id: int, path: Path, is_image: bool) -> None:
        if is_image:
            sent_artifacts.append(path)
        await original_send_artifact(chat_id, path, is_image)

    service._send_markdown = capture_markdown  # type: ignore[method-assign]
    service._send_artifact = capture_artifact  # type: ignore[method-assign]

    try:
        identity = await service.bot.client._call("getMe")
        if not isinstance(identity, dict) or not identity.get("id"):
            raise RuntimeError("Telegram getMe did not return a bot identity")

        offset = await drain_pending_updates(service)
        await service.bot.client.send_message(
            user_id,
            "<b>Gladiator image E2E is ready.</b>\n"
            "Send one NEW non-sensitive photo or image file now.\n\n"
            f"The test will use <code>{model}</code> to describe it, crop the largest centered square, "
            "verify the crop dimensions, and send the square image back.",
        )

        incoming = await wait_for_new_image(service, user_id=user_id, offset=offset)
        source = incoming.image_paths[0]
        with Image.open(source) as image:
            source_size = image.size

        output_path = workspace / OUTPUT_NAME
        task = IncomingTask(
            text=(
                "Interactive multimodal E2E test. Inspect the attached image yourself and describe what you see accurately. "
                f"The downloaded source image is also available at {source.resolve()}. Its decoded size is {source_size[0]}x{source_size[1]}. "
                "Then use bash and Python Pillow to crop the image to the largest possible CENTERED 1:1 square without stretching or adding borders. "
                f"Save the result exactly to {output_path.resolve()}. Re-open the saved image and verify width == height. "
                f"Then send the cropped image back to the user with `gladiator send {output_path.resolve()}` as a sole bash command. "
                f"After the artifact is sent, finish with a concise final response whose first line begins exactly `{FINAL_PREFIX}` followed by your visual description. "
                "On later lines state the original dimensions and the final square dimensions."
            ),
            image_paths=incoming.image_paths,
        )
        await service.handle_task(user_id, task)

        if not final_outputs or not final_outputs[-1].lstrip().startswith(FINAL_PREFIX):
            raise RuntimeError("Model did not return the required image-description final response")
        if not output_path.exists():
            raise RuntimeError("Agent did not create the expected square crop")
        with Image.open(output_path) as cropped:
            width, height = cropped.size
        if width != height or width <= 0:
            raise RuntimeError(f"Agent crop is not square: {width}x{height}")
        if output_path.resolve() not in {path.resolve() for path in sent_artifacts}:
            raise RuntimeError("Agent created the crop but did not send it back through Gladiator")

        await service.bot.client.send_message(
            user_id,
            "<b>Gladiator IMAGE E2E PASS</b>\n"
            f"Vision input ✓\nDescription ✓\nCentered square crop ✓ ({width}×{height})\nTelegram image return ✓",
        )
        print(f"GLADIATOR_IMAGE_E2E_PASS source={source_size[0]}x{source_size[1]} crop={width}x{height}")
    finally:
        await service.bot.client.close()


if __name__ == "__main__":
    asyncio.run(main())
