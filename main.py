from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import time
import uuid
import weakref
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger, star
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import File, Image, Plain
from astrbot.api.platform import MessageType
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import StarTools
from mcp.types import CallToolResult, ImageContent, TextContent

from .store import (
    IMAGE_FORMAT_UNRECOGNIZED_ERROR,
    IMAGE_TOO_LARGE_ERROR,
    ImageLocatorStore,
    ImageLookup,
    StoredImage,
)

PLUGIN_NAME = "astrbot_plugin_group_context_image_locator"
TOOL_NAME = "resolve_context_images"
FORWARD_TOOL_NAME = "forward_context_images"
LOCATOR_RE = re.compile(r"\bctximg:[0-9a-f]{16}\b")
LOCATOR_SUFFIX_RE = re.compile(r"[0-9a-f]{16}")
CONTEXT_MARKERS_EXTRA = "_group_context_image_locator_markers"
MODEL_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


class _LocalValidationError(ValueError):
    """A fixed, plugin-owned validation message that is safe to show and store."""


class GroupContextImageLocatorPlugin(star.Star):
    def __init__(self, context: star.Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        data_root = Path(StarTools.get_data_dir(PLUGIN_NAME))
        self.store = ImageLocatorStore(data_root)
        self._locator_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._cleanup_lock = asyncio.Lock()
        self._last_cleanup = 0.0

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=1)
    async def stage_group_context_image_locators(
        self,
        event: AstrMessageEvent,
    ) -> None:
        """Stage adjacent locator markers before native group-context recording."""
        _clear_staged_markers(event)
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return
        if not self._group_context_is_enabled(event):
            return

        messages = event.get_messages()
        if not isinstance(messages, list):
            return

        await self._maybe_prune()

        scope = _scope_for_event(event)
        message_id = _message_id_for_event(event)
        markers: list[Plain] = []
        image_index = 0
        staged: list[object] = []
        max_capture_images = self._bounded_int(
            self.config.get("max_capture_images_per_message", 10),
            1,
            50,
            10,
        )
        for component in tuple(messages):
            staged.append(component)
            if not isinstance(component, Image):
                continue
            if image_index >= max_capture_images:
                image_index += 1
                continue

            locator = _make_locator(scope, message_id, image_index)
            marker = Plain(text=f" [该图片提取码={locator}]")
            staged.append(marker)
            markers.append(marker)
            source = ""

            lock = self._locator_locks.get(locator)
            if lock is None:
                lock = asyncio.Lock()
                self._locator_locks[locator] = lock

            async with lock:
                existing = await asyncio.to_thread(
                    self.store.resolve_many,
                    scope,
                    [locator],
                )
                if not existing or existing[0].image is None:
                    await self._capture_component(
                        component=component,
                        scope=scope,
                        locator=locator,
                        message_id=message_id,
                        image_index=image_index,
                        source=source,
                    )
            image_index += 1

        if markers:
            messages[:] = staged
            event.set_extra(CONTEXT_MARKERS_EXTRA, markers)

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=-1)
    async def clear_staged_group_context_markers(
        self,
        event: AstrMessageEvent,
    ) -> None:
        """Remove staged markers after native group-context recording."""
        _clear_staged_markers(event)

    @filter.on_llm_request(priority=-1000)
    async def gate_locator_tool(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        """Hide locator tools when the effective request has no locator."""
        # A fail-safe for hosts or plugins that stop the normal message-handler chain
        # before the priority=-1 cleanup handler runs.
        _clear_staged_markers(event)
        if not self._config_bool("only_expose_when_referenced", True):
            return
        if _request_contains_locator(req):
            return
        if req.func_tool:
            req.func_tool.remove_tool(TOOL_NAME)
            req.func_tool.remove_tool(FORWARD_TOOL_NAME)

    @filter.llm_tool(name=TOOL_NAME)
    async def resolve_context_images(
        self,
        event: AstrMessageEvent,
        refs: list[str] | None = None,
    ) -> CallToolResult:
        """按群聊感知提取码取回原图供本轮模型查看；本工具不向聊天发送图片。

        契约：h ::= 16 位十六进制；r ::= "ctximg:" + h | h；
        R ::= [r1,...,rn]，1 <= n <= N；resolve(R) -> 与 R 同序的 results。
        status=resolved => 同一结果中附带对应原图；status=unavailable => 不附图。
        r 仅来自群聊感知图片描述中的“提取码=...”，
        r 不属于当前消息临时 ID、QQ file_id、"current" 或外部图片生成插件提取码域。

        Args:
            refs(list[string]): R；按需要查看的顺序填写群聊感知提取码。
        """
        max_images = self._bounded_int(
            self.config.get("max_images_per_call", 5),
            1,
            20,
            5,
        )
        normalized_refs = _normalize_refs(refs)
        if not normalized_refs:
            return _error_result("至少需要一个 ctximg 提取码。")
        if len(normalized_refs) > max_images:
            return _error_result(
                f"一次最多读取 {max_images} 张图片；本次收到 {len(normalized_refs)} 个提取码。"
            )

        scope = _scope_for_event(event)
        valid_refs = [ref for ref in normalized_refs if LOCATOR_RE.fullmatch(ref)]
        valid_lookups = await asyncio.to_thread(
            self.store.resolve_many,
            scope,
            valid_refs,
        )
        valid_lookup_iter = iter(valid_lookups)
        lookups = [
            next(valid_lookup_iter)
            if LOCATOR_RE.fullmatch(ref)
            else ImageLookup(ref, None, None, "", None, "提取码格式不正确")
            for ref in normalized_refs
        ]

        manifest: list[dict[str, Any]] = []
        content: list[TextContent | ImageContent] = []
        resolved_images: list[tuple[ImageLookup, bytes]] = []
        image_ordinal = 0
        for lookup in lookups:
            if lookup.image is None:
                manifest.append(
                    {
                        "ref": lookup.locator,
                        "status": "unavailable",
                        "error": lookup.error or "原图当前不可用",
                    }
                )
                continue
            if lookup.image.mime_type not in MODEL_IMAGE_MIME_TYPES:
                manifest.append(
                    {
                        "ref": lookup.locator,
                        "status": "unavailable",
                        "error": "该图片格式不能安全交给当前模型查看",
                    }
                )
                continue
            max_read_bytes = self._bounded_int(
                self.config.get("max_capture_image_mb", 32),
                1,
                128,
                32,
            ) * 1024 * 1024
            if lookup.image.size > max_read_bytes:
                manifest.append(
                    {
                        "ref": lookup.locator,
                        "status": "unavailable",
                        "error": "原图超过插件允许读取的单图大小",
                    }
                )
                continue
            try:
                data = await asyncio.to_thread(lookup.image.file_path.read_bytes)
            except OSError as exc:
                manifest.append(
                    {
                        "ref": lookup.locator,
                        "status": "unavailable",
                        "error": _safe_error(exc, "读取原图失败"),
                    }
                )
                continue
            if len(data) > max_read_bytes:
                manifest.append(
                    {
                        "ref": lookup.locator,
                        "status": "unavailable",
                        "error": IMAGE_TOO_LARGE_ERROR,
                    }
                )
                continue
            detected_mime_type = _detect_image_mime(data)
            if (
                detected_mime_type not in MODEL_IMAGE_MIME_TYPES
                or detected_mime_type != lookup.image.mime_type
            ):
                manifest.append(
                    {
                        "ref": lookup.locator,
                        "status": "unavailable",
                        "error": "该图片格式不能安全交给当前模型查看",
                    }
                )
                continue

            resolved_images.append((lookup, data))
            image_ordinal += 1
            manifest.append(
                {
                    "ref": lookup.locator,
                    "status": "resolved",
                    "image_ordinal": image_ordinal,
                    "mime_type": lookup.image.mime_type,
                    "size": len(data),
                    "sha256": lookup.image.blob_hash,
                }
            )

        content.append(
            TextContent(
                type="text",
                text=json.dumps({"results": manifest}, ensure_ascii=False),
            )
        )
        for lookup, data in resolved_images:
            assert lookup.image is not None
            content.append(
                ImageContent(
                    type="image",
                    data=base64.b64encode(data).decode("ascii"),
                    mimeType=lookup.image.mime_type,
                )
            )

        return CallToolResult(
            content=content,
            structuredContent={"results": manifest},
            isError=not resolved_images,
        )

    @filter.llm_tool(name=FORWARD_TOOL_NAME)
    async def forward_context_images(
        self,
        event: AstrMessageEvent,
        refs: list[str] | None = None,
        delivery: str = "inline",
    ) -> CallToolResult:
        """按群聊感知提取码把图片发送到当前聊天；本工具会产生真实发送。

        契约：r 与 R 的定义同 resolve_context_images；
        d ::= "inline" | "original_file"；forward(R,d) -> {sent, results}。
        d=inline => 作为聊天图片发送，QQ 可能压缩；
        d=original_file => 作为原文件发送，不压缩。
        只需转发已知 r 时可直接调用；需要先看图或按内容选择时，先 resolve(R)。
        r 不属于当前消息临时 ID、QQ file_id、"current" 或外部图片生成插件提取码域。

        Args:
            refs(list[string]): R；按实际发送顺序填写群聊感知提取码。
            delivery(string): d；inline 或 original_file。
        """
        mode = str(delivery or "inline").strip().lower()
        if mode not in {"inline", "original_file"}:
            return _error_result("delivery 只能是 inline 或 original_file。")
        max_images = self._bounded_int(
            self.config.get("max_images_per_call", 5), 1, 20, 5
        )
        normalized_refs = _normalize_refs(refs)
        if not normalized_refs:
            return _error_result("至少需要一个 ctximg 提取码。")
        if len(normalized_refs) > max_images:
            return _error_result(
                f"一次最多发送 {max_images} 张图片；本次收到 {len(normalized_refs)} 个提取码。"
            )

        scope = _scope_for_event(event)
        valid_refs = [ref for ref in normalized_refs if LOCATOR_RE.fullmatch(ref)]
        valid_lookups = await asyncio.to_thread(self.store.resolve_many, scope, valid_refs)
        valid_lookup_iter = iter(valid_lookups)
        lookups = [
            next(valid_lookup_iter)
            if LOCATOR_RE.fullmatch(ref)
            else ImageLookup(ref, None, None, "", None, "提取码格式不正确")
            for ref in normalized_refs
        ]

        manifest: list[dict[str, Any]] = []
        sent = 0
        for lookup in lookups:
            if lookup.image is None:
                manifest.append(
                    {
                        "ref": lookup.locator,
                        "status": "unavailable",
                        "error": lookup.error or "原图当前不可用",
                    }
                )
                continue
            try:
                component = _delivery_component(lookup, mode)
                message_id = await _send_component(event, component)
            except Exception as exc:
                logger.error(
                    "context image forwarding failed locator=%s",
                    lookup.locator,
                    exc_info=True,
                )
                manifest.append(
                    {
                        "ref": lookup.locator,
                        "status": "send_failed",
                        "error": _safe_error(exc, "发送失败"),
                    }
                )
                continue
            sent += 1
            manifest.append(
                {
                    "ref": lookup.locator,
                    "status": "sent",
                    "delivery": mode,
                    "message_id": message_id,
                    "sha256": lookup.image.blob_hash,
                }
            )

        payload = {"delivery": mode, "sent": sent, "results": manifest}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
            structuredContent=payload,
            isError=sent == 0,
        )

    async def _capture_component(
        self,
        *,
        component: Image,
        scope: str,
        locator: str,
        message_id: str,
        image_index: int,
        source: str,
    ) -> StoredImage | None:
        try:
            file_path = Path(await component.convert_to_file_path())
            max_bytes = self._bounded_int(
                self.config.get("max_capture_image_mb", 32),
                1,
                128,
                32,
            ) * 1024 * 1024
            size = await asyncio.to_thread(lambda: file_path.stat().st_size)
            if size > max_bytes:
                raise _LocalValidationError(IMAGE_TOO_LARGE_ERROR)
            data = await asyncio.to_thread(file_path.read_bytes)
            if len(data) > max_bytes:
                raise _LocalValidationError(IMAGE_TOO_LARGE_ERROR)
            mime_type = _detect_image_mime(data)
            if mime_type == "application/octet-stream":
                raise _LocalValidationError(IMAGE_FORMAT_UNRECOGNIZED_ERROR)
            return await asyncio.to_thread(
                self.store.put,
                scope=scope,
                locator=locator,
                message_id=message_id,
                image_index=image_index,
                source="",
                data=data,
                mime_type=mime_type,
            )
        except Exception as exc:
            error = _safe_error(exc, "图片捕获失败")
            await asyncio.to_thread(
                self.store.record_unresolved,
                scope=scope,
                locator=locator,
                message_id=message_id,
                image_index=image_index,
                source="",
                error=error,
            )
            logger.warning(
                "group context image capture failed: locator=%s error=%s",
                locator,
                error,
            )
            return None

    async def _maybe_prune(self) -> None:
        interval = self._bounded_int(
            self.config.get("cleanup_interval_seconds", 3600),
            60,
            86400,
            3600,
        )
        now = time.monotonic()
        if now - self._last_cleanup < interval or self._cleanup_lock.locked():
            return

        async with self._cleanup_lock:
            now = time.monotonic()
            if now - self._last_cleanup < interval:
                return
            retention_hours = self._bounded_int(
                self.config.get("retention_hours", 168),
                0,
                8760,
                168,
            )
            max_cache_mb = self._bounded_int(
                self.config.get("max_cache_mb", 1024),
                0,
                102400,
                1024,
            )
            try:
                result = await asyncio.to_thread(
                    self.store.prune,
                    retention_seconds=retention_hours * 3600,
                    max_bytes=max_cache_mb * 1024 * 1024,
                )
            except Exception:
                logger.warning(
                    "group context image cleanup failed; capture continues",
                    exc_info=True,
                )
                return
            self._last_cleanup = now
            if result.blobs_removed:
                logger.info(
                    "group context image cache pruned: occurrences=%s blobs=%s bytes=%s",
                    result.occurrences_removed,
                    result.blobs_removed,
                    result.bytes_removed,
                )

    def _config_bool(self, key: str, default: bool) -> bool:
        value = self.config.get(key, default)
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "off", "no"}
        return bool(value)

    def _group_context_is_enabled(self, event: AstrMessageEvent) -> bool:
        try:
            config = self.context.get_config(umo=event.unified_msg_origin)
            settings = config.get("provider_ltm_settings", {})
            return bool(settings.get("group_icl_enable", False))
        except Exception:
            logger.warning(
                "group context image capture skipped: group_icl_enable unavailable",
                exc_info=True,
            )
            return False

    @staticmethod
    def _bounded_int(
        value: Any,
        minimum: int,
        maximum: int,
        default: int,
    ) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, parsed))


def _scope_for_event(event: AstrMessageEvent) -> str:
    raw_scope = "\0".join(
        (
            str(event.get_platform_id() or event.get_platform_name()),
            str(event.get_self_id()),
            str(event.get_group_id()),
        )
    )
    return hashlib.sha256(raw_scope.encode("utf-8")).hexdigest()


def _message_id_for_event(event: AstrMessageEvent) -> str:
    message_id = str(getattr(event.message_obj, "message_id", "") or "").strip()
    if message_id:
        return message_id
    generated = event.get_extra("_group_context_image_locator_message_id")
    if not generated:
        generated = uuid.uuid4().hex
        event.set_extra("_group_context_image_locator_message_id", generated)
    return str(generated)


def _make_locator(scope: str, message_id: str, image_index: int) -> str:
    payload = f"{scope}\0{message_id}\0{image_index}".encode()
    digest = hashlib.blake2s(payload, digest_size=8).hexdigest()
    return f"ctximg:{digest}"


def _clear_staged_markers(event: AstrMessageEvent) -> None:
    markers = event.get_extra(CONTEXT_MARKERS_EXTRA, [])
    if not isinstance(markers, list) or not markers:
        return
    messages = event.get_messages()
    if isinstance(messages, list):
        marker_ids = {id(marker) for marker in markers}
        messages[:] = [component for component in messages if id(component) not in marker_ids]
    event.set_extra(CONTEXT_MARKERS_EXTRA, [])


def _normalize_refs(refs: list[str] | Any) -> list[str]:
    if not isinstance(refs, list):
        return []
    normalized: list[str] = []
    for value in refs:
        ref = str(value).strip().lower()
        if not ref:
            continue
        # Models sometimes preserve only the compact suffix shown beside an
        # image.  Both spellings denote the same locator; canonicalize once at
        # the tool boundary instead of duplicating lookup branches downstream.
        if LOCATOR_SUFFIX_RE.fullmatch(ref):
            ref = f"ctximg:{ref}"
        normalized.append(ref)
    return normalized


def _request_contains_locator(req: ProviderRequest) -> bool:
    values = [
        getattr(req, "prompt", None),
        getattr(req, "contexts", None),
        getattr(req, "extra_user_content_parts", None),
    ]
    return any(_value_contains_locator(value) for value in values)


def _value_contains_locator(value: Any, depth: int = 0) -> bool:
    if depth > 8 or value is None:
        return False
    if isinstance(value, str):
        return bool(LOCATOR_RE.search(value))
    if isinstance(value, dict):
        return any(_value_contains_locator(item, depth + 1) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_value_contains_locator(item, depth + 1) for item in value)
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return bool(LOCATOR_RE.search(text))
    content = getattr(value, "content", None)
    if content is not None and content is not value:
        return _value_contains_locator(content, depth + 1)
    return False


def _detect_image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    return "application/octet-stream"


def _delivery_component(lookup: ImageLookup, mode: str):
    assert lookup.image is not None
    path = lookup.image.file_path
    if mode == "original_file":
        suffix = path.suffix or ".bin"
        return File(name=f"{lookup.locator.replace(':', '_')}{suffix}", file=str(path))
    return Image.fromFileSystem(str(path))


async def _send_component(event: AstrMessageEvent, component) -> str | None:
    """Send once through AstrBot's public event route."""
    chain = MessageChain(chain=[component])
    await event.send(chain)
    return None


def _safe_error(exc: Exception, fallback: str) -> str:
    if isinstance(exc, _LocalValidationError):
        return str(exc)
    if isinstance(exc, TimeoutError):
        category = "超时"
    elif isinstance(exc, OSError):
        category = "I/O"
    elif isinstance(exc, ValueError):
        category = "数据无效"
    else:
        category = "内部错误"
    return f"{fallback}（{category}）"


def _error_result(message: str) -> CallToolResult:
    payload = {"results": [], "error": message}
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(payload, ensure_ascii=False),
            )
        ],
        structuredContent=payload,
        isError=True,
    )
