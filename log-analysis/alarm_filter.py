#!/usr/bin/env python3
"""PicMe 告警聚合过滤脚本。

聚合层级：
    message -> params["子功能"] -> 归一化后的错误描述 -> count

支持两种输入方式：
1. 直接请求告警接口；
2. 读取已经保存的原始 JSON 文件，便于离线调试。

在线请求使用代码中暂时固定的接口地址和 Token。
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:  # 只有在线请求模式才需要 requests
    requests = None  # type: ignore[assignment]

ALARM_URL = "https://admin-api.picme.one/api/admin/open/alarm/list"
ALARM_TOKEN = "tOHJKf7PwRkmNDYahx3VO4P"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "filtered_alarms.json"
NO_SUBFUNCTION = None

# detail 中常见的动态字段。它们不应参与错误分类。
WORK_ID_RE = re.compile(r"(?i)\bWorkID\s*[:=]\s*[^\s'\"},]+")
WORK_ID_VALUE_RE = re.compile(
    r"(?i)\bWorkID\s*[:=]\s*['\"]?(?P<value>[^\s'\"},;]+)"
)
MASKED_ASSET_NAME_RE = re.compile(
    r'''(?i)(["']Name["']\s*:\s*["'])\*+[A-Za-z0-9._~-]*(["'])'''
)
GROUP_ID_RE = re.compile(
    r'''(?i)(["']GroupId["']\s*:\s*["'])[^"']+(["'])'''
)
EXEC_TIME_RE = re.compile(
    r'''(?i)(["']?exec_time["']?\s*:\s*)\d+(?:\.\d+)?'''
)
X_DATE_RE = re.compile(
    r"(?i)(\bx_date\s*:\s*)\d{8}T\d{6}(?:\.\d+)?Z\b"
)
X_CONTENT_SHA256_RE = re.compile(
    r"(?i)(\bx_content_sha256\s*:\s*)[0-9a-f]{32,}\b"
)
BYTE_COUNTS_RE = re.compile(
    r"(?i)(received\s+)\d+(\s+bytes,\s+expected\s+)\d+"
)
DYNAMIC_TASK_PATH_RE = re.compile(r"/task_[A-Za-z0-9_-]+/")
URL_QUERY_VALUE_RE = re.compile(
    r'''([?&][A-Za-z0-9_.~-]+=)[^&\s'"}]+'''
)
API_KEY_RE = re.compile(r"(?i)\bapi[_ -]?key\s*[:=]\s*[^\s,;]+")
REQUEST_ID_RE = re.compile(r"(?i)\brequest[ _-]?id\s*[:=]\s*[^\s'\"},]+")
TASK_ID_RE = re.compile(r"(?i)(['\"]?task_id['\"]?\s*[:=]\s*['\"])[^'\"]+(['\"])")
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
ASSET_ID_RE = re.compile(r"\basset-\d{8,}-[A-Za-z0-9_-]+\b", re.IGNORECASE)
LONG_NUMBER_RE = re.compile(r"\b\d{8,}\b")
WORK_STYLE_ID_RE = re.compile(r"\b\d{10,}-\d{10,}-[A-Za-z0-9_-]+\b")
API_KEY_VALUE_RE = re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_*.-]{8,}\b")
MORE_INFO_RE = re.compile(r"(?i)For more information check:\s*https?://\S+")
ATTEMPT_RE = re.compile(r"(?i)Occurred on attempt\s+\d+")
WHITESPACE_RE = re.compile(r"\s+")

# 优先提取 detail 中显式的 error_message 字段。
ERROR_MESSAGE_PATTERNS = (
    re.compile(r"['\"]error_message['\"]\s*:\s*'(?P<value>.*?)(?<!\\)'", re.DOTALL),
    re.compile(r"['\"]error_message['\"]\s*:\s*\"(?P<value>.*?)(?<!\\)\"", re.DOTALL),
)
MARKDOWN_ERROR_RE = re.compile(r"(?:\*\*)?错误信息(?:\*\*)?\s*:\s*(?P<value>.*)", re.DOTALL)
FALLBACK_PREFIX_RE = re.compile(r"^\s*\[Fallback Triggered:\s*Rule\s*[^\]]+\]\s*", re.IGNORECASE)
ERROR_CAUSE_RE = re.compile(
    r'''["']Error["']\s*:\s*\{.*?'''
    r'''["']Code["']\s*:\s*["'](?P<code>[^"']+)["'].*?'''
    r'''["']Message["']\s*:\s*["'](?P<message>[^"']+)["']''',
    re.IGNORECASE | re.DOTALL,
)


def parse_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("日期格式必须为 YYYY-MM-DD") from exc

    if value != parsed.isoformat():
        raise argparse.ArgumentTypeError("日期格式必须为 YYYY-MM-DD")
    return parsed


def resolve_date_range(
    start_date: date | None,
    end_date: date | None,
    *,
    today: date | None = None,
) -> tuple[date, date]:
    start_date = start_date or today or datetime.now().astimezone().date()
    end_date = end_date or start_date + timedelta(days=1)
    if end_date <= start_date:
        raise ValueError("--end-date 必须晚于 --start-date")
    return start_date, end_date


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 message、子功能和错误描述前缀聚合 PicMe 告警。"
    )
    parser.add_argument("--input", help="离线原始 JSON 文件；设置后不请求接口")
    parser.add_argument(
        "--start-date",
        type=parse_date,
        help="开始日期，格式 YYYY-MM-DD；默认按本机时区取运行当天",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        help="结束日期，格式 YYYY-MM-DD；默认取开始日期的下一天",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出 JSON 文件；默认保存在脚本所在目录",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP 超时秒数")
    parser.add_argument("--retries", type=int, default=3, help="HTTP 失败重试次数")
    parser.add_argument(
        "--prefix-chars",
        type=int,
        default=240,
        help="用于分组的错误签名最多保留的字符数，默认 240；不截断输出 error",
    )
    parser.add_argument(
        "--include-example",
        action="store_true",
        help="在每个错误组中额外保留一个脱敏后的代表样本；默认只输出错误类别和数量",
    )
    parser.add_argument(
        "--keep-fallback-rule",
        action="store_true",
        help="保留 [Fallback Triggered: Rule ...] 前缀；默认移除",
    )
    args = parser.parse_args(argv)
    if not args.input:
        try:
            args.start_date, args.end_date = resolve_date_range(
                args.start_date,
                args.end_date,
            )
        except ValueError as exc:
            parser.error(str(exc))
    return args


def load_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.input:
        path = Path(args.input)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(f"输入文件不存在: {path}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"输入文件不是合法 JSON: {path}: {exc}") from exc

    if requests is None:
        raise RuntimeError("缺少 requests，请先执行: pip install requests")

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=max(args.retries, 0))
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        response = session.get(
            ALARM_URL,
            params={
                "startDate": args.start_date.isoformat(),
                "endDate": args.end_date.isoformat(),
            },
            headers={"token": ALARM_TOKEN},
            timeout=args.timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"请求告警接口失败: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("接口返回内容不是合法 JSON") from exc

    return payload


def extract_error_text(detail: Any) -> str:
    """从 detail 中提取最值得分类的错误文本。"""
    text = html.unescape(str(detail or "")).strip()
    if not text:
        return "<empty detail>"

    for pattern in ERROR_MESSAGE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group("value").strip()

    match = MARKDOWN_ERROR_RE.search(text)
    if match:
        return match.group("value").strip()

    return text


def extract_work_id(item: dict[str, Any], params: dict[str, Any]) -> str | None:
    """从结构化字段或 detail 中提取一条可用于排查的 WorkID。"""
    candidate_keys = ("workId", "workID", "work_id", "WorkID")
    for source in (params, item):
        for key in candidate_keys:
            value = source.get(key)
            if value not in (None, ""):
                work_id = str(value).strip()
                if work_id:
                    return work_id

    detail = html.unescape(str(item.get("detail") or ""))
    match = WORK_ID_VALUE_RE.search(detail)
    return match.group("value") if match else None


def redact_dynamic_values(text: str) -> str:
    """移除会导致同类错误无法聚合的动态字段，并做基础脱敏。"""
    text = html.unescape(text)
    text = MASKED_ASSET_NAME_RE.sub(r"\1<asset-name>\2", text)
    text = GROUP_ID_RE.sub(r"\1<group-id>\2", text)
    text = EXEC_TIME_RE.sub(r"\1<seconds>", text)
    text = X_DATE_RE.sub(r"\1<timestamp>", text)
    text = X_CONTENT_SHA256_RE.sub(r"\1<sha256>", text)
    text = BYTE_COUNTS_RE.sub(r"\1<n>\2<n>", text)
    text = DYNAMIC_TASK_PATH_RE.sub("/task_<id>/", text)
    text = URL_QUERY_VALUE_RE.sub(r"\1<value>", text)
    text = WORK_ID_RE.sub("", text)
    text = API_KEY_RE.sub("", text)
    text = REQUEST_ID_RE.sub("", text)
    text = TASK_ID_RE.sub(r"\1<redacted>\2", text)
    text = API_KEY_VALUE_RE.sub("<api-key>", text)
    text = WORK_STYLE_ID_RE.sub("<work-id>", text)
    text = UUID_RE.sub("<uuid>", text)
    text = IP_RE.sub("<ip>", text)
    text = EMAIL_RE.sub("<email>", text)
    text = ASSET_ID_RE.sub("<asset-id>", text)
    text = LONG_NUMBER_RE.sub("<long-number>", text)
    text = MORE_INFO_RE.sub("", text)
    text = ATTEMPT_RE.sub("Occurred on attempt <n>", text)
    return WHITESPACE_RE.sub(" ", text).strip(" \n\r\t,;")


def truncate_at_stable_boundary(text: str, prefix_chars: int) -> str:
    """保留错误开头的稳定描述，舍弃后方的大型响应体。"""
    # 这些标记之后通常是完整响应体、动态参数或重复堆栈。
    boundaries = (
        " Response:",
        " response:",
        " Response={",
        " Details: Content is empty",
        " Details: {",
        " Traceback (most recent call last):",
        "\nTraceback",
    )
    for marker in boundaries:
        pos = text.find(marker)
        if pos >= 20:
            text = text[:pos]
            break

    if len(text) <= prefix_chars:
        return text.strip()

    chunk = text[:prefix_chars]
    # 尽量在句号、分号或逗号处截断，不产生半个单词。
    candidates = [chunk.rfind(mark) for mark in ("。", ". ", "; ", "；", ", ", "，")]
    cut = max(candidates)
    if cut >= int(prefix_chars * 0.55):
        chunk = chunk[: cut + 1]
    else:
        last_space = chunk.rfind(" ")
        if last_space >= int(prefix_chars * 0.75):
            chunk = chunk[:last_space]
    return chunk.rstrip(" ,;，；") + "…"


def build_group_signature(text: str, prefix_chars: int) -> str:
    """生成内部聚合签名，并保留响应中的业务错误原因。"""
    signature = truncate_at_stable_boundary(text, prefix_chars)
    match = ERROR_CAUSE_RE.search(text)
    if not match:
        return signature

    code = WHITESPACE_RE.sub(" ", match.group("code")).strip()
    message = WHITESPACE_RE.sub(" ", match.group("message")).strip()
    return f"{signature} | Error.Code={code} | Error.Message={message}"


def normalize_error(detail: Any, prefix_chars: int, keep_fallback_rule: bool) -> tuple[str, str]:
    """返回 (内部聚合签名, 未截断的规范化错误文本)。"""
    extracted = extract_error_text(detail)
    cleaned = redact_dynamic_values(extracted)

    if not keep_fallback_rule:
        cleaned = FALLBACK_PREFIX_RE.sub("", cleaned)

    cleaned = cleaned or "<empty error>"
    signature = build_group_signature(cleaned, prefix_chars)
    return signature, cleaned


def aggregate(
    payload: dict[str, Any],
    prefix_chars: int = 240,
    keep_fallback_rule: bool = False,
    include_example: bool = False,
) -> dict[str, Any]:
    records = payload.get("data")
    if not isinstance(records, list):
        raise RuntimeError("返回 JSON 中 data 不是数组")

    # message -> subfunction -> signature -> stats
    buckets: dict[str, dict[str | None, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    skipped = 0

    for item in records:
        if not isinstance(item, dict):
            skipped += 1
            continue

        message = str(item.get("message") or "<empty message>").strip()
        params = item.get("params")
        if not isinstance(params, dict):
            params = {}

        raw_subfunction = (
            params.get("子功能")
            or params.get("subFunction")
            or params.get("sub_function")
        )
        subfunction = str(raw_subfunction).strip() if raw_subfunction not in (None, "") else NO_SUBFUNCTION
        work_id = extract_work_id(item, params)
        signature, cleaned_error = normalize_error(
            item.get("detail"),
            prefix_chars=prefix_chars,
            keep_fallback_rule=keep_fallback_rule,
        )

        error_map = buckets[message][subfunction]
        stats = error_map.get(signature)
        if stats is None:
            stats = {
                "error": cleaned_error,
                "workId": work_id,
                "count": 0,
            }
            if include_example:
                stats["example"] = cleaned_error
            error_map[signature] = stats
        elif stats["workId"] is None and work_id is not None:
            stats["error"] = cleaned_error
            stats["workId"] = work_id
            if include_example:
                stats["example"] = cleaned_error
        stats["count"] += 1

    message_groups: list[dict[str, Any]] = []
    total_error_groups = 0

    for message, subfunction_map in buckets.items():
        children: list[dict[str, Any]] = []
        message_count = 0

        for subfunction, error_map in subfunction_map.items():
            errors = sorted(
                error_map.values(),
                key=lambda row: (-row["count"], row["error"]),
            )
            child_count = sum(row["count"] for row in errors)
            message_count += child_count
            total_error_groups += len(errors)
            children.append(
                {
                    "subFunction": subfunction,
                    "count": child_count,
                    "errorGroups": errors,
                }
            )

        children.sort(
            key=lambda row: (
                -row["count"],
                row["subFunction"] is None,
                row["subFunction"] or "",
            )
        )
        message_groups.append(
            {
                "message": message,
                "count": message_count,
                "children": children,
            }
        )

    message_groups.sort(key=lambda row: (-row["count"], row["message"]))

    return {
        "meta": {
            "sourceRecordCount": len(records),
            "skippedRecordCount": skipped,
            "messageGroupCount": len(message_groups),
            "errorGroupCount": total_error_groups,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "grouping": [
                "message",
                "params.子功能",
                "normalized prefix + structured Error.Code/Error.Message",
            ],
            "errorText": "full normalized detail/error_message",
            "prefixChars": prefix_chars,
        },
        "groups": message_groups,
    }


def main() -> int:
    args = parse_args()
    try:
        payload = load_payload(args)
        result = aggregate(
            payload,
            prefix_chars=max(args.prefix_chars, 40),
            keep_fallback_rule=args.keep_fallback_rule,
            include_example=args.include_example,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"完成: {result['meta']['sourceRecordCount']} 条原始告警 -> "
        f"{result['meta']['errorGroupCount']} 个错误组，输出 {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
