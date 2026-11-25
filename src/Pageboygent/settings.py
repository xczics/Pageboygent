"""
网页上的“设置”：把散落在 .env / pageboygent.toml / 用户配置文件里的配置集中展示与修改。

- 按用途分组、带中文说明；显示每项的当前值与来源（环境变量 / .env / 项目配置 / 用户配置 / 默认值）。
- 保存写入用户配置文件（~/.config/pageboygent/config.toml，与 config.py 的读取路径一致）：
  写前备份，写后权限设为 600（含密钥）。若某项被更高优先级的来源覆盖，页面会提示“保存了也不会生效”。
- 密钥永远不回传给浏览器：只显示是否已设置及掩码；留空表示不修改。
- 大多数模块在启动时读取配置，保存后需重启服务生效。
"""
from __future__ import annotations

import datetime as dt
import os
import shutil
from pathlib import Path
from typing import Any

from . import config as cfg

# (分组, [(配置名, 中文名, 是否必填)])；未列出的已注册配置项自动归入“其他”
GROUPS: list[tuple[str, str, list[tuple[str, str, bool]]]] = [
    ("model", "大模型 API", [
        ("ds_api_key", "DeepSeek API Key", True),
        ("ds_url", "API 地址", False),
        ("model_name_cheap", "Flash 模型名", False),
        ("model_name_professional", "Pro 模型名", False),
    ]),
    ("zotero", "Zotero", [
        ("zotero_id", "Zotero 用户 ID", True),
        ("zotero_key", "Zotero Web API Key", False),
        ("zotero_local_api_key", "Zotero 本地写入授权 Key", False),
        ("zotero_storage_path", "Zotero storage 目录", False),
        ("pdf_require_collection", "缺 PDF 时放入的集合", False),
        ("tag_for_ai_notes", "AI 笔记标签", False),
        ("tag_for_quick_summary", "快速总结标签", False),
        ("tag_for_detailed_reading", "精读笔记标签前缀", False),
    ]),
    ("pdf", "PDF → 全文（MinerU）", [
        ("pdf2md_mode", "转换方式（api / manual）", False),
        ("mineru_token", "MinerU Token", False),
        ("mineru_access_key", "OpenXLab Access Key", False),
        ("mineru_secret_key", "OpenXLab Secret Key", False),
        ("mineru_api_base", "MinerU API 地址", False),
        ("mineru_model_version", "MinerU 模型", False),
        ("mineru_upload_accelerate", "上传加速", False),
        ("mineru_poll_interval_seconds", "轮询间隔（秒）", False),
        ("mineru_max_attempts", "最大重试次数", False),
        ("mineru_url", "本地 MinerU 服务地址", False),
        ("mineru_output_dir", "本地 MinerU 输出目录", False),
        ("mineru_result_glob_pattern", "本地 MinerU 结果匹配", False),
    ]),
    ("embedding", "向量与检索", [
        ("embedding_url", "Embedding 服务地址（OpenAI 兼容接口，Ollama 填 …:11434/v1）", False),
        ("embedding_model", "Embedding 模型", False),
        ("embedding_key", "Embedding API Key", False),
        ("embedding_dim", "向量维度", False),
        ("max_chunk_size", "切块最大字符数", False),
    ]),
    ("feed", "文献追踪（RSS）", [
        ("feed_fetch_interval_hours", "抓取间隔（小时）", False),
        ("feed_retention_days", "条目保留天数", False),
        ("feed_root_collection", "Zotero 入库集合", False),
        ("feed_auto_ingest_per_run", "每次最多自动入库", False),
        ("feed_llm_batch_size", "每次 LLM 判断篇数", False),
    ]),
    ("ai", "AI 检索 / 对话 / 综述", [
        ("lit_search_max_rounds", "检索最大轮数", False),
        ("lit_search_context_budget_tokens", "上下文软阈值（tokens）", False),
        ("lit_search_context_hard_tokens", "上下文硬上限（tokens）", False),
        ("lit_search_deep_read_max_chars", "精读全文最大字符数", False),
        ("review_pipeline_concurrency", "综述精读并发数", False),
        ("review_pipeline_max_output_tokens", "综述单次最大输出", False),
    ]),
    ("price", "费用估算（元 / 百万 tokens）", [
        ("price_cheap_input_cache_hit", "Flash 输入 · 缓存命中", False),
        ("price_cheap_input_cache_miss", "Flash 输入 · 未命中", False),
        ("price_cheap_output", "Flash 输出", False),
        ("price_professional_input_cache_hit", "Pro 输入 · 缓存命中", False),
        ("price_professional_input_cache_miss", "Pro 输入 · 未命中", False),
        ("price_professional_output", "Pro 输出", False),
    ]),
    ("system", "系统", [
        ("workstation_dir", "数据目录", False),
        ("storage_heartbeat_interval_seconds", "后台心跳间隔（秒）", False),
    ]),
]

SOURCE_LABELS = {"env": "环境变量", "dotenv": ".env 文件", "project": "项目 pageboygent.toml",
                 "user": "用户配置文件", "default": "默认值", "unset": "未设置"}


def _source_of(item: cfg.ConfigItem) -> str:
    """与 ConfigRegistry._resolve_value 相同的优先级，只是返回值来自哪里。"""
    env_name = item.get_env_name()
    if cfg._is_non_empty(os.environ.get(env_name)):
        return "env"
    dotenv = cfg._read_dotenv(cfg._get_dotenv_path())
    if cfg._is_non_empty(dotenv.get(item.name)) or cfg._is_non_empty(dotenv.get(env_name)):
        return "dotenv"
    profile = cfg._get_active_profile()
    if cfg._get_from_cfg(cfg._read_toml(cfg._get_project_config_path()), item.name, profile) is not None:
        return "project"
    if cfg._get_from_cfg(cfg._read_toml(cfg._get_user_config_path()), item.name, profile) is not None:
        return "user"
    return "default" if item.default is not None else "unset"


def _type_name(item: cfg.ConfigItem) -> str:
    return {bool: "bool", int: "int", float: "float"}.get(item.type_, "str")


def describe() -> dict:
    """设置页面的全部数据（密钥只给掩码）。"""
    items = cfg._registry._items
    listed: set[str] = set()
    groups = []
    for group_id, title, entries in GROUPS:
        rows = []
        for name, label, required in entries:
            if name in items:
                listed.add(name)
                rows.append(_row(items[name], label, required))
        groups.append({"id": group_id, "title": title, "items": rows})
    others = [_row(item, name, False) for name, item in sorted(items.items()) if name not in listed]
    if others:
        groups.append({"id": "other", "title": "其他", "items": others})
    missing = [row for group in groups for row in group["items"] if row["required"] and not row["is_set"]]
    return {"groups": groups, "missing": missing, "user_config_path": str(cfg._get_user_config_path()),
            "profile": cfg._get_active_profile()}


def _row(item: cfg.ConfigItem, label: str, required: bool) -> dict:
    value = cfg.get_config(item.name)
    source = _source_of(item)
    is_set = cfg._is_non_empty(value)
    return {
        "name": item.name, "label": label, "description": item.description, "type": _type_name(item),
        "secret": item.is_secret, "required": required, "is_set": bool(is_set), "source": source,
        "source_label": SOURCE_LABELS[source],
        # 被用户配置文件之上的来源覆盖时，在网页上保存不会生效
        "shadowed": source in {"env", "dotenv", "project"},
        "value": (cfg.masked(str(value)) if is_set else "") if item.is_secret else ("" if value is None else value),
        "default": None if item.is_secret else item.default,
    }


# ---------------------------------------------------------------------------
# 写入用户配置文件（最小 TOML 写入器：只处理配置文件实际用到的结构）
# ---------------------------------------------------------------------------

def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_toml_key(k)} = {_toml_value(v)}" for k, v in value.items()) + " }"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return f'"{text}"'


def _toml_key(key: str) -> str:
    import re

    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else _toml_value(key)


def dump_toml(data: dict) -> str:
    lines = ["# Pageboygent 用户配置（可在网页“主页 → 设置”中修改）", ""]
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    lines += [f"{_toml_key(k)} = {_toml_value(v)}" for k, v in scalars.items()]

    def _emit(prefix: str, table: dict) -> None:
        inner_scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        inner_tables = {k: v for k, v in table.items() if isinstance(v, dict)}
        if inner_scalars or not inner_tables:
            lines.append("")
            lines.append(f"[{prefix}]")
            lines.extend(f"{_toml_key(k)} = {_toml_value(v)}" for k, v in inner_scalars.items())
        for key, sub in inner_tables.items():
            _emit(f"{prefix}.{_toml_key(key)}", sub)

    for key, table in tables.items():
        _emit(_toml_key(key), table)
    return "\n".join(lines).rstrip() + "\n"


def _coerce(item: cfg.ConfigItem, raw: Any) -> Any:
    if item.type_ is bool:
        return raw if isinstance(raw, bool) else str(raw).strip().lower() in {"1", "true", "yes", "on"}
    if item.type_ is int:
        return int(str(raw).strip())
    if item.type_ is float:
        return float(str(raw).strip())
    return str(raw).strip()


def save(changes: dict[str, Any], *, clear: list[str] | None = None, path: Path | None = None) -> dict:
    """保存修改到用户配置文件。changes 中密钥为空字符串表示不修改；clear 中的项删除（回到默认值）。"""
    items = cfg._registry._items
    path = path or cfg._get_user_config_path()
    data = cfg._read_toml(path) if path.exists() else {}
    profile = cfg._get_active_profile()
    profile_table = data.setdefault("profiles", {}).setdefault(profile, {})
    applied, errors = [], {}
    for name, raw in changes.items():
        item = items.get(name)
        if item is None:
            errors[name] = "未知配置项"
            continue
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            continue  # 留空：不修改（对密钥尤其重要）
        try:
            value = _coerce(item, raw)
        except (TypeError, ValueError):
            errors[name] = "格式不正确"
            continue
        # 已经写在 profile 段里的、或是密钥：写到 profile 段（与现有文件结构一致）；否则写到顶层
        if name in profile_table or item.is_secret:
            profile_table[name] = value
            data.pop(name, None)
        else:
            data[name] = value
        applied.append(name)
    for name in clear or []:
        if name in items:
            data.pop(name, None)
            profile_table.pop(name, None)
            applied.append(name)
    if not profile_table:
        data["profiles"].pop(profile, None)
        if not data["profiles"]:
            data.pop("profiles")
    if applied:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            backup = path.with_name(path.name + f".bak-{dt.datetime.now():%Y%m%d-%H%M%S}")
            shutil.copy2(path, backup)
            for old in sorted(path.parent.glob(path.name + ".bak-*"))[:-5]:
                old.unlink()  # 只保留最近 5 份备份
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(dump_toml(data), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(path)
        cfg._registry._cache.clear()
    return {"applied": applied, "errors": errors, "restart_required": bool(applied)}
