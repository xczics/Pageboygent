"""
界面多语言（中文 / English）。

两种 key：
- 语义 key（推荐，新代码一律用它）：“模块:语义”，例如 t("search:scope.all")、t("home:tab.reader")。
  语言文件里按模块嵌套存放：{"search": {"scope.all": "All"}}，加载时展开成 "search:scope.all"。
  每种语言一份：locales/zh.json、locales/en.json。查找顺序：当前语言 → 中文 → key 本身。
- 中文原文 key（过渡用）：t("新文章")。中文界面原样返回；英文界面查 en.json 顶层的 {"新文章": "New papers"}，
  查不到就显示中文。同一句中文在不同地方要译成不同英文时，不要用中文原文作 key，改用语义 key。

用法：Python 里 from .i18n import t；Jinja 模板里 {{ _("…") }}；页面脚本里 tr('…')（由 pb_master.html 注入）。
带变量：t("library:papers.count", n=3) / tr('library:papers.count', {n: 3})，译文里写 {n}。

界面语言（ui_language）每次请求时读取，切换后刷新页面即生效；AI 回答语言（ai_language）写在提示词里，
重启服务后生效。
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import get_config

LANGUAGES = {"zh": "中文", "en": "English"}
_LOCALE_DIR = Path(__file__).parent / "locales"


def ui_language() -> str:
    lang = str(get_config("ui_language") or "zh").strip().lower()
    return lang if lang in LANGUAGES else "zh"


def ai_language() -> str:
    """AI 生成内容的语言：没有单独设置时跟随界面语言。"""
    lang = str(get_config("ai_language") or "").strip().lower()
    return lang if lang in LANGUAGES else ui_language()


_SEMANTIC_KEY = re.compile(r"^[a-z][a-z0-9_]*:[A-Za-z0-9_.\-]+$")


def is_semantic_key(text: str) -> bool:
    return bool(_SEMANTIC_KEY.match(text or ""))


def _flatten(data: dict) -> dict[str, str]:
    """顶层的字符串值是中文原文 key；顶层的对象是模块分组，展开成 “模块:语义”（可多层，内层用 . 连接）。"""
    out: dict[str, str] = {}

    def walk(prefix: str, node: dict) -> None:
        for key, value in node.items():
            name = f"{prefix}.{key}" if prefix.count(":") else f"{prefix}:{key}"
            if isinstance(value, dict):
                walk(name, value)
            elif value:
                out[name] = str(value)

    for key, value in data.items():
        if isinstance(value, dict):
            walk(key, value)
        elif value:
            out[key] = str(value)
    return out


@lru_cache(maxsize=None)
def catalog(lang: str) -> dict[str, str]:
    path = _LOCALE_DIR / f"{lang}.json"
    if not path.exists():
        return {}
    return _flatten(json.loads(path.read_text(encoding="utf-8")))


def t(text: str, lang: str | None = None, **params: Any) -> str:
    """翻译一句界面文字（语义 key 或中文原文）；params 填进 {name} 占位。"""
    if text is None:
        return ""
    lang = lang or ui_language()
    if is_semantic_key(text):
        out = catalog(lang).get(text) or catalog("zh").get(text) or text
    else:
        out = catalog(lang).get(text, text) if lang != "zh" else text
    if params:
        try:
            out = out.format(**params)
        except (KeyError, IndexError, ValueError):
            pass
    return out


def js_catalog() -> dict[str, str]:
    """注入页面的译文表：中文的语义 key 作底，当前语言覆盖其上（中文界面时只有语义 key）。"""
    lang = ui_language()
    merged = {k: v for k, v in catalog("zh").items() if is_semantic_key(k)}
    if lang != "zh":
        merged.update(catalog(lang))
    return merged


def answer_language_name(lang: str | None = None) -> str:
    """写进提示词的“回答语言”。"""
    return {"zh": "中文", "en": "英文（English）"}[lang or ai_language()]
