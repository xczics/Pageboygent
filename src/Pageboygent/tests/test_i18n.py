"""界面多语言：代码里用到的每个 key 都要有译文；译文保留占位符；英文界面能正常打开。"""
import ast
import re
from pathlib import Path

import pytest

from Pageboygent import i18n

PKG = Path(i18n.__file__).parent
CJK = re.compile(r"[一-鿿]")
PLACEHOLDER = re.compile(r"\{(\w+)\}")
_LIT = r"""'((?:[^'\\\n]|\\.)*)'|"((?:[^"\\\n]|\\.)*)\""""


def _template_keys() -> set[str]:
    keys = set()
    for f in (PKG / "templates" / "admin").glob("*.html"):
        for m in re.finditer(r"\b(?:tr|_)\(\s*(?:" + _LIT + ")", f.read_text()):
            raw, quote = (m.group(1), "'") if m.group(1) is not None else (m.group(2), '"')
            try:
                keys.add(ast.literal_eval(quote + raw + quote))
            except (SyntaxError, ValueError):
                keys.add(raw)
    return keys


def _modules() -> set[str]:
    return {k.split(":", 1)[0] for k in list(i18n.catalog("zh")) + list(i18n.catalog("en")) if i18n.is_semantic_key(k)}


def _python_keys() -> set[str]:
    keys = set()
    for f in PKG.glob("*.py"):
        if f.name == "i18n.py":
            continue
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "t" and node.args \
                    and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                keys.add(node.args[0].value)
            # 标签字典、工具名等以语义 key 形式写在常量里（只认语言文件里已有的模块，避免把 "cluster:1" 之类当成 key）
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and i18n.is_semantic_key(node.value) \
                    and node.value.split(":", 1)[0] in _modules():
                keys.add(node.value)
    return keys


def _used_keys() -> set[str]:
    return {k for k in _template_keys() | _python_keys() if CJK.search(k) or i18n.is_semantic_key(k)}


def test_every_chinese_key_has_an_english_translation():
    en = i18n.catalog("en")
    missing = sorted(k for k in _used_keys() if not i18n.is_semantic_key(k) and k not in en)
    assert not missing, f"{len(missing)} 条界面文字缺英文译文，例如：{missing[:10]}"


def test_every_semantic_key_exists_in_both_languages():
    zh, en = i18n.catalog("zh"), i18n.catalog("en")
    semantic = {k for k in _used_keys() if i18n.is_semantic_key(k)}
    assert not sorted(semantic - set(zh)), f"zh.json 缺少：{sorted(semantic - set(zh))[:10]}"
    assert not sorted(semantic - set(en)), f"en.json 缺少：{sorted(semantic - set(en))[:10]}"


def test_translations_keep_placeholders():
    zh, en = i18n.catalog("zh"), i18n.catalog("en")
    bad = []
    for key, value in en.items():
        source = zh.get(key, key) if i18n.is_semantic_key(key) else key
        # 译文可以省略原文的占位符（英文不需要量词等），但不能出现原文没有的占位符
        if not set(PLACEHOLDER.findall(value)) <= set(PLACEHOLDER.findall(source)):
            bad.append(key)
    assert not bad, f"译文里有原文没有的占位符：{bad[:10]}"


def test_t_falls_back_and_formats(monkeypatch):
    monkeypatch.setattr(i18n, "ui_language", lambda: "en")
    assert i18n.t("这句话没有译文") == "这句话没有译文"            # 查不到：显示中文，不显示 key
    assert i18n.t("no_module:missing.key") == "no_module:missing.key"
    assert i18n.t("feed:progress.llm", done=3, total=8).count("3") == 1


@pytest.fixture
def english_client(monkeypatch):
    from Pageboygent import ui

    monkeypatch.setattr(i18n, "ui_language", lambda: "en")
    app = ui.create_storage_admin_app()
    app.config["TESTING"] = True
    return app.test_client()


@pytest.mark.parametrize("path", ["/admin/", "/admin/tracking/", "/admin/chat/", "/admin/litsearch/", "/admin/review/",
                                  "/admin/database/", "/admin/paper/"])
def test_english_pages_render(english_client, path):
    resp = english_client.get(path)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert 'window.PB_LANG = "en"' in html
    assert "Paper tracking" in html and "Database" in html       # 顶部导航已是英文
