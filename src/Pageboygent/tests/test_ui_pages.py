"""页面级导航：三个 AI 页面 + 数据库（看板 + 二级菜单）。"""
from __future__ import annotations

import re

import pytest

from Pageboygent import ui


@pytest.fixture(scope="module")
def client():
    return ui.create_storage_admin_app().test_client()


def _top_nav(html: str) -> list[str]:
    items = re.findall(r'<li class="nav-item pb-top-item( active)?">\s*<a class="nav-link" href="[^"]+">([^<]+)</a>', html)
    return [label + ("*" if active else "") for active, label in items]


def _sub_nav(html: str) -> list[str]:
    items = re.findall(r'<nav class="pb-subnav".*?</nav>', html, re.S)
    if not items:
        return []
    return [label.strip() + ("*" if active else "")
            for active, label in re.findall(r'<a href="[^"]+" class="(active)?"[^>]*>\s*([^<\n]+)', items[0])]


@pytest.mark.parametrize("path, active", [
    ("/admin/chat/", "AI 对话"),
    ("/admin/litsearch/", "AI 文献检索"),
    ("/admin/review/", "AI 综述"),
    ("/admin/rag/", "RAG 搜索"),
])
def test_ai_pages_are_top_level(client, path, active):
    html = client.get(path).get_data(as_text=True)
    assert _top_nav(html) == [name + ("*" if name == active else "")
                              for name in ["AI 对话", "AI 文献检索", "AI 综述", "RAG 搜索", "数据库"]]
    assert _sub_nav(html) == []          # AI 页面没有数据库二级菜单
    assert "home-tabs" not in html       # 旧的页内标签栏已移除


def test_database_dashboard_and_subnav(client):
    html = client.get("/admin/database/").get_data(as_text=True)
    assert _top_nav(html)[-1] == "数据库*"
    sub = _sub_nav(html)
    assert sub[0] == "看板*" and "文献" in sub and "MinerU 任务" in sub
    for section in ["处理流水线", "待处理", "出版年份分布", "主题总结进度", "最近加入的文献", "最近的 AI 会话"]:
        assert section in html


@pytest.mark.parametrize("path, active", [
    ("/admin/paper/", "文献"),
    ("/admin/theme/", "主题"),
    ("/admin/pdfstatus/", "PDF 状态"),
    ("/admin/mineru_tasks/", "MinerU 任务"),
])
def test_table_pages_live_under_database(client, path, active):
    html = client.get(path).get_data(as_text=True)
    assert _top_nav(html)[-1] == "数据库*"     # 数据表页面不出现在顶部导航，但高亮“数据库”
    assert "文献" not in _top_nav(html)
    assert f"{active}*" in _sub_nav(html)


def test_home_is_paper_tracking_dashboard_reached_via_brand(client):
    html = client.get("/admin/").get_data(as_text=True)
    assert 'id="fh-root"' in html and "订阅源" not in _top_nav(html)
    assert "主页" not in _top_nav(html)                      # 主页不进导航栏
    assert 'class="navbar-brand pb-brand" href="/admin/"' in html  # 点 “Pageboygent” 进主页
    location = client.get("/admin/?tab=lit&lit_session=abc").headers["Location"]
    assert location.endswith("/admin/litsearch/?lit_session=abc")  # 旧链接仍可用


def test_year_histogram_groups_early_years():
    hist = ui._year_histogram([1975, 1985, 1990, 1992, 1992], start=1990)
    assert [b["short"] for b in hist["bars"]] == ["更早", "1990", "1991", "1992"]
    assert [b["value"] for b in hist["bars"]] == [2, 1, 0, 2]
    assert hist["max"] == 2
