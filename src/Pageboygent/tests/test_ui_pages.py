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


NAV = ["文献追踪", "AI 对话", "AI 文献检索", "AI 综述", "数据库"]


@pytest.mark.parametrize("path, active", [
    ("/admin/tracking/", "文献追踪"),
    ("/admin/chat/", "AI 对话"),
    ("/admin/litsearch/", "AI 文献检索"),
    ("/admin/review/", "AI 综述"),
    ("/admin/", None),                   # 主页（深度搜索）不在导航里
])
def test_top_level_pages_and_nav_order(client, path, active):
    html = client.get(path).get_data(as_text=True)
    assert _top_nav(html) == [name + ("*" if name == active else "") for name in NAV]
    assert _sub_nav(html) == []          # 这些页面没有数据库二级菜单
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


def test_home_is_deep_search_and_tracking_is_first_in_nav(client):
    html = client.get("/admin/").get_data(as_text=True)
    assert 'id="ds-root"' in html                                     # 主页 = 深度搜索
    assert 'class="navbar-brand pb-brand" href="/admin/"' in html      # 点 “Pageboygent” 进主页
    assert 'id="fh-root"' in client.get("/admin/tracking/").get_data(as_text=True)
    # 旧链接仍可用
    assert client.get("/admin/?tab=lit&lit_session=abc").headers["Location"].endswith("/admin/litsearch/?lit_session=abc")
    assert client.get("/admin/?view=reports").headers["Location"].endswith("/admin/tracking/?view=reports")
    assert client.get("/admin/search/?q=zinc").headers["Location"].endswith("/admin/?q=zinc")
    assert client.get("/admin/rag/?q=zinc").headers["Location"].endswith("/admin/?q=zinc")


def test_year_histogram_groups_early_years():
    hist = ui._year_histogram([1975, 1985, 1990, 1992, 1992], start=1990)
    assert [b["short"] for b in hist["bars"]] == ["更早", "1990", "1991", "1992"]
    assert [b["value"] for b in hist["bars"]] == [2, 1, 0, 2]
    assert hist["max"] == 2
