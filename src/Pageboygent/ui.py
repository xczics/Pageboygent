from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Mapping, Sequence
import webbrowser

from flask import Flask, Response, flash, jsonify, redirect, request, url_for
from flask_admin import Admin, AdminIndexView, BaseView, expose
from flask_admin.theme import Bootstrap4Theme
from flask_admin.actions import action
from flask_admin.contrib.peewee import ModelView
from markupsafe import Markup, escape

try:
    from markdown import markdown as _markdown_render_impl  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency fallback
    _markdown_render_impl = None

from .config import get_config
from .storage import BaseModel, MDStatus, Paper, PDFStatus, Theme, ThemePaper, TracedCollection
from . import storage
from . import rag
from . import write_queue
from . import mineru_tasks
from .mineru_tasks import MinerUTask


LIST_VIEW_CSS = """
table.model-list {
    width: auto;
    min-width: 100%;
    table-layout: fixed;
}

table.model-list th,
table.model-list td {
    white-space: normal;
    vertical-align: top;
}

table.model-list th {
    position: relative;
}

table.model-list th .pb-col-resize-handle {
    position: absolute;
    top: 0;
    right: -4px;
    width: 8px;
    height: 100%;
    cursor: col-resize;
    user-select: none;
    touch-action: none;
    z-index: 2;
}

table.model-list th .pb-col-resize-handle::after {
    content: '';
    position: absolute;
    top: 20%;
    bottom: 20%;
    left: 3px;
    width: 2px;
    background: rgba(108, 117, 125, 0.25);
    transition: background-color 120ms ease-in-out;
}

table.model-list th:hover .pb-col-resize-handle::after,
table.model-list th .pb-col-resize-handle.pb-active::after {
    background: rgba(33, 37, 41, 0.55);
}

table.model-list .pb-cell-clamp {
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
    overflow: hidden;
    line-height: 1.4;
    max-height: 2.8em;
    word-break: break-word;
    overflow-wrap: anywhere;
}

table.model-list .pb-cell-clamp > a,
table.model-list .pb-cell-clamp > span,
table.model-list .pb-cell-clamp > div,
table.model-list .pb-cell-clamp > p {
    display: inline;
}
""".strip()


LIST_VIEW_JS = """
(function () {
    const MIN_COLUMN_WIDTH = 72;
    const MAX_COLUMN_WIDTH = 720;

    function shouldSkipCell(cell) {
        return cell.matches('.list-checkbox-column, .list-buttons-column')
            || cell.querySelector('input[type="checkbox"], button, .btn, form') !== null;
    }

    function getColumnWidths(table, headerCells) {
        const colgroup = table.querySelector('colgroup[data-pb-auto-width="1"]');
        if (!colgroup) {
            return headerCells.map((header) => Math.max(Math.round(header.getBoundingClientRect().width), MIN_COLUMN_WIDTH));
        }

        const cols = Array.from(colgroup.querySelectorAll('col'));
        return headerCells.map((header, index) => {
            const col = cols[index];
            if (col && col.style.width.endsWith('px')) {
                const parsed = Number.parseFloat(col.style.width);
                if (Number.isFinite(parsed) && parsed > 0) {
                    return parsed;
                }
            }
            return Math.max(Math.round(header.getBoundingClientRect().width), MIN_COLUMN_WIDTH);
        });
    }

    function getOrCreateColgroup(table) {
        let colgroup = table.querySelector('colgroup[data-pb-auto-width="1"]');
        if (!colgroup) {
            colgroup = document.createElement('colgroup');
            colgroup.setAttribute('data-pb-auto-width', '1');
            table.insertBefore(colgroup, table.firstChild);
        }
        return colgroup;
    }

    function applyColumnWidths(table, widths) {
        const colgroup = getOrCreateColgroup(table);
        colgroup.replaceChildren();

        widths.forEach((width) => {
            const col = document.createElement('col');
            col.style.width = Math.round(width) + 'px';
            colgroup.appendChild(col);
        });

        const containerWidth = Math.max(table.parentElement ? table.parentElement.clientWidth : 0, 960);
        const totalWidth = widths.reduce((sum, width) => sum + width, 0);
        table.style.width = Math.max(containerWidth, totalWidth) + 'px';
        table.style.minWidth = totalWidth + 'px';
    }

    function ensureClampWrapper(cell) {
        if (shouldSkipCell(cell) || cell.querySelector('.pb-cell-clamp')) {
            return;
        }

        const wrapper = document.createElement('div');
        wrapper.className = 'pb-cell-clamp';

        while (cell.firstChild) {
            wrapper.appendChild(cell.firstChild);
        }

        const text = wrapper.textContent.trim();
        if (text && !wrapper.getAttribute('title')) {
            wrapper.setAttribute('title', text);
        }

        cell.appendChild(wrapper);
    }

    function estimateColumnWidth(header, cells) {
        const samples = cells
            .slice(0, 12)
            .map((cell) => (cell.textContent || '').replace(/\\s+/g, ' ').trim())
            .filter(Boolean);

        const headerLength = (header.textContent || '').trim().length;
        const sampleLength = samples.reduce((sum, text) => sum + Math.min(text.length, 80), 0);
        const averageLength = samples.length ? sampleLength / samples.length : headerLength;
        const effectiveLength = Math.max(headerLength * 1.2, averageLength);

        return Math.max(96, Math.min(320, Math.round(48 + effectiveLength * 7)));
    }

    function distributeWidths(widths, targetWidth, minWidth, maxWidth) {
        const nextWidths = widths.slice();
        let totalWidth = nextWidths.reduce((sum, width) => sum + width, 0);

        if (totalWidth > targetWidth) {
            const shrinkable = nextWidths.reduce((sum, width) => sum + Math.max(0, width - minWidth), 0);
            if (shrinkable > 0) {
                const overflow = totalWidth - targetWidth;
                nextWidths.forEach((width, index) => {
                    const room = Math.max(0, width - minWidth);
                    const shrink = Math.round((room / shrinkable) * overflow);
                    nextWidths[index] = Math.max(minWidth, width - shrink);
                });
            }
        } else if (totalWidth < targetWidth) {
            const growable = nextWidths.reduce((sum, width) => sum + Math.max(0, maxWidth - width), 0);
            if (growable > 0) {
                const extra = targetWidth - totalWidth;
                nextWidths.forEach((width, index) => {
                    const room = Math.max(0, maxWidth - width);
                    const grow = Math.round((room / growable) * extra);
                    nextWidths[index] = Math.min(maxWidth, width + grow);
                });
            }
        }

        return nextWidths;
    }

    function applyAutoWidths(table) {
        const headerCells = Array.from(table.querySelectorAll('thead th'));
        if (!headerCells.length) {
            return;
        }

        const manualWidthsRaw = table.dataset.pbManualWidths;
        if (manualWidthsRaw) {
            try {
                const manualWidths = JSON.parse(manualWidthsRaw);
                if (Array.isArray(manualWidths) && manualWidths.length === headerCells.length) {
                    applyColumnWidths(
                        table,
                        manualWidths.map((width) => {
                            const normalized = Number(width);
                            if (!Number.isFinite(normalized)) {
                                return MIN_COLUMN_WIDTH;
                            }
                            return Math.min(MAX_COLUMN_WIDTH, Math.max(MIN_COLUMN_WIDTH, normalized));
                        }),
                    );
                    return;
                }
            } catch (_error) {
                delete table.dataset.pbManualWidths;
            }
        }

        const bodyRows = Array.from(table.querySelectorAll('tbody tr'));
        const containerWidth = Math.max(table.parentElement ? table.parentElement.clientWidth : 0, table.clientWidth, 960);
        const minWidth = 112;
        const maxWidth = 360;
        const widths = [];
        let fixedWidth = 0;

        headerCells.forEach((header, columnIndex) => {
            const cells = bodyRows
                .map((row) => row.children[columnIndex])
                .filter((cell) => cell instanceof HTMLTableCellElement);

            const skip = shouldSkipCell(header) || cells.some((cell) => shouldSkipCell(cell));
            if (skip) {
                const measuredWidth = Math.max(header.offsetWidth, ...cells.map((cell) => cell.offsetWidth), 56);
                widths.push(measuredWidth);
                fixedWidth += measuredWidth;
                return;
            }

            widths.push(estimateColumnWidth(header, cells));
        });

        const adjustableIndexes = widths
            .map((width, index) => ({ width, index }))
            .filter(({ index }) => {
                const header = headerCells[index];
                const cells = bodyRows
                    .map((row) => row.children[index])
                    .filter((cell) => cell instanceof HTMLTableCellElement);
                return !(shouldSkipCell(header) || cells.some((cell) => shouldSkipCell(cell)));
            });

        const availableWidth = Math.max(containerWidth - fixedWidth, adjustableIndexes.length * minWidth);
        const distributed = distributeWidths(
            adjustableIndexes.map(({ width }) => width),
            availableWidth,
            minWidth,
            maxWidth,
        );

        distributed.forEach((width, offset) => {
            const targetIndex = adjustableIndexes[offset].index;
            widths[targetIndex] = width;
        });

        applyColumnWidths(table, widths);
    }

    function enableColumnResize(table) {
        if (table.dataset.pbResizeReady === '1') {
            return;
        }

        const headerCells = Array.from(table.querySelectorAll('thead th'));
        if (!headerCells.length) {
            return;
        }

        headerCells.forEach((header, columnIndex) => {
            if (shouldSkipCell(header) || header.querySelector('.pb-col-resize-handle')) {
                return;
            }

            const handle = document.createElement('span');
            handle.className = 'pb-col-resize-handle';
            handle.setAttribute('role', 'separator');
            handle.setAttribute('aria-orientation', 'vertical');
            handle.setAttribute('title', 'Drag to resize column');
            header.appendChild(handle);

            handle.addEventListener('mousedown', (event) => {
                if (!(event instanceof MouseEvent)) {
                    return;
                }

                event.preventDefault();
                event.stopPropagation();

                const currentHeaders = Array.from(table.querySelectorAll('thead th'));
                const startWidths = getColumnWidths(table, currentHeaders);
                const startWidth = startWidths[columnIndex] || MIN_COLUMN_WIDTH;
                const startX = event.clientX;

                const onMove = (moveEvent) => {
                    if (!(moveEvent instanceof MouseEvent)) {
                        return;
                    }

                    const deltaX = moveEvent.clientX - startX;
                    const nextWidth = Math.min(MAX_COLUMN_WIDTH, Math.max(MIN_COLUMN_WIDTH, startWidth + deltaX));
                    startWidths[columnIndex] = nextWidth;
                    applyColumnWidths(table, startWidths);
                };

                const onUp = () => {
                    document.removeEventListener('mousemove', onMove);
                    document.removeEventListener('mouseup', onUp);
                    handle.classList.remove('pb-active');
                    table.dataset.pbManualWidths = JSON.stringify(startWidths.map((width) => Math.round(width)));
                };

                handle.classList.add('pb-active');
                document.addEventListener('mousemove', onMove);
                document.addEventListener('mouseup', onUp);
            });
        });

        table.dataset.pbResizeReady = '1';
    }

    function enhanceTables() {
        const tables = document.querySelectorAll('table.model-list');
        tables.forEach((table) => {
            table.querySelectorAll('tbody td').forEach(ensureClampWrapper);
            applyAutoWidths(table);
            enableColumnResize(table);
        });
    }

    window.addEventListener('load', enhanceTables);
    window.addEventListener('resize', enhanceTables);
})();
""".strip()


RAG_HOME_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Pageboygent RAG Search</title>
    <style>
        :root {
            --bg: #f6f7fb;
            --card: #ffffff;
            --text: #1f2937;
            --muted: #6b7280;
            --line: #d1d5db;
            --accent: #0f766e;
            --accent-weak: #ccfbf1;
            --danger: #b91c1c;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text);
            background: radial-gradient(circle at top right, #ecfeff 0%, var(--bg) 40%, #eef2ff 100%);
        }

        .container {
            width: min(1100px, 92vw);
            margin: 28px auto;
        }

        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
        }

        h1 {
            margin: 0;
            font-size: 1.45rem;
            letter-spacing: 0.2px;
        }

        .admin-link {
            color: var(--accent);
            text-decoration: none;
            font-weight: 600;
        }

        .panel {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 14px;
            box-shadow: 0 8px 30px rgba(31, 41, 55, 0.08);
        }

        .search-form {
            display: grid;
            grid-template-columns: 1fr 140px 120px;
            gap: 10px;
            align-items: stretch;
        }

        .search-form textarea,
        .search-form input {
            width: 100%;
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 10px;
            font-size: 14px;
            line-height: 1.4;
            resize: vertical;
            background: #ffffff;
        }

        .search-form textarea {
            min-height: 78px;
        }

        .search-form button {
            border: none;
            border-radius: 10px;
            padding: 10px 14px;
            font-size: 14px;
            font-weight: 700;
            color: #ffffff;
            background: linear-gradient(135deg, #0f766e, #0e7490);
            cursor: pointer;
        }

        .form-hint {
            margin-top: 8px;
            color: var(--muted);
            font-size: 13px;
        }

        .error {
            margin-top: 10px;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #fecaca;
            color: var(--danger);
            background: #fef2f2;
        }

        .results {
            margin-top: 16px;
            display: grid;
            gap: 10px;
        }

        .result-item {
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 12px;
            background: #ffffff;
        }

        .meta {
            font-size: 12px;
            color: var(--muted);
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 8px;
        }

        .score {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            background: var(--accent-weak);
            color: #115e59;
            font-weight: 700;
        }

        .snippet {
            margin: 0;
            word-break: break-word;
            line-height: 1.55;
        }

        .snippet p {
            margin: 0.35rem 0;
        }

        .snippet pre {
            margin: 0.5rem 0;
            padding: 0.6rem 0.8rem;
            border-radius: 8px;
            background: #f8fafc;
            overflow-x: auto;
            border: 1px solid #e2e8f0;
        }

        .snippet code {
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.92em;
        }

        .snippet ul,
        .snippet ol {
            margin: 0.35rem 0 0.35rem 1.2rem;
            padding: 0;
        }

        .empty {
            margin-top: 12px;
            color: var(--muted);
            font-size: 14px;
        }

        @media (max-width: 760px) {
            .search-form {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RAG 搜索</h1>
            <a class="admin-link" href="{{ url_for('admin.index') }}">进入管理后台</a>
        </div>

        <div class="panel">
            <form method="get" action="{{ url_for('index') }}" class="search-form">
                <textarea name="q" placeholder="输入问题或检索关键词...">{{ query }}</textarea>
                <input type="number" min="1" max="50" name="top_k" value="{{ top_k }}" placeholder="top_k">
                <button type="submit">开始检索</button>
            </form>
            <div class="form-hint">默认返回前 {{ top_k }} 条结果。可在上方调整。</div>

            {% if error %}
            <div class="error">{{ error }}</div>
            {% endif %}

            {% if searched and not results and not error %}
            <div class="empty">未检索到结果，请尝试更具体的关键词。</div>
            {% endif %}

            {% if results %}
            <div class="results">
                {% for item in results %}
                <article class="result-item">
                    <div class="meta">
                        <span class="score">score: {{ '%.4f'|format(item.score if item.score is not none else 0.0) }}</span>
                        <span>paper_id: {{ item.paper_id }}</span>
                        <span>md_key: {{ item.md_key }}</span>
                        <span>line: {{ item.line_number }}</span>
                    </div>
                    <div class="snippet">{{ item.rendered_text }}</div>
                </article>
                {% endfor %}
            </div>
            {% endif %}
        </div>
    </div>
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true,
            },
            options: {
                processHtmlClass: 'snippet',
                ignoreHtmlClass: 'no-mathjax',
            },
        };
    </script>
    <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</body>
</html>
""".strip()


def _render_markdown_html(value: object | None) -> Markup | str:
    if not value:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    escaped_text = str(escape(text))
    if _markdown_render_impl is None:
        return Markup(f"<pre style=\"margin:0;white-space:pre-wrap;\">{escaped_text}</pre>")

    html = _markdown_render_impl(
        escaped_text,
        extensions=["extra", "sane_lists", "fenced_code", "tables", "nl2br"],
    )
    return Markup(html)


def _flash_queued(description: str) -> None:
    pending = write_queue.pending_count()
    flash(f"已提交写入队列：{description}（待处理 {pending} 个），将在 heartbeat 空闲时执行，稍后刷新查看结果。", "info")


def _same_form_value(a: object, b: object) -> bool:
    # 表单里空字符串与数据库中的 NULL 视为相同，避免把未改动的空字段当成修改
    return (a if a != "" else None) == (b if b != "" else None)


class DatabaseSectionMixin:
    """属于"数据库"分区的页面：不出现在顶部导航，只从数据库的二级菜单进入。"""

    pb_section = "database"

    def is_visible(self) -> bool:
        return False

    def pb_index_url(self) -> str:
        # 不能用 get_url(".index_view")：Flask 会相对“当前请求的页面”解析，而不是相对这个视图
        return url_for(f"{self.endpoint}.index_view")  # type: ignore[attr-defined]


class AdminModelView(DatabaseSectionMixin, ModelView):
    """所有写操作（增/改/删）都作为 quick 任务提交到 write_queue，由 heartbeat 线程执行（空闲时，或 heart_beat 内的 checkpoint 插队）。"""

    extra_css = ["/admin-assets/list-clamp.css"]
    extra_js = ["/admin-assets/list-autowidth.js"]

    def _model_field_values(self, form, *, only_changed: bool) -> dict[str, object]:
        # 在请求线程里把表单值取出来（外键字段在此时解析成模型实例），任务里只做赋值
        model_fields = self.model._meta.fields  # type: ignore[attr-defined]
        values: dict[str, object] = {}
        for field in form:
            if field.name not in model_fields:
                continue
            if only_changed and _same_form_value(field.data, field.object_data):
                continue
            values[field.name] = field.data
        return values

    def create_model(self, form):
        values = self._model_field_values(form, only_changed=False)
        model_cls = self.model
        description = f"create {model_cls.__name__}"

        def _task() -> None:
            model = model_cls()
            for name, value in values.items():
                setattr(model, name, value)
            self._on_model_change(form, model, True)
            model.save(force_insert=True)
            self.after_model_change(form, model, True)

        write_queue.submit(description, _task)
        _flash_queued(description)
        return True

    def update_model(self, form, model):
        # 只写回用户改动过的字段：执行时重新读取记录，避免覆盖 heartbeat 在此期间写入的其他字段
        values = self._model_field_values(form, only_changed=True)
        pk = self.get_pk_value(model)
        description = f"update {self.model.__name__} #{pk} ({', '.join(values) or 'no changes'})"
        if not values:
            return True

        def _task() -> None:
            fresh = self.get_one(pk)
            for name, value in values.items():
                setattr(fresh, name, value)
            self._on_model_change(form, fresh, False)
            fresh.save()
            self.after_model_change(form, fresh, False)

        write_queue.submit(description, _task)
        _flash_queued(description)
        return True

    def delete_model(self, model):
        pk = self.get_pk_value(model)
        description = f"delete {self.model.__name__} #{pk}"

        def _task() -> None:
            target = self.get_one(pk)
            self.on_model_delete(target)
            target.delete_instance(recursive=True)
            self.after_model_delete(target)

        write_queue.submit(description, _task)
        _flash_queued(description)
        return True

    @action(
        "delete",
        "Delete",
        "Are you sure you want to delete selected records?",
    )
    def action_delete(self, ids):
        model_cls = self.model
        pk_field = getattr(model_cls, self._primary_key)
        ids = list(ids)
        description = f"delete {len(ids)} {model_cls.__name__} record(s)"

        def _task() -> None:
            for record in list(model_cls.select().where(pk_field << ids)):
                self.on_model_delete(record)
                record.delete_instance(recursive=True)

        write_queue.submit(description, _task)
        _flash_queued(description)

    @staticmethod
    def _render_markdown_detail(value: object | None) -> Markup | str:
        html = _render_markdown_html(value)
        if not html:
            return ""
        return Markup(
            '<div style="max-width:min(960px,75vw);overflow:auto;line-height:1.5;">'
            f"{html}"
            "</div>"
        )


class TracedCollectionModelView(AdminModelView):
    column_list = ("name", "note", "last_traced_at", "last_paper_count")
    form_excluded_columns = ("last_traced_at", "last_paper_count")
    column_default_sort = "name"

    @action(
        "trace_collection",
        "Trace Now",
        "Run trace_collection for selected collections? This may take a while.",
    )
    def action_trace_collection(self, ids):
        for collection_id in ids:

            def _task(collection_id=collection_id) -> None:
                count = TracedCollection.get_by_id(collection_id).trace()
                if count is None:
                    print(f"Zotero actions paused; skipped tracing collection {collection_id}.")
                else:
                    print(f"Traced {count} paper(s) from collection {collection_id}.")

            write_queue.submit(f"trace collection #{collection_id}", _task, quick=False)
        _flash_queued(f"trace {len(ids)} collection(s)")


class ThemeModelView(AdminModelView):
    can_view_details = True
    details_modal = True

    def _summary_prompt_markdown(self, _context, model: Theme, _name):
        return self._render_markdown_detail(model.summary_prompt)

    column_formatters_detail = {
        "summary_prompt": _summary_prompt_markdown,
    }

    @action(
        "update_theme_papers",
        "Update Theme Papers",
        "Run update_theme_papers for selected themes?",
    )
    def action_update_theme_papers(self, ids):
        for theme_id in ids:

            def _task(theme_id=theme_id) -> None:
                Theme.get_by_id(theme_id).update_theme_papers()

            write_queue.submit(f"update theme papers #{theme_id}", _task, quick=False)
        _flash_queued(f"update theme papers for {len(ids)} theme(s)")
    
    @action(
        "download_theme_summaries",
        "Download Theme Summaries",
        "Download summaries (both theme-specific and quick) for the first selected theme?",
    )
    def action_download_theme_summaries(self, ids):
        from flask import make_response
        from urllib.parse import quote

        for theme_id in ids:
            try:
                theme = Theme.get_by_id(theme_id)
                summary_text = theme.retreive_summary(attach_quick_summary=True)

                if summary_text:
                    # 构造安全文件名
                    safe_name = "".join(
                        c for c in theme.name if c.isalnum() or c in (" ", "-", "_")
                    ).strip().replace(" ", "_") or "theme"

                    filename = f"{safe_name}_{theme.id}_summary.txt"
                    encoded_filename = quote(filename, safe="")

                    response = make_response(summary_text)
                    response.headers["Content-Type"] = "text/plain; charset=utf-8"
                    response.headers["Content-Disposition"] = (
                        f"attachment; filename*=UTF-8''{encoded_filename}"
                    )
                    return response

            except Exception as e:
                print(e)
                pass

        # 如果没有可下载的内容
        from flask import flash
        flash("No summaries available to download.", "warning")


class PaperModelView(AdminModelView):
    can_view_details = True
    details_modal = True

    column_list = (
        "key_in_zotero",
        "title",
        "year",
        "first_author",
        "doi",
        "abstract_preview",
        "quick_summary_preview",
        "md_key_in_zotero_display",
        "pdf_key_in_zotero_display",
        "md_status_display",
        "pdf_status_display",
    )

    column_details_list = [
        "key_in_zotero",
        "title",
        "year",
        "first_author",
        "doi",
        "abstract",
        "quick_summary",
        "md_key_in_zotero_display",
        "pdf_key_in_zotero_display",
        "md_status_display",
        "pdf_status_display",
    ]

    column_labels = {
        "abstract_preview": "Abstract",
        "quick_summary_preview": "Quick Summary",
        "md_key_in_zotero_display": "Md Key In Zotero",
        "pdf_key_in_zotero_display": "Pdf Key In Zotero",
        "md_status_display": "MD_Status",
        "pdf_status_display": "PDF_Status",
    }

    @staticmethod
    def _render_full_text(value: object | None) -> Markup | str:
        if not value:
            return ""
        text = str(value).strip()
        return Markup(f'<span title="{escape(text)}">{escape(text)}</span>')

    def _get_md_status(self, model: Paper):
        md_filter = model.md_status  # type: ignore[attr-defined]
        if not md_filter.exists():
            return None
        return md_filter.get()

    def _get_pdf_status(self, model: Paper):
        pdf_filter = model.pdf_status  # type: ignore[attr-defined]
        if not pdf_filter.exists():
            return None
        return pdf_filter.get()

    def _md_key_in_zotero_display(self, _context, model: Paper, _name):
        md_status = self._get_md_status(model)
        if not md_status or not md_status.md_key_in_zotero:
            return ""
        return str(md_status.md_key_in_zotero)

    def _abstract_preview(self, _context, model: Paper, _name):
        return self._render_full_text(model.abstract)

    def _quick_summary_preview(self, _context, model: Paper, _name):
        return self._render_full_text(model.quick_summary)

    def _quick_summary_detail(self, _context, model: Paper, _name):
        return self._render_markdown_detail(model.quick_summary)

    def _pdf_key_in_zotero_display(self, _context, model: Paper, _name):
        pdf_status = self._get_pdf_status(model)
        if not pdf_status or not pdf_status.pdf_key_in_zotero:
            return ""
        return str(pdf_status.pdf_key_in_zotero)

    def _md_status_display(self, _context, model: Paper, _name):
        md_status = self._get_md_status(model)
        if not md_status:
            return "Zotero❎Locally❎, transform not submitted"

        if md_status.has_md_in_zotero and md_status.has_md_locally:
            return "Zotero✔️ Locally✔️"

        transform_text = "submitted" if md_status.pdf_2_md_submitted else "not submitted"
        return f"Zotero❎Locally❎, transform {transform_text}"

    def _pdf_status_display(self, _context, model: Paper, _name):
        pdf_status = self._get_pdf_status(model)
        if not pdf_status:
            return "not required"

        if pdf_status.has_pdf_locally:
            return "local✔️"
        if pdf_status.has_pdf_in_zotero:
            return "Zotero✔️"
        return "Requiring" if pdf_status.pdf_require_submitted else "not required"

    column_formatters = {
        "abstract_preview": _abstract_preview,
        "quick_summary_preview": _quick_summary_preview,
        "md_key_in_zotero_display": _md_key_in_zotero_display,
        "pdf_key_in_zotero_display": _pdf_key_in_zotero_display,
        "md_status_display": _md_status_display,
        "pdf_status_display": _pdf_status_display,
    }

    column_formatters_detail = {
        "quick_summary": _quick_summary_detail,
    }

class ThemePaperModelView(AdminModelView):
    can_view_details = True
    details_modal = True
    # 规定不显示id
    def _display_paper(self, _context, model: ThemePaper, _name):
        paper = model.paper  # type: ignore[attr-defined]
        if not paper:
            return ""
        return Markup(f"<small>{escape(paper.first_author)} et al., {escape(str(paper.year))}</small>")
    def _display_theme(self, _context, model: ThemePaper, _name):
        theme = model.theme  # type: ignore[attr-defined]
        if not theme:
            return ""
        return escape(theme.name)
    def _display_theme_specific_summary(self, _context, model: ThemePaper, _name):
        summary = model.theme_specific_summary  # type: ignore[attr-defined]
        if not summary:
            return ""
        return Markup(f'<div style="max-width:400px; white-space:pre-wrap;">{escape(summary)}</div>')

    def _display_theme_specific_summary_detail(self, _context, model: ThemePaper, _name):
        return self._render_markdown_detail(model.theme_specific_summary)  # type: ignore[attr-defined]

    column_list = (
        "theme",
        "paper",
        "theme_specific_summary",
    )
    column_labels = {
        "theme": "Theme",
        "paper": "Paper",
        "theme_specific_summary": "Theme Summary",
    }
    column_formatters = {
        "paper": _display_paper,
        "theme": _display_theme,
        "theme_specific_summary": _display_theme_specific_summary,
    }

    column_formatters_detail = {
        "theme_specific_summary": _display_theme_specific_summary_detail,
    }


class MinerUTaskModelView(DatabaseSectionMixin, ModelView):
    """MinerU API PDF->MD 任务（mineru_tasks.db）。只读视图 + Retry / Fallback to manual 两个操作。

    这里直接写 mineru_tasks.db、不走 write_queue：该库与 pageboygent.db 分离，
    且所有状态迁移都带 WHERE state 条件，与 mineru-worker 进程并发安全。
    """

    extra_css = ["/admin-assets/list-clamp.css"]
    extra_js = ["/admin-assets/list-autowidth.js"]
    can_create = False
    can_edit = False
    can_delete = False
    can_view_details = True
    details_modal = True

    column_list = (
        "paper_title",
        "pdf_key",
        "state",
        "remote_state",
        "progress",
        "attempts",
        "err_msg",
        "model_version",
        "created_at",
        "submitted_at",
        "finished_at",
        "updated_at",
    )
    column_labels = {"paper_title": "Paper", "remote_state": "MinerU State", "err_msg": "Error"}
    column_default_sort = ("updated_at", True)
    column_filters = ("state",)
    column_searchable_list = ("paper_title", "pdf_key")

    def _progress(self, _context, model: MinerUTask, _name):
        if model.total_pages:
            return f"{model.extracted_pages or 0}/{model.total_pages}"
        return ""

    column_formatters = {"progress": _progress}

    @action("retry", "Retry", "Re-queue selected failed tasks (attempts reset to 0)?")
    def action_retry(self, ids):
        count = mineru_tasks.retry(int(i) for i in ids)
        flash(f"Re-queued {count} task(s). Only failed tasks can be retried.", "success" if count else "warning")

    @action(
        "fallback_manual",
        "Fallback to manual",
        "Move the PDFs of selected queued/failed tasks to human_upload_pdfs for manual conversion?",
    )
    def action_fallback_manual(self, ids):
        count = mineru_tasks.fallback_to_manual(int(i) for i in ids)
        flash(f"Moved {count} task(s) to the manual flow. Only queued/failed tasks can be moved.", "success" if count else "warning")


def _normalize_paper_id(raw_paper_id: object) -> int | None:
    if raw_paper_id is None:
        return None
    if isinstance(raw_paper_id, bool):
        return None
    if isinstance(raw_paper_id, int):
        return raw_paper_id
    if isinstance(raw_paper_id, str):
        try:
            return int(raw_paper_id)
        except ValueError:
            return None
    try:
        return int(str(raw_paper_id))
    except (TypeError, ValueError):
        return None


def _load_bib_full_info_map(raw_paper_ids: list[object]) -> dict[int, str]:
    normalized_ids = {
        paper_id
        for raw_paper_id in raw_paper_ids
        if (paper_id := _normalize_paper_id(raw_paper_id)) is not None
    }
    if not normalized_ids:
        return {}

    paper_rows = Paper.select().where(Paper.id.in_(sorted(normalized_ids)))  # type: ignore[attr-defined]
    bib_map: dict[int, str] = {}
    for paper in paper_rows:
        try:
            bib_map[int(paper.id)] = paper.bib_full_info()
        except Exception:
            # bib_full_info may fail if Zotero export is unavailable.
            bib_map[int(paper.id)] = ""
    return bib_map


def _build_rag_search_context() -> dict:
    query = (request.args.get("q") or "").strip()
    top_k_raw = (request.args.get("top_k") or "10").strip()

    try:
        top_k = int(top_k_raw)
    except ValueError:
        top_k = 10
    top_k = max(1, min(50, top_k))

    results: list[dict] = []
    error = ""
    searched = bool(query)

    if query:
        try:
            results = rag.retrieve(query=query, top_k=top_k)
            bib_full_info_map = _load_bib_full_info_map([item.get("paper_id") for item in results])
            normalized_results: list[dict] = []
            for item in results:
                row = dict(item)
                paper_id = _normalize_paper_id(item.get("paper_id"))
                row["bib_full_info"] = bib_full_info_map.get(paper_id, "") if paper_id is not None else ""
                row["rendered_text"] = _render_markdown_html(item.get("text"))
                normalized_results.append(row)
            results = normalized_results
        except Exception as exc:
            error = f"检索失败: {exc}"

    return {
        "query": query,
        "top_k": top_k,
        "results": results,
        "error": error,
        "searched": searched,
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# AI 文献检索（lit_search）
# ---------------------------------------------------------------------------

_LIT_CITE_HTML_RE = re.compile(r"\[(E\d+|R\d+|P\d+|\?)\]")
_LIT_KIND_LABELS = {"fulltext": "原文", "summary": "AI 快速总结", "theme": "主题总结", "note": "笔记", "reference": "参考文献"}
_LIT_RELEVANCE_LABELS = {"high": "高相关", "medium": "中相关", "low": "低相关"}


def _lit_render_with_citations(text: str, result: Mapping[str, Any], turn_id: str) -> str:
    """Markdown 渲染后，把 [E#] 变成跳转到证据片段的角标，把 [P#] 变成数据库生成的短引用。"""
    evidence = result.get("evidence", {})
    paper_cites = result.get("paper_cites", {})

    # 模型常写成 "[P184] Hansen (2009)"，而 [P184] 本身就会渲染成 "Hansen (2009)"：去掉紧随其后的重复引用
    def _drop_duplicate_cite(match: re.Match) -> str:
        cite = paper_cites.get(match.group(1))
        if not cite:
            return match.group(0)
        author = re.escape(cite.split(" (")[0])
        year = re.escape(cite.rsplit("(", 1)[-1].rstrip(")"))
        pattern = rf"\s*{author}(?:\s+(?:et al\.?|and \S+|& \S+))?\s*[（(]?\s*{year}\s*[）)]?"
        rest = re.sub(rf"^{pattern}", "", match.group(2), count=1)
        return f"[P{match.group(1)}]{rest}"

    text = re.sub(r"\[P(\d+)\]([^\n\[]{0,60})", _drop_duplicate_cite, text or "")
    html = str(_render_markdown_html(text))

    def _replace(match: re.Match) -> str:
        ref = match.group(1)
        if ref == "?":
            return '<span class="lit-cite lit-cite-bad" title="无效引用（已被系统剔除）">?</span>'
        if ref[0] in "ER":
            ev = evidence.get(ref)
            if not ev:
                return f'<span class="lit-cite">{ref}</span>'
            cite = paper_cites.get(str(ev["paper_id"]), f"P{ev['paper_id']}")
            title = escape(f"{cite} · {ev['label']}")
            return f'<a class="lit-cite" href="#lit-{turn_id}-{ref}" title="{title}">{ref}</a>'
        pid = ref[1:]
        cite = paper_cites.get(pid)
        if not cite:
            return f"[{ref}]"
        return f'<a class="lit-cite-paper" href="#lit-{turn_id}-P{pid}">{escape(cite)}</a>'

    return _LIT_CITE_HTML_RE.sub(_replace, html)


def _lit_render_evidence(ev: Mapping[str, Any]) -> dict:
    return {
        "eid": ev["eid"],
        "kind": ev["kind"],
        "kind_label": _LIT_KIND_LABELS.get(ev["kind"], ev["kind"]),
        "label": ev["label"],
        "methods": ev.get("methods", []),
        "text_html": str(_render_markdown_html(ev["text"])),
    }


def _lit_render_turn(turn: Mapping[str, Any]) -> dict:
    rendered: dict[str, Any] = {
        "id": turn["id"],
        "question": turn["question"],
        "status": turn["status"],
        "error": turn.get("error", ""),
        "created_at": turn.get("created_at", ""),
        "use_pro": bool(turn.get("use_pro")),
        "trace": turn.get("trace", []),
        "usage": turn.get("usage") or ((turn.get("result") or {}).get("stats") or {}).get("usage") or {},
        "result": None,
    }
    result = turn.get("result")
    if not result:
        return rendered
    evidence = result.get("evidence", {})
    candidates = []
    for cand in result.get("candidates", []):
        candidates.append({
            "paper": cand["paper"],
            "relevance": cand["relevance"],
            "relevance_label": _LIT_RELEVANCE_LABELS.get(cand["relevance"], cand["relevance"]),
            "reason_html": _lit_render_with_citations(cand.get("reason", ""), result, turn["id"]),
            "auto_evidence": cand.get("auto_evidence", False),
            "summary_only": cand.get("summary_only", False),
            "evidence": [_lit_render_evidence(evidence[eid]) for eid in cand.get("evidence_ids", []) if eid in evidence],
        })
    orphans = []
    for eid in result.get("orphan_evidence_ids", []):
        if eid in evidence:
            item = _lit_render_evidence(evidence[eid])
            item["paper_cite"] = result.get("paper_cites", {}).get(str(evidence[eid]["paper_id"]), "")
            orphans.append(item)
    secondary = []
    for ref in result.get("secondary_references", []):
        secondary.append({
            "ref_ids": ref["ref_ids"],
            "raw": ref["raw"],
            "fields": ref.get("fields", {}),
            "doi_url": ref.get("doi_url"),
            "relevance": ref["relevance"],
            "relevance_label": _LIT_RELEVANCE_LABELS.get(ref["relevance"], ref["relevance"]),
            "reason_html": _lit_render_with_citations(ref.get("reason", ""), result, turn["id"]),
            "cited_by": ref.get("cited_by", []),
            "in_library": ref.get("in_library"),
            "context_evidence": [_lit_render_evidence(evidence[e]) for e in ref.get("context_evidence_ids", []) if e in evidence],
        })
    from . import lit_search

    # 旧记录没有 short_answer：从详细报告开头截取，保证"先给简短回答"对历史结果也成立
    short_answer = result.get("short_answer") or lit_search._derive_short_answer(result.get("overall_judgement", ""))
    rendered["result"] = {
        "short_answer_html": _lit_render_with_citations(short_answer, result, turn["id"]),
        "short_answer_derived": not result.get("short_answer"),
        "judgement_html": _lit_render_with_citations(result.get("overall_judgement", ""), result, turn["id"]),
        "judgement_chars": len(result.get("overall_judgement", "")),
        "gaps_html": _lit_render_with_citations(result.get("gaps", ""), result, turn["id"]),
        "confidence": result.get("confidence", "low"),
        "confidence_label": {"high": "高", "medium": "中", "low": "低"}.get(result.get("confidence", "low"), "低"),
        "search_summary": result.get("search_summary", ""),
        "candidates": candidates,
        "secondary_references": secondary,
        "orphan_evidence": orphans,
        "warnings": result.get("warnings", []),
        "stats": result.get("stats", {}),
    }
    return rendered


def _chat_render_turn(turn: Mapping[str, Any]) -> dict:
    """AI 对话的一轮：回答（带可点击的 [E#]/[P#]/[R#]）+ 参考文献 + 引用片段。"""
    rendered: dict[str, Any] = {
        "id": turn["id"],
        "question": turn["question"],
        "status": turn["status"],
        "error": turn.get("error", ""),
        "created_at": turn.get("created_at", ""),
        "use_pro": bool(turn.get("use_pro")),
        "trace": turn.get("trace", []),
        "usage": turn.get("usage") or ((turn.get("result") or {}).get("stats") or {}).get("usage") or {},
        "result": None,
    }
    result = turn.get("result")
    if not result:
        return rendered
    evidence = result.get("evidence", {})
    rendered["result"] = {
        "answer_html": _lit_render_with_citations(result.get("answer", ""), result, turn["id"]),
        "legacy": bool(result.get("legacy")),
        "cited_papers": result.get("cited_papers", []),
        "evidence": [_lit_render_evidence(ev) for ev in evidence.values()],
        "warnings": result.get("warnings", []),
        "stats": result.get("stats", {}),
    }
    return rendered


def _agent_state_payload(store: Any, session_id: str, render_turn: Any) -> dict:
    """文献检索与 AI 对话共用：会话内容 + 会话列表 + token 仪表数据。"""
    from . import lit_search

    session = store.get(session_id)
    turns = session.get("turns", [])

    def _turn_model(turn: Mapping[str, Any]) -> str | None:
        stats = (turn.get("result") or {}).get("stats") or {}
        return turn.get("model") or stats.get("model") or (
            get_config("model_name_professional") if turn.get("use_pro") else get_config("model_name_cheap"))

    breakdowns = [
        lit_search.usage_breakdown(
            turn.get("usage") or ((turn.get("result") or {}).get("stats") or {}).get("usage"), _turn_model(turn))
        for turn in turns
    ]
    current = None
    if turns:
        # "当前一轮"：正在运行的那一轮，否则为最近一轮
        index = next((i for i, t in enumerate(turns) if t.get("status") == "running"), len(turns) - 1)
        current = {
            **breakdowns[index],
            "question": turns[index]["question"],
            "status": turns[index]["status"],
            "model": _turn_model(turns[index]),
            "turn_index": index + 1,
        }
    usage_panel = {
        "current": current,
        "session": {**lit_search.merge_breakdowns(breakdowns), "turn_count": len(turns)},
        "rates": lit_search.price_table(get_config("model_name_cheap")),
        "pro_rates": lit_search.price_table(get_config("model_name_professional")),
    }
    return {
        "ok": True,
        "session_id": session_id,
        "is_running": session.get("is_running", False),
        "usage_panel": usage_panel,
        "turns": [render_turn(turn) for turn in turns],
        "sessions": store.list_sessions(),
    }


def _agent_views() -> dict[str, tuple[Any, Any]]:
    """tab 名 -> (会话存储, 单轮渲染函数)。"""
    from . import chat_agent, lit_search

    return {
        "chat": (chat_agent.get_chat_store(), _chat_render_turn),
        "lit": (lit_search.get_store(), _lit_render_turn),
    }


def _build_session_context(tab: str) -> str:
    """页面初次加载时选中的会话：URL 指定的 > 新建 > 最近一个。"""
    from . import lit_search

    store, _ = _agent_views()[tab]
    session_id = lit_search.safe_session_id(request.values.get(f"{tab}_session"))
    if request.values.get(f"{tab}_new") == "1" or session_id is None:
        sessions = [] if request.values.get(f"{tab}_new") == "1" else store.list_sessions(limit=1)
        session_id = sessions[0]["id"] if sessions else store.new_session_id()
    return session_id


class HomeView(AdminIndexView):
    """/admin/：默认进入 AI 对话；兼容旧的 ?tab=chat|lit|rag 链接。"""

    def is_visible(self) -> bool:
        return False

    @expose("/")
    def index(self):
        target = {"lit": "litsearch.index", "rag": "rag.index"}.get((request.args.get("tab") or "").lower(), "chat.index")
        params = {k: v for k, v in request.args.items() if k != "tab"}
        return redirect(url_for(target, **params))


class AgentPageView(BaseView):
    """AI 功能页面（对话 / 文献检索 / RAG 搜索），共用 rag_home.html。"""

    pb_section = "ai"
    tab = "chat"

    @expose("/")
    def index(self):
        context = {
            "active_tab": self.tab,
            **_build_rag_search_context(),
            "chat_session_id": _build_session_context("chat") if self.tab == "chat" else "",
            "lit_session_id": _build_session_context("lit") if self.tab == "lit" else "",
            "review_project_id": (request.args.get("review_project") or "") if self.tab == "review" else "",
        }
        return self.render("admin/rag_home.html", **context)


class ChatPageView(AgentPageView):
    tab = "chat"


class LitSearchPageView(AgentPageView):
    tab = "lit"


class RagPageView(AgentPageView):
    tab = "rag"


class ReviewPageView(AgentPageView):
    tab = "review"


# ---------------------------------------------------------------------------
# AI 综述（review_pipeline）
# ---------------------------------------------------------------------------

_REVIEW_STAGE_KEYS = {"plan": "规划", "final": "最终综述"}


def _review_stage_label(key: str | None) -> str:
    if not key:
        return ""
    if key.startswith("notes:"):
        return "精读 · " + key.split(":", 1)[1]
    if key.startswith("review:"):
        return "子课题综述 · " + key.split(":", 1)[1]
    return _REVIEW_STAGE_KEYS.get(key, key)


def _review_state_payload(project_id: str) -> dict:
    from . import lit_search, review_pipeline

    store = review_pipeline.get_store()
    project = store.load(project_id)
    status = store.status(project_id, project)
    model = project["model"]
    base = store.project_dir(project_id)
    plan = project.get("plan") or {}
    progress = project.get("progress", {})

    subtopics = []
    notes_done = notes_total = reviews_done = 0
    for topic in plan.get("subtopics", []):
        topic_notes = progress.get("notes", {}).get(topic["dir"], {})
        papers = []
        for pid in topic["pids"]:
            meta = project["papers"].get(str(pid), {})
            note_path = f"subtopics/{topic['dir']}/notes/P{pid}.md"
            state = "done" if (base / note_path).exists() else topic_notes.get(str(pid), "pending")
            notes_total += 1
            notes_done += state == "done"
            papers.append({"pid": pid, "cite": f"{meta.get('first_author') or 'Unknown'} ({meta.get('year') or 'n.d.'})",
                           "title": meta.get("title", ""), "status": state, "path": note_path})
        review_path = f"subtopics/{topic['dir']}/review.md"
        review_state = "done" if (base / review_path).exists() else progress.get("reviews", {}).get(topic["dir"], "pending")
        reviews_done += review_state == "done"
        subtopics.append({"index": topic["index"], "label": topic["label"], "dir": topic["dir"],
                          "summary": topic.get("subprojectsummary", ""), "papers": papers,
                          "review_status": review_state, "review_path": review_path})

    final_path = "final/review.md"
    final_text = store.read_document(project_id, final_path)
    running_stage = project.get("stage") if status == "running" else None

    imported = bool(project.get("imported"))

    def _stage(key: str, done: bool, detail: str) -> dict:
        state = "done" if done else ("running" if running_stage == key else "pending")
        if status == "error" and project.get("stage") == key and not done:
            state = "error"
        if imported and not done:
            # 历史导入：旧流程没有这一步，或只留下了部分产物
            state, detail = ("done", detail) if detail else ("skipped", "旧流程无此步")
        return {"key": key, "label": review_pipeline.STAGE_LABELS[key], "state": state, "detail": detail}

    stages = [
        _stage("collect", bool(project.get("papers")),
               f"{len(project.get('papers', {}))} 篇可用 · {len(project.get('excluded', []))} 篇排除" if project.get("papers") else ""),
        _stage("plan", bool(plan), f"{len(subtopics)} 个子课题" if plan else ""),
        _stage("notes", bool(plan) and notes_total > 0 and notes_done == notes_total,
               (f"{notes_done} / {notes_total}" + (f"（缺 {notes_total - notes_done} 篇）"
                                                  if imported and notes_done < notes_total else "")) if plan else ""),
        _stage("reviews", bool(plan) and reviews_done == len(subtopics) and bool(subtopics),
               f"{reviews_done} / {len(subtopics)}" if plan else ""),
        _stage("final", final_text is not None, "已完成" if final_text else ""),
    ]

    # Token 仪表：整个项目 + 当前（或最近）阶段
    usage = project.get("usage", {})
    breakdowns = {key: lit_search.usage_breakdown(value, model) for key, value in usage.items()}
    current_key = project.get("current_stage_key") or (list(usage)[-1] if usage else None)
    current = None
    if current_key and current_key in breakdowns:
        current = {**breakdowns[current_key], "question": _review_stage_label(current_key),
                   "status": "running" if status == "running" else "done", "model": model, "turn_index": 0}
    usage_panel = {
        "current": current,
        "session": {**lit_search.merge_breakdowns(list(breakdowns.values())), "turn_count": len(breakdowns)},
        "rates": lit_search.price_table(get_config("model_name_cheap")),
        "pro_rates": lit_search.price_table(get_config("model_name_professional")),
    }
    return {
        "ok": True,
        "project": {
            "id": project_id, "name": project["name"], "topic": project["topic"], "collections": project["collections"],
            "model": model, "use_pro": project.get("use_pro"), "status": status, "error": project.get("error", ""),
            "created_at": project.get("created_at", "")[:16].replace("T", " "),
            "stages": stages, "subtopics": subtopics, "excluded": project.get("excluded", []),
            "plan_warnings": plan.get("warnings", []),
            "imported": bool(project.get("imported")),
            "source_note": project.get("source_note", ""),
            # 附件：原始规划文件等（只列出确实存在的）
            "attachments": [a for a in ([{"label": "规划（模型原始输出）", "path": "plan/plan.xml"}]
                                        + project.get("attachments", []))
                            if store.read_document(project_id, a["path"]) is not None],
            "final": {"path": final_path, "html": str(_render_markdown_html(final_text)) if final_text else ""},
            "trace": project.get("trace", [])[-200:],
        },
        "usage_panel": usage_panel,
        "projects": store.list_projects(),
    }


class DatabaseDashboardView(BaseView):
    """数据库 → 看板。"""

    pb_section = "database"

    def pb_index_url(self) -> str:
        return url_for(f"{self.endpoint}.index")

    @expose("/")
    def index(self):
        return self.render("admin/pb_dashboard.html", stats=_dashboard_stats())


# ---------------------------------------------------------------------------
# 数据看板
# ---------------------------------------------------------------------------

def _count(query) -> int:
    try:
        return int(query.count())
    except Exception as exc:  # pragma: no cover - 看板不应因为某个统计失败而整页报错
        print(f"[dashboard] count failed: {exc}")
        return 0


def _ai_usage_summary(limit_recent: int = 6) -> dict:
    """累计 AI 花费与最近会话（对话 + 文献检索）。只读会话文件，不读体积较大的 .agent.json。"""
    from . import chat_agent, lit_search

    totals = {"tokens": 0, "cost": 0.0, "turns": 0, "sessions": {"chat": 0, "lit": 0}, "unpriced": 0,
              "by_kind": {"chat": {"tokens": 0, "cost": 0.0}, "lit": {"tokens": 0, "cost": 0.0}}}
    recent: list[dict] = []
    stores = {"chat": chat_agent.get_chat_store(), "lit": lit_search.get_store()}
    for kind, store in stores.items():
        for path in store.directory.glob("*.json"):
            session_id = lit_search.safe_session_id(path.stem)
            if not session_id:
                continue
            try:
                session = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            turns = session.get("turns", [])
            if not turns:
                continue
            totals["sessions"][kind] += 1
            for turn in turns:
                stats = (turn.get("result") or {}).get("stats") or {}
                usage = turn.get("usage") or stats.get("usage")
                if not usage:
                    continue
                model = turn.get("model") or stats.get("model") or (
                    get_config("model_name_professional") if turn.get("use_pro") else get_config("model_name_cheap"))
                breakdown = lit_search.usage_breakdown(usage, model)
                totals["tokens"] += breakdown["total_tokens"]
                totals["by_kind"][kind]["tokens"] += breakdown["total_tokens"]
                totals["turns"] += 1
                if breakdown["cost"] is None:
                    totals["unpriced"] += 1
                else:
                    totals["cost"] += breakdown["cost"]["total"]
                    totals["by_kind"][kind]["cost"] += breakdown["cost"]["total"]
            endpoint = "chat.index" if kind == "chat" else "litsearch.index"
            recent.append({
                "kind": kind,
                "kind_label": "对话" if kind == "chat" else "文献检索",
                "title": turns[0].get("question", "")[:60],
                "turns": len(turns),
                "mtime": path.stat().st_mtime,
                "updated": datetime.fromtimestamp(path.stat().st_mtime).strftime("%m-%d %H:%M"),
                "url": url_for(endpoint, **{f"{kind}_session": session_id}),
            })
    recent.sort(key=lambda item: item["mtime"], reverse=True)
    totals["recent"] = recent[:limit_recent]
    return totals


def _ai_spend() -> dict:
    """AI 花费 = 对话/文献检索的会话用量 + 用量账本（总结、精读、综述流水线等）。"""
    from . import llm_usage

    sessions = _ai_usage_summary()
    ledger = llm_usage.summarize()
    parts = [
        {"label": "AI 对话", "tokens": sessions["by_kind"]["chat"]["tokens"], "cost": sessions["by_kind"]["chat"]["cost"]},
        {"label": "AI 文献检索", "tokens": sessions["by_kind"]["lit"]["tokens"], "cost": sessions["by_kind"]["lit"]["cost"]},
    ] + [{"label": e["category"], "tokens": e["tokens"], "cost": e["cost"]} for e in ledger["categories"]]
    parts = [part for part in parts if part["tokens"]]
    total_cost = sum(part["cost"] for part in parts)
    for part in parts:
        part["pct"] = round(100.0 * part["cost"] / total_cost, 1) if total_cost else 0.0
    parts.sort(key=lambda part: part["cost"], reverse=True)
    return {
        **sessions,
        "cost": total_cost,
        "tokens": sum(part["tokens"] for part in parts),
        "unpriced": sessions["unpriced"] + ledger["unpriced_calls"],
        "parts": parts,
        "ledger_since": ledger["since"],
    }


def _year_histogram(years: list[int], start: int = 1990) -> dict:
    """出版年份分布：start 之前合并为一根"更早"的柱子。"""
    if not years:
        return {"bars": [], "max": 0}
    last = max(years)
    counts: dict[int, int] = {}
    earlier = 0
    for year in years:
        if year < start:
            earlier += 1
        else:
            counts[year] = counts.get(year, 0) + 1
    bars = ([{"label": f"{start} 年以前", "short": "更早", "value": earlier, "year": None}] if earlier else [])
    bars += [{"label": f"{year} 年", "short": str(year), "value": counts.get(year, 0), "year": year}
             for year in range(start, last + 1)]
    peak = max(bars, key=lambda bar: bar["value"])
    return {"bars": bars, "max": peak["value"], "peak": peak, "total": len(years)}


def _dashboard_stats() -> dict:
    from .note_storage import NoteRecord

    total = _count(Paper.select())
    has_pdf = _count(PDFStatus.select().where((PDFStatus.has_pdf_locally == True) | (PDFStatus.has_pdf_in_zotero == True)))  # noqa: E712
    has_md = _count(MDStatus.select().where(MDStatus.has_md_locally == True))  # noqa: E712
    has_summary = _count(Paper.select().where(Paper.quick_summary.is_null(False)))
    try:
        embedded = _count(rag.EmbedStatus.select().where(rag.EmbedStatus.status == rag.EmbedStatus.STATUS_DONE))
        embedding_backlog = _count(rag.EmbedStatus.select().where(rag.EmbedStatus.status != rag.EmbedStatus.STATUS_DONE))
        chunks = int(rag.DBcollection.count())
    except Exception as exc:  # pragma: no cover
        print(f"[dashboard] rag stats failed: {exc}")
        embedded, embedding_backlog, chunks = 0, 0, 0

    def _pct(value: int) -> float:
        return round(100.0 * value / total, 1) if total else 0.0

    pipeline = [
        {"label": "已追踪", "hint": "Zotero 中被追踪的文献", "value": total},
        {"label": "有 PDF", "hint": "本地或 Zotero 中已有 PDF", "value": has_pdf},
        {"label": "全文 Markdown", "hint": "PDF 已转换为可检索的全文", "value": has_md},
        {"label": "AI 快速总结", "hint": "已生成 quick summary", "value": has_summary},
        {"label": "已入向量库", "hint": "原文已切块并写入 RAG 向量库", "value": embedded},
    ]
    for step in pipeline:
        step["pct"] = _pct(step["value"])

    pdf_waiting = _count(PDFStatus.select().where(
        (PDFStatus.pdf_require_submitted == True) & (PDFStatus.has_pdf_in_zotero == False)))  # noqa: E712
    md_converting = _count(MDStatus.select().where(
        (MDStatus.pdf_2_md_submitted == True) & (MDStatus.has_md_locally == False)))  # noqa: E712
    summary_waiting = _count(Paper.select().join(MDStatus).where(
        (MDStatus.has_md_locally == True) & Paper.quick_summary.is_null(True)))  # noqa: E712
    mineru_failed = _count(MinerUTask.select().where(MinerUTask.state == "failed"))
    mineru_active = _count(MinerUTask.select().where(MinerUTask.state.not_in(["done", "failed"])))
    notes_unsynced = _count(NoteRecord.select().where(NoteRecord.needs_sync == True))  # noqa: E712
    queue_pending = write_queue.pending_count()

    def _item(label: str, value: int, severity_if_any: str, hint: str, url: str | None = None) -> dict:
        return {"label": label, "value": value, "severity": severity_if_any if value else "good", "hint": hint, "url": url}

    pending = [
        _item("PDF 待获取", pdf_waiting, "warning", "已加入 PDF 需求集合，等待在 Zotero 中补充 PDF",
              url_for("pdfstatus.index_view")),
        _item("MinerU 失败任务", mineru_failed, "critical", "PDF→Markdown 转换失败，可在 MinerU 任务中重试",
              url_for("mineru_tasks.index_view")),
        _item("笔记待同步", notes_unsynced, "warning", "本地生成的笔记尚未写回 Zotero"),
        _item("全文转换中", md_converting + mineru_active, "info", "PDF→Markdown 转换任务进行中",
              url_for("mineru_tasks.index_view")),
        _item("等待生成 AI 总结", summary_waiting, "info", "全文就绪后约 10 天自动生成 quick summary"),
        _item("向量库待入库", embedding_backlog, "info", "等待切块、筛选并写入向量库"),
        _item("写入队列", queue_pending, "info", "网页端提交、等待后台执行的写操作"),
    ]
    attention = sum(1 for item in pending if item["severity"] in {"warning", "critical"})

    themes = []
    for theme in Theme.select().order_by(Theme.name):
        papers = _count(ThemePaper.select().where(ThemePaper.theme == theme))
        done = _count(ThemePaper.select().where(
            (ThemePaper.theme == theme) & ThemePaper.theme_specific_summary.is_null(False)))
        themes.append({"name": theme.name, "papers": papers, "done": done,
                       "pct": round(100.0 * done / papers, 1) if papers else 0.0,
                       "url": url_for("theme.details_view", id=theme.id)})
    themes.sort(key=lambda t: t["papers"], reverse=True)
    theme_papers = sum(t["papers"] for t in themes)
    theme_done = sum(t["done"] for t in themes)

    collections = list(TracedCollection.select())
    last_traced = max((c.last_traced_at for c in collections if c.last_traced_at), default=None)

    recent_papers = []
    paper_rows = list(Paper.select().order_by(Paper.id.desc()).limit(8))
    pdf_map = {row.paper_id: row for row in PDFStatus.select().where(PDFStatus.paper.in_([p.id for p in paper_rows]))}
    md_map = {row.paper_id: row for row in MDStatus.select().where(MDStatus.paper.in_([p.id for p in paper_rows]))}
    try:
        embedded_ids = {row.paper_id for row in rag.EmbedStatus.select(rag.EmbedStatus.paper_id).where(
            (rag.EmbedStatus.status == rag.EmbedStatus.STATUS_DONE) & rag.EmbedStatus.paper_id.in_([p.id for p in paper_rows]))}
    except Exception:  # pragma: no cover
        embedded_ids = set()
    for paper in paper_rows:
        pdf, md = pdf_map.get(paper.id), md_map.get(paper.id)
        recent_papers.append({
            "id": paper.id,
            "cite": f"{paper.first_author or 'Unknown'} ({paper.year or 'n.d.'})",
            "title": paper.title,
            "url": url_for("paper.details_view", id=paper.id),
            "steps": [
                ("PDF", bool(pdf and (pdf.has_pdf_locally or pdf.has_pdf_in_zotero))),
                ("全文", bool(md and md.has_md_locally)),
                ("总结", bool(paper.quick_summary)),
                ("向量库", paper.id in embedded_ids),
            ],
        })

    years = [int(p.year) for p in Paper.select(Paper.year).where(Paper.year.is_null(False))
             if 1900 < int(p.year) <= datetime.now().year + 1]
    return {
        "total": total,
        "complete": embedded,
        "chunks": chunks,
        "pipeline": pipeline,
        "pending": pending,
        "attention": attention,
        "themes": themes,
        "theme_papers": theme_papers,
        "theme_done": theme_done,
        "theme_pct": round(100.0 * theme_done / theme_papers, 1) if theme_papers else 0.0,
        "collections": len(collections),
        "last_traced": last_traced.strftime("%Y-%m-%d") if last_traced else None,
        "notes": _count(NoteRecord.select()),
        "notes_unsynced": notes_unsynced,
        "recent_papers": recent_papers,
        "years": _year_histogram(years),
        "ai": _ai_spend(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


@dataclass(frozen=True)
class BindTarget:
    host: str
    port: int


def parse_bind_target(bind: str | None, default_host: str, default_port: int) -> BindTarget:
    if not bind:
        return BindTarget(host=default_host, port=default_port)

    if ":" not in bind:
        raise ValueError(f"Invalid bind target '{bind}'. Expected format: host:port")

    host, port_text = bind.rsplit(":", 1)
    host = host.strip()
    port_text = port_text.strip()

    if not host:
        raise ValueError(f"Invalid bind target '{bind}'. Host cannot be empty")
    if not port_text.isdigit():
        raise ValueError(f"Invalid bind target '{bind}'. Port must be an integer")

    port = int(port_text)
    if port < 1 or port > 65535:
        raise ValueError(f"Invalid bind target '{bind}'. Port must be in range 1-65535")

    return BindTarget(host=host, port=port)


def create_storage_admin_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "pageboygent-storage-admin"

    def _register_agent_api(tab: str, url_prefix: str, empty_error: str) -> None:
        def _send() -> Any:
            from . import lit_search

            store, render_turn = _agent_views()[tab]
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                payload = request.form.to_dict(flat=True)
            question = str(payload.get("question") or "").strip()
            if not question:
                return jsonify({"ok": False, "error": empty_error}), 400
            session_id = lit_search.safe_session_id(str(payload.get("session_id") or "")) or store.new_session_id()
            try:
                store.start_turn(session_id, question, use_pro=_as_bool(payload.get("use_pro")))
            except RuntimeError as exc:
                return jsonify({"ok": False, "error": str(exc)}), 409
            return jsonify(_agent_state_payload(store, session_id, render_turn))

        def _state() -> Any:
            from . import lit_search

            store, render_turn = _agent_views()[tab]
            session_id = lit_search.safe_session_id(request.args.get("session_id"))
            if session_id is None:
                return jsonify({"ok": False, "error": "session_id 无效。"}), 400
            return jsonify(_agent_state_payload(store, session_id, render_turn))

        app.add_url_rule(f"{url_prefix}/send", f"{tab}_api_send", _send, methods=["POST"])
        app.add_url_rule(f"{url_prefix}/state", f"{tab}_api_state", _state, methods=["GET"])

    _register_agent_api("chat", "/admin-api/chat", "请输入问题后再发送。")
    _register_agent_api("lit", "/admin-api/litsearch", "请输入检索需求。")

    # ---- AI 综述 ----
    def _review_store():
        from . import review_pipeline

        return review_pipeline.get_store()

    def _review_id(raw: Any) -> str | None:
        from . import review_pipeline

        return review_pipeline.safe_project_id(str(raw or ""))

    @app.get("/admin-api/review/collections")
    def review_api_collections() -> Any:
        return jsonify({"ok": True, "collections": _review_store().backend.collection_names()})

    @app.post("/admin-api/review/preview")
    def review_api_preview() -> Any:
        payload = request.get_json(silent=True) or {}
        names = [str(n) for n in payload.get("collections") or [] if str(n).strip()]
        if not names:
            return jsonify({"ok": False, "error": "请至少选择一个 Collection。"}), 400
        backend = _review_store().backend
        try:
            keys = list(dict.fromkeys(k for name in names for k in backend.collection_keys(name)))
        except Exception as exc:
            return jsonify({"ok": False, "error": f"读取 Zotero Collection 失败：{exc}"}), 502
        found = backend.papers_by_keys(keys)
        usable = sum(1 for meta in found.values() if meta["md_path"])
        return jsonify({"ok": True, "items": len(keys), "usable": usable,
                        "no_fulltext": len(found) - usable, "untracked": len(keys) - len(found)})

    @app.post("/admin-api/review/create")
    def review_api_create() -> Any:
        payload = request.get_json(silent=True) or {}
        names = [str(n).strip() for n in payload.get("collections") or [] if str(n).strip()]
        topic = str(payload.get("topic") or "").strip()
        if not names or not topic:
            return jsonify({"ok": False, "error": "请选择 Collection 并填写综述主题。"}), 400
        store = _review_store()
        project_id = store.create(name=str(payload.get("name") or ""), collections=names, topic=topic,
                                  use_pro=_as_bool(payload.get("use_pro")))
        store.start(project_id)
        return jsonify(_review_state_payload(project_id))

    @app.get("/admin-api/review/state")
    def review_api_state() -> Any:
        project_id = _review_id(request.args.get("project_id"))
        if project_id is None:
            return jsonify({"ok": True, "project": None, "projects": _review_store().list_projects(),
                            "usage_panel": None})
        try:
            return jsonify(_review_state_payload(project_id))
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "项目不存在。"}), 404

    @app.post("/admin-api/review/resume")
    def review_api_resume() -> Any:
        project_id = _review_id((request.get_json(silent=True) or {}).get("project_id"))
        if project_id is None:
            return jsonify({"ok": False, "error": "project_id 无效。"}), 400
        try:
            _review_store().start(project_id)
        except RuntimeError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 409
        return jsonify(_review_state_payload(project_id))

    @app.post("/admin-api/review/cancel")
    def review_api_cancel() -> Any:
        project_id = _review_id((request.get_json(silent=True) or {}).get("project_id"))
        if project_id is None:
            return jsonify({"ok": False, "error": "project_id 无效。"}), 400
        _review_store().cancel(project_id)
        return jsonify(_review_state_payload(project_id))

    @app.get("/admin-api/review/doc")
    def review_api_doc() -> Any:
        project_id = _review_id(request.args.get("project_id"))
        path = str(request.args.get("path") or "")
        text = _review_store().read_document(project_id, path) if project_id else None
        if text is None:
            return jsonify({"ok": False, "error": "文档不存在或尚未生成。"}), 404
        if request.args.get("download") == "1":
            from urllib.parse import quote

            project = _review_store().load(project_id)
            filename = f"{project['slug']}-{Path(path).stem}.md"
            response = Response(text, mimetype="text/markdown; charset=utf-8")
            response.headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{quote(filename)}"
            return response
        # 笔记开头的 <!-- P# · 作者 (年份) · 标题 --> 是给文件读者看的说明，网页上不显示
        text = re.sub(r"\A\s*<!--.*?-->\s*", "", text, flags=re.S)
        return jsonify({"ok": True, "path": path, "html": str(_render_markdown_html(text))})

    @app.get("/admin-assets/list-clamp.css")
    def admin_list_css() -> Response:
        return Response(LIST_VIEW_CSS, mimetype="text/css")

    @app.get("/admin-assets/list-autowidth.js")
    def admin_list_js() -> Response:
        return Response(LIST_VIEW_JS, mimetype="application/javascript")

    admin = Admin(
        app,
        name="Pageboygent",
        # Flask-Admin 2.x：基础模板通过主题指定；所有页面（含数据表的增删改查页）都套用 pb_master.html
        theme=Bootstrap4Theme(base_template="admin/pb_master.html"),
        index_view=HomeView(url="/admin", endpoint="admin", name="Home"),
    )

    # 顶部导航：三个 AI 页面 + 数据库（看板）
    top_views = [
        ChatPageView(name="AI 对话", endpoint="chat", url="/admin/chat"),
        LitSearchPageView(name="AI 文献检索", endpoint="litsearch", url="/admin/litsearch"),
        ReviewPageView(name="AI 综述", endpoint="review", url="/admin/review"),
        RagPageView(name="RAG 搜索", endpoint="rag", url="/admin/rag"),
        DatabaseDashboardView(name="数据库", endpoint="database", url="/admin/database"),
    ]
    for view in top_views:
        admin.add_view(view)
    dashboard_view = top_views[-1]

    # 数据库的二级菜单（按这个顺序）；其余 BaseModel 子类自动追加在后面，保证新表也有 CRUD 页面
    db_views: list[Any] = [
        PaperModelView(Paper, name="文献"),
        ThemeModelView(Theme, name="主题"),
        ThemePaperModelView(ThemePaper, name="主题-文献"),
        TracedCollectionModelView(TracedCollection, name="追踪集合"),
        AdminModelView(PDFStatus, name="PDF 状态"),
        AdminModelView(MDStatus, name="全文状态"),
    ]
    known = {view.model for view in db_views}
    for model in sorted(BaseModel.__subclasses__(), key=lambda m: m.__name__):
        if model not in known:
            db_views.append(AdminModelView(model))
    db_views.append(MinerUTaskModelView(MinerUTask, name="MinerU 任务", endpoint="mineru_tasks"))
    for view in db_views:
        admin.add_view(view)

    def pb_top_nav(admin_view: Any) -> list[dict]:
        section = getattr(admin_view, "pb_section", "")
        return [{"label": view.name, "url": url_for(f"{view.endpoint}.index"),
                 "active": admin_view is view or (view is dashboard_view and section == "database")}
                for view in top_views]

    def pb_db_nav(admin_view: Any) -> list[dict]:
        items = [{"label": "看板", "url": dashboard_view.pb_index_url(), "active": admin_view is dashboard_view,
                  "count": None}]
        for view in db_views:
            items.append({"label": view.name, "url": view.pb_index_url(), "active": admin_view is view,
                          "count": _count(view.model.select())})
        return items

    app.jinja_env.globals.update(
        pb_top_nav=pb_top_nav,
        pb_db_nav=pb_db_nav,
        pb_home_url=lambda: url_for("chat.index"),
    )

    @app.get("/")
    def index_redirect():
        return redirect(url_for("admin.index"))

    return app


def _resolve_heartbeat_interval_seconds() -> int:
    raw_value = get_config("storage_heartbeat_interval_seconds")
    if raw_value is None:
        return 60
    try:
        interval = int(raw_value)
    except (TypeError, ValueError):
        print(
            "Invalid config 'storage_heartbeat_interval_seconds' "
            f"value: {raw_value!r}. Falling back to 60 seconds."
        )
        return 60
    if interval <= 0:
        print(
            "Invalid config 'storage_heartbeat_interval_seconds' "
            f"value: {interval}. Falling back to 60 seconds."
        )
        return 60
    return interval


def _start_heartbeat_worker() -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()
    interval_seconds = _resolve_heartbeat_interval_seconds()

    def _worker() -> None:
        print(f"Storage heartbeat worker started (interval={interval_seconds}s)")
        write_queue.set_consumer_thread()
        next_beat = time.monotonic() + interval_seconds
        while not stop_event.is_set():
            # heartbeat 空闲期：处理 Web 端提交的写任务，直到下一次 heart_beat
            write_queue.process_until(next_beat, stop_event)
            if stop_event.is_set():
                break
            try:
                storage.heart_beat()
                rag.heart_beat()
            except KeyboardInterrupt:
                # 内部 embedding 任务收到关闭信号并已保存状态，安全退出循环
                break
            except Exception as exc:
                print(f"heart_beat execution failed: {exc}")
            next_beat = time.monotonic() + interval_seconds
        write_queue.drain(shutting_down=True)
        print("Storage heartbeat worker stopped.")

    thread = threading.Thread(target=_worker, name="pageboygent-heartbeat", daemon=False)
    thread.start()
    return stop_event, thread


def _start_mineru_worker(global_args: Sequence[str]) -> subprocess.Popen | None:
    """api 模式下拉起独立的 mineru-worker 进程（PDF 上传、轮询、下载都在该进程中完成）。"""
    if get_config("pdf2md_mode") != "api":
        print("pdf2md_mode is not 'api'; MinerU worker not started (manual PDF->MD flow).")
        return None
    cmd = [sys.executable, "-m", "Pageboygent", *global_args, "mineru-worker", "--parent-pid", str(os.getpid())]
    proc = subprocess.Popen(cmd)
    print(f"MinerU worker process started (pid={proc.pid}).")
    return proc


def _stop_mineru_worker(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("警告：MinerU worker 未在10秒内退出，强制结束。")
        proc.kill()


def serve_storage(
    bind: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    worker_global_args: Sequence[str] = (),
) -> int:
    try:
        target = parse_bind_target(bind=bind, default_host=host, default_port=port)
    except ValueError as exc:
        print(exc)
        return 2

    app = create_storage_admin_app()
    heartbeat_stop_event, heartbeat_thread = _start_heartbeat_worker()
    from . import lit_search
    lit_search.warmup_indexes()
    mineru_worker = _start_mineru_worker(worker_global_args)
    home_url = f"http://{target.host}:{target.port}/"
    admin_url = f"http://{target.host}:{target.port}/admin/"
    print(f"Serving storage home at {home_url}")
    print(f"Serving storage admin at {admin_url}")

    def _on_shutdown(signum: int, frame: object) -> None:
        print(f"\n收到信号 {signum}，通知 embedding 任务保存状态后退出...")
        rag.request_shutdown()       # 在 embedding 循环中注入 KeyboardInterrupt
        heartbeat_stop_event.set()   # 阻止心跳线程启动新任务
        write_queue.wake()           # 唤醒正在等待写队列的心跳线程
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _on_shutdown)

    if open_browser:
        webbrowser.open(admin_url)

    try:
        app.run(host=target.host, port=target.port)
    finally:
        rag.request_shutdown()
        heartbeat_stop_event.set()
        write_queue.wake()
        _stop_mineru_worker(mineru_worker)
        print("等待 embedding 任务保存状态，最多等待60秒...")
        heartbeat_thread.join(timeout=60)
        if heartbeat_thread.is_alive():
            print("警告：heartbeat 线程在超时时间内未完全退出。")
        else:
            print("Heartbeat worker 已安全退出。")
    return 0
