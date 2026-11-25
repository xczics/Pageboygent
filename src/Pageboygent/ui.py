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
from flask_admin import Admin, AdminIndexView, expose
from flask_admin.actions import action
from flask_admin.contrib.peewee import ModelView
from markupsafe import Markup, escape

try:
    from markdown import markdown as _markdown_render_impl  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency fallback
    _markdown_render_impl = None

from .config import get_config
from .storage import BaseModel, Paper, Theme, ThemePaper, TracedCollection
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


class AdminModelView(ModelView):
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


class MinerUTaskModelView(ModelView):
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


class RagSearchAdminIndexView(AdminIndexView):
    @expose("/", methods=("GET", "POST"))
    def index(self):
        active_tab = (request.values.get("tab") or "chat").strip().lower()
        if active_tab not in {"chat", "rag", "lit"}:
            active_tab = "chat"

        context = {
            "active_tab": active_tab,
            **_build_rag_search_context(),
            "chat_session_id": _build_session_context("chat") if active_tab == "chat" else "",
            "lit_session_id": _build_session_context("lit") if active_tab == "lit" else "",
        }
        return self.render("admin/rag_home.html", **context)
    
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

    @app.get("/admin-assets/list-clamp.css")
    def admin_list_css() -> Response:
        return Response(LIST_VIEW_CSS, mimetype="text/css")

    @app.get("/admin-assets/list-autowidth.js")
    def admin_list_js() -> Response:
        return Response(LIST_VIEW_JS, mimetype="application/javascript")

    admin = Admin(
        app,
        name="Pageboygent Storage",
        index_view=RagSearchAdminIndexView(url="/admin", endpoint="admin", name="Home"),
    )

    # Register every Peewee model declared under BaseModel so all storage tables
    # get a CRUD view without manually maintaining model lists.
    for model in sorted(BaseModel.__subclasses__(), key=lambda m: m.__name__):
        if model is Paper:
            admin.add_view(PaperModelView(model))
            continue
        if model is Theme:
            admin.add_view(ThemeModelView(model))
            continue
        if model is ThemePaper:
            admin.add_view(ThemePaperModelView(model))
            continue
        if model is TracedCollection:
            admin.add_view(TracedCollectionModelView(model))
            continue
        else:
            admin.add_view(AdminModelView(model))
    admin.add_view(MinerUTaskModelView(MinerUTask, name="MinerU Tasks", endpoint="mineru_tasks"))

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
