"""
RSS学术论文追踪器

自动抓取 RSS Feed → AI 初筛 → 标注管理 → 准备注入 Zotero 流水线

数据流：
  高频（每1-2小时）→ fetch_all_feeds() → 新文章入库 (status=pending)
                          → screen_pending_articles() → AI 评分
                          → 更新 status (interested / not_interested)
  低频（每天/用户请求）→ get_digest() → 呈现筛选结果
  标注操作 → mark_interested() / mark_not_interested() / mark_favorite()

配置源：pageboygent.toml / .env (通过 config.py 统一管理)
数据库：rss_tracker.db (SQLite via Peewee，与现有 pageboygent.db 独立)
"""

from __future__ import annotations

import datetime
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import feedparser
import json
from peewee import (
    BooleanField,
    CharField,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    SqliteDatabase,
    TextField,
    fn,
)

from .config import get_config, register_config

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# 配置注册 —— 这些配置项可通过 pageboygent.toml / .env 设定
# ═══════════════════════════════════════════════════════════

register_config(
    "rss_feeds",
    type_=list,
    default=[],
    description="List of RSS feed URLs to track. Each entry: {name, url, category}",
)
register_config(
    "rss_check_interval_minutes",
    type_=int,
    default=60,
    description="RSS feed check interval in minutes (how often fetch_all runs)",
)
register_config(
    "rss_ai_screening_enabled",
    type_=bool,
    default=True,
    description="Enable AI-based screening for new articles",
)
register_config(
    "rss_ai_threshold",
    type_=int,
    default=7,
    description="Minimum AI score (0-10) for auto-marking as interested",
)
register_config(
    "rss_digest_interval_hours",
    type_=int,
    default=24,
    description="How often to surface screening results to the user",
)
register_config(
    "rss_interest_profile",
    type_=str,
    default="",
    description="Research interest profile for the AI screening prompt",
)

# ═══════════════════════════════════════════════════════════
# 数据库 & 模型
# ═══════════════════════════════════════════════════════════

DB = SqliteDatabase(None)  # lazy init, call init_db() first


def _get_db_path() -> Path:
    workstation = Path(get_config("workstation_dir")).expanduser()
    return workstation / "rss_tracker.db"


class BaseModel(Model):
    class Meta:
        database = DB


class RssFeed(BaseModel):
    """RSS Feed 源"""
    name = CharField(max_length=255)
    url = CharField(max_length=1024, unique=True)
    category = CharField(max_length=128, null=True)
    enabled = BooleanField(default=True)
    last_checked_at = DateTimeField(null=True)
    created_at = DateTimeField(default=datetime.datetime.now)

    class Meta:
        table_name = "rss_feed"


class RssArticle(BaseModel):
    """从 RSS 抓取到的学术论文/文献条目"""
    feed = ForeignKeyField(RssFeed, backref="articles", on_delete="CASCADE")
    guid = CharField(max_length=512, unique=True, index=True)
    title = TextField()
    authors = TextField(null=True)
    link = CharField(max_length=1024, null=True)
    doi = CharField(max_length=255, null=True, index=True)
    journal = CharField(max_length=512, null=True)
    published = DateTimeField(null=True)
    summary = TextField(null=True)

    # AI 初筛结果
    ai_score = IntegerField(null=True)       # 0-10
    ai_reason = TextField(null=True)         # AI 筛选理由

    # 标注状态
    #   pending        = 未筛选（默认，也是"感兴趣"的默认态）
    #   interested     = 感兴趣（AI 自动标记 or 用户确认）
    #   not_interested = 不感兴趣
    #   favorite       = 收藏精读
    status = CharField(
        max_length=32,
        default="pending",
        index=True,
    )
    first_seen_at = DateTimeField(default=datetime.datetime.now)
    status_changed_at = DateTimeField(null=True)
    created_at = DateTimeField(default=datetime.datetime.now)

    class Meta:
        table_name = "rss_article"


class ProfileMeta(BaseModel):
    """用户偏好元数据——记录偏好刷新状态"""
    key = CharField(max_length=64, primary_key=True)
    value = TextField()
    updated_at = DateTimeField(default=datetime.datetime.now)

    class Meta:
        table_name = "rss_profile_meta"


# ── 偏好相关的默认 key ──────────────────────────────
_PROFILE_KEY_LAST_REFRESH = "last_profile_refresh_at"
_PROFILE_KEY_ANNOTATIONS_SINCE = "annotations_since_refresh"


def init_db() -> None:
    """初始化数据库并建表（幂等）"""
    DB.init(str(_get_db_path()))
    DB.connect()
    DB.create_tables([RssFeed, RssArticle, ProfileMeta], safe=True)


# ═══════════════════════════════════════════════════════════
# Feed 管理
# ═══════════════════════════════════════════════════════════

def add_feed(name: str, url: str, category: str | None = None) -> RssFeed:
    """添加一个 RSS Feed 源（已存在则更新信息）"""
    feed, created = RssFeed.get_or_create(
        url=url,
        defaults={"name": name, "category": category},
    )
    if not created:
        feed.name = name
        if category is not None:
            feed.category = category
        feed.save()
    return feed


def remove_feed(feed_id: int) -> bool:
    """删除一个 RSS Feed 源及其所有文章"""
    rows = RssFeed.delete().where(RssFeed.id == feed_id).execute()
    return rows > 0


def list_feeds(enabled_only: bool = True) -> list[RssFeed]:
    """列出所有 RSS Feed（默认仅启用的）"""
    query = RssFeed.select()
    if enabled_only:
        query = query.where(RssFeed.enabled == True)
    return list(query)


def load_feeds_from_config() -> list[RssFeed]:
    """
    从配置 'rss_feeds' 加载 feeds 到数据库。
    配置格式示例（pageboygent.toml）：
        [[rss_feeds]]
        name = "arXiv Earth and Planetary Astrophysics"
        url = "https://rss.arxiv.org/rss/astro-ph.EP"
        category = "astro-ph"

        [[rss_feeds]]
        name = "GCA"
        url = "https://www.sciencedirect.com/journal/geochimica-et-cosmochimica-acta/rss"
        category = "geochemistry"
    """
    feeds_config = get_config("rss_feeds")
    if not feeds_config:
        logger.warning("No rss_feeds configured. Add them via pageboygent.toml [rss_feeds].")
        return []

    feeds = []
    for entry in feeds_config:
        name = entry.get("name", entry["url"])
        url = entry["url"]
        category = entry.get("category")
        feed = add_feed(name, url, category)
        feeds.append(feed)
    return feeds


# ═══════════════════════════════════════════════════════════
# RSS 抓取
# ═══════════════════════════════════════════════════════════

def fetch_feed(feed: RssFeed) -> list[RssArticle]:
    """
    抓取单个 RSS Feed，返回本次新入库的文章列表。
    已存在的文章（按 guid 去重）不会重复插入。
    """
    logger.info("Fetching feed: %s (%s)", feed.name, feed.url)
    parsed = feedparser.parse(feed.url)

    if parsed.bozo and not parsed.entries:
        logger.error("Feed parse error for %s: %s", feed.name, parsed.bozo_exception)
        return []

    new_articles: list[RssArticle] = []
    for entry in parsed.entries:
        guid = entry.get("id") or entry.get("link", "")
        if not guid:
            continue

        doi = _extract_doi(entry.get("link", ""))
        if not doi:
            doi = _extract_doi(entry.get("summary", ""))

        article, created = RssArticle.get_or_create(
            guid=guid,
            defaults={
                "feed": feed,
                "title": entry.get("title", "(no title)"),
                "authors": _extract_authors(entry),
                "link": entry.get("link", ""),
                "doi": doi,
                "journal": _extract_journal(parsed, entry),
                "published": _parse_date(entry),
                "summary": entry.get("summary", ""),
            },
        )
        if created:
            new_articles.append(article)

    feed.last_checked_at = datetime.datetime.now()
    feed.save()

    logger.info("  → %d new article(s) from '%s'", len(new_articles), feed.name)
    return new_articles


def fetch_all_feeds() -> list[RssArticle]:
    """抓取所有启用的 RSS Feed，返回所有新入库的文章"""
    all_new: list[RssArticle] = []
    for feed in list_feeds(enabled_only=True):
        try:
            all_new.extend(fetch_feed(feed))
        except Exception:
            logger.exception("Error fetching feed '%s'", feed.name)
    return all_new


# ═══════════════════════════════════════════════════════════
# AI 初筛
# ═══════════════════════════════════════════════════════════

def screen_article(article: RssArticle) -> RssArticle:
    """
    用 AI 对单篇文章初筛。
    调用 DeepSeek 评估论文相关性，输出 0-10 评分 + 理由。
    AI 不可用时默认标记为 interested（宁多勿漏）。
    """
    if not get_config("rss_ai_screening_enabled"):
        article.status = "interested"
        article.status_changed_at = datetime.datetime.now()
        article.save()
        return article

    prompt = _build_screening_prompt(
        title=article.title,
        summary=article.summary or "",
        journal=article.journal or "",
        authors=article.authors or "",
    )

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=get_config("ds_api_key"),
            base_url=get_config("ds_url"),
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个学术文献筛选助手。"
                        "根据用户的研究兴趣判断论文是否值得阅读。"
                        "请严格按照格式输出：评分和理由。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        result = response.choices[0].message.content.strip()
        score, reason = _parse_ai_result(result)

        article.ai_score = score
        article.ai_reason = reason

        threshold = get_config("rss_ai_threshold")
        article.status = "interested" if score >= threshold else "not_interested"

    except Exception:
        logger.exception("AI screening failed for article %s", article.id)
        article.ai_score = -1
        article.ai_reason = "Screening unavailable (API error)"
        article.status = "interested"  # 宁可多看

    article.status_changed_at = datetime.datetime.now()
    article.save()
    return article


def screen_pending_articles(limit: int | None = None) -> list[RssArticle]:
    """筛选所有状态为 pending 的文章"""
    query = RssArticle.select().where(RssArticle.status == "pending")
    if limit is not None:
        query = query.limit(limit)

    articles = list(query)
    for article in articles:
        screen_article(article)
    return articles


# ═══════════════════════════════════════════════════════════
# 用户标注操作
# ═══════════════════════════════════════════════════════════

def mark_interested(article_id: int) -> RssArticle:
    """标记为感兴趣"""
    article = RssArticle.get_by_id(article_id)
    article.status = "interested"
    article.status_changed_at = datetime.datetime.now()
    article.save()
    # 标注后更新计数，触发偏好刷新检查
    _increment_annotation_counter()
    maybe_refresh_profile()
    return article


def mark_not_interested(article_id: int) -> RssArticle:
    """标记为不感兴趣"""
    article = RssArticle.get_by_id(article_id)
    article.status = "not_interested"
    article.status_changed_at = datetime.datetime.now()
    article.save()
    # 标注后更新计数，触发偏好刷新检查
    _increment_annotation_counter()
    maybe_refresh_profile()
    return article


def mark_favorite(article_id: int) -> RssArticle:
    """标记为收藏精读"""
    article = RssArticle.get_by_id(article_id)
    article.status = "favorite"
    article.status_changed_at = datetime.datetime.now()
    article.save()
    # 标注后更新计数，触发偏好刷新检查
    _increment_annotation_counter()
    maybe_refresh_profile()
    return article


# ═══════════════════════════════════════════════════════════
# 查询 & 摘要呈现
# ═══════════════════════════════════════════════════════════

@dataclass
class ScreeningDigest:
    """筛选摘要——用于每天/用户请求时呈现"""
    total_new: int                  # 期间内新抓取文章总数
    ai_screened: int                # AI 已筛选数
    interested: list[RssArticle]    # 感兴趣的文章
    favorites: list[RssArticle]     # 收藏精读的文章（不限于本期）
    from_time: datetime.datetime
    to_time: datetime.datetime


def get_digest(
    since: datetime.datetime | None = None,
) -> ScreeningDigest:
    """
    获取筛选摘要——呈现"感兴趣"和"收藏精读"的文章。
    since: 默认为 24 小时前（由 rss_digest_interval_hours 控制）。
    """
    if since is None:
        since = datetime.datetime.now() - datetime.timedelta(
            hours=get_config("rss_digest_interval_hours"),
        )

    # 本期新文章中标记为 interested 的
    interested = list(
        RssArticle.select()
        .where(
            (RssArticle.first_seen_at >= since)
            & (RssArticle.status == "interested")
        )
        .order_by(RssArticle.ai_score.desc(nulls="last"))
    )

    # 收藏精读（不限时间，但显示最近变更的在前）
    favorites = list(
        RssArticle.select()
        .where(RssArticle.status == "favorite")
        .order_by(RssArticle.status_changed_at.desc())
        .limit(50)
    )

    total_new = (
        RssArticle.select()
        .where(RssArticle.first_seen_at >= since)
        .count()
    )

    ai_screened = (
        RssArticle.select()
        .where(
            (RssArticle.first_seen_at >= since)
            & RssArticle.ai_score.is_null(False)
        )
        .count()
    )

    now = datetime.datetime.now()
    return ScreeningDigest(
        total_new=total_new,
        ai_screened=ai_screened,
        interested=interested,
        favorites=favorites,
        from_time=since,
        to_time=now,
    )


def get_unread_count() -> int:
    """待处理（未标记、未被标记为不感兴趣）的文章数"""
    return (
        RssArticle.select()
        .where(
            (RssArticle.status == "pending")
            | (RssArticle.status == "interested")
        )
        .count()
    )


def list_articles_by_status(
    status: str,
    limit: int = 50,
) -> list[RssArticle]:
    """按状态查询文章列表"""
    return list(
        RssArticle.select()
        .where(RssArticle.status == status)
        .order_by(RssArticle.created_at.desc())
        .limit(limit)
    )


# ═══════════════════════════════════════════════════════════
# 定时脚本入口
# ═══════════════════════════════════════════════════════════

def run_scheduled_check() -> dict[str, Any]:
    """
    定时检查入口（供 cron 调用）：
    1. 抓取所有 feeds → 新文章入库
    2. AI 初筛所有 pending 文章
    3. 返回统计信息（不直接呈现给用户）

    设计为每 1-2 小时执行一次。
    """
    from .lock import lock_it

    init_db()

    with lock_it("rss_fetch"):
        load_feeds_from_config()
        new_articles = fetch_all_feeds()

    with lock_it("rss_screen"):
        screened = screen_pending_articles()

    return {
        "new_articles": len(new_articles),
        "screened_articles": len(screened),
        "unread_count": get_unread_count(),
        "checked_at": datetime.datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════
# heart_beat —— 取代 cron 的定时驱动
# ═══════════════════════════════════════════════════════════

_heart_beat_last_check: datetime.datetime | None = None


def heart_beat() -> dict[str, Any]:
    """
    心跳函数 —— 被外部系统频繁调用（如每 1-5 分钟）。
    内部判断：
      - 距离上次抓取是否超过 rss_check_interval_minutes
      - 若是，则执行一次抓取 + 初筛
    返回结果说明是否真正执行了检查。
    """
    init_db()

    global _heart_beat_last_check
    now = datetime.datetime.now()

    # 优先用数据库记录的时间，兜底用内存变量
    last_check: datetime.datetime | None = (
        RssFeed.select(fn.MAX(RssFeed.last_checked_at))
        .where(RssFeed.enabled == True)
        .scalar()
    )
    if last_check is None:
        last_check = _heart_beat_last_check

    interval = get_config("rss_check_interval_minutes")

    if last_check is not None:
        elapsed_min = (now - last_check).total_seconds() / 60
        if elapsed_min < interval:
            return {
                "action": "skipped",
                "reason": f"Next check in {interval - elapsed_min:.0f} min",
            }

    # 执行检查
    _heart_beat_last_check = now
    result = run_scheduled_check()
    result["action"] = "checked"

    # 检查后也顺带看看是否需要刷新偏好
    try:
        maybe_refresh_profile()
    except Exception:
        logger.exception("Profile refresh in heart_beat failed")

    return result


# ═══════════════════════════════════════════════════════════
# 用户偏好自动维护
# ═══════════════════════════════════════════════════════════

def _get_profile_path() -> Path:
    """用户偏好文件路径（工作站目录下）"""
    workstation = Path(get_config("workstation_dir")).expanduser()
    return workstation / "user_profile.md"


def load_user_profile() -> str:
    """
    读取用户偏好文件。
    如果文件不存在，返回 rss_interest_profile 配置中的初始值。
    """
    path = _get_profile_path()
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    # 兜底：从配置读取初始兴趣描述
    return get_config("rss_interest_profile") or ""


def _set_profile_meta(key: str, value: str) -> None:
    ProfileMeta.replace(key=key, value=value, updated_at=datetime.datetime.now()).execute()


def _get_profile_meta(key: str, default: str = "") -> str:
    try:
        return ProfileMeta.get_by_id(key).value
    except ProfileMeta.DoesNotExist:
        return default


def _count_annotations_since_refresh() -> int:
    """统计自上次偏好刷新以来的标注次数"""
    raw = _get_profile_meta(_PROFILE_KEY_ANNOTATIONS_SINCE, "0")
    return int(raw)


def _increment_annotation_counter() -> int:
    """标注计数 +1，返回新值"""
    count = _count_annotations_since_refresh() + 1
    _set_profile_meta(_PROFILE_KEY_ANNOTATIONS_SINCE, str(count))
    return count


def _reset_annotation_counter() -> None:
    _set_profile_meta(_PROFILE_KEY_ANNOTATIONS_SINCE, "0")


def _needs_profile_refresh() -> bool:
    """
    判断是否需要刷新用户偏好：
    条件 A：距离上次刷新超过 3 天
    条件 B：自上次刷新以来标注了 >= 20 篇
    """
    # 条件 A：时间间隔
    last_refresh_str = _get_profile_meta(_PROFILE_KEY_LAST_REFRESH, "")
    if last_refresh_str:
        try:
            last_refresh = datetime.datetime.fromisoformat(last_refresh_str)
            days_elapsed = (datetime.datetime.now() - last_refresh).total_seconds() / 86400
            if days_elapsed >= 3:
                return True
        except (ValueError, TypeError):
            pass
    else:
        # 从未刷新过
        return True

    # 条件 B：标注数量
    annotation_count = _count_annotations_since_refresh()
    if annotation_count >= 20:
        return True

    return False


def _refresh_user_profile() -> str:
    """
    用 LLM 提炼用户偏好：
    1. 读取最近标注的文献（感兴趣 + 不感兴趣）
    2. 发给 DeepSeek 分析模式
    3. 输出更新后的偏好描述
    4. 写入 user_profile.md
    5. 重置计数器并更新刷新时间
    """
    # 收集最近标注的文献
    recent_interested = list(
        RssArticle.select()
        .where(RssArticle.status == "interested")
        .order_by(RssArticle.status_changed_at.desc())
        .limit(15)
    )
    recent_not_interested = list(
        RssArticle.select()
        .where(RssArticle.status == "not_interested")
        .order_by(RssArticle.status_changed_at.desc())
        .limit(15)
    )

    current_profile = load_user_profile()

    # 构建 prompt
    interested_lines = "\n".join(
        f"- [{a.ai_score}/10] {a.title}"
        + (f" ({a.journal})" if a.journal else "")
        + (f"\n  摘要: {a.summary[:200]}" if a.summary else "")
        for a in recent_interested
    )
    not_interested_lines = "\n".join(
        f"- {a.title}" + (f" ({a.journal})" if a.journal else "")
        for a in recent_not_interested
    )

    prompt = f"""你是一个学术研究者的偏好分析助手。请分析以下数据，提炼用户的研究兴趣偏好。

当前偏好描述：
{current_profile or "(无)"}

用户标记为**感兴趣**的近期论文（含 AI 评分）：
{interested_lines or "(暂无)"}

用户标记为**不感兴趣**的近期论文：
{not_interested_lines or "(暂无)"}

请分析：
1. 用户真正关心哪些研究方向、方法、关键词？
2. 用户对哪些类型的论文明显不感兴趣？
3. 当前的偏好描述是否需要更新？

请输出更新后的偏好描述（一段流畅的中文或英文，描述用户的研究兴趣范围，
用于 AI 筛选新论文时的判断依据。200-500 字。不要包含格式标记，只输出纯文本描述。）
"""

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=get_config("ds_api_key"),
            base_url=get_config("ds_url"),
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个学术偏好分析助手。分析用户的标注模式，提炼其研究兴趣偏好。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=800,
        )
        new_profile = response.choices[0].message.content.strip()
    except Exception:
        logger.exception("LLM profile refresh failed")
        return current_profile  # 刷新失败则保持原样

    # 写入偏好文件
    path = _get_profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_profile, encoding="utf-8")

    # 更新元数据
    _set_profile_meta(_PROFILE_KEY_LAST_REFRESH, datetime.datetime.now().isoformat())
    _reset_annotation_counter()

    logger.info("User profile refreshed via LLM")
    return new_profile


def maybe_refresh_profile(force: bool = False) -> bool:
    """
    判断并执行用户偏好刷新。
    返回 True 表示执行了刷新。
    force=True 则无条件刷新。
    """
    if force or _needs_profile_refresh():
        _refresh_user_profile()
        return True
    return False


# ═══════════════════════════════════════════════════════════
# 内部工具函数
# ═══════════════════════════════════════════════════════════

def _extract_doi(text: str | None) -> str | None:
    """从文本中提取 DOI"""
    if not text:
        return None
    m = re.search(r"(10\.\d{4,}/[^\s,;<>\"'()\[\]]+)", text)
    return m.group(1).rstrip(".") if m else None


def _extract_authors(entry: dict) -> str | None:
    """从 RSS 条目中提取作者"""
    authors = entry.get("authors", [])
    if authors:
        names = [a.get("name", "") for a in authors if a.get("name")]
        if names:
            return "; ".join(names)
    tags = entry.get("tags", [])
    author_tags = [
        t.get("term", "")
        for t in tags
        if t.get("term") and not t.get("scheme")
    ]
    if author_tags:
        return "; ".join(author_tags)
    return None


def _extract_journal(feed_parsed: dict, entry: dict) -> str | None:
    """提取期刊/来源名称"""
    src = entry.get("journal") or entry.get("source")
    if src:
        return src.get("title") if isinstance(src, dict) else str(src)
    return feed_parsed.get("feed", {}).get("title")


def _parse_date(entry: dict) -> datetime.datetime | None:
    """解析发布日期"""
    for key in ("published_parsed", "updated_parsed"):
        ts = entry.get(key)
        if ts:
            try:
                return datetime.datetime(*ts[:6])
            except (TypeError, ValueError):
                pass
    for key in ("published", "updated"):
        ds = entry.get(key)
        if ds:
            try:
                from dateutil.parser import parse
                return parse(ds)
            except Exception:
                pass
    return None


def _build_screening_prompt(
    title: str,
    summary: str,
    journal: str,
    authors: str,
) -> str:
    """构建 AI 筛选的 prompt"""
    profile = get_config("rss_interest_profile")
    if not profile:
        profile = "地球化学、天体物理、地球物理、行星科学"

    return f"""判断以下学术论文是否与你的研究兴趣相关。

研究兴趣领域：{profile}

论文标题：{title}
摘要：{summary or "(无摘要)"}
期刊/来源：{journal or "(未知)"}
作者：{authors or "(未知)"}

请按以下格式输出：
评分：[0-10 之间的整数，10 = 非常相关]
理由：[一句话说明判断理由]
"""


def _parse_ai_result(text: str) -> tuple[int, str]:
    """解析 AI 输出，返回 (score, reason)"""
    score = 5
    reason = ""

    m = re.search(r"评分[：:]\s*(\d+)", text)
    if m:
        score = max(0, min(10, int(m.group(1))))

    m = re.search(r"理由[：:]\s*(.*?)$", text, re.DOTALL)
    if m:
        reason = m.group(1).strip()
    else:
        reason = text.strip()

    return score, reason


# ═══════════════════════════════════════════════════════════
# CLI 后端入口（后续对接 Web 前端）
# ═══════════════════════════════════════════════════════════

def cli_check() -> int:
    """pageboygent rss-check: 立即执行一次抓取+筛选"""
    result = run_scheduled_check()
    print(f"📡 RSS 检查完成")
    print(f"   新文章: {result['new_articles']}")
    print(f"   已筛选: {result['screened_articles']}")
    print(f"   待处理: {result['unread_count']}")
    return 0


def cli_digest() -> int:
    """pageboygent rss-digest: 显示筛选摘要"""
    init_db()
    digest = get_digest()
    print(
        f"📡 RSS 筛选摘要"
        f" ({digest.from_time.strftime('%m-%d %H:%M')}"
        f" ~ {digest.to_time.strftime('%m-%d %H:%M')})"
    )
    print(f"   总抓取: {digest.total_new} 篇")
    print(f"   AI 已筛选: {digest.ai_screened} 篇")
    print(f"   感兴趣: {len(digest.interested)} 篇")
    print(f"   收藏精读: {len(digest.favorites)} 篇")

    if digest.interested:
        print(f"\n{'─' * 60}")
        print("📖 感兴趣的文章")
        print(f"{'─' * 60}")
        for i, art in enumerate(digest.interested, 1):
            score = f"[{art.ai_score}/10]" if art.ai_score is not None else "[?]"
            print(f"\n{i}. {score} {art.title}")
            if art.authors:
                print(f"   作者: {art.authors}")
            if art.journal:
                print(f"   期刊: {art.journal}")
            if art.doi:
                print(f"   DOI: {art.doi}")
            if art.ai_reason:
                print(f"   筛选理由: {art.ai_reason}")
            print(f"   ID: {art.id} | 状态: {art.status}")

    if digest.favorites:
        print(f"\n{'─' * 60}")
        print("⭐ 收藏精读")
        print(f"{'─' * 60}")
        for art in digest.favorites:
            print(f"  · {art.title} (ID: {art.id})")

    return 0


def cli_mark(article_id: int, new_status: str) -> int:
    """pageboygent rss-mark <id> <status>: 标注文章"""
    init_db()
    fn = {
        "interested": mark_interested,
        "not_interested": mark_not_interested,
        "favorite": mark_favorite,
    }
    handler = fn.get(new_status)
    if not handler:
        print(f"❌ 无效状态: {new_status} (可选: interested, not_interested, favorite)")
        return 1
    article = handler(article_id)
    print(f"✅ 文章 #{article.id} 「{article.title[:60]}」已标记为: {article.status}")
    return 0
