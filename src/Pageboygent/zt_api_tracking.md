# Zotero API 追踪表

维护 [zt_tools.py](zt_tools.py) 用的检查清单。Zotero 客户端和 pyzotero 库都更新得越来越
频繁，这份表格的目的是让"升级之后到底要不要改代码"这件事变成一个可以照着勾的清单，而不用
每次都重新读一遍 pyzotero 源码。

## 版本矩阵 (2026-09-11 记录，已用 [Zotero官方文档](https://www.zotero.org/support/dev/web_api/v3/local_api) 和 [pyzotero官方文档](https://pyzotero.readthedocs.io/en/latest/) 核实)

| 项目 | 版本/要求 | 备注 |
|---|---|---|
| Zotero 桌面客户端 (读) | Zotero 7+ | 必须在 `Zotero > 设置 > 高级 > "允许其他应用与Zotero通信" (Allow other applications on this computer to communicate with Zotero)` 打开 |
| Zotero 桌面客户端 (**写**) | **Zotero 10+** | 官方文档原话: "Write requests (Zotero 10+) require a local API key"。只打开上面那个开关不够，写入这条能力是更新的版本才有的，7~9 之间的Zotero只能读、不能写本地API |
| pyzotero | `>=1.15.0`（当前环境实测 1.15.1，已写进 [setup.py](../../setup.py)） | 本地API写入支持是这个大版本区间新加的能力；更早的版本 `local=True` 只能读 |
| 本地API地址 | `http://localhost:23119/api` | 硬编码在 pyzotero 里，不需要我们配置 |

**每次升级 Zotero 或 pyzotero 之后，先确认上面这一行版本矩阵是不是还成立，再往下走清单。** 如果写入
全面失败，第一件事是确认 Zotero 桌面端版本号，而不是先怀疑代码。

## 这次重构做了什么 (2026-09-11)

- 之前：`zot`（远程Web API，负责写）+ `zot_local`（本地API `local=True`，只负责读）两套客户端并存。
  写笔记、传附件全部走远程Web API，传完MD附件后还要手动把文件拷贝进本地 `storage/<key>/` 目录，
  否则本地Zotero"看不到"新文件——这就是之前那套别扭的"远程写入 -> 拿到新ID -> 再手动放回本地文件夹"的补丁。
- 现在：pyzotero 的本地API已经原生支持写入（`create_items` / `update_item` / `add_tags` /
  `addto_collection` / `deletefrom_collection` / `attachment_simple` 等所有写方法，在
  `local=True` 的客户端上会自动带上 `Zotero-Server-ID` 头，走 `http://localhost:23119/api`）。
  `zt_tools.py` 里现在只有一个 `zot` 客户端；`zot_local` 只是指向同一个对象的别名，留着是为了不用
  同时改 `note_storage.py` / `migrate_paper_notes_html.py` 等外部模块里的旧引用。
- 写入需要额外的一次性授权：`zot.authorize_local(app_name)` 会让 Zotero 弹一个 "Allow / Always
  Allow / Deny" 对话框；选 "Always Allow" 拿到的 key 长期有效。`zt_tools.py` 会把这个 key 缓存到
  `<workstation_dir>/zotero_local_api_key.txt`，以后启动都会自动复用，不会重复弹窗。也可以用配置项
  `zotero_local_api_key`（环境变量 `PAGEBOYGENT_ZOTERO_LOCAL_API_KEY`）预先注入一个已经拿到的 key，
  用于无人值守环境。
- `upload_md_to_zotero` 里手动拷贝文件进 storage 目录的那段代码已经删掉：本地API写入后，Zotero自己
  的实例立刻把文件落盘并更新数据库，读回来是即时一致的，不再需要那个补丁。

### 2026-09-11 补充修复：`item_template()` / `attachment_simple()` 在本地模式下根本不能用

第一次实现完之后，让用户实测 `--write` 冒烟测试，第一步 `create_collections` 就报了
`401 Invalid or expired API key`——排查后发现那是测试脚本自己的bug（直接裸调
`zot.create_collections`，没走 `_perform_local_write`，所以从来没触发过 `authorize_local`）,
已经在 [zt_smoke_test.py](zt_smoke_test.py) 里修掉。但借着这次"重新读文档"，用户又提了一个
更关键的问题：**本地写入和以前用的"简便"方法是不是两码事？** 查了 pyzotero 1.15.1 的源码
(`_client.py` 里 `item_template()` 的实现)、官方 [local API 文档](https://www.zotero.org/support/dev/web_api/v3/local_api)、
以及 pyzotero 自己的测试 (`tests/test_local_write.py::ItemTemplateTests`) 和 MCP server 参考实现
(`pyzotero/mcp_server.py` 里的 `add_attachment`)，确认了:

> **本地API根本没有实现 `/items/new` 这个端点。** `zot.item_template(...)` 在 `local=True`
> 时会直接抛 `CallDoesNotExistError`，报错信息原文是: *"The local Zotero API doesn't implement
> the /items/new template endpoint, so item_template() (and the attachment_simple() /
> attachment_both() methods that rely on it) are unavailable in local mode. Build the item
> dict directly and pass it to create_items(), or to upload_attachments() for a file
> attachment."*

也就是说，**"本地文件更新"（本地读storage目录/本地API写入item）和"（以前）远程上传文件"用的
根本不是同一套便捷函数** —— `attachment_simple()`/`attachment_both()`/`item_template()`
这几个"拿模板再填内容"的便捷方法，本质上是靠一个只有Web API才有的 `/items/new` 端点实现的，
迁移到本地API之后必须换成"自己手写最小字典 + `create_items()`/`upload_attachments()`"这条更
底层、但本地/远程通用的路径。第一版重构里，下面三处还在用旧的便捷方法，全部改掉了:

| 位置 | 之前 (在本地模式下会抛 `CallDoesNotExistError`) | 现在 |
|---|---|---|
| `_create_note_in_zotero` | `zot.item_template("note")` | 手写 `{"itemType": "note", "note": ..., "tags": [], "relations": {}}`，直接传给 `zot.create_items(...)` |
| `upload_md_to_zotero` | `zot.attachment_simple([str(md_path)], item_key)` | 手写 `{"itemType": "attachment", "linkMode": "imported_file", "title": ..., "filename": ..., "note": "", "tags": [], "relations": {}}`，传给 `zot.upload_attachments([...], item_key)`（顺便把标题直接写成"Full Text MD"，省掉了原来"先传再改名"的第二次写入往返） |
| `insert_reference_to_collection` | `zot.item_template(type)` | 手写 `{"itemType": type, "tags": [], "creators": [], "relations": {}}`，再按需塞 `publicationTitle`/`creators`/`title`/`DOI` |

上面这几个手写字典的形状，是照抄 pyzotero 自带的 MCP server (`pyzotero/mcp_server.py` 里的
`add_attachment`) 手写附件模板的方式来的，不是我们瞎猜的。`create_items()`/`upload_attachments()`
本身不会像 `item_template()` 那样帮你校验字段名对不对——不认识的字段会体现在返回值的 `failed`
里，我们已有的"检查 `success`/`unchanged` 是否为空"的失败处理逻辑本来就能兜住这种情况，不用
额外加校验。

## 函数 -> 底层API 对照表

| zt_tools.py 里的函数 | 调用的 pyzotero 方法 | 读/写 | 升级后要重点检查什么 |
|---|---|---|---|
| `get_pdf_from_item` | `zot.children`, `zot.item` | 读 | 附件的 `title`/`contentType` 字段名有没有变 |
| `dump_file_from_local` | 直接读文件系统 `zotero_storage_path/<key>/` | - | Zotero storage 目录结构（按 key 分子目录）有没有变 |
| `dump_file_from_zotero` | `zot.file` | 读 | `/items/<key>/file` 端点是否仍可用 |
| `check_md_full_text_exits` / `upload_md_to_zotero` | `zot.children`, `zot.item`, `zot.upload_attachments`（手写attachment字典,**不是** `attachment_simple`,见上面"补充修复"） | 读+**写** | 附件上传三步握手（注册意图 -> 拿上传授权 -> 上传字节）是否仍然一次性打通；手写的 `imported_file` 附件字典字段是否还被接受 |
| `check_literature_exists` / `search_literature_by_doi` | `zot.item`, `zot.items`, `zot.everything` | 读 | 分页机制（100条一页）是否还是 100 |
| `get_info` | `zot.item` | 读 | - |
| `get_collections` / `fetch_keys_by_collection` / `fetch_ref_keys_by_collection` | `zot.collections`, `zot.collections_sub`, `zot.collection_items` | 读 | 同上分页逻辑 |
| `_find_collection_key` / `add_item_to_collection` / `rm_item_from_collection` | `zot.collections`, `zot.addto_collection`, `zot.deletefrom_collection` | 读+**写** | `addto_collection`/`deletefrom_collection` 的入参形状（collection key + 完整item dict） |
| `retreive_note_keys_from_item` / `retreive_note_from_item` / `get_notes_by_tags` | `zot.children`, `zot.item`, `zot.items` | 读 | note 的 HTML/Markdown 判断逻辑（`<p>`/`<div>` 探测）是否还够用 |
| `_create_note_in_zotero`（供 `add_note_to_item` 和 `NoteCache.check_and_update_cache` 共用） | `zot.create_items`（手写note字典,**不是** `zot.item_template("note")`） `zot.item`, `zot.add_tags` | **写** | `create_items` 返回体里 `success`/`successful`/`unchanged` 的键名和结构；手写note字典里的 `itemType`/`note`/`tags`/`relations` 这几个key是否还被接受 |
| `bib_export` | `zot.item(..., format="bib")` | 读 | CSL style 名字 `earth-and-planetary-science-letters` 是否还被 Zotero 识别 |
| `_label_notes_in_collections`（调试用，未被其他模块引用） | `zot.collection_items`, `zot.item_tags`, `zot.item`, `zot.add_tags` | 读+**写** | 仅调试用，非关键路径 |
| `re_label_note_tags` | `zot.item`, `zot.update_item` | **写** | - |
| `insert_reference_to_collection` | `_find_collection_key`, `search_literature_by_doi`, `zot.create_items`（手写条目字典,**不是** `zot.item_template(type)`）, `zot.addto_collection` | 读+**写** | Zotero服务端对未知itemType/字段名的处理方式（体现在返回值的 `failed` 里）有没有变 |

## 本地写入的授权/恢复机制（`_perform_local_write` / `_ensure_local_write_authorized`）

- 首次写入前自动调用 `authorize_local`，弹窗由用户在 Zotero 里点 "Always Allow"，key 缓存到
  `<workstation_dir>/zotero_local_api_key.txt`。
- `LocalAPIKeyRequiredError`（比如用户在 Zotero 里手动收回了授权，或者只点了一次性的 "Allow"）：
  清掉缓存文件、重新走一次授权流程，再重试一次写入。
- `ServerIDMismatchError`（连接到了不同的 Zotero 实例/数据库，例如换了 Zotero profile 或重装了
  Zotero）：重置 `zot.server_id`，再重试一次。
- `LocalAPIDeniedError`（用户在弹窗里点了"拒绝"）或本地API整体连不上：统一走 `_pause_action`
  暂停并记录到 `zotero_guard.log`，不会让异常直接炸穿整个调用链。

**如果升级Zotero后所有写入都开始报 401/`LocalAPIKeyRequiredError`**：先手动删掉
`<workstation_dir>/zotero_local_api_key.txt`，重新触发一次授权弹窗；如果还不行，再看是不是
Zotero版本变了导致 `authorize_local` 的行为变化（见下方坑清单）。

## 已知的坑 / 升级时容易踩的点

1. **`pyzotero.zotero_errors` 不是 `pyzotero.errors` 的完整镜像。** 本地写入相关的新异常类
   （`LocalAPIDeniedError`、`LocalAPIKeyRequiredError`、`ServerIDMismatchError`、
   `ServerIDRequiredError`）只在 `pyzotero.errors` 里，旧的兼容模块 `pyzotero.zotero_errors`
   没有重新导出它们。`zt_tools.py` 已经改成 `from pyzotero import errors as ze`，**不要**升级时
   手滑改回 `zotero_errors`，否则 `_is_local_api_denied` 等判断函数会永远返回 `False`。
2. **`authorize_local` 有频率限制**，pyzotero 官方文档明确说"不要把它放在重试循环里"。
   `_perform_local_write` 里对授权失败只重试一次，升级时如果要加更多重试逻辑，记得避开这一点。
3. **`zot.item_template(...)`（以及依赖它的 `attachment_simple()`/`attachment_both()`）在本地
   模式下是彻底不可用的**，不是"可能有坑"，是**一定会抛 `CallDoesNotExistError`**：本地API根本
   没有实现 `/items/new` 这个端点。已确认过 `item_types()`/`item_type_fields()` 这两个"schema"
   接口本地是能用的（pyzotero自己的 `validate_items()` 辅助函数就是这么用的），只有 `/items/new`
   这一个端点没有本地实现。`zt_tools.py` 里已经把所有用到 `item_template`/`attachment_simple`
   的地方都换成了"手写字典 + `create_items()`/`upload_attachments()`"（细节见上面"2026-09-11
   补充修复"一节）。**升级后如果又哪里手滑加回了 `item_template()`/`attachment_simple()`/
   `attachment_both()` 调用，本地模式下会立刻报错，报错信息里会直接提示改用 `create_items()`/
   `upload_attachments()`。**
4. **`server_id` 是跟着某一个具体的 Zotero 实例/数据库走的**，如果换了机器、重装了Zotero、或者
   恢复了数据库备份，旧的 `server_id`/版本号就不能再复用。目前 `zt_tools.py` 没有跨进程持久化
   `server_id`（每次启动都是重新握手拿一次），所以正常使用不受影响；但如果以后要做"增量同步/按
   version缓存"之类的功能，要记得把 `server_id` 一起存、一起分区，不能跨实例混用。
5. **一次性授权 vs 永久授权**：如果用户在弹窗里选的是 "Allow"（不是 "Always Allow"），拿到的 key
   只能用一次，用完当次写入就失效。这种情况下 `authorized.get("remember")` 为 `False`，我们不会
   缓存这个 key，下次写入会再弹一次窗——这是预期行为，不是bug。

## 手动烟雾测试清单（升级 Zotero 桌面端或 pyzotero 之后跑一遍）

跑之前先确认：Zotero 桌面端开着、`设置 > 高级 > 允许其他应用与Zotero通信` 打开。

1. `python -c "from Pageboygent.zt_tools import zot; print(zot.top(limit=1))"` —— 本地API的读
   通不通。
2. 删掉 `<workstation_dir>/zotero_local_api_key.txt`（如果存在），触发一次全新授权：调用任意写函数
   （比如对着一个测试用的 item 调 `add_note_to_item`），确认 Zotero 弹出授权对话框，点 "Always
   Allow" 后写入成功，且缓存文件被重新写入。
3. 再跑一次同一个写函数，确认这次**不会**再弹窗（说明缓存的 key 生效了）。
4. `upload_md_to_zotero`：找一个测试用的 item，上传一个小的 `.md` 文件，确认：
   - Zotero 里能立刻看到新增的 "Full Text MD" 附件（不需要手动刷新/同步）；
   - 附件对应的 storage 目录里能找到文件本体；
   - 再调用一次 `upload_md_to_zotero`，确认走的是"已存在，跳过上传"分支，不会重复创建。
5. `insert_reference_to_collection`：新建一条测试文献，确认手写的 `{"itemType": type, ...}` 字典
   还能被 `create_items` 正常接受（没有落进返回值的 `failed` 里），`journal`/`author`/`title`/`DOI`
   这几个字段正常赋值，条目创建后确实出现在目标collection里。
6. `add_item_to_collection` / `rm_item_from_collection`：确认基于 collection 名字的查找和增删仍然
   正常。
7. 把 Zotero 桌面端整个退出，再跑一次任意读/写函数，确认走的是 `_pause_action` 的
   "本地Zotero API不可达"分支（打印 `[PAUSED] ...` 并写入 `zotero_guard.log`），而不是一个
   看不懂的裸异常。
