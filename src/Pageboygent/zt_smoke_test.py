"""
zt_tools.py 的人工烟雾测试脚本 (smoke test)。

用途: 重构/升级 Zotero 或 pyzotero 之后，跑一遍这个脚本，确认本地API的读、写、
     授权、附件上传/下载这些关键路径还能正常工作，不用每次都手工在Zotero里点来点去。

安全性说明 (重要):
  - 默认(不加任何参数)只做只读检查: 连接测试、读collections、读一个item。
    绝对不会创建/修改/删除你Zotero库里的任何东西。
  - 只有显式加上 --write 参数，才会做写入相关的测试。这部分会:
      1. 新建一个**明确标记为测试用**的临时collection和一个临时item
         (标题/collection名里都带 "Pageboygent smoke test" 字样和随机后缀，
         不会跟你已有的任何条目/collection重名或混淆)。
      2. 在这个临时item下面创建笔记、上传一个几个字节的临时.md附件、改一下
         collection归属、改一下笔记标签 —— 全部只发生在这个临时item/collection上，
         不会touch你已有的任何文献条目、笔记或collection。
      3. 测试跑完后，不管成功还是失败，都会在 finally 里把这次创建的临时item和
         临时collection**清理掉**(硬删除，不会进回收站，这是Zotero本地API删除的
         固有行为，见 zt_api_tracking.md)。清理结果会打印出来；如果有清理失败的情况，
         会明确打印出剩下的key/名字，方便你自己去Zotero里手动删掉。
  - 本脚本完全不会导入/调用 storage.py 或 note_storage.py，所以你用来追踪文献阅读
    进度的那个sqlite数据库 (pageboygent.db 里的 Paper/PDFStatus等表) 从头到尾都不会
    被打开或修改。
  - 本脚本会像正常使用一样复用/写入 workstation_dir 下的 zotero_local_api_key.txt
    (本地写入授权缓存) 和 note_cache.db (笔记提交失败时的重试缓存)；对 note_cache.db
    的影响仅限于"这次测试用的临时item_key对应的那一行"，脚本会在结束时主动把它删掉，
    不会跟你真正在排队等待提交的笔记搞混。

用法:
    # 只读检查(安全, 推荐先跑这个)
    python -m Pageboygent.zt_smoke_test

    # 完整测试, 包含写入(会先打印将要做的事情, 要求你确认)
    python -m Pageboygent.zt_smoke_test --write

    # 完整测试 + 跳过确认(比如接入自动化脚本时用)
    python -m Pageboygent.zt_smoke_test --write --yes

    # 完整测试, 但保留临时item/collection不清理, 方便你自己去Zotero里肉眼检查
    python -m Pageboygent.zt_smoke_test --write --yes --keep

跑之前请确认: Zotero桌面客户端正在运行, 且 设置 > 高级 > "允许其他应用与Zotero通信"
已经打开。
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
import uuid
from pathlib import Path


def _hr(title: str) -> None:
    print("\n" + "=" * 8 + f" {title} " + "=" * 8)


def _confirm(prompt: str) -> bool:
    reply = input(f"{prompt} 输入 yes 继续,其他任意输入都会取消: ").strip().lower()
    return reply == "yes"


def run_read_checks(zt) -> bool:
    """只读检查: 确认本地API连得上、能读collections/item。不会修改任何数据。"""
    _hr("只读检查 (read-only checks)")
    ok = True

    print(f"[config] zotero_id={zt.zt_id!r}  zotero_storage_path={zt.zt_storage_path!r}")
    print(f"[config] workstation_dir={zt.get_config('workstation_dir')!r}")
    print(f"[client] zot.local={zt.zot.local}  zot.endpoint={zt.zot.endpoint}")
    print(f"[client] local_api_key cached: {bool(zt.zot.local_api_key)}")

    try:
        top_items = zt.zot.top(limit=1)
        print(f"[OK] zot.top(limit=1) 成功,本地Zotero API可以读取, 返回 {len(top_items)} 条。")
    except Exception as e:
        print(f"[FAIL] 读取本地Zotero API失败: {e}")
        print(
            "       请检查: Zotero桌面客户端是否在运行? "
            "设置 > 高级 > \"允许其他应用与Zotero通信\" 是否已打开?"
        )
        return False

    try:
        collections = zt.zot.collections(limit=5)
        print(f"[OK] zot.collections(limit=5) 成功, 示例: "
              f"{[c['data']['name'] for c in collections]}")
    except Exception as e:
        print(f"[FAIL] 读取collections失败: {e}")
        ok = False

    if top_items:
        sample_key = top_items[0]["data"]["key"]
        try:
            info = zt.get_info(sample_key, ["title", "itemType"])
            print(f"[OK] get_info('{sample_key}', ...) 成功: {info}")
        except Exception as e:
            print(f"[FAIL] get_info('{sample_key}', ...) 失败: {e}")
            ok = False

    return ok


def run_write_checks(zt, auto_yes: bool, keep: bool) -> bool:
    """写入相关检查。全程只操作脚本自己创建的临时item/collection,并在结束时清理。"""
    _hr("写入检查 (write checks) —— 会创建并清理一个临时item和临时collection")

    run_tag = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    marker = f"[Pageboygent smoke test - safe to delete] {run_tag}"
    collection_name_1 = f"Pageboygent Smoke Test {run_tag}"
    collection_name_2 = f"Pageboygent Smoke Test {run_tag} (moved)"

    print("即将创建下面这些临时对象用于测试,测试结束后会自动删除:")
    print(f"  - 临时collection: {collection_name_1!r}")
    print(f"  - 临时collection: {collection_name_2!r}")
    print(f"  - 临时item (标题里带这个标记): {marker!r}")
    print("不会碰你已有的任何item/collection/笔记。")
    if not keep:
        print("测试结束后会硬删除以上临时对象(本地API的删除是硬删除,不进回收站)。")
    else:
        print("你传了 --keep,测试结束后不会清理,方便你自己去Zotero里看一眼再手动删。")

    if not auto_yes and not _confirm("确认要开始写入测试吗?"):
        print("已取消,没有做任何修改。")
        return False

    created = {
        "collection_key_1": None,
        "collection_key_2": None,
        "item_key": None,
        "note_key": None,
    }
    ok = True

    try:
        # 1. 建临时collection,并通过 insert_reference_to_collection 建item(会顺带
        #    练到 _find_collection_key / 手工拼item字典(不走item_template,本地
        #    API没有/items/new) / zot.create_items / zot.addto_collection /
        #    本地写入授权 这一整条链路)。
        col_result = zt._perform_local_write(
            f"create_collection('{collection_name_1}')",
            zt.zot.create_collections, [{"name": collection_name_1}],
        )
        col_success = col_result.get("success", {})
        assert col_success, f"创建临时collection失败: {col_result}"
        created["collection_key_1"] = col_success["0"]
        print(f"[OK] 创建临时collection '{collection_name_1}' -> {created['collection_key_1']}")

        # itemType用journalArticle而不是document: fetch_ref_keys_by_collection /
        # insert_reference_to_collection 内部按 "book || journalArticle || thesis
        # || preprint" 过滤,document不在这个列表里,会导致新item"查不到"的假报错。
        item_key = zt.insert_reference_to_collection(
            collection_name_1, type="journalArticle", title=marker
        )
        assert item_key, "insert_reference_to_collection 没有返回item_key"
        created["item_key"] = item_key
        print(f"[OK] insert_reference_to_collection 创建了临时item -> {item_key}")

        members = zt.fetch_ref_keys_by_collection(created["collection_key_1"])
        assert item_key in members, "新item没有出现在临时collection里"
        print("[OK] 确认临时item已经在临时collection里")

        # 2. 笔记的增/读/改标签
        note_key = zt.add_note_to_item(
            item_key, "这是Pageboygent的自动化烟雾测试笔记,可以放心删除。",
            tags=["pageboygent-smoke-test"],
        )
        assert note_key, "add_note_to_item 没有返回note_key"
        created["note_key"] = note_key
        print(f"[OK] add_note_to_item 创建了笔记 -> {note_key}")

        note_text = zt.retreive_note_from_item(item_key, ["pageboygent-smoke-test"])
        assert note_text and "烟雾测试" in note_text, "读回来的笔记内容对不上"
        print("[OK] retreive_note_from_item 读回的内容一致")

        relabel_ok = zt.re_label_note_tags(note_key, ["pageboygent-smoke-test", "renamed"])
        assert relabel_ok, "re_label_note_tags 返回了失败"
        print("[OK] re_label_note_tags 修改标签成功")

        # 3. 上传一个几字节的临时.md附件,再读回来做一次内容校验
        md_content = f"# Pageboygent smoke test\n\nrun_tag={run_tag}\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            upload_src = tmp_dir_path / "smoke_test.md"
            upload_src.write_text(md_content, encoding="utf-8")

            att_key = zt.upload_md_to_zotero(item_key, upload_src)
            assert att_key, "upload_md_to_zotero 没有返回附件key"
            print(f"[OK] upload_md_to_zotero 上传了附件 -> {att_key}")

            found_key = zt.check_md_full_text_exits(item_key)
            assert found_key == att_key, "check_md_full_text_exits 没有找到刚上传的附件"
            print("[OK] check_md_full_text_exits 能找到刚上传的附件")

            download_via_disk = tmp_dir_path / "roundtrip_disk.md"
            zt.dump_file_from_local(att_key, download_via_disk)
            assert download_via_disk.read_text(encoding="utf-8") == md_content, (
                "dump_file_from_local 读回的内容和原始内容不一致"
            )
            print("[OK] dump_file_from_local 内容往返一致")

            download_via_api = tmp_dir_path / "roundtrip_api.md"
            zt.dump_file_from_zotero(att_key, download_via_api)
            assert download_via_api.read_text(encoding="utf-8") == md_content, (
                "dump_file_from_zotero 读回的内容和原始内容不一致"
            )
            print("[OK] dump_file_from_zotero (本地API /file 接口) 内容往返一致")

        # 4. collection迁移: 新建第二个临时collection, 把item加过去再挪走
        col2_result = zt._perform_local_write(
            f"create_collection('{collection_name_2}')",
            zt.zot.create_collections, [{"name": collection_name_2}],
        )
        col2_success = col2_result.get("success", {})
        assert col2_success, f"创建第二个临时collection失败: {col2_result}"
        created["collection_key_2"] = col2_success["0"]
        print(f"[OK] 创建第二个临时collection '{collection_name_2}' -> {created['collection_key_2']}")

        assert zt.add_item_to_collection(item_key, collection_name_2), "add_item_to_collection 失败"
        members_2 = zt.fetch_ref_keys_by_collection(created["collection_key_2"])
        assert item_key in members_2, "add_item_to_collection 之后item没有出现在第二个collection里"
        print("[OK] add_item_to_collection 成功")

        assert zt.rm_item_from_collection(item_key, collection_name_1), "rm_item_from_collection 失败"
        members_1_after = zt.fetch_ref_keys_by_collection(created["collection_key_1"])
        assert item_key not in members_1_after, "rm_item_from_collection 之后item还留在原collection里"
        print("[OK] rm_item_from_collection 成功")

        # 5. 顺带确认一下bib_export这类只读导出没有被写入逻辑影响到
        bib = zt.bib_export(item_key)
        assert isinstance(bib, str) and bib.strip(), "bib_export 返回了空内容"
        print("[OK] bib_export 正常返回引用格式文本")

        print("\n所有写入检查都通过了。")

    except Exception as e:
        ok = False
        print(f"\n[FAIL] 写入检查过程中出错: {e}")

    finally:
        if keep:
            print("\n--keep 已指定,不做清理。请记得测试完自己在Zotero里手动删除:")
            for label, key in created.items():
                if key:
                    print(f"  - {label}: {key}")
        else:
            _cleanup(zt, created)

    return ok


def _cleanup(zt, created: dict) -> None:
    _hr("清理临时测试数据")

    # 1. 不管笔记是否成功创建, 都先把这次测试用的item_key在NoteCache里的残留行清掉,
    #    避免它跟你真正在排队等待提交的笔记混在一起。
    item_key = created.get("item_key")
    if item_key:
        try:
            deleted = zt.NoteCache.delete().where(zt.NoteCache.item_key == item_key).execute()
            if deleted:
                print(f"[OK] 清理了 note_cache.db 里残留的 {deleted} 行测试数据")
        except Exception as e:
            print(f"[WARN] 清理 note_cache.db 残留数据时出错(不影响你的真实笔记缓存): {e}")

    # 2. 删除测试item(会级联删掉它的笔记和附件)。
    if item_key:
        try:
            item_obj = zt.zot.item(item_key)
            # 保险起见,删除前再确认一遍这确实是我们自己创建的测试item,而不是别的东西。
            # 用小写比较,避免大小写被Zotero/插件改写导致这个安全检查误判成"不是我们的数据"。
            if "pageboygent smoke test" in item_obj.get("data", {}).get("title", "").lower():
                zt._perform_local_write(f"delete_item('{item_key}')", zt.zot.delete_item, item_obj)
                print(f"[OK] 已删除临时item {item_key}")
            else:
                print(f"[WARN] item {item_key} 的标题看起来不像我们自己创建的测试数据,"
                      f"为安全起见跳过删除,请手动检查。")
        except Exception as e:
            print(f"[WARN] 删除临时item {item_key} 失败,请手动去Zotero里删除: {e}")

    # 3. 删除两个临时collection。
    for label in ("collection_key_1", "collection_key_2"):
        col_key = created.get(label)
        if not col_key:
            continue
        try:
            col_obj = zt.zot.collection(col_key)
            if "pageboygent smoke test" in col_obj.get("data", {}).get("name", "").lower():
                zt._perform_local_write(f"delete_collection('{col_key}')", zt.zot.delete_collection, col_obj)
                print(f"[OK] 已删除临时collection {col_key}")
            else:
                print(f"[WARN] collection {col_key} 的名字看起来不像我们自己创建的测试数据,"
                      f"为安全起见跳过删除,请手动检查。")
        except Exception as e:
            print(f"[WARN] 删除临时collection {col_key} 失败,请手动去Zotero里删除: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="额外运行写入相关的测试(默认只做只读检查)")
    parser.add_argument("--yes", action="store_true", help="跳过写入测试前的交互式确认")
    parser.add_argument("--keep", action="store_true", help="写入测试结束后不清理临时数据,方便手动检查")
    args = parser.parse_args()

    from . import zt_tools as zt  # 延迟导入: 导入时会连接/初始化本地Zotero客户端

    read_ok = run_read_checks(zt)
    if not read_ok:
        print("\n只读检查没有全部通过,建议先解决上面的问题,再考虑跑 --write。")
        return 1

    if not args.write:
        print("\n只读检查全部通过。如果想测试写入路径,加上 --write 参数重新运行。")
        return 0

    write_ok = run_write_checks(zt, auto_yes=args.yes, keep=args.keep)
    return 0 if write_ok else 1


if __name__ == "__main__":
    sys.exit(main())
