from Pageboygent.zotero_merge import replaced_key_map


def test_replaced_key_map_follows_merge_chains():
    items = [
        {"key": "MASTER", "data": {"relations": {"dc:replaces": ["http://zotero.org/users/1/items/MID", "http://zotero.org/users/1/items/OTHER"]}}},
        {"key": "MID2", "data": {"relations": {"dc:replaces": "http://zotero.org/users/1/items/OLD"}}},
        {"key": "PLAIN", "data": {}},
    ]
    # OLD 先并进 MID2，MID2 之后又……这里 MID2 仍在库里，所以 OLD → MID2；MID、OTHER → MASTER
    assert replaced_key_map(items) == {"MID": "MASTER", "OTHER": "MASTER", "OLD": "MID2"}
    chained = items + [{"key": "TOP", "data": {"relations": {"dc:replaces": "http://zotero.org/users/1/items/MID2"}}}]
    assert replaced_key_map(chained)["OLD"] == "TOP"     # 多次合并：一直找到最终的主条目
