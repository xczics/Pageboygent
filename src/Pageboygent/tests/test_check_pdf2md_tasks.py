from __future__ import annotations

from pathlib import Path

import Pageboygent.storage as storage


def test_check_pdf2md_tasks_moves_pdfs_when_md_already_exists(tmp_path: Path, monkeypatch):
    ws = tmp_path / "ws"
    for d in ("human_upload_pdfs", "out_mds", "old_human_pdfs"):
        (ws / d).mkdir(parents=True)
    mineru_out = tmp_path / "MinerU"
    monkeypatch.setattr(storage, "_workstation", ws)
    monkeypatch.setattr(storage, "mineru_ouput_dir", mineru_out)
    monkeypatch.setattr(storage, "mineru_glob_pattern", "{pdf_name}.pdf-*/full.md")

    # API 产出：md 已在 out_mds（以前这里第一次就触发 free variable 'target_md_path' 报错）
    (ws / "human_upload_pdfs" / "APIKEY.pdf").write_bytes(b"%PDF")
    (ws / "out_mds" / "APIKEY.md").write_text("# api")
    # 手动流程：md 在 MinerU 输出目录，需要复制
    (ws / "human_upload_pdfs" / "MANKEY.pdf").write_bytes(b"%PDF")
    (mineru_out / "MANKEY.pdf-123").mkdir(parents=True)
    (mineru_out / "MANKEY.pdf-123" / "full.md").write_text("# manual")

    storage.check_pdf2md_tasks()

    assert not list((ws / "human_upload_pdfs").glob("*.pdf"))
    assert {p.name for p in (ws / "old_human_pdfs").glob("*.pdf")} == {"APIKEY.pdf", "MANKEY.pdf"}
    assert (ws / "out_mds" / "MANKEY.md").read_text() == "# manual"
