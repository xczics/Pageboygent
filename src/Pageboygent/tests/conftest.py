"""测试的公共设置：不受本机配置文件影响。"""
import pytest

from Pageboygent import i18n


@pytest.fixture(autouse=True)
def _chinese_interface_by_default(monkeypatch):
    """界面语言固定为中文（断言里用的是中文原文）。专门测英文界面的用例会自己再改成 "en"。"""
    monkeypatch.setattr(i18n, "ui_language", lambda: "zh")
