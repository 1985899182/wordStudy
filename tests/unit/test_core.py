"""core 模块单元测试。"""
from word_study.core.schemas import (
    ExtractRequest, ManualMeaning, WordSaveItem, SaveRequest,
)


class TestSchemas:

    def test_extract_request(self):
        req = ExtractRequest(text="hello world")
        assert req.text == "hello world"

    def test_manual_meaning(self):
        m = ManualMeaning(pos="noun", meaning="测试")
        assert m.pos == "noun"
        assert m.meaning == "测试"
        assert m.group_key == ""

    def test_word_save_item(self):
        item = WordSaveItem(
            word="test",
            resource="手动录入",
            manual_meanings=[
                ManualMeaning(pos="noun", meaning="测试"),
            ],
        )
        assert item.word == "test"
        assert len(item.manual_meanings) == 1

    def test_save_request(self):
        req = SaveRequest(selections=[])
        assert req.selections == []
