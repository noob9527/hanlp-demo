import unittest

from src.analysis.analysis import _filter_terms, _filter_named_entities, \
    analysis


class TestKeywordsAnalysis(unittest.TestCase):
    def test_filter_named_entities(self):
        # Test case with mixed entity types
        test_entities = [
            ('英伟达', 'ORGANIZATION', 9, 10),
            ('西欧', 'LOCATION', 27, 28),
            ('谷歌', 'ORGANIZATION', 30, 31),
            ('中国', 'LOCATION', 40, 41)
        ]

        expected_result = [
            ('英伟达', 'ORGANIZATION', 9, 10),
            ('谷歌', 'ORGANIZATION', 30, 31)
        ]

        result = _filter_named_entities(test_entities)
        self.assertEqual(result, expected_result)

    def test_filter_named_entities_empty_input(self):
        # Test with empty input
        result = _filter_named_entities([])
        self.assertEqual(result, [])

    def test_filter_terms_empty_input(self):
        # Test with empty input
        result = _filter_terms([])
        self.assertEqual(result, [])

    def test_keywords_analysis_basic(self):
        # Test with a simple Chinese text
        test_text = """
        Q4：咱们一共有多少个供应商？都是谁？
        A：现在有三大类的AI芯片供应商。第一大类是国际的供应商，以英伟达为首，英特尔、AMD也是，谷歌税控已经退出，我们跟它在西欧市场有合作，在中国市场进行了退出。2023年，从出货量上来讲，英伟达占了68.4%，出货金额占73.2%；英特尔2023年出货量占了0.3%，AMD占了0.4%，这三家加起来占了69%左右。第二大类是国内相对独立的AI芯片厂商，像HW、寒武纪、碧润、摩尔、东菱、海光、天数、燧原等。HW的份额排在第一位，2023年出货量占比在15%到17%之间。第三类是以互联网大厂自研为主的，像百度的昆仑芯，占了6.1%的份额，还包括阿里的含光，腾讯的自销，以及给字节做的定制化。
        """
        result = analysis(test_text)
        print(result)


if __name__ == '__main__':
    unittest.main()
