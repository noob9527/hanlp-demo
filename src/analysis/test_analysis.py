import unittest

from src.analysis.analysis import _filter_terms, _filter_named_entities, \
    fine_analysis, coarse_analysis, fine_coarse_analysis, \
    _should_use_paragraph_pipeline, TEXT_LENGTH_THRESHOLD, has_gpu


class TestAnalysis(unittest.TestCase):
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

    def test_fine_analysis(self):
        """
        测试细粒度分词分析
        """
        text = "英伟达和谷歌是世界知名的科技公司"
        result = fine_analysis(text)
        # 确保返回了分词结果
        self.assertTrue(len(result.terms) > 0)
        # 确保识别出了组织实体
        self.assertTrue(any(ne.entity == "英伟达" for ne in result.named_entities))
        self.assertTrue(any(ne.entity == "谷歌" for ne in result.named_entities))

    def test_coarse_analysis(self):
        """
        测试粗粒度分词分析

        注意，因为分句会丢失上下文信息，所以可以在一定程度上对分词结果有不好的影响
        e.g. 2）大众点评、天猫、百度负责医美广告的； 3）更美、美呗等竞对；
        在粗分情况下 2)和3) 在不在一行，决定了"更美"和"美呗"能否被分对。。
        """
        text = """麻烦找一下新氧相关的专家：
        1）新氧公司的各种专家，包括高管、负责内容的、负责BD的等；
        2）大众点评、天猫、百度负责医美广告的； 3）更美、美呗等竞对；
        4）医美机构负责广告投放的"""
        result = coarse_analysis(text)
        print(result)
        # 确保返回了分词结果
        self.assertTrue(len(result.terms) > 0)
        # 检查粗分是否正确处理了"更美"和"美呗"
        self.assertTrue(any(term.token == "更美" for term in result.terms))
        self.assertTrue(any(term.token == "美呗" for term in result.terms))

    def test_fine_coarse_analysis(self):
        """
        测试同时进行细粒度和粗粒度分词
        """
        text = "杭州甘其食是一家连锁包子店"
        result = fine_coarse_analysis(text)

        # 确保两种分析都返回了结果
        self.assertTrue(len(result.fine.terms) > 0)
        self.assertTrue(len(result.coarse.terms) > 0)

        # 细分可能会将"甘其食"分开
        fine_tokens = [term.token for term in result.fine.terms]
        # 粗分可能会将"甘其食"作为整体
        coarse_tokens = [term.token for term in result.coarse.terms]

        self.assertNotEqual(fine_tokens, coarse_tokens, "细分和粗分的结果应该有所不同")

    def test_pos_filtering(self):
        """
        测试词性过滤功能
        """
        text = "英伟达公司的人工智能技术"
        # 只保留名词
        result = fine_analysis(text, allow_pos_ctb={'NN', 'NR'})

        # 确保所有返回的词都是名词
        for term in result.terms:
            self.assertIn(term.pos_ctb, {'NN', 'NR'},
                         f"词 '{term.token}' 的词性 '{term.pos_ctb}' 不是名词")

    def test_should_use_paragraph_pipeline(self):
        """
        Test the logic for determining whether to use paragraph pipeline
        """
        # Test short text without newlines
        short_text = "这是一个短文本"
        self.assertFalse(_should_use_paragraph_pipeline(short_text))

        # Test text exactly at threshold
        text_at_threshold = "字" * TEXT_LENGTH_THRESHOLD
        self.assertFalse(_should_use_paragraph_pipeline(text_at_threshold))

        # Test text exceeding threshold
        long_text = "字" * (TEXT_LENGTH_THRESHOLD + 1)
        self.assertTrue(_should_use_paragraph_pipeline(long_text))

        # Test short text with newlines
        short_text_with_newline = "第一行\n第二行"
        self.assertFalse(_should_use_paragraph_pipeline(short_text_with_newline))

    def test_gpu_check(self):
        """Test that GPU availability check runs without error"""
        gpu_available = has_gpu()
        print(gpu_available)
        assert isinstance(gpu_available, bool)


if __name__ == '__main__':
    unittest.main()
