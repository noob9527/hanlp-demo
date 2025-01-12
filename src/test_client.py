import unittest

from hanlp_restful import HanLPClient

_BASE_URL = 'http://localhost:5012'


class TestClient(unittest.TestCase):

    def setUp(self) -> None:
        self.HanLP = HanLPClient(_BASE_URL,
                                 auth=None,
                                 timeout=60 * 10,
                                 )  # Fill in your auth

    def test_raw_text(self):
        text = '2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。'
        doc = self.HanLP.parse(text)
        print(doc)

    def test_sents(self):
        text = ['2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。',
                '阿婆主来到北京立方庭参观自然语义科技公司。']
        doc = self.HanLP(text)
        print(doc)

    # def test_tokens(self):
    #     tokens = [
    #         ["2021年", "HanLPv2.1", "为", "生产", "环境", "带来", "次", "世代", "最",
    #          "先进", "的", "多语种", "NLP", "技术", "。"],
    #         ["英", "首相", "与", "特朗普", "通", "电话", "讨论", "华为", "与",
    #          "苹果", "公司", "。"]
    #     ]
    #     doc = self.HanLP(tokens=tokens, tasks=['ner*', 'srl', 'dep'])
    #     print(doc)
    #
    # def test_sents_mul(self):
    #     text = [
    #         'In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environment.',
    #         '2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。',
    #         '2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。']
    #     doc = self.HanLP.parse(text, language='mul')
    #
    # def test_tokenize(self):
    #     print(self.HanLP.tokenize(
    #         '商品和服务。阿婆主来到北京立方庭参观自然语义科技公司'))
    #     print(self.HanLP.tokenize(
    #         '商品和服务。阿婆主来到北京立方庭参观自然语义科技公司', coarse=True))
    #     print(self.HanLP.tokenize(['商品和服务。', '当下雨天地面积水分外严重']))
    #     print(self.HanLP.tokenize(
    #         '奈須きのこは1973年11月28日に千葉県円空山で生まれ、ゲーム制作会社「ノーツ」の設立者だ。',
    #         language='ja', coarse=True))
    #     print(self.HanLP.tokenize(
    #         [
    #             'In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environment.',
    #             '2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。',
    #             '2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。'],
    #         language='mul'))


if __name__ == '__main__':
    unittest.main()
