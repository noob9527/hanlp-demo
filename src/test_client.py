import unittest

import requests

_BASE_URL = 'http://localhost:5012/api/v1'


class TestClient(unittest.TestCase):
    def test_fine_analysis(self):
        """Test single text fine analysis"""
        url = f"{_BASE_URL}/analysis/fine"
        data = {
            "text": "英伟达和谷歌是世界知名的科技公司",
            "allow_pos_ctb": ["NN", "NR"]
        }
        response = requests.post(url, json=data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("terms", result)
        self.assertIn("named_entities", result)

    def test_fine_analysis_batch(self):
        """Test batch fine analysis"""
        url = f"{_BASE_URL}/analysis/fine/batch"
        data = {
            "texts": [
                "英伟达和谷歌是世界知名的科技公司",
                "苹果公司是一家创新科技企业",
                "阿里巴巴是中国最大的电商平台"
            ],
            "allow_pos_ctb": ["NN", "NR"]
        }
        response = requests.post(url, json=data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 3)
        for analysis in result["results"]:
            self.assertIn("terms", analysis)
            self.assertIn("named_entities", analysis)

    def test_coarse_analysis(self):
        """Test single text coarse analysis"""
        url = f"{_BASE_URL}/analysis/coarse"
        data = {
            "text": "更美和美呗是医美行业的竞争对手",
            "allow_pos_ctb": ["NN", "NR"]
        }
        response = requests.post(url, json=data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("terms", result)
        self.assertIn("named_entities", result)

    def test_coarse_analysis_batch(self):
        """Test batch coarse analysis"""
        url = f"{_BASE_URL}/analysis/coarse/batch"
        data = {
            "texts": [
                "更美和美呗是医美行业的竞争对手",
                "大众点评和天猫都有医美广告",
                "医美机构需要做广告投放"
            ],
            "allow_pos_ctb": ["NN", "NR"]
        }
        response = requests.post(url, json=data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 3)
        for analysis in result["results"]:
            self.assertIn("terms", analysis)
            self.assertIn("named_entities", analysis)

    def test_fine_coarse_analysis(self):
        """Test single text fine-coarse analysis"""
        url = f"{_BASE_URL}/analysis/fine-coarse"
        data = {
            "text": "杭州甘其食是一家连锁包子店",
            "allow_pos_ctb": ["NN", "NR"]
        }
        response = requests.post(url, json=data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("fine", result)
        self.assertIn("coarse", result)
        for analysis in [result["fine"], result["coarse"]]:
            self.assertIn("terms", analysis)
            self.assertIn("named_entities", analysis)

    def test_fine_coarse_analysis_batch(self):
        """Test batch fine-coarse analysis"""
        url = f"{_BASE_URL}/analysis/fine-coarse/batch"
        data = {
            "texts": [
                "杭州甘其食是一家连锁包子店",
                "巴比食品是包子行业的领导者",
                "三津汤包也是知名包子品牌"
            ],
            "allow_pos_ctb": ["NN", "NR"]
        }
        response = requests.post(url, json=data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 3)
        for combined_analysis in result["results"]:
            self.assertIn("fine", combined_analysis)
            self.assertIn("coarse", combined_analysis)
            for analysis in [combined_analysis["fine"], combined_analysis["coarse"]]:
                self.assertIn("terms", analysis)
                self.assertIn("named_entities", analysis)


if __name__ == '__main__':
    unittest.main()
