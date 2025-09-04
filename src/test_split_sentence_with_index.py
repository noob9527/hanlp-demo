import unittest
from .split_sentence import split_sentence_with_index


def compute_expected_indices(text: str, sentences: list[str]) -> list[int]:
    indices = []
    cursor = 0
    for s in sentences:
        idx = text.find(s, cursor)
        if idx == -1:
            # Fallback: if not found due to spacing/quotes, try without leading spaces
            idx = text.replace("\n", "").find(s, cursor)
        indices.append(idx)
        cursor = idx + len(s) if idx >= 0 else cursor
    return indices


class TestSplitSentenceWithIndex(unittest.TestCase):
    """Test cases for split_sentence_with_index mirroring split_sentence tests."""

    def assert_sentences_and_indices(self, text: str, expected_sentences: list[str]):
        result = list(split_sentence_with_index(text))
        got_sentences = [s for s, _ in result]
        self.assertEqual(got_sentences, expected_sentences)
        expected_indices = compute_expected_indices(text, expected_sentences)
        got_indices = [i for _, i in result]
        self.assertEqual(got_indices, expected_indices)

    def test_basic_english_sentences(self):
        text = "Hello world. How are you? I am fine!"
        expected = ["Hello world.", "How are you?", "I am fine!"]
        self.assert_sentences_and_indices(text, expected)

    def test_chinese_punctuation(self):
        text = "你好世界。你好吗？我很好！"
        expected = ["你好世界。", "你好吗？", "我很好！"]
        self.assert_sentences_and_indices(text, expected)

    def test_mixed_chinese_english(self):
        text = "Hello世界。How are you今天？"
        expected = ["Hello世界。", "How are you今天？"]
        self.assert_sentences_and_indices(text, expected)

    def test_ellipsis_handling(self):
        text = "Wait......Let me think. Something...else here."
        expected = ["Wait......", "Let me think.", "Something...else here."]
        self.assert_sentences_and_indices(text, expected)

    def test_chinese_ellipsis(self):
        text = "等等……让我想想。还有其他事情。"
        expected = ["等等……", "让我想想。", "还有其他事情。"]
        self.assert_sentences_and_indices(text, expected)

    def test_quotes_with_punctuation(self):
        text = 'He said "Hello!" Then he left. She replied "Goodbye."'
        expected = ['He said "Hello!" Then he left.', 'She replied "Goodbye."']
        self.assert_sentences_and_indices(text, expected)

    def test_chinese_quotes_with_punctuation(self):
        text = '他说"你好！"然后离开了。她回答"再见。"'
        expected = ['他说"你好！', '"然后离开了。', '她回答"再见。', '"']
        self.assert_sentences_and_indices(text, expected)

    def test_abbreviations_not_split(self):
        text = "Mr. Smith went to Dr. Johnson's office. They discussed the project."
        expected = ["Mr. Smith went to Dr. Johnson's office.", "They discussed the project."]
        self.assert_sentences_and_indices(text, expected)

    def test_acronyms_not_split(self):
        text = "The U.S.A. is a country. The U.K. is another country."
        expected = ["The U.S.A. is a country.", "The U.K. is another country."]
        self.assert_sentences_and_indices(text, expected)

    def test_best_mode_vs_simple_mode(self):
        text = "Mr. Smith said hello. Dr. Johnson replied."
        result_best = list(split_sentence_with_index(text, best=True))
        expected_best = ["Mr. Smith said hello.", "Dr. Johnson replied."]
        self.assertEqual([s for s, _ in result_best], expected_best)
        # Validate indices are positions of sentences
        self.assertEqual([i for _, i in result_best], compute_expected_indices(text, expected_best))

        result_simple = list(split_sentence_with_index(text, best=False))
        self.assertTrue(len(result_simple) >= 1)
        # Indices must be non-decreasing and valid
        indices = [i for _, i in result_simple]
        self.assertTrue(all(i >= 0 for i in indices))
        self.assertTrue(all(indices[i] <= indices[i+1] for i in range(len(indices)-1)))

    def test_empty_text(self):
        text = ""
        result = list(split_sentence_with_index(text))
        self.assertEqual(result, [])

    def test_whitespace_only(self):
        text = "   \n\t  "
        result = list(split_sentence_with_index(text))
        self.assertEqual(result, [])

    def test_single_word(self):
        text = "Hello"
        expected = ["Hello"]
        self.assert_sentences_and_indices(text, expected)

    def test_no_punctuation_sentence(self):
        text = "This is a sentence without punctuation"
        expected = ["This is a sentence without punctuation"]
        self.assert_sentences_and_indices(text, expected)

    def test_multiple_newlines(self):
        text = "First sentence.\n\n\nSecond sentence."
        expected = ["First sentence.", "Second sentence."]
        self.assert_sentences_and_indices(text, expected)

    def test_complex_mixed_content(self):
        text = "Dr. Smith说：'Hello world！'然后他离开了。U.S.A.是一个国家......真的吗？"
        result = list(split_sentence_with_index(text))
        self.assertTrue(len(result) >= 2)
        self.assertTrue(any("Dr. Smith" in s for s, _ in result))
        # Validate indices correspond to positions
        self.assertTrue(all(text.find(s, i) == i for s, i in result if i >= 0))

    def test_long_sentence_with_abbreviations(self):
        text = "The U.S.A. and U.K. signed a treaty with Dr. Smith and Mr. Johnson present."
        expected = ["The U.S.A. and U.K. signed a treaty with Dr. Smith and Mr. Johnson present."]
        self.assert_sentences_and_indices(text, expected)

    def test_repeated_punctuation(self):
        text = "Really??? Yes!!! Absolutely..."
        expected = ["Really?", "??", "Yes!!!", "Absolutely..."]
        self.assert_sentences_and_indices(text, expected)

    def test_numbers_with_periods(self):
        text = "The value is 3.14159. Another value is 2.71828."
        expected = ["The value is 3.14159.", "Another value is 2.71828."]
        self.assert_sentences_and_indices(text, expected)

    def test_special_characters(self):
        text = "Price: $100.50. Percentage: 95.5%. Time: 10:30 a.m."
        result = list(split_sentence_with_index(text))
        self.assertTrue(len(result) >= 2)
        self.assertTrue(all(i >= 0 for _, i in result))


if __name__ == '__main__':
    unittest.main(verbosity=2)


