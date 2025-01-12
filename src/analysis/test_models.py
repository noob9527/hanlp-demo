import json
import unittest

from src.analysis.models import AnalysisResponse, Term, NamedEntity


class TestKeywordsAnalysisResponse(unittest.TestCase):
    def test_json_serializable(self):
        # Create sample data
        terms = [
            Term(
                token="test",
                pos_ctb9="n",
                pos_pku="n",
                # term_frequency=1,
            )
        ]
        named_entities = [
            NamedEntity(
                entity="John Doe",
                type="PERSON",
                offset=(1, 2),
                # term_frequency=1
            )
        ]

        # Create response object
        response = AnalysisResponse(
            terms=terms,
            named_entities=named_entities
        )

        # Test serialization
        try:
            json_str = response.model_dump_json()
            print(json_str)
            # Verify we can parse it back
            parsed = json.loads(json_str)
            self.assertIsInstance(parsed, dict)
            self.assertIn("terms", parsed)
            self.assertIn("named_entities", parsed)
        except Exception as e:
            self.fail(f"JSON serialization failed: {str(e)}")


if __name__ == "__main__":
    unittest.main()
