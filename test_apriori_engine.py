import unittest
from apriori_engine import apriori_from_scratch, generate_rules, transactions_from_dataframe

class TestApriori(unittest.TestCase):
    def setUp(self):
        self.transactions = [
            ["Milk", "Bread"],
            ["Milk", "Bread", "Butter"],
            ["Milk", "Butter"],
            ["Bread", "Butter"],
        ]

    def test_support_and_pair(self):
        itemsets, _ = apriori_from_scratch(self.transactions, min_support=0.5, max_len=2)
        self.assertAlmostEqual(itemsets[frozenset(["Milk"])], 0.75)
        self.assertAlmostEqual(itemsets[frozenset(["Bread"])], 0.75)
        self.assertAlmostEqual(itemsets[frozenset(["Milk", "Bread"])], 0.5)

    def test_rule_metrics(self):
        itemsets, _ = apriori_from_scratch(self.transactions, min_support=0.5, max_len=2)
        rules = generate_rules(itemsets, min_confidence=0.5, min_lift=0)
        milk_to_bread = rules[
            rules["antecedents"].apply(lambda x: x == frozenset(["Milk"]))
            & rules["consequents"].apply(lambda x: x == frozenset(["Bread"]))
        ].iloc[0]
        self.assertAlmostEqual(milk_to_bread["confidence"], 2 / 3)
        self.assertAlmostEqual(milk_to_bread["lift"], (2 / 3) / 0.75)

    def test_dataframe_transactions(self):
        import pandas as pd
        df = pd.DataFrame({
            "Fault_Acronym": ["KeyError", "IndexError"],
            "Project": ["pandas", "numpy"],
            "Python_Version": ["3.8.10", "3.9"],
            "Bug_Description": ["free text", "other text"],
        })
        txs = transactions_from_dataframe(df)
        self.assertEqual(len(txs), 2)
        self.assertIn("Fault_Acronym=KeyError", txs[0])
        self.assertIn("PythonVersion=3.8", txs[0])
        self.assertFalse(any("Bug_Description" in item for tx in txs for item in tx))

if __name__ == "__main__":
    unittest.main()
