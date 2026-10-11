
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

from preprocess import load_and_preprocess


# Mining thresholds
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.50


def mine_association_rules():
    # Step 1: Load and preprocess the dataset
    data, transactions = load_and_preprocess()

    # Step 2: Convert transactions into a Boolean table
    encoder = TransactionEncoder()
    encoded_array = encoder.fit(transactions).transform(transactions)

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.columns_
    )

    print("\nEncoded transaction table:", encoded_df.shape)

    # Step 3: Find frequent itemsets using Apriori
    frequent_itemsets = apriori(
        encoded_df,
        min_support=MIN_SUPPORT,
        use_colnames=True
    )

    if frequent_itemsets.empty:
        print("No frequent itemsets found. Try lowering MIN_SUPPORT.")
        return

    # Save readable frequent itemsets
    frequent_itemsets_output = frequent_itemsets.copy()

    frequent_itemsets_output["itemsets"] = (
        frequent_itemsets_output["itemsets"].apply(
            lambda items: ", ".join(sorted(items))
        )
    )

    frequent_itemsets_output["length"] = (
        frequent_itemsets_output["length"]
        if "length" in frequent_itemsets_output.columns
        else frequent_itemsets_output["itemsets"].apply(
            lambda items: len(items.split(", "))
        )
    )

    frequent_itemsets_output.to_csv(
        "frequent_itemsets.csv",
        index=False
    )

    print("\nFrequent itemsets found:", len(frequent_itemsets))

    # Step 4: Generate association rules
    rules = association_rules(
        frequent_itemsets,
        num_itemsets=len(transactions),
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    )

    if rules.empty:
        print("\nNo association rules found.")
        print("Try lowering MIN_CONFIDENCE or MIN_SUPPORT.")
        return

    # Step 5: Format antecedents and consequents
    rules["antecedents"] = rules["antecedents"].apply(
        lambda items: ", ".join(sorted(items))
    )

    rules["consequents"] = rules["consequents"].apply(
        lambda items: ", ".join(sorted(items))
    )

    # Step 6: Sort by lift and confidence
    rules = rules.sort_values(
        by=["lift", "confidence"],
        ascending=False
    ).reset_index(drop=True)

    # Step 7: Save all association rules
    rules.to_csv(
        "association_rules.csv",
        index=False
    )

    # Step 8: Display top 15 rules separately
    print("\nAssociation rules found:", len(rules))
    print("\nTop 15 association rules:")

    for index, (_, row) in enumerate(
        rules.head(15).iterrows(), start=1
    ):
        print(f"\nRule {index}:")
        print(f"IF: {row['antecedents']}")
        print(f"THEN: {row['consequents']}")
        print(f"Support: {row['support']:.4f}")
        print(f"Confidence: {row['confidence']:.2%}")
        print(f"Lift: {row['lift']:.4f}")

    print("\nFiles created successfully:")
    print("- frequent_itemsets.csv")
    print("- association_rules.csv")


if __name__ == "__main__":
    mine_association_rules()