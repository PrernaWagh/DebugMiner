
import pandas as pd

df = pd.read_excel("PyResBugs.xlsx")

# Search historical descriptions and code for KeyError evidence
columns = [
    "Bug_Description",
    "Implementation-Level Description",
    "Contextual-Level Description",
    "High-Level Description",
    "Faulty Code",
]

for column in columns:
    df[column] = df[column].fillna("").astype(str)

df["combined_text"] = df[columns].agg(" ".join, axis=1)

matches = df[
    df["combined_text"].str.contains(
        r"\bKeyError\b|key does not exist|missing dictionary key",
        case=False,
        regex=True,
    )
]

print("Total historical records:", len(df))
print("Records mentioning KeyError or a missing key:", len(matches))

for _, row in matches.head(10).iterrows():
    print("\nDescription:", row["Bug_Description"])
    print("Fault category:", row.get("Fault_Acronym", ""))
    print("Project:", row.get("Project", ""))
    print("Reference:", row.get("Url", ""))
    print("-" * 50)