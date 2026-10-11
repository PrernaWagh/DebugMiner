
import pandas as pd

df = pd.read_excel("PyResBugs.xlsx")

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

# Only records with an explicit KeyError mention
matches = df[
    df["combined_text"].str.contains(
        r"\bKeyError\b",
        case=False,
        regex=True,
        na=False
    )
].copy()

print("Records explicitly mentioning KeyError:", len(matches))

# Show the shortest descriptions first so we can inspect
# whether the records actually relate to dictionary lookups.
matches["description_length"] = (
    matches["Bug_Description"].str.len()
)

matches = matches.sort_values("description_length")

for _, row in matches.head(20).iterrows():
    print("\nDescription:", row["Bug_Description"][:500])
    print("Project:", row.get("Project", ""))
    print("Fault category:", row.get("Fault_Acronym", ""))
    print("URL:", row.get("Url", ""))
    print("-" * 50)