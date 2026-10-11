
import pandas as pd

df = pd.read_excel("PyresBugs.xlsx")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\n--- Records with KeyError in Bug_Description ---")

mask = df["Bug_Description"].fillna("").str.contains(
    r"\bKeyError\b|missing key|dictionary key",
    case=False,
    regex=True,
)

matches = df.loc[mask]

print("Matching descriptions:", len(matches))

for _, row in matches.head(15).iterrows():
    print("\n" + "=" * 70)
    print("DESCRIPTION:", row.get("Bug_Description", ""))
    print("PROJECT:", row.get("Project", ""))
    print("FAULTY CODE:", str(row.get("Faulty Code", ""))[:500])
    print("FIXED METHOD:", str(row.get("Fixed_Method", ""))[:500])
    print("FAULT-FREE CODE:", str(row.get("Fault Free Code", ""))[:500])
    print("URL:", row.get("Url", ""))