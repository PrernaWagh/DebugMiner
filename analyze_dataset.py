
import pandas as pd

# Load the Excel dataset
df = pd.read_excel("PyResBugs.xlsx")

print("\n========== DATASET OVERVIEW ==========")
print("Rows and columns:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum().to_string())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

columns = [
    "Fault_Acronym",
    "Project",
    "Python_Version",
    "Dataset_input",
    "Impact"
]

for col in columns:
    if col in df.columns:
        print(f"\n========== {col} DISTRIBUTION ==========")
        print(df[col].fillna("MISSING").value_counts().to_string())

print("\n========== ANALYSIS COMPLETE ==========")