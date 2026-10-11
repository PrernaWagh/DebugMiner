
import pandas as pd


def load_and_preprocess(file_path="PyResBugs.xlsx"):
    # Load the original dataset
    df = pd.read_excel(file_path)

    print("Original dataset shape:", df.shape)

    # Keep only the columns useful for association-rule mining
    data = df[
        [
            "Fault_Acronym",
            "Project",
            "Dataset_input",
            "Python_Version",
        ]
    ].copy()

    # Remove rows missing any selected feature
    data = data.dropna()

    # Normalize text values
    for column in ["Fault_Acronym", "Project", "Dataset_input"]:
        data[column] = data[column].astype(str).str.strip()

    # Normalize Python versions to major.minor
    def normalize_version(version):
        parts = str(version).strip().split(".")
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
        return str(version).strip()

    data["Python_Version"] = data["Python_Version"].apply(
        normalize_version
    )

    # Convert each row into a transaction of categorical items
    transactions = []

    for _, row in data.iterrows():
        transaction = [
            "Fault=" + row["Fault_Acronym"],
            "Project=" + row["Project"],
            "Source=" + row["Dataset_input"],
            "PythonVersion=" + row["Python_Version"],
        ]
        transactions.append(transaction)

    print("Rows available for mining:", len(data))
    print("Example transaction:")
    print(transactions[0])

    return data, transactions


if __name__ == "__main__":
    data, transactions = load_and_preprocess()
    print("\nPreprocessing completed successfully!")