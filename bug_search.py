"""TF-IDF search over historical bug records; separate from Apriori mining."""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TEXT_COLUMNS = [
    "Bug_Description", "Implementation-Level Description",
    "Contextual-Level Description", "High-Level Description", "Faulty Code",
]

class BugSearch:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        for col in TEXT_COLUMNS:
            if col not in self.df.columns:
                self.df[col] = ""
            self.df[col] = self.df[col].fillna("").astype(str)
        # Weight descriptions more heavily than code/context.
        self.df["_search_text"] = (
            (self.df["Bug_Description"] + " ") * 3
            + (self.df["Implementation-Level Description"] + " ") * 2
            + self.df["Contextual-Level Description"] + " "
            + self.df["High-Level Description"] + " "
            + self.df["Faulty Code"]
        )
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", max_features=30000,
            ngram_range=(1, 2), sublinear_tf=True
        )
        try:
            self.matrix = self.vectorizer.fit_transform(self.df["_search_text"])
        except ValueError:
            self.matrix = None

    def search(self, query, top_k=5):
        query = (query or "").strip()
        if not query or self.matrix is None:
            return pd.DataFrame()
        scores = cosine_similarity(
            self.vectorizer.transform([query]), self.matrix
        ).ravel()
        exception_match = re.search(r"\b([A-Za-z_]\w*(?:Error|Exception))\b", query)
        candidates = list(range(len(self.df)))
        if exception_match and "Fault_Acronym" in self.df.columns:
            error_type = exception_match.group(1)
            fault_values = self.df["Fault_Acronym"].fillna("").astype(str)
            mask = fault_values.str.contains(
                r"\b" + re.escape(error_type) + r"\b", case=False, regex=True
            )
            exact = [i for i, matched in enumerate(mask) if matched]
            if exact:
                candidates = exact
        candidates.sort(key=lambda i: scores[i], reverse=True)
        rows, seen = [], set()
        for i in candidates:
            if scores[i] <= 0:
                continue
            row = self.df.iloc[i]
            desc = str(row.get("Bug_Description", "")).strip()
            if not desc:
                desc = str(row.get("High-Level Description", "")).strip()
            normalized = " ".join(desc.lower().split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            rows.append({
                "Bug_Description": desc,
                "Fault_Acronym": row.get("Fault_Acronym", ""),
                "Project": row.get("Project", ""),
                "Python_Version": row.get("Python_Version", ""),
                "Dataset_input": row.get("Dataset_input", ""),
                "similarity_score": float(scores[i]),
                "Url": row.get("Url", ""),
            })
            if len(rows) >= top_k:
                break
        return pd.DataFrame(rows)
