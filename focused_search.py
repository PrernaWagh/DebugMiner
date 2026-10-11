
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TEXT_COLUMNS = [
    "Bug_Description",
    "Implementation-Level Description",
    "Contextual-Level Description",
    "High-Level Description",
    "Faulty Code",
]

ERROR_TERMS = {
    "KeyError": [
        "keyerror",
        "missing key",
        "dictionary key",
        "key not found",
        "key does not exist",
    ],
    "IndexError": [
        "indexerror",
        "index out of range",
        "list index out of range",
        "tuple index out of range",
    ],
    "TypeError": [
        "typeerror",
        "unsupported operand type",
        "unexpected keyword argument",
        "unexpected argument",
    ],
    "AttributeError": [
        "attributeerror",
        "has no attribute",
        "none type",
    ],
    "ValueError": [
        "valueerror",
        "invalid literal",
        "invalid value",
        "could not convert",
    ],
}


class FocusedBugSearch:
    def __init__(self, file_path="PyresBugs.xlsx"):
        print("Loading historical records...")

        self.df = pd.read_excel(file_path)

        for column in TEXT_COLUMNS + [
            "Fault Free Code",
            "Fixed_Method",
            "Project",
            "Fault_Acronym",
            "Url",
        ]:
            if column not in self.df.columns:
                self.df[column] = ""

            self.df[column] = (
                self.df[column].fillna("").astype(str)
            )

        # Keep titles and detailed context separate.
        self.df["description_text"] = (
            self.df["Bug_Description"].str.lower()
        )

        self.df["context_text"] = (
            self.df[
                [
                    "Implementation-Level Description",
                    "Contextual-Level Description",
                    "High-Level Description",
                ]
            ]
            .agg(" ".join, axis=1)
            .str.lower()
        )

        self.df["all_text"] = (
            self.df["description_text"]
            + " "
            + self.df["context_text"]
            + " "
            + self.df["Faulty Code"].str.lower()
        )

        # Separate vectorizers prevent a large code field from
        # completely dominating the bug description.
        self.title_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=15000,
            sublinear_tf=True,
        )

        self.context_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=30000,
            sublinear_tf=True,
        )

        self.title_matrix = self.title_vectorizer.fit_transform(
            self.df["description_text"]
        )

        # Empty context fields are supported by using a fallback.
        context_values = self.df["context_text"].copy()
        context_values = context_values.where(
            context_values.str.strip().ne(""),
            self.df["description_text"],
        )

        self.context_matrix = (
            self.context_vectorizer.fit_transform(context_values)
        )

        print(f"Loaded {len(self.df)} records.")
        print("Separate title and context indexes created.")

    def search(self, query, error_type=None, top_k=5):
        query = query.strip()

        if not query:
            return pd.DataFrame()

        if error_type:
            for known_error in ERROR_TERMS:
                if known_error.lower() == error_type.lower():
                    error_type = known_error
                    break
        else:
            match = re.search(
                r"\b([A-Za-z_]\w*(?:Error|Exception))\b",
                query,
                re.IGNORECASE,
            )
            error_type = match.group(1) if match else None

        title_query = self.title_vectorizer.transform([query])
        context_query = self.context_vectorizer.transform([query])

        title_scores = cosine_similarity(
            title_query, self.title_matrix
        ).flatten()

        context_scores = cosine_similarity(
            context_query, self.context_matrix
        ).flatten()

        terms = ERROR_TERMS.get(error_type, [])
        ranked = []

        for i, row in self.df.iterrows():
            description = row["description_text"]
            context = row["context_text"]
            full_text = row["all_text"]

            title_matches = [
                term for term in terms if term in description
            ]

            context_matches = [
                term for term in terms if term in context
            ]

            code_matches = [
                term for term in terms
                if term in row["Faulty Code"].lower()
            ]

            # Require some semantic overlap, not merely the
            # presence of an exception word somewhere in the file.
            if title_scores[i] <= 0 and context_scores[i] <= 0:
                continue

            if terms and not (
                title_matches or context_matches or code_matches
            ):
                continue

            # Descriptions carry more weight than context.
            semantic_score = (
                0.65 * title_scores[i]
                + 0.35 * context_scores[i]
            )

            evidence_score = (
                0.60 * bool(title_matches)
                + 0.25 * bool(context_matches)
                + 0.15 * bool(code_matches)
            )

            final_score = (
                0.80 * semantic_score
                + 0.20 * evidence_score
            )

            # Keep scores transparent; these are ranking values,
            # not probabilities or accuracy percentages.
            ranked.append((
                i,
                final_score,
                title_scores[i],
                context_scores[i],
                title_matches + context_matches + code_matches,
            ))

        ranked.sort(key=lambda item: item[1], reverse=True)

        results = []
        seen = set()

        for i, final_score, title_score, context_score, evidence in ranked:
            row = self.df.iloc[i]
            description = row["Bug_Description"].strip()
            normalized = " ".join(description.lower().split())

            if not normalized or normalized in seen:
                continue

            # Avoid returning extremely weak semantic matches.
            if max(title_score, context_score) < 0.03:
                continue

            seen.add(normalized)

            results.append({
                "Bug_Description": description,
                "Project": row["Project"],
                "Fixed_Method": row["Fixed_Method"],
                "Faulty_Code": row["Faulty Code"],
                "Fixed_Code": row["Fault Free Code"],
                "Historical_URL": row["Url"],
                "Title_Similarity": round(float(title_score), 4),
                "Context_Similarity": round(float(context_score), 4),
                "Ranking_Score": round(float(final_score), 4),
                "Evidence": ", ".join(evidence),
            })

            if len(results) >= top_k:
                break

        return pd.DataFrame(results)


if __name__ == "__main__":
    searcher = FocusedBugSearch()

    query = input(
        "Enter the error and problem description: "
    )

    error_type = input(
        "Enter the Python exception name (e.g. KeyError): "
    ).strip()

    results = searcher.search(
        query,
        error_type=error_type,
        top_k=5,
    )

    if results.empty:
        print("\nNo sufficiently relevant historical matches found.")
    else:
        for number, (_, row) in enumerate(
            results.iterrows(), start=1
        ):
            print("\n" + "=" * 70)
            print(f"HISTORICAL MATCH {number}")
            print("=" * 70)
            print("Bug:", row["Bug_Description"])
            print("Project:", row["Project"])
            print("Fixed method:", row["Fixed_Method"])
            print("Title similarity:", row["Title_Similarity"])
            print("Context similarity:", row["Context_Similarity"])
            print("Ranking score:", row["Ranking_Score"])
            print("Evidence:", row["Evidence"])
            print("\nFaulty code:\n", row["Faulty_Code"][:1500])
            print("\nHistorical fixed code:\n", row["Fixed_Code"][:1500])
            print("\nProject URL:", row["Historical_URL"])