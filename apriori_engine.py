"""Apriori from scratch and optional library benchmarks for DebugMiner."""
from itertools import combinations
from collections import Counter
from time import perf_counter
import math
import pandas as pd

PREFERRED_COLUMNS = ["Fault_Acronym", "Project", "Dataset_input", "Python_Version"]
IGNORED_COLUMNS = {
    "Bug_Description", "Implementation-Level Description",
    "Contextual-Level Description", "High-Level Description",
    "Faulty Code", "Fault Free Code", "Fixed_Method", "Url",
    "Commit_URL", "Commit_sha", "Diff_patch", "Test_File_Path",
    "CVE-ID", "Bug_Type", "Impact", "index", "Unnamed: 0",
}

def _clean_value(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null", "unknown"}:
        return None
    return value

def transactions_from_dataframe(df, columns=None, max_categories=100):
    """Convert each row into a transaction of categorical field=value items.

    Defaults to the four selected categorical fields. If those are unavailable,
    uses low-cardinality categorical columns, excluding free-text/code/URL fields.
    """
    if df is None or df.empty:
        return []
    if columns:
        cols = [c for c in columns if c in df.columns]
    else:
        cols = [c for c in PREFERRED_COLUMNS if c in df.columns]
        if not cols:
            cols = []
            for c in df.columns:
                if c in IGNORED_COLUMNS:
                    continue
                if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
                    nunique = df[c].nunique(dropna=True)
                    if 1 < nunique <= max_categories:
                        cols.append(c)
    transactions = []
    for _, row in df.iterrows():
        items = []
        for col in cols:
            value = _clean_value(row.get(col))
            if value is None:
                continue
            if col == "Python_Version":
                # Normalize 3.8.10 and 3.8 to a major.minor version when possible.
                parts = value.split(".")
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    value = f"{parts[0]}.{parts[1]}"
                item_name = "PythonVersion"
            else:
                item_name = col
            items.append(f"{item_name}={value}")
        # One item per field is expected; skip records with too little information.
        if len(items) >= 2:
            transactions.append(sorted(set(items)))
    return transactions

def _count_support(transactions, candidates):
    counts = Counter()
    candidate_sets = [(candidate, set(candidate)) for candidate in candidates]
    for tx in transactions:
        tx_set = set(tx)
        for candidate, candidate_set in candidate_sets:
            if candidate_set.issubset(tx_set):
                counts[candidate] += 1
    return counts

def apriori_from_scratch(transactions, min_support=0.1, max_len=4):
    """Return {frozenset(items): support_fraction}, and mining statistics."""
    txs = [set(tx) for tx in transactions if tx]
    n = len(txs)
    if n == 0:
        return {}, {"candidates_generated": 0, "pruned": 0, "transaction_count": 0}
    if not 0 <= min_support <= 1:
        raise ValueError("min_support must be between 0 and 1")
    if max_len < 1:
        raise ValueError("max_len must be at least 1")
    min_count = math.ceil(min_support * n - 1e-12)
    item_counts = Counter(item for tx in txs for item in tx)
    candidates_generated = len(item_counts)
    pruned = 0
    frequent = {}
    current_level = []
    for item, count in item_counts.items():
        if count >= min_count:
            fs = frozenset([item])
            frequent[fs] = count / n
            current_level.append(fs)
        else:
            pruned += 1

    k = 2
    while current_level and k <= max_len:
        prev = sorted([tuple(sorted(s)) for s in current_level])
        candidates = set()
        prev_set = set(current_level)
        for i in range(len(prev)):
            for j in range(i + 1, len(prev)):
                if prev[i][:k-2] != prev[j][:k-2]:
                    continue
                merged = tuple(sorted(set(prev[i]) | set(prev[j])))
                if len(merged) != k:
                    continue
                # Apriori property: every (k-1)-subset must be frequent.
                if all(frozenset(sub) in prev_set
                       for sub in combinations(merged, k - 1)):
                    candidates.add(merged)
                else:
                    pruned += 1
        candidates_generated += len(candidates)
        if not candidates:
            break
        counts = _count_support(txs, candidates)
        next_level = []
        for cand in candidates:
            count = counts.get(cand, 0)
            if count >= min_count:
                fs = frozenset(cand)
                frequent[fs] = count / n
                next_level.append(fs)
            else:
                pruned += 1
        current_level = next_level
        k += 1

    return frequent, {
        "candidates_generated": candidates_generated,
        "pruned": pruned,
        "transaction_count": n,
        "total_frequent": len(frequent),
    }

def generate_rules(frequent_itemsets, min_confidence=0.5, min_lift=1.0):
    """Generate all valid X -> Y rules from frequent itemsets."""
    rows = []
    for itemset, itemset_support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        ordered_items = sorted(itemset)
        for size in range(1, len(ordered_items)):
            for ant_tuple in combinations(ordered_items, size):
                antecedent = frozenset(ant_tuple)
                consequent = itemset - antecedent
                ant_support = frequent_itemsets.get(antecedent)
                cons_support = frequent_itemsets.get(consequent)
                if not ant_support or not cons_support:
                    continue
                confidence = itemset_support / ant_support
                lift = confidence / cons_support
                if confidence + 1e-12 < min_confidence or lift + 1e-12 < min_lift:
                    continue
                rows.append({
                    "antecedents": antecedent,
                    "consequents": consequent,
                    "support": itemset_support,
                    "confidence": confidence,
                    "lift": lift,
                    "support_count": None,
                })
    return pd.DataFrame(rows, columns=[
        "antecedents", "consequents", "support", "confidence", "lift", "support_count"
    ])

def run_apriori(transactions, min_support=0.1, min_confidence=0.5, min_lift=1.0, max_len=4):
    start = perf_counter()
    frequent, stats = apriori_from_scratch(transactions, min_support, max_len)
    rules = generate_rules(frequent, min_confidence, min_lift)
    elapsed_ms = (perf_counter() - start) * 1000
    n = len([tx for tx in transactions if tx])
    for row in rules.to_dict("records"):
        row["support_count"] = round(row["support"] * n)
    if not rules.empty:
        rules["support_count"] = (rules["support"] * n).round().astype(int)
    itemset_rows = [
        {"itemsets": fs, "support": sup, "length": len(fs),
         "support_count": round(sup * n)}
        for fs, sup in frequent.items()
    ]
    itemsets_df = pd.DataFrame(itemset_rows, columns=["itemsets", "support", "length", "support_count"])
    if not itemsets_df.empty:
        itemsets_df = itemsets_df.sort_values(
            ["length", "support"], ascending=[True, False]
        ).reset_index(drop=True)
    stats.update({
        "time_ms": round(elapsed_ms, 3),
        "total_frequent": len(frequent),
        "rules": len(rules),
        "transaction_count": n,
    })
    return itemsets_df, rules, stats

def _encode(transactions):
    from mlxtend.preprocessing import TransactionEncoder
    enc = TransactionEncoder()
    array = enc.fit(transactions).transform(transactions)
    return pd.DataFrame(array, columns=enc.columns_)

def compare_algorithms(transactions, min_support=0.1, max_len=4):
    """Benchmark algorithms; timings include encoding for library implementations."""
    results = []
    t0 = perf_counter()
    ours, _ = apriori_from_scratch(transactions, min_support, max_len)
    results.append({"Algorithm": "Apriori (from scratch)",
                    "Time (ms)": round((perf_counter() - t0) * 1000, 3),
                    "Frequent itemsets": len(ours)})
    try:
        from mlxtend.frequent_patterns import apriori as mlx_apriori, fpgrowth
        t0 = perf_counter()
        encoded = _encode(transactions)
        fi = mlx_apriori(encoded, min_support=min_support, use_colnames=True, max_len=max_len)
        results.append({"Algorithm": "Apriori (mlxtend)",
                        "Time (ms)": round((perf_counter() - t0) * 1000, 3),
                        "Frequent itemsets": len(fi)})
        t0 = perf_counter()
        encoded = _encode(transactions)
        fi = fpgrowth(encoded, min_support=min_support, use_colnames=True, max_len=max_len)
        results.append({"Algorithm": "FP-Growth",
                        "Time (ms)": round((perf_counter() - t0) * 1000, 3),
                        "Frequent itemsets": len(fi)})
    except ImportError:
        results.append({"Algorithm": "Library benchmarks unavailable",
                        "Time (ms)": None, "Frequent itemsets": None})
    return pd.DataFrame(results)

def format_itemset(value):
    if isinstance(value, (set, frozenset, list, tuple)):
        return ", ".join(sorted(map(str, value)))
    return str(value)
