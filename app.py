
"""DebugMiner: Apriori-based frequent pattern mining for software bug records."""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from apriori_engine import (
    run_apriori, compare_algorithms, transactions_from_dataframe, format_itemset,
)
from error_analyzer import identify_error, get_guidance, build_transaction_from_error


# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="DebugMiner | Bug Pattern Mining",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------
# DESIGN SYSTEM
# -------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080d18;
    --panel: #101827;
    --border: #26354b;
    --text: #edf2fb;
    --muted: #92a2b9;
    --accent: #8b9cff;
    --green: #64e6b4;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
      radial-gradient(ellipse at 80% 0%, #192344 0%, transparent 38%),
      var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] { background: rgba(8,13,24,0.8); }

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

section[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

h1, h2, h3 {
    color: var(--text) !important;
    letter-spacing: -0.035em;
}

p, label, .stMarkdown { color: #c2cede; }

.brand {
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.8px;
    color: #f5f7ff;
}
.brand span { color: var(--accent); }

.hero-title {
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -2px;
    color: #f3f6ff;
    margin: 0;
}

.hero-description {
    color: #a5b4ca;
    font-size: 1rem;
    line-height: 1.8;
    margin-top: 12px;
    max-width: 820px;
}

.metric-card {
    background: linear-gradient(145deg, #141f32, #0f1726);
    border: 1px solid #2b3a53;
    border-radius: 15px;
    padding: 20px;
    min-height: 122px;
}
.metric-value {
    font-size: 1.9rem;
    line-height: 1.5;
    font-weight: 800;
    letter-spacing: -1px;
    color: #f3f6ff;
}
.metric-note { color: #8fa2be; font-size: 0.78rem; }

.mini-label {
    color: #94a6c0;
    text-transform: uppercase;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.2px;
}

.feature-card {
    background: #111b2b;
    border: 1px solid #283850;
    border-radius: 15px;
    padding: 19px;
    min-height: 130px;
}
.feature-title {
    font-weight: 700;
    font-size: 0.98rem;
    color: #eef3ff;
    margin-bottom: 6px;
}
.feature-description {
    color: #9babc2;
    font-size: 0.83rem;
    line-height: 1.65;
}

.result-card {
    background: #101827;
    border: 1px solid #2b3b52;
    border-radius: 13px;
    padding: 17px;
    margin: 9px 0;
}

.status-good { color: #6ee7b7; font-size: 0.85rem; font-weight: 600; }

div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    background: #0b1321 !important;
    color: #edf2fb !important;
    border: 1px solid #30405a !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace;
}

.stButton > button {
    background: linear-gradient(100deg, #7588f8, #9c8bff);
    color: #080d18 !important;
    font-weight: 800;
    border: none;
    border-radius: 10px;
    padding: 0.62rem 1.1rem;
    min-height: 43px;
}
.stButton > button:hover {
    filter: brightness(1.12);
    transform: translateY(-1px);
}

div[data-testid="stExpander"] {
    background: #101827;
    border: 1px solid #2b3b52;
    border-radius: 12px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #293950;
    border-radius: 10px;
}

hr { border-color: #26354b; }

a { color: #a6b4ff !important; }

.footer {
    color: #72839d;
    font-size: 0.76rem;
    border-top: 1px solid #26354b;
    padding-top: 18px;
    margin-top: 38px;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# DATASET HELPERS
# -------------------------------------------------------
DATASET_CANDIDATES = [
    "PyresBugs.xlsx", "PyResBugs.xlsx",
    "PyresBugs.csv", "PyresBugs.csv",
]


@st.cache_data(show_spinner=False)
def load_dataset(path, modified_time):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    return pd.read_excel(path)


def find_dataset():
    for candidate in DATASET_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def readable_itemsets(df):
    out = df.copy()
    if "itemsets" in out:
        out["itemsets"] = out["itemsets"].apply(format_itemset)
    return out


def readable_rules(df):
    out = df.copy()
    for col in ("antecedents", "consequents"):
        if col in out:
            out[col] = out[col].apply(format_itemset)
    return out


def get_current_transactions(source, manual_text, uploaded_file=None):
    if source == "Manual transactions":
        return [[x.strip() for x in line.split(",") if x.strip()]
                for line in manual_text.splitlines() if line.strip()]
    if source == "Uploaded CSV":
        if uploaded_file is None:
            return []
        frame = pd.read_csv(uploaded_file, low_memory=False)
        return transactions_from_dataframe(frame)
    path = find_dataset()
    if not path:
        return []
    frame = load_dataset(path, Path(path).stat().st_mtime)
    return transactions_from_dataframe(frame)


# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand">🐛 Debug<span>Miner</span></div>',
                unsafe_allow_html=True)
    st.caption("Apriori-Based Bug Pattern Mining")
    st.divider()
    page = st.radio("NAVIGATION", [
        "Overview",
        "Code Analyzer",
        "Input Builder",
        "Apriori Mining",
        "Rule Explorer",
        "Threshold Analyzer",
        "Algorithm Comparison",
        "Dataset Statistics",
    ])
    st.divider()
    ds = find_dataset()
    if ds:
        st.markdown(f'<span class="status-good">● Dataset: {ds}</span>',
                    unsafe_allow_html=True)
    else:
        st.warning("Place PyresBugs.xlsx or PyresBugs.csv beside app.py.")
    st.caption("Python · Streamlit · Apriori")


# -------------------------------------------------------
# PAGE: OVERVIEW
# -------------------------------------------------------
if page == "Overview":
    st.title("DebugMiner")
    st.caption("Software bug pattern mining using association rules")

    path = find_dataset()
    if path:
        df = load_dataset(path, Path(path).stat().st_mtime)
        transactions = transactions_from_dataframe(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bug records", f"{len(df):,}")
        c2.metric("Dataset fields", f"{len(df.columns):,}")
        c3.metric("Transactions", f"{len(transactions):,}")
        c4.metric("Mining algorithm", "Apriori")
        st.caption(f"Dataset file: {path}")
    else:
        st.warning("Dataset not found. Place PyresBugs.xlsx or PyresBugs.csv beside app.py.")

    st.subheader("Workflow")
    w1, w2, w3, w4 = st.columns(4)
    w1.markdown("**1. Prepare data**\n\nConvert each bug record into a transaction.")
    w2.markdown("**2. Mine itemsets**\n\nGenerate candidates and prune infrequent combinations.")
    w3.markdown("**3. Generate rules**\n\nEvaluate rules using support, confidence and lift.")
    w4.markdown("**4. Review results**\n\nExplore frequent patterns and compare implementations.")

    st.subheader("Project modules")
    st.dataframe(pd.DataFrame([
        {"Module": "Input Builder", "Purpose": "Convert a traceback or manual input into a transaction."},
        {"Module": "Apriori Mining", "Purpose": "Discover frequent itemsets from the selected transaction source."},
        {"Module": "Rule Explorer", "Purpose": "Inspect association rules and their metrics."},
        {"Module": "Threshold Analyzer", "Purpose": "Observe how minimum support changes the number of rules."},
        {"Module": "Algorithm Comparison", "Purpose": "Compare manual Apriori with the library Apriori implementation."},
        {"Module": "Dataset Statistics", "Purpose": "Inspect dataset size, fields, missing values and category counts."},
    ]), width="stretch", hide_index=True)
    st.caption("Association rules show co-occurrence; they do not establish causation.")

# -------------------------------------------------------
# PAGE: CODE ANALYZER  (kept — bug guidance + related records)
# -------------------------------------------------------
elif page == "Code Analyzer":
    st.markdown('<div class="mini-label">01 / DIAGNOSTICS</div>',
                unsafe_allow_html=True)
    st.title("Code & Error Analyzer")
    st.caption("Understand a Python exception and retrieve related historical bugs.")

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        code = st.text_area(
            "Python source code", height=230,
            placeholder='data = {"name": "Sneha"}\nprint(data["age"])',
            key="analyzer_code",
        )
        error_text = st.text_area(
            "Exception or traceback", height=140,
            placeholder="KeyError: 'age'", key="analyzer_error",
        )
        analyze = st.button("Analyze and find historical fixes",
                            type="primary", width="stretch")

    with right:
        st.markdown("#### How it works")
        st.info(
            "1. Identify the exception type from the traceback.\n\n"
            "2. Show general guidance and an example fix.\n\n"
            "3. Retrieve textually similar historical bug records."
        )

    if analyze:
        if not code.strip() or not error_text.strip():
            st.warning("Please enter both code and the error message.")
        else:
            error_type = identify_error(error_text)
            guidance = get_guidance(error_type)

            st.markdown("---")
            st.markdown("### Diagnosis")
            st.markdown(f"**Detected exception:** `{error_type}`")
            st.markdown("**Explanation**")
            st.write(guidance["explanation"])
            st.markdown("**Suggested fix**")
            st.write(guidance["fix"])
            st.markdown("**Example approach**")
            st.code(guidance["example"], language="python")
            st.caption("Guidance is general — verify the fix and test your code.")

# -------------------------------------------------------
# PAGE: INPUT BUILDER
# -------------------------------------------------------
elif page == "Input Builder":
    st.markdown('<div class="mini-label">02 / TRANSACTION BUILDER</div>',
                unsafe_allow_html=True)
    st.title("Input Builder")
    st.caption("Turn an error or manual entries into Apriori transactions.")

    mode = st.radio("Input mode",
                    ["From Error Traceback", "Manual Transactions", "Full Dataset"],
                    horizontal=True)

    if mode == "From Error Traceback":
        err = st.text_area("Paste traceback or error message",
                           placeholder="KeyError: 'age'", height=120)
        col1, col2 = st.columns(2)
        project = col1.text_input("Project (optional)", placeholder="pandas")
        pyver = col2.text_input("Python version (optional)", placeholder="3.8")

        if st.button("Analyze error and build transaction", type="primary"):
            if not err.strip():
                st.warning("Enter an error message or traceback.")
            else:
                error_type = identify_error(err)
                guidance = get_guidance(error_type)
                st.markdown(f"### Detected exception: `{error_type}`")
                st.write("**Explanation:**", guidance["explanation"])
                st.write("**Suggested check:**", guidance["fix"])
                st.code(guidance["example"], language="python")

                tx = build_transaction_from_error(error_type, project, pyver)
                if not tx:
                    st.warning("No usable items were detected.")
                else:
                    st.write("**Transaction built**")
                    st.code("{" + ", ".join(tx) + "}")
                    st.session_state["new_bug_transaction"] = tx
                    st.success("Transaction saved for matching against mined patterns.")

    elif mode == "Manual Transactions":
        default = (
            "Fault_Acronym=KeyError, Project=pandas, PythonVersion=3.8\n"
            "Fault_Acronym=KeyError, Project=pandas, PythonVersion=3.8\n"
            "Fault_Acronym=KeyError, Project=numpy, PythonVersion=3.8\n"
            "Fault_Acronym=IndexError, Project=numpy, PythonVersion=3.8\n"
            "Fault_Acronym=IndexError, Project=numpy, PythonVersion=3.8\n"
            "Fault_Acronym=IndexError, Project=numpy, PythonVersion=3.7\n"
            "Fault_Acronym=TypeError, Project=black, PythonVersion=3.6\n"
            "Fault_Acronym=TypeError, Project=black, PythonVersion=3.6\n"
            "Fault_Acronym=NameError, Project=pandas, PythonVersion=3.8\n"
            "Fault_Acronym=ValueError, Project=numpy, PythonVersion=3.8"
        )
        manual = st.text_area(
            "One transaction per line; separate items with commas",
            value=st.session_state.get("manual_text", default), height=260,
        )
        st.session_state["manual_text"] = manual
        st.caption("Small sample to verify the algorithm before mining the full dataset.")
        st.write(f"Transactions entered: "
                 f"{sum(bool(line.strip()) for line in manual.splitlines())}")

    else:
        path = find_dataset()
        if path:
            df = load_dataset(path, Path(path).stat().st_mtime)
            st.success(f"Loaded {len(df):,} records and {len(df.columns)} columns.")
            st.dataframe(df.head(10), width="stretch")
            st.write("Mining fields used:",
                     [c for c in ["Fault_Acronym", "Project",
                                  "Dataset_input", "Python_Version"]
                      if c in df.columns])
        else:
            st.error("Dataset file not found.")

# -------------------------------------------------------
# PAGE: APRIORI MINING
# -------------------------------------------------------
elif page == "Apriori Mining":
    st.markdown('<div class="mini-label">03 / ASSOCIATION ANALYTICS</div>',
                unsafe_allow_html=True)
    st.title("Apriori Mining")
    st.caption("Mine frequent itemsets and association rules.")

    source = st.selectbox("Transaction source",
                          ["Full PyResBugs dataset", "Manual transactions", "Uploaded CSV"])
    uploaded = None
    if source == "Uploaded CSV":
        uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])

    manual_text = st.session_state.get("manual_text", "")
    transactions = get_current_transactions(source, manual_text, uploaded)

    if transactions:
        st.success(f"{len(transactions):,} transactions loaded.")
        with st.expander("Preview first 10 transactions"):
            st.write(transactions[:10])
    else:
        st.info("Choose or provide a transaction source to begin.")

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    min_sup = c1.slider("Minimum support", 0.01, 1.0, 0.05, 0.01,
                        help="Minimum fraction of transactions containing an itemset.")
    min_conf = c2.slider("Minimum confidence", 0.0, 1.0, 0.5, 0.05,
                         help="P(Y|X) for a rule X → Y.")
    min_lift = c3.slider("Minimum lift", 0.0, 5.0, 1.0, 0.1,
                         help="Lift above 1 indicates positive association.")
    max_len = c4.slider("Maximum itemset size", 2, 6, 3)

    if st.button("Run Apriori", type="primary", disabled=len(transactions) < 2):
        with st.spinner("Mining frequent itemsets and rules..."):
            itemsets, rules, stats = run_apriori(
                transactions, min_sup, min_conf, min_lift, max_len)
        st.session_state["last_itemsets_raw"] = itemsets
        st.session_state["last_rules_raw"] = rules
        st.session_state["last_stats"] = stats
        st.session_state["last_source"] = source

    if ("last_itemsets_raw" in st.session_state
            and st.session_state.get("last_source") == source):
        itemsets = st.session_state["last_itemsets_raw"]
        rules = st.session_state["last_rules_raw"]
        stats = st.session_state["last_stats"]

        a, b, c, d = st.columns(4)
        a.metric("Transactions", f'{stats["transaction_count"]:,}')
        b.metric("Frequent itemsets", stats["total_frequent"])
        c.metric("Association rules", len(rules))
        d.metric("Mining time", f'{stats["time_ms"]:.2f} ms')
        st.caption(f'Candidates generated: {stats["candidates_generated"]:,} · '
                   f'Pruned: {stats["pruned"]:,}')

        st.markdown("### Frequent itemsets")
        item_display = readable_itemsets(itemsets)
        st.dataframe(item_display, width="stretch", hide_index=True)
        if not item_display.empty:
            top = item_display.sort_values("support", ascending=False).head(15)
            fig = px.bar(top, x="support", y="itemsets", orientation="h",
                         color="support", title="Top 15 frequent itemsets",
                         labels={"support": "Support", "itemsets": "Itemset"})
            fig.update_layout(yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, width="stretch")
            st.download_button(
                "Download frequent itemsets CSV",
                item_display.to_csv(index=False).encode("utf-8"),
                "frequent_itemsets.csv", "text/csv",
            )

        st.markdown("### Association rules")
        rule_display = readable_rules(rules)
        if rule_display.empty:
            st.warning("No rules meet the selected thresholds. "
                       "Try lowering support, confidence or lift.")
        else:
            rule_display = rule_display.sort_values(
                ["lift", "confidence"], ascending=False)
            st.dataframe(
                rule_display.round({"support": 4, "confidence": 4, "lift": 4}),
                width="stretch", hide_index=True,
            )
            fig = px.scatter(rule_display, x="support", y="confidence",
                             color="lift", hover_data=["antecedents", "consequents"],
                             title="Support vs confidence (color = lift)")
            st.plotly_chart(fig, width="stretch")
            st.download_button(
                "Download association rules CSV",
                rule_display.to_csv(index=False).encode("utf-8"),
                "bug_association_rules.csv", "text/csv",
            )


# -------------------------------------------------------
# PAGE: RULE EXPLORER
# -------------------------------------------------------
elif page == "Rule Explorer":
    st.markdown('<div class="mini-label">04 / RULE ANALYSIS</div>',
                unsafe_allow_html=True)
    st.title("Rule Explorer")
    rules = st.session_state.get("last_rules_raw")

    if rules is None or rules.empty:
        st.info("Run Apriori on the Apriori Mining page first.")
    else:
        display = readable_rules(rules)
        c1, c2, c3 = st.columns(3)
        left = c1.text_input("Antecedent contains",
                             placeholder="Fault_Acronym=KeyError")
        right = c2.text_input("Consequent contains",
                              placeholder="Project=pandas")
        sort_by = c3.selectbox("Sort by", ["lift", "confidence", "support"])

        if left:
            display = display[display["antecedents"].str.contains(
                left, case=False, regex=False)]
        if right:
            display = display[display["consequents"].str.contains(
                right, case=False, regex=False)]
        display = display.sort_values(sort_by, ascending=False)

        st.write(f"{len(display)} matching rules")
        st.dataframe(
            display.round({"support": 4, "confidence": 4, "lift": 4}),
            width="stretch", hide_index=True,
        )

        if not display.empty:
            top = display.iloc[0]
            st.markdown("### Interpretation of strongest matching rule")
            st.write(f"**IF** {top['antecedents']} **THEN** {top['consequents']}")
            st.write(f"Support: {top['support']:.3f} — "
                     f"{top['support']:.1%} of transactions contain both sides.")
            st.write(f"Confidence: {top['confidence']:.3f} — "
                     f"among transactions with the antecedent, "
                     f"{top['confidence']:.1%} also contain the consequent.")
            st.write(f"Lift: {top['lift']:.3f} — the consequent occurs "
                     f"{top['lift']:.2f}× its baseline rate.")
            st.caption("Association is not causation.")


# -------------------------------------------------------
# PAGE: THRESHOLD ANALYZER
# -------------------------------------------------------
elif page == "Threshold Analyzer":
    st.markdown('<div class="mini-label">05 / SENSITIVITY</div>',
                unsafe_allow_html=True)
    st.title("Threshold Impact Analyzer")
    st.write("Compare how minimum support changes itemsets, rules and runtime.")

    source = st.selectbox("Dataset for experiment",
                          ["Full PyResBugs dataset", "Manual transactions"],
                          key="threshold_source")
    transactions = get_current_transactions(
        source, st.session_state.get("manual_text", ""))

    if not transactions:
        st.warning("No transactions available.")
    else:
        conf = st.slider("Minimum confidence", 0.0, 1.0, 0.5, 0.05,
                         key="threshold_conf")
        lift = st.slider("Minimum lift", 0.0, 5.0, 1.0, 0.1,
                         key="threshold_lift")
        max_len = st.slider("Maximum itemset size", 2, 5, 3,
                            key="threshold_max_len")
        support_values = st.multiselect(
            "Support thresholds to compare",
            [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
            default=[0.02, 0.05, 0.10],
        )
        if st.button("Run threshold experiment", type="primary",
                     disabled=not support_values):
            rows = []
            bar = st.progress(0)
            for i, support in enumerate(sorted(support_values)):
                itemsets, rules, stats = run_apriori(
                    transactions, support, conf, lift, max_len)
                rows.append({"Minimum support": support,
                             "Frequent itemsets": len(itemsets),
                             "Rules": len(rules),
                             "Runtime (ms)": stats["time_ms"]})
                bar.progress((i + 1) / len(support_values))
            result = pd.DataFrame(rows)
            st.dataframe(result, width="stretch", hide_index=True)
            st.plotly_chart(
                px.line(result, x="Minimum support", y="Frequent itemsets",
                        markers=True,
                        title="Support threshold vs frequent itemsets"),
                width="stretch")
            st.plotly_chart(
                px.line(result, x="Minimum support", y="Rules", markers=True,
                        title="Support threshold vs association rules"),
                width="stretch")
            st.download_button(
                "Download threshold results",
                result.to_csv(index=False).encode("utf-8"),
                "threshold_experiment.csv", "text/csv",
            )


# -------------------------------------------------------
# PAGE: ALGORITHM COMPARISON
# -------------------------------------------------------
elif page == "Algorithm Comparison":
    st.markdown('<div class="mini-label">07 / BENCHMARKS</div>',
                unsafe_allow_html=True)
    st.title("Algorithm Comparison")
    st.write("Compare the from-scratch Apriori implementation with the library Apriori implementation using identical transactions and settings.")

    source = st.selectbox("Transaction source",
                          ["Full PyResBugs dataset", "Manual transactions"],
                          key="compare_source")
    transactions = get_current_transactions(
        source, st.session_state.get("manual_text", ""))

    c1, c2 = st.columns(2)
    support = c1.slider("Minimum support", 0.01, 0.5, 0.05, 0.01,
                        key="compare_support")
    max_len = c2.slider("Maximum itemset size", 2, 5, 3, key="compare_max_len")

    if st.button("Compare algorithms", type="primary",
                 disabled=len(transactions) < 2):
        with st.spinner("Running benchmarks..."):
            comparison = compare_algorithms(transactions, support, max_len)
        st.dataframe(comparison, width="stretch", hide_index=True)

        if comparison["Time (ms)"].notna().any():
            st.plotly_chart(
                px.bar(comparison.dropna(subset=["Time (ms)"]),
                       x="Algorithm", y="Time (ms)", color="Algorithm",
                       title="Runtime comparison"),
                width="stretch",
            )
        if comparison["Frequent itemsets"].notna().all() and comparison["Frequent itemsets"].nunique() == 1:
            st.success("Both implementations returned the same number of frequent itemsets.")
        elif comparison["Frequent itemsets"].notna().all():
            st.warning("The itemset counts differ. Check preprocessing and the implementation settings.")
        st.caption("Timings vary by machine and run.")


# -------------------------------------------------------
# PAGE: DATASET STATISTICS
# -------------------------------------------------------
elif page == "Dataset Statistics":
    st.markdown('<div class="mini-label">08 / DATA PROFILING</div>',
                unsafe_allow_html=True)
    st.title("Dataset Statistics")
    path = find_dataset()
    if not path:
        st.error("Dataset not found.")
    else:
        df = load_dataset(path, Path(path).stat().st_mtime)
        a, b = st.columns(2)
        a.metric("Rows", f"{len(df):,}")
        b.metric("Columns", len(df.columns))

        st.dataframe(pd.DataFrame({
            "Column": df.columns,
            "Non-missing values": [int(df[c].notna().sum()) for c in df.columns],
            "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
            "Data type": [str(df[c].dtype) for c in df.columns],
        }), width="stretch", hide_index=True)

        candidates = [c for c in df.columns if df[c].nunique(dropna=True) <= 100]
        chosen = st.selectbox("Inspect categorical distribution",
                              candidates or list(df.columns))
        counts = (df[chosen].fillna("(missing)").astype(str)
                  .value_counts().head(20)
                  .rename_axis(chosen).reset_index(name="Count"))
        st.plotly_chart(
            px.bar(counts, x="Count", y=chosen, orientation="h",
                   title=f"Top values in {chosen}"),
            width="stretch",
        )
        st.dataframe(counts, width="stretch", hide_index=True)


# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------
st.markdown(
    '<div class="footer">DEBUGMINER · APRIORI-BASED BUG PATTERN MINING · '
    'SUPPORT · CONFIDENCE · LIFT</div>',
    unsafe_allow_html=True,
)