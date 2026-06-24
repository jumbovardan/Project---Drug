import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import f_oneway, ttest_ind, chi2_contingency
from statsmodels.stats.weightstats import ztest


st.set_page_config(
    page_title="Drug Treatment Analysis",
    page_icon="app",
    layout="wide",
)


def create_notebook_dataset() -> pd.DataFrame:
    data = {
        "Age": [45, 50, 42, 48, 52, 46, 44, 49, 51],
        "Gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M"],
        "Treatment": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "BP_Before": [150, 160, 145, 155, 165, 150, 158, 162, 170],
        "BP_After": [135, 140, 130, 125, 130, 120, 118, 122, 125],
        "PainScore": [3, 2, 4, 1, 2, 1, 1, 2, 1],
        "RecoveryDays": [10, 12, 9, 7, 8, 6, 5, 6, 5],
        "Recovered": ["No", "No", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes"],
    }
    df = pd.DataFrame(data)

    np.random.seed(42)
    n = 39
    new_data = {
        "Age": np.random.randint(35, 65, n),
        "Gender": np.random.choice(["M", "F"], n),
        "Treatment": np.random.choice(["A", "B", "C"], n),
        "BP_Before": np.random.randint(140, 180, n),
        "BP_After": np.random.randint(110, 150, n),
        "PainScore": np.random.randint(1, 5, n),
        "RecoveryDays": np.random.randint(4, 14, n),
        "Recovered": np.random.choice(["Yes", "No"], n),
    }
    new_df = pd.DataFrame(new_data)

    df = pd.concat([df, new_df], ignore_index=True)
    df["BP_Reduction"] = df["BP_Before"] - df["BP_After"]
    return df


def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return create_notebook_dataset()

    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Treatment" in df.columns:
        drug_map = {
            "A": "Drug A",
            "B": "Drug B",
            "C": "Drug C",
            "0": "Drug A",
            "1": "Drug B",
            "2": "Drug C",
        }
        df["drug"] = (
            df["Treatment"].astype(str).str.strip().replace(drug_map)
        )

    if "drug" in df.columns:
        df["drug"] = (
            df["drug"].astype(str)
            .str.strip()
            .replace({"A": "Drug A", "B": "Drug B", "C": "Drug C"})
        )

    if {"BP_Before", "BP_After"}.issubset(df.columns):
        df["bp_reduction"] = df["BP_Before"] - df["BP_After"]

    if "BP_Reduction" in df.columns and "bp_reduction" not in df.columns:
        df["bp_reduction"] = df["BP_Reduction"]

    if "RecoveryDays" in df.columns:
        df = df.rename(columns={"RecoveryDays": "recovery_days"})

    if "recovered" not in df.columns and "Recovered" in df.columns:
        df["recovered"] = (
            df["Recovered"].astype(str)
            .str.strip()
            .str.lower()
            .isin(["yes", "y", "true", "1", "t"])
        )

    if "recovered" in df.columns and df["recovered"].dtype == object:
        df["recovered"] = (
            df["recovered"].astype(str)
            .str.strip()
            .str.lower()
            .isin(["yes", "y", "true", "1", "t"])
        )

    if "Treatment" not in df.columns and "drug" in df.columns:
        treatment_map = {
            "Drug A": "A",
            "Drug B": "B",
            "Drug C": "C",
            "A": "A",
            "B": "B",
            "C": "C",
        }
        df["Treatment"] = df["drug"].astype(str).map(treatment_map)

    if "side_effects" not in df.columns:
        df["side_effects"] = "Unknown"

    return df


def validate_data(df: pd.DataFrame) -> list[str]:
    if df is None:
        return ["No dataset uploaded. Please upload the actual 48-patient dataset to continue."]

    required_columns = {
        "drug",
        "bp_reduction",
        "recovery_days",
        "recovered",
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        return [f"Missing required column(s): {', '.join(missing)}"]
    return []


def summarize_drugs(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("drug", as_index=False)
        .agg(
            patients=("drug", "size"),
            avg_bp_reduction=("bp_reduction", "mean"),
            median_bp_reduction=("bp_reduction", "median"),
            avg_recovery_days=("recovery_days", "mean"),
            recovery_rate=("recovered", "mean"),
        )
        .sort_values(
            ["recovery_rate", "avg_bp_reduction", "avg_recovery_days"],
            ascending=[False, False, True],
        )
    )
    summary["recovery_rate"] = summary["recovery_rate"] * 100
    return summary


def score_drugs(summary: pd.DataFrame) -> pd.DataFrame:
    scored = summary.copy()
    scored["bp_score"] = scored["avg_bp_reduction"].rank(pct=True)
    scored["recovery_score"] = (1 / scored["avg_recovery_days"]).rank(pct=True)
    scored["outcome_score"] = scored["recovery_rate"].rank(pct=True)
    scored["overall_score"] = (
        scored["bp_score"] * 0.4
        + scored["recovery_score"] * 0.25
        + scored["outcome_score"] * 0.35
    )
    return scored.sort_values("overall_score", ascending=False)


def run_anova(df: pd.DataFrame) -> tuple[float, float]:
    a = df[df["Treatment"] == "A"]["bp_reduction"]
    b = df[df["Treatment"] == "B"]["bp_reduction"]
    c = df[df["Treatment"] == "C"]["bp_reduction"]
    return f_oneway(a, b, c)


def run_ttest(df: pd.DataFrame) -> tuple[float, float, float, float, float]:
    a = df[df["Treatment"] == "A"]["bp_reduction"]
    b = df[df["Treatment"] == "B"]["bp_reduction"]
    t_stat, p_value = ttest_ind(a, b, equal_var=False)
    a_mean = a.mean()
    b_mean = b.mean()
    pct_diff = ((b_mean - a_mean) / a_mean) * 100 if a_mean != 0 else 0.0
    return t_stat, p_value, a_mean, b_mean, pct_diff


def run_ztest(df: pd.DataFrame, value: float = 20.0) -> tuple[float, float]:
    return ztest(df["bp_reduction"], value=value)


def run_chi2(df: pd.DataFrame) -> tuple[float, float, int, pd.DataFrame]:
    if "Recovered" in df.columns:
        recovered_series = df["Recovered"].astype(str)
    else:
        recovered_series = df["recovered"].map({True: "Yes", False: "No"})
    table = pd.crosstab(df["Treatment"], recovered_series)
    chi2, p_value, dof, expected = chi2_contingency(table)
    return chi2, p_value, dof, table


def display_statistical_tests(df: pd.DataFrame) -> None:
    anova_f, anova_p = run_anova(df)
    t_stat, t_p, a_mean, b_mean, pct_diff = run_ttest(df)
    z_stat, z_p = run_ztest(df, value=20.0)
    chi2, chi2_p, dof, contingency = run_chi2(df)

    st.subheader("Statistical Tests")
    with st.expander("ANOVA: BP reduction across treatments"):
        st.write(f"F statistic: {anova_f:.4f}")
        st.write(f"P value: {anova_p:.4f}")
        if anova_p < 0.05:
            st.success("There is a significant difference between treatments.")
        else:
            st.info("No statistically significant difference between treatments.")

    with st.expander("T-Test: Treatment A vs Treatment B"):
        st.write(f"T statistic: {t_stat:.4f}")
        st.write(f"P value: {t_p:.4f}")
        st.write(f"Treatment A mean bp reduction: {a_mean:.2f}")
        st.write(f"Treatment B mean bp reduction: {b_mean:.2f}")
        st.write(f"Percentage difference: {pct_diff:.2f}%")
        if t_p < 0.05:
            st.success("The difference between Treatment A and Treatment B is statistically significant.")
        else:
            st.info("The difference between Treatment A and Treatment B is not statistically significant.")

    with st.expander("Z-Test: Average BP reduction vs 20"):
        st.write(f"Z statistic: {z_stat:.4f}")
        st.write(f"P value: {z_p:.4f}")
        if z_p < 0.05:
            st.success("The average BP reduction differs from 20 units.")
        else:
            st.info("The average BP reduction does not differ significantly from 20 units.")

    with st.expander("Chi-square: Treatment vs recovery status"):
        st.write(contingency)
        st.write(f"Chi-square: {chi2:.4f}")
        st.write(f"P value: {chi2_p:.4f}")
        if chi2_p < 0.05:
            st.success("Recovery status depends on treatment.")
        else:
            st.info("Recovery status does not depend significantly on treatment.")


st.title("Drug Treatment Analysis")
st.caption(
    "Evaluate Drugs A, B, and C using blood pressure reduction, recovery time, "
    "and patient recovery outcomes."
)

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader(
        "Upload the actual 48-patient CSV or Excel dataset", type=["csv", "xlsx"]
    )
    st.caption(
        "Expected columns: drug, bp_reduction, recovery_days, recovered, side_effects. "
        "Alternative column names such as Treatment, BP_Before, BP_After, and Recovered "
        "are also accepted."
    )

df = load_data(uploaded_file)
if df is not None:
    df = normalize_data(df)
errors = validate_data(df)

if errors:
    st.error(errors[0])
    st.stop()

df["recovered"] = df["recovered"].astype(bool)
summary = summarize_drugs(df)
scored = score_drugs(summary)
best_drug = scored.iloc[0]

metric_cols = st.columns(4)
metric_cols[0].metric("Patients", f"{len(df):,}")
metric_cols[1].metric("Best Drug", best_drug["drug"])
metric_cols[2].metric(
    "Avg BP Reduction",
    f"{best_drug['avg_bp_reduction']:.1f} mmHg",
)
metric_cols[3].metric("Recovery Rate", f"{best_drug['recovery_rate']:.1f}%")

st.divider()

left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("Drug Comparison")
    st.dataframe(
        summary.style.format(
            {
                "avg_bp_reduction": "{:.1f}",
                "median_bp_reduction": "{:.1f}",
                "avg_recovery_days": "{:.1f}",
                "recovery_rate": "{:.1f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(
        summary.set_index("drug")[["avg_bp_reduction", "avg_recovery_days"]],
        use_container_width=True,
    )

with right:
    st.subheader("Recommendation")
    st.success(
        f"{best_drug['drug']} is the strongest option in this analysis, with "
        f"{best_drug['avg_bp_reduction']:.1f} mmHg average blood pressure reduction, "
        f"{best_drug['avg_recovery_days']:.1f} average recovery days, and "
        f"{best_drug['recovery_rate']:.1f}% recovery rate."
    )

    side_effects = pd.crosstab(df["drug"], df["side_effects"], normalize="index") * 100
    st.subheader("Side Effects")
    st.dataframe(side_effects.round(1), use_container_width=True)

st.subheader("Statistical Tests")
display_statistical_tests(df)

st.subheader("Patient Data")
selected_drug = st.multiselect(
    "Filter by drug",
    options=sorted(df["drug"].unique()),
    default=sorted(df["drug"].unique()),
)
filtered = df[df["drug"].isin(selected_drug)]
st.dataframe(filtered, use_container_width=True, hide_index=True)
