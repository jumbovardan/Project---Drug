import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency, f_oneway, ttest_ind
from statsmodels.stats.weightstats import ztest


st.set_page_config(
    page_title="Drug Treatment Analysis",
    page_icon="D",
    layout="wide",
)


def load_notebook_data() -> pd.DataFrame:
    """Return the complete 49-patient dataset created in Drugs.ipynb."""
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
    n = 40
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
    df["Recovered_Flag"] = df["Recovered"].map({"No": 0, "Yes": 1})
    df["Treatment_Name"] = "Drug " + df["Treatment"]
    return df


def treatment_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["Treatment", "Treatment_Name"], as_index=False)
        .agg(
            Patients=("Treatment", "size"),
            Avg_Age=("Age", "mean"),
            Avg_BP_Before=("BP_Before", "mean"),
            Avg_BP_After=("BP_After", "mean"),
            Avg_BP_Reduction=("BP_Reduction", "mean"),
            Median_BP_Reduction=("BP_Reduction", "median"),
            Avg_Pain_Score=("PainScore", "mean"),
            Avg_Recovery_Days=("RecoveryDays", "mean"),
            Recovery_Rate=("Recovered_Flag", "mean"),
        )
        .sort_values(
            ["Avg_BP_Reduction", "Recovery_Rate", "Avg_Recovery_Days"],
            ascending=[False, False, True],
        )
    )
    summary["Recovery_Rate"] = summary["Recovery_Rate"] * 100
    return summary


def run_anova(df: pd.DataFrame) -> tuple[float, float]:
    groups = [
        df.loc[df["Treatment"] == treatment, "BP_Reduction"]
        for treatment in sorted(df["Treatment"].unique())
    ]
    return f_oneway(*groups)


def run_ttest(df: pd.DataFrame) -> tuple[float, float, float, float, float]:
    drug_a = df.loc[df["Treatment"] == "A", "BP_Reduction"]
    drug_b = df.loc[df["Treatment"] == "B", "BP_Reduction"]
    t_stat, p_value = ttest_ind(drug_a, drug_b)
    a_mean = drug_a.mean()
    b_mean = drug_b.mean()
    percentage_diff = ((b_mean - a_mean) / a_mean) * 100
    return t_stat, p_value, a_mean, b_mean, percentage_diff


def run_ztest(df: pd.DataFrame, target: float = 20.0) -> tuple[float, float]:
    return ztest(df["BP_Reduction"], value=target)


def run_chi_square(df: pd.DataFrame) -> tuple[float, float, int, pd.DataFrame]:
    table = pd.crosstab(df["Treatment"], df["Recovered"])
    chi2, p_value, dof, _expected = chi2_contingency(table)
    return chi2, p_value, dof, table


def train_recovery_models(df: pd.DataFrame) -> dict:
    model_df = df.copy()
    model_df["Gender_Code"] = model_df["Gender"].map({"M": 0, "F": 1})
    model_df["Treatment_Code"] = model_df["Treatment"].map({"A": 0, "B": 1, "C": 2})

    feature_columns = [
        "Age",
        "Gender_Code",
        "Treatment_Code",
        "BP_Before",
        "BP_After",
        "PainScore",
        "RecoveryDays",
        "BP_Reduction",
    ]
    x = model_df[feature_columns]
    y = model_df["Recovered_Flag"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
    )

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(x_train, y_train)
    logistic_predictions = logistic_model.predict(x_test)

    forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
    forest_model.fit(x_train, y_train)
    forest_predictions = forest_model.predict(x_test)

    return {
        "feature_columns": feature_columns,
        "x_test": x_test,
        "y_test": y_test,
        "logistic": {
            "name": "Logistic Regression",
            "model": logistic_model,
            "predictions": logistic_predictions,
            "accuracy": accuracy_score(y_test, logistic_predictions),
            "confusion_matrix": confusion_matrix(y_test, logistic_predictions),
            "classification_report": classification_report(
                y_test,
                logistic_predictions,
                target_names=["Not Recovered", "Recovered"],
                output_dict=True,
                zero_division=0,
            ),
        },
        "random_forest": {
            "name": "Random Forest",
            "model": forest_model,
            "predictions": forest_predictions,
            "accuracy": accuracy_score(y_test, forest_predictions),
            "confusion_matrix": confusion_matrix(y_test, forest_predictions),
            "classification_report": classification_report(
                y_test,
                forest_predictions,
                target_names=["Not Recovered", "Recovered"],
                output_dict=True,
                zero_division=0,
            ),
            "feature_importance": pd.DataFrame(
                {
                    "Feature": feature_columns,
                    "Importance": forest_model.feature_importances_,
                }
            ).sort_values("Importance", ascending=False),
        },
    }


def confusion_matrix_frame(matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        matrix,
        index=["Actual Not Recovered", "Actual Recovered"],
        columns=["Predicted Not Recovered", "Predicted Recovered"],
    )


def classification_report_frame(report: dict) -> pd.DataFrame:
    return pd.DataFrame(report).transpose().round(3)


def decision_text(p_value: float, positive: str, negative: str) -> None:
    if p_value < 0.05:
        st.success(positive)
    else:
        st.info(negative)


df = load_notebook_data()
summary = treatment_summary(df)
best = summary.iloc[0]
model_results = train_recovery_models(df)

st.title("Drug Treatment Analysis")
st.caption(
    "Streamlit version of Drugs.ipynb using the complete 49-patient dataset created in the notebook."
)

metric_cols = st.columns(4)
metric_cols[0].metric("Patients", f"{len(df)}")
metric_cols[1].metric("Treatments", f"{df['Treatment'].nunique()}")
metric_cols[2].metric("Highest Avg BP Reduction", best["Treatment_Name"])
metric_cols[3].metric("Avg Reduction", f"{best['Avg_BP_Reduction']:.1f} mmHg")

st.divider()

left, right = st.columns([1.2, 0.8])

with left:
    st.subheader("Notebook Dataset")
    st.dataframe(df.drop(columns=["Recovered_Flag"]), width="stretch", hide_index=True)

with right:
    st.subheader("Data Checks")
    st.metric("Missing Values", int(df.isna().sum().sum()))
    st.metric("Duplicate Rows", int(df.duplicated().sum()))
    st.write("Shape:", df.shape)
    st.write("Columns:", ", ".join(df.drop(columns=["Recovered_Flag"]).columns))

st.subheader("Treatment Summary")
st.dataframe(
    summary.style.format(
        {
            "Avg_Age": "{:.1f}",
            "Avg_BP_Before": "{:.1f}",
            "Avg_BP_After": "{:.1f}",
            "Avg_BP_Reduction": "{:.1f}",
            "Median_BP_Reduction": "{:.1f}",
            "Avg_Pain_Score": "{:.1f}",
            "Avg_Recovery_Days": "{:.1f}",
            "Recovery_Rate": "{:.1f}%",
        }
    ),
    width="stretch",
    hide_index=True,
)

chart_cols = st.columns(2)
with chart_cols[0]:
    st.subheader("BP Reduction by Drug")
    st.bar_chart(summary.set_index("Treatment_Name")["Avg_BP_Reduction"])

with chart_cols[1]:
    st.subheader("Recovery Days by Drug")
    st.bar_chart(summary.set_index("Treatment_Name")["Avg_Recovery_Days"])

chart_cols = st.columns(2)
with chart_cols[0]:
    st.subheader("Recovery Status")
    recovery_counts = pd.crosstab(df["Treatment_Name"], df["Recovered"])
    st.bar_chart(recovery_counts)

with chart_cols[1]:
    st.subheader("Age and Recovery Days")
    scatter_data = df.rename(columns={"Age": "age", "RecoveryDays": "recovery_days"})
    st.scatter_chart(scatter_data, x="age", y="recovery_days", color="Treatment_Name")

st.subheader("Statistical Tests")

anova_f, anova_p = run_anova(df)
t_stat, t_p, a_mean, b_mean, percentage_diff = run_ttest(df)
z_stat, z_p = run_ztest(df)
chi2, chi_p, dof, contingency = run_chi_square(df)

test_cols = st.columns(2)

with test_cols[0]:
    with st.expander("ANOVA: BP reduction across Drugs A, B, and C", expanded=True):
        st.write(f"F Statistic: {anova_f:.4f}")
        st.write(f"P Value: {anova_p:.4f}")
        decision_text(
            anova_p,
            "There is a statistically significant difference between the treatments.",
            "There is no statistically significant difference between the treatments.",
        )

    with st.expander("T-Test: Drug A vs Drug B", expanded=True):
        st.write(f"T Statistic: {t_stat:.4f}")
        st.write(f"P Value: {t_p:.4f}")
        st.write(f"Drug A mean BP reduction: {a_mean:.2f}")
        st.write(f"Drug B mean BP reduction: {b_mean:.2f}")
        st.write(f"Percentage difference: {percentage_diff:.2f}%")
        decision_text(
            t_p,
            "Drugs A and B are significantly different.",
            "There is no statistically significant difference between Drugs A and B.",
        )

with test_cols[1]:
    with st.expander("Z-Test: Average BP reduction vs target of 20", expanded=True):
        st.write(f"Z Statistic: {z_stat:.4f}")
        st.write(f"P Value: {z_p:.4f}")
        decision_text(
            z_p,
            "The average BP reduction differs from the 20-unit target.",
            "The average BP reduction does not differ significantly from the 20-unit target.",
        )

    with st.expander("Chi-square: Treatment and recovery status", expanded=True):
        st.dataframe(contingency, width="stretch")
        st.write(f"Chi-square: {chi2:.4f}")
        st.write(f"P Value: {chi_p:.4f}")
        st.write(f"Degrees of freedom: {dof}")
        decision_text(
            chi_p,
            "Recovery status depends on treatment.",
            "Recovery status does not depend significantly on treatment.",
        )

st.subheader("Machine Learning Models")
st.caption(
    "Recovery prediction models from the notebook using encoded Gender, Treatment, "
    "vital measurements, pain score, recovery days, and BP reduction."
)

model_metric_cols = st.columns(3)
model_metric_cols[0].metric("Training Rows", f"{len(df) - len(model_results['y_test'])}")
model_metric_cols[1].metric("Test Rows", f"{len(model_results['y_test'])}")
model_metric_cols[2].metric(
    "Best Model",
    (
        "Random Forest"
        if model_results["random_forest"]["accuracy"]
        >= model_results["logistic"]["accuracy"]
        else "Logistic Regression"
    ),
)

model_tabs = st.tabs(["Logistic Regression", "Random Forest", "Model Comparison"])

with model_tabs[0]:
    logistic = model_results["logistic"]
    st.metric("Accuracy", f"{logistic['accuracy'] * 100:.1f}%")
    cols = st.columns(2)
    with cols[0]:
        st.write("Confusion Matrix")
        st.dataframe(
            confusion_matrix_frame(logistic["confusion_matrix"]),
            width="stretch",
        )
    with cols[1]:
        st.write("Classification Report")
        st.dataframe(
            classification_report_frame(logistic["classification_report"]),
            width="stretch",
        )

with model_tabs[1]:
    random_forest = model_results["random_forest"]
    st.metric("Accuracy", f"{random_forest['accuracy'] * 100:.1f}%")
    cols = st.columns(2)
    with cols[0]:
        st.write("Confusion Matrix")
        st.dataframe(
            confusion_matrix_frame(random_forest["confusion_matrix"]),
            width="stretch",
        )
    with cols[1]:
        st.write("Feature Importance")
        st.dataframe(random_forest["feature_importance"], width="stretch", hide_index=True)
        st.bar_chart(random_forest["feature_importance"].set_index("Feature")["Importance"])

    st.write("Classification Report")
    st.dataframe(
        classification_report_frame(random_forest["classification_report"]),
        width="stretch",
    )

with model_tabs[2]:
    comparison = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Random Forest"],
            "Accuracy": [
                model_results["logistic"]["accuracy"],
                model_results["random_forest"]["accuracy"],
            ],
        }
    )
    st.dataframe(
        comparison.style.format({"Accuracy": "{:.1%}"}),
        width="stretch",
        hide_index=True,
    )
    st.bar_chart(comparison.set_index("Model")["Accuracy"])
    st.info(
        "These models predict the notebook target variable `Recovered`. Because the "
        "dataset has 49 rows, model accuracy should be read as a small-data result, "
        "not as a clinical guarantee."
    )

st.subheader("Recommendation")
st.info(
    f"{best['Treatment_Name']} has the highest average BP reduction in the given "
    f"notebook data ({best['Avg_BP_Reduction']:.1f} mmHg). Because the statistical "
    "tests do not prove a significant treatment difference, this "
    "should be treated as an observed trend rather than proof that one drug is "
    "definitively better."
)
