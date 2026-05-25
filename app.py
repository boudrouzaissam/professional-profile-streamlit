import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency

st.set_page_config(
    page_title="Professional Profile Modeling",
    page_icon="📊",
    layout="wide",
)

st.title("Professional Profile Modeling")
st.caption("English Streamlit version of the MPF document")

st.markdown(
    """
This interactive application presents a quantitative and visual modeling of a professional profile for an economist / data analyst role. 
The objective is to translate the original report into English and make it easier to present through a Streamlit dashboard.
"""
)

# --------------------------
# Data
# --------------------------
model_terms = pd.DataFrame({
    "Variable": ["Constant", "Education (E)", "Experience (Ex)", "Analytical & creativity skills (A)", "Technical / communication skills (C)", "Mismatch with economist role (I)", "Education lag 1", "Education lag 2", "Experience lag 1", "Experience lag 2", "Analytical skills lag 1", "Analytical skills lag 2", "Skills lag 1", "Skills lag 2"],
    "Coefficient": [1.3188, 2.0444, 1.4871, 1.3537, 1.6764, -1.9909, 1.8371, 1.5563, 1.2242, 1.1472, 1.3319, 0.0509, 1.4628, 1.1755],
    "p-value": [0.125, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.008, 0.000, 0.000],
    "Interpretation": ["Not statistically significant", "Positive contribution", "Strong positive contribution", "Positive contribution", "Positive contribution", "Negative effect", "Positive delayed effect", "Positive delayed effect", "Positive delayed effect", "Positive delayed effect", "Positive delayed effect", "Positive but weak delayed effect", "Positive delayed contribution", "Positive delayed contribution"]
})

model_stats = pd.DataFrame({
    "Indicator": ["R-squared", "Adjusted R-squared", "F-statistic", "Prob(F-statistic)", "Durbin-Watson", "Omnibus probability", "Jarque-Bera probability"],
    "Value": ["0.994", "0.994", "1777", "4.61e-143", "2.135", "0.066", "0.0824"],
    "Interpretation": ["Excellent fit", "Robust model fit", "Overall model is significant", "Explanatory variables are jointly significant", "No strong evidence of autocorrelation", "Slight deviation from normality", "Acceptable normality"]
})

months = np.arange(0, 13)
impact_df = pd.DataFrame({
    "Month after hiring": months,
    "Productivity": [50, 55, 60, 65, 70, 75, 80, 85, 90, 96, 100, 105, 110],
    "Stakeholder satisfaction": [60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108],
    "Project monitoring": [55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91],
})
impact_long = impact_df.melt(id_vars="Month after hiring", var_name="Indicator", value_name="Performance (%)")

x = np.linspace(0, 100, 101)
elasticity_df = pd.DataFrame({"Professional profile improvement (%)": x, "Economist performance (%)": (x/100) ** 1.55 * 100})

np.random.seed(7)
other_candidates = pd.DataFrame({
    "Candidate": "Other candidates",
    "Job fit": np.random.normal(50, 9, 45).clip(28, 72),
    "Potential performance": np.random.normal(62, 14, 45).clip(20, 88)
})
aissam = pd.DataFrame({"Candidate": ["Aissam Boudrouz"], "Job fit": [90], "Potential performance": [95]})
scatter_df = pd.concat([other_candidates, aissam], ignore_index=True)

chi_table = pd.DataFrame({
    "Competency": ["Analytical skills", "Applied economics", "Writing", "Creativity", "Synthesis", "Data analysis", "French and English", "Project management"],
    "Matched with economist role": [20, 18, 16, 15, 14, 17, 20, 19],
    "Not matched": [0, 0, 0, 0, 0, 0, 0, 0]
})

# Chi-square test requires positive expected counts; the original table has zeros in one column.
# For a stable demonstration, we add a small continuity value only for computation.
chi_for_calc = chi_table[["Matched with economist role", "Not matched"]].replace(0, 0.5)
chi2, p, dof, expected = chi2_contingency(chi_for_calc)

# --------------------------
# Sidebar
# --------------------------
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Model specification",
        "Regression results",
        "Impact projection",
        "Elasticity relationship",
        "Candidate benchmarking",
        "Chi-square test",
        "Academic & professional timeline",
        "Conclusion",
    ],
)

# --------------------------
# Sections
# --------------------------
if section == "Overview":
    st.header("1. Overview")
    st.markdown(
        """
The dashboard models a professional profile using education, professional experience, analytical capacity, technical skills, communication, and the degree of mismatch between the profile and an economist position.

The original document included: a document plan, a profile model, regression output, interpretation of the coefficients, several figures, and a chi-square test between the professional profile and the economist role.
"""
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", "0.994")
    c2.metric("F-statistic", "1777")
    c3.metric("Durbin-Watson", "2.135")
    c4.metric("JB p-value", "0.0824")

elif section == "Model specification":
    st.header("2. Model specification")
    st.latex(r"P_t = \beta_0 + \sum_{p=1}^{N}\beta_p X_{p,t} + \sum_{p=1}^{M}\alpha_p X_{p,t-1} + \varepsilon_t")
    st.markdown(
        """
Where:

- **P** represents the professional profile score.
- **X** represents explanatory variables affecting the professional profile.
- **E** captures education level.
- **Ex** captures professional experience.
- **A** captures analytical thinking and creativity.
- **C** captures technical, data analysis, software, communication, and personal skills.
- **I** captures the mismatch between the profile and the economist position.
- **β** and **α** are coefficients for contemporaneous and lagged effects.
- **ε** is the error term.
"""
    )

elif section == "Regression results":
    st.header("3. Regression results")
    st.subheader("Model quality indicators")
    st.dataframe(model_stats, use_container_width=True, hide_index=True)
    st.subheader("Coefficient interpretation")
    st.dataframe(model_terms, use_container_width=True, hide_index=True)
    fig = px.bar(model_terms[model_terms["Variable"] != "Constant"], x="Coefficient", y="Variable", orientation="h", text="Coefficient", title="Estimated contribution of profile dimensions")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

elif section == "Impact projection":
    st.header("4. Impact projection after hiring")
    st.markdown("The figure illustrates a projected improvement in productivity, stakeholder satisfaction, and project monitoring during the 12 months following hiring.")
    fig = px.line(impact_long, x="Month after hiring", y="Performance (%)", color="Indicator", markers=True, title="Projected impact on organizational performance")
    fig.add_hline(y=50, line_dash="dash", annotation_text="Initial level")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(impact_df, use_container_width=True, hide_index=True)

elif section == "Elasticity relationship":
    st.header("5. Positive elasticity relationship")
    st.markdown("This curve represents the positive relationship between professional profile improvement and expected performance in an economist role.")
    fig = px.line(elasticity_df, x="Professional profile improvement (%)", y="Economist performance (%)", title="Elasticity between profile improvement and economist performance")
    fig.add_hline(y=50, line_dash="dash", annotation_text="Reference level: 50%")
    fig.add_vline(x=50, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

elif section == "Candidate benchmarking":
    st.header("6. Candidate benchmarking")
    st.markdown("This figure compares job fit and potential performance for Aissam Boudrouz relative to other candidates.")
    fig = px.scatter(scatter_df, x="Job fit", y="Potential performance", color="Candidate", symbol="Candidate", size=np.where(scatter_df["Candidate"].eq("Aissam Boudrouz"), 16, 8), title="Job fit and potential performance")
    fig.add_hline(y=70, line_dash="dash", annotation_text="Average performance: 70%")
    fig.add_vline(x=55, line_dash="dash", annotation_text="Average fit: 55%")
    st.plotly_chart(fig, use_container_width=True)

elif section == "Chi-square test":
    st.header("7. Chi-square test between profile and economist role")
    st.markdown("The original document presents a chi-square test to assess the association between the professional competencies and the economist position.")
    st.dataframe(chi_table, use_container_width=True, hide_index=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Original chi-square", "312.45")
    c2.metric("Degrees of freedom", "7")
    c3.metric("Original p-value", "0.0000")
    st.info("Note: the original table contains a zero column for non-matching competencies. In a real statistical application, the chi-square setup should be checked carefully because expected frequencies should not be zero.")

elif section == "Academic & professional timeline":
    st.header("8. Academic and professional timeline")
    timeline = pd.DataFrame({
        "Activity": ["Master in Development and Applied Economics", "Freelance research and empirical studies", "Statistics officer", "Part-time doctoral researcher"],
        "Start": [2020, 2020, 2021, 2022],
        "End": [2020, 2025, 2024, 2025],
    })
    fig = px.timeline(timeline, x_start="Start", x_end="End", y="Activity", title="Academic and professional experience")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(timeline, use_container_width=True, hide_index=True)

elif section == "Conclusion":
    st.header("9. Conclusion")
    st.markdown(
        """
The model suggests that education, professional experience, analytical capacity, technical skills, communication, and delayed effects contribute positively to the professional profile. The mismatch indicator has a negative sign, which is consistent with the idea that lower mismatch improves suitability for an economist position.

Overall, the dashboard presents the profile as quantitatively aligned with the economist / data analyst role, while also highlighting the importance of interpreting this type of illustrative model carefully.
"""
    )

st.divider()
st.caption("Built with Streamlit, pandas, numpy, Plotly, and scipy.")
