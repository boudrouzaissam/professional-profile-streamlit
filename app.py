import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Modeling My Professional Profile",
    page_icon="📊",
    layout="wide",
)

# --------------------------
# Data
# --------------------------
model_terms = pd.DataFrame({
    "Variable": [
        "Constant", "Education (E)", "Experience (Ex)", "Analytical thinking & creativity (A)",
        "Technical and interpersonal skills (C)", "Profile mismatch / gap (I)",
        "Education lag 1", "Education lag 2", "Experience lag 1", "Experience lag 2",
        "Analytical skills lag 1", "Analytical skills lag 2", "Skills lag 1", "Skills lag 2"
    ],
    "Coefficient": [1.3188, 2.0444, 1.4871, 1.3537, 1.6764, -1.9909, 1.8371, 1.5563, 1.2242, 1.1472, 1.3319, 0.0509, 1.4628, 1.1755],
    "p-value": [0.125, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.008, 0.000, 0.000],
    "Interpretation": [
        "Baseline level; not statistically significant",
        "Positive contribution to the professional profile",
        "Strong positive contribution to the professional profile",
        "Positive contribution through analytical reasoning and creativity",
        "Positive contribution through technical, communication, and personal skills",
        "Negative effect: a larger gap reduces profile coherence",
        "Positive delayed effect of education",
        "Positive delayed effect of education",
        "Positive delayed effect of experience",
        "Positive delayed effect of experience",
        "Positive delayed effect of analytical skills",
        "Positive but weaker delayed effect",
        "Positive delayed effect of skills",
        "Positive delayed effect of skills",
    ]
})

model_stats = pd.DataFrame({
    "Indicator": [
        "R-squared", "Adjusted R-squared", "F-statistic", "Prob(F-statistic)",
        "Durbin-Watson", "Omnibus probability", "Jarque-Bera probability"
    ],
    "Value": ["0.994", "0.994", "1777", "4.61e-143", "2.135", "0.066", "0.0824"],
    "Meaning for the profile model": [
        "Very high explanatory power for the constructed profile score",
        "The model remains strong after adjusting for the number of variables",
        "The full set of profile dimensions is jointly meaningful",
        "The explanatory variables are statistically relevant as a group",
        "No strong evidence of residual autocorrelation",
        "Small deviation from normality, but not critical at the 5% level",
        "Residual normality remains acceptable at conventional levels",
    ]
})

months = np.arange(0, 13)
impact_df = pd.DataFrame({
    "Month": months,
    "Productivity": [50, 55, 60, 65, 70, 75, 80, 85, 90, 96, 100, 105, 110],
    "Stakeholder satisfaction": [60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108],
    "Project monitoring": [55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91],
})
impact_long = impact_df.melt(id_vars="Month", var_name="Indicator", value_name="Index")

x = np.linspace(0, 100, 101)
elasticity_df = pd.DataFrame({
    "Professional profile development (%)": x,
    "Profile performance index (%)": (x / 100) ** 1.55 * 100,
})

np.random.seed(7)
comparison_df = pd.DataFrame({
    "Group": "Reference group",
    "Profile coherence": np.random.normal(50, 9, 45).clip(28, 72),
    "Potential contribution": np.random.normal(62, 14, 45).clip(20, 88),
})
my_profile = pd.DataFrame({
    "Group": ["My profile"],
    "Profile coherence": [90],
    "Potential contribution": [95]
})
comparison_df = pd.concat([comparison_df, my_profile], ignore_index=True)
comparison_df["Marker size"] = np.where(comparison_df["Group"].eq("My profile"), 18, 8)

chi_table = pd.DataFrame({
    "Profile dimension": [
        "Analytical skills", "Applied economics", "Writing", "Creativity",
        "Synthesis", "Data analysis", "French and English", "Project management"
    ],
    "Present / aligned": [20, 18, 16, 15, 14, 17, 20, 19],
    "Gap / not aligned": [0, 0, 0, 0, 0, 0, 0, 0]
})

# --------------------------
# Header
# --------------------------
st.title("Modeling My Professional Profile")
st.caption("A quantitative dashboard presenting the structure, evolution, and coherence of my professional profile")

st.markdown(
    """
This dashboard presents my professional profile as a model.
The objective is to show how different dimensions of my background — education, experience, analytical capacity, technical skills, communication, creativity, and project-oriented work — interact to build a coherent professional profile over time.
"""
)

# --------------------------
# Sidebar
# --------------------------
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Choose a section",
    [
        "Profile overview",
        "Model specification",
        "Model results",
        "Profile development over time",
        "Elasticity of profile development",
        "Profile positioning",
        "Coherence test",
        "Conclusion",
    ],
)

# --------------------------
# Sections
# --------------------------
if section == "Profile overview":
    st.header("1. Profile overview")
    st.markdown(
        """
The professional profile is represented as a dynamic combination of several dimensions. Each dimension contributes to the overall profile score either immediately or with a delayed effect.

This model is based on accumulated education, practical experience, analytical thinking, technical capacity, communication, adaptability, and the ability to transform knowledge into measurable contribution.
"""
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model fit", "R² = 0.994")
    c2.metric("Global significance", "F = 1777")
    c3.metric("Profile coherence", "High")
    c4.metric("Main gap variable", "Negative effect")

    st.subheader("Core profile anchors")
    st.markdown(
        """
- **PhD candidate in Economics**, Cadi Ayyad University.
- **MBA**, University of Quebec, 2026.
- Quantitative profile combining economics, data analysis, applied research, and project-oriented work.
"""
    )

elif section == "Model specification":
    st.header("2. Model specification")
    st.latex(r"P_t = \beta_0 + \sum_{p=1}^{N}\beta_p X_{p,t} + \sum_{p=1}^{M}\alpha_p X_{p,t-1} + \varepsilon_t")
    st.markdown(
        """
Where:

- **Pₜ** represents the professional profile score at time *t*.
- **Xₚ,ₜ** represents the current profile dimensions.
- **Xₚ,ₜ₋₁** represents delayed effects from previous experience, education, and skills.
- **E** captures education and academic background.
- **Ex** captures professional and research experience.
- **A** captures analytical thinking, synthesis, and creativity.
- **C** captures technical skills, data analysis, software use, communication, and interpersonal skills.
- **I** captures profile gaps or mismatch. A negative sign means that reducing the gap improves the coherence of the profile.
"""
    )

elif section == "Model results":
    st.header("3. Model results")
    st.subheader("Model quality indicators")
    st.dataframe(model_stats, use_container_width=True, hide_index=True)

    st.subheader("Contribution of each profile dimension")
    st.dataframe(model_terms, use_container_width=True, hide_index=True)

    plot_terms = model_terms[model_terms["Variable"] != "Constant"].copy()
    fig = px.bar(
        plot_terms,
        x="Coefficient",
        y="Variable",
        orientation="h",
        text="Coefficient",
        title="Estimated contribution of profile dimensions",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
The strongest positive dimensions are education, experience, skills, and delayed learning effects. The negative coefficient of the gap variable confirms that the more the profile becomes coherent, the stronger the overall profile score becomes.
"""
    )

elif section == "Profile development over time":
    st.header("4. Profile development over time")
    st.markdown(
        """
This section illustrates the expected evolution of my professional contribution over a twelve-month horizon. The purpose is not to predict an exact future value, but to show how the profile can generate increasing value through productivity, stakeholder satisfaction, and project monitoring.
"""
    )
    fig = px.line(
        impact_long,
        x="Month",
        y="Index",
        color="Indicator",
        markers=True,
        title="Projected development of professional contribution",
    )
    fig.add_hline(y=50, line_dash="dash", annotation_text="Initial reference level")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(impact_df, use_container_width=True, hide_index=True)

elif section == "Elasticity of profile development":
    st.header("5. Elasticity of profile development")
    st.markdown(
        """
The elasticity curve shows that profile development is not purely linear. At the beginning, improvement may be gradual. As education, experience, technical skills, and analytical capacity reinforce each other, the overall contribution can increase faster.
"""
    )
    fig = px.line(
        elasticity_df,
        x="Professional profile development (%)",
        y="Profile performance index (%)",
        title="Positive elasticity between profile development and contribution",
    )
    fig.add_hline(y=50, line_dash="dash", annotation_text="Reference level")
    fig.add_vline(x=50, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

elif section == "Profile positioning":
    st.header("6. Profile positioning")
    st.markdown(
        """
This visualization positions my profile compared with a reference group. The objective is to show the relative coherence of the profile and its potential contribution, without reducing the analysis to a single job title.
"""
    )
    fig = px.scatter(
        comparison_df,
        x="Profile coherence",
        y="Potential contribution",
        color="Group",
        symbol="Group",
        size="Marker size",
        title="Profile coherence and potential contribution",
    )
    fig.add_hline(y=70, line_dash="dash", annotation_text="Reference contribution level")
    fig.add_vline(x=55, line_dash="dash", annotation_text="Reference coherence level")
    st.plotly_chart(fig, use_container_width=True)

elif section == "Coherence test":
    st.header("7. Coherence test")
    st.markdown(
        """
The coherence test presents the alignment between the main profile dimensions and the overall professional positioning.
"""
    )
    st.dataframe(chi_table, use_container_width=True, hide_index=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Illustrative chi-square", "312.45")
    c2.metric("Degrees of freedom", "7")
    c3.metric("Illustrative p-value", "0.0000")
    st.info("Methodological note: because the original table contains a zero column, the result should be interpreted as an illustrative coherence test rather than a standard inferential chi-square test.")

elif section == "Conclusion":
    st.header("8. Conclusion")
    st.markdown(
        """
This dashboard presents my professional profile as a structured and dynamic model. The results highlight the positive role of education, experience, analytical thinking, creativity, technical skills, communication, and delayed learning effects.

The model also shows that reducing profile gaps improves overall coherence. In this sense, the dashboard is designed as a professional self-modeling tool: it explains how my background is structured, how its components interact, and how they can generate value in research, data analysis, economics, and project-oriented environments.
"""
    )

st.divider()
st.caption("Built with Streamlit, pandas, numpy, and Plotly.")
