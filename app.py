
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Women's Diabetes Classification",
    page_icon="💜",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #faf7ff;
    }

    h1, h2, h3 {
        color: #6a1b9a;
    }

    div[data-testid="stMetric"] {
        background-color: #f0e6ff;
        padding: 15px;
        border-radius: 12px;
    }

    .stButton > button {
        background-color: #7b2cbf;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }

    .stButton > button:hover {
        background-color: #5a189a;
        color: white;
    }

    .notice {
        background-color: #f0e6ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #7b2cbf;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL, SCALER AND DATASET
# --------------------------------------------------
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
df = pd.read_csv("diabetes.csv")
# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.title("💜 Women's Diabetes Classification System")
col15, col16 = st.columns([1,1])
with col15:
    with st.expander("🚗 About Model"):
        st.markdown("""An educational machine-learning application that explores diabetes classification and demonstrates how trained models make predictions from selected health-related measurements. """)
        st.markdown("""<div class="notice">
        <b>Important:</b> This is an educational prototype, not a medical diagnostic tool. It has not been clinically validated for Ugandan women. Predictions must not replace professional medical assessment. </div> """, unsafe_allow_html=True)


    with st.expander("📞 Contacts"):
        st.write("For more information about this project, contact the development team.")
        st.write("📧 Email: carpriceprediction@example.com")
        st.write("📱 Phone: +256 XXX XXX XXX")

with col16:
    st.image('UICT.jpg', caption='@uict', width=140)




# --------------------------------------------------
# CHECK DATASET COLUMNS
# --------------------------------------------------

feature_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]
# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "🩺 Classification",
    "📊 Dataset Exploration",
    "🤖 Model Performance"
])

# ==================================================
# TAB 1: CLASSIFICATION
# ==================================================

with st.form("diabetes_form"):

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=30,
            value=1,
            step=1,
            help="Number of times the patient has been pregnant. Enter 0 if never pregnant."
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0.0,
            max_value=400.0,
            value=120.0,
            step=1.0,
            help="Plasma glucose concentration (mg/dL) measured 2 hours after an oral glucose tolerance test. "
                 "Below 140 is normal; 140-199 suggests pre-diabetes; 200+ is in the diabetic range."
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            max_value=250.0,
            value=70.0,
            step=1.0,
            help="Diastolic blood pressure (mm Hg), the lower number in a blood pressure reading. "
                 "Around 60-80 is normal."
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0.0,
            max_value=150.0,
            value=20.0,
            step=1.0,
            help="Triceps skinfold thickness (mm), measured on the back of the upper arm. "
                 "It is used to estimate body fat."
        )

    with col2:
        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            max_value=1000.0,
            value=80.0,
            step=1.0,
            help="2-hour serum insulin level (mu U/ml). Insulin is the hormone that moves "
                 "sugar from the blood into the cells."
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=0.1,
            help="Body Mass Index = weight (kg) / height (m)². "
                 "Under 18.5 underweight, 18.5-24.9 normal, 25-29.9 overweight, 30+ obese."
        )

        diabetes_pedigree_function = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.01,
            help="A score showing how likely diabetes is based on family history. "
                 "A higher value means more relatives with diabetes, and closer ones."
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=120,
            value=30,
            step=1,
            help="Age of the patient in years."
        )

    submitted = st.form_submit_button(
        "Classify",
        use_container_width=True
    )

    if submitted:

        # Keep feature names and order consistent with training
        input_data = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree_function,
                age
            ]],
            columns=feature_names
        )

        # Apply the same fitted scaler used during training
        input_scaled = scaler.transform(input_data)

        # Predict outcome
        prediction = model.predict(input_scaled)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.warning(
                "The model classified this input as Outcome 1 "
                "(This implies that diabetes is positive)."
            )
        else:
            st.success(
                "The model classified this input as Outcome 0 "
                "(This implies that diabetes is negative)."
            )

        # Display probability only if the model supports it
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_scaled)[0]

            positive_classes = np.where(model.classes_ == 1)[0]

            if len(positive_classes) > 0:
                positive_probability = probabilities[
                    positive_classes[0]
                ]

                st.metric(
                    "Model-estimated probability for Outcome 1",
                    f"{positive_probability * 100:.1f}%"
                )

                st.caption(
                    "This is a model estimate, not a confirmed "
                    "individual medical risk."
                )

# ==================================================
# TAB 2: DATASET EXPLORATION
# ==================================================

with tab2:

    st.header("Dataset Exploration")

    st.write(
        "Explore the dataset used for the diabetes classification project."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(df))

    with col2:
        st.metric("Input Features", len(feature_names))

    with col3:
        st.metric("Dataset Columns", len(df.columns))

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Dataset Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.write("**Data types**")
        st.dataframe(
            df.dtypes.astype(str).rename("Data Type"),
            use_container_width=True
        )

    with info_col2:
        st.write("**Missing values**")
        missing_values = df.isnull().sum().rename("Missing Values")
        st.dataframe(
            missing_values,
            use_container_width=True
        )

    st.subheader("Outcome Distribution")

    outcome_counts = (
        df["Outcome"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    outcome_chart = pd.DataFrame({
        "Outcome": ["0 — Negative", "1 — Positive"],
        "Records": [
            int(outcome_counts.loc[0]),
            int(outcome_counts.loc[1])
        ]
    })

    st.bar_chart(
        outcome_chart.set_index("Outcome")
    )

    st.subheader("Feature Distribution")

    selected_feature = st.selectbox(
        "Choose a feature",
        feature_names
    )

    st.bar_chart(
        df[selected_feature].value_counts(
            bins=20
        ).sort_index()
    )

    st.subheader("Descriptive Statistics")

    st.dataframe(
        df[feature_names].describe().round(2),
        use_container_width=True
    )

# ==================================================
# TAB 3: MODEL PERFORMANCE
# ==================================================

with tab3:

    st.header("Model Performance")

    st.write("""
    The following are the previously reported test-set accuracy results
    for the four models evaluated in this project.
    """)

    performance_data = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Support Vector Machine"
        ],
        "Accuracy (%)": [
            71.43,
            79.22,
            75.97,
            75.32
        ],
        "Precision (Positive Class)": [
            0.61,
            0.70,
            0.68,
            0.66
        ],
        "Recall (Positive Class)": [
            0.52,
            0.70,
            0.59,
            0.61
        ],
        "F1 Score (Positive Class)": [
            0.56,
            0.70,
            0.63,
            0.63
        ]
    })

    st.dataframe(
        performance_data,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Accuracy Comparison")

    chart_data = performance_data[
        ["Model", "Accuracy (%)"]
    ].set_index("Model")

    st.bar_chart(chart_data)

    st.subheader("Selected Model")

    st.success(
        "Decision Tree was selected based on the highest reported "
        "test-set accuracy of 79.22% among the four evaluated models."
    )

    st.write("""
    **Interpretation:** Accuracy describes the proportion of test
    predictions that were correct. Precision, recall and F1 score
    provide additional information about positive-class performance.
    """)

    st.warning("""
    These results are the previously reported evaluation figures.
    They are displayed as project results, not recalculated from the
    currently loaded model. Model performance alone does not establish
    clinical reliability.
    """)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Women's Diabetes Classification Project | "
    "Educational machine-learning prototype"
)
# SIDEBAR — DEVELOPERS
with st.sidebar:

    st.header("👨‍💻 Developers")

    st.write("Project Development Team")

    st.markdown(
        """
        <style>
        .developer-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        .developer-table th,
        .developer-table td {
            border: 1px solid #888;
            padding: 6px;
            text-align: left;
        }

        .developer-table th {
            font-weight: bold;
        }
        </style>

        <table class="developer-table">
            <tr>
                <th>Name</th>
                <th>Registration Number</th>
            </tr>
            <tr>
                <td>YABIRAKU REAGAN</td>
                <td>2025/DSMA/DAY/1562/G</td>
            </tr>
            <tr>
                <td>NAKATE JANE</td>
                <td>2025/DSMA/DAY/1564/G</td>
            </tr>
            <tr>
                <td>KATENDE DOUGLAS</td>
                <td>2025/DSMA/DAY/1560/G</td>
            </tr>
            <tr>
                <td>AKULLU GLADYS</td>
                <td>2025/DSMA/DAY/2503/G</td>
            </tr>
            <tr>
                <td>SSEMBOWA DOMINIC VICENT</td>
                <td>2025/DSMA/DAY/1487/G</td>
            </tr>
        </table>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    with st.expander('👨‍💻Dev Info'):
        st.write('This prediction model was developed by the above students wwho are currently persuing a Diploma in Data Science Management and Analytics at Uganda Institute of Information and Communication Technology(UICT).')
        st.write('The above students created a group called GROUP C that enabled them to work together and come up with such an amazing project')

    st.caption(
         "Women's Diabetes Classification Project | "
    "Educational machine-learning prototype"
        
    )
