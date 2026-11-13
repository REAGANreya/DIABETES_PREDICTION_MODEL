# 💜 Women's Diabetes Classification System

An interactive machine learning application that uses health-related measurements to classify diabetes outcomes in women. The project combines data analysis, machine learning, and interactive visualization through a Streamlit web application.

## 📌 Project Overview

The Women's Diabetes Classification System is a data science project developed to demonstrate how machine learning can be applied to diabetes classification.

The system accepts eight health-related input variables and uses a trained classification model to predict whether an observation belongs to the positive or negative diabetes outcome class.

Beyond prediction, the application provides an interactive environment for exploring the dataset and comparing the performance of different machine learning algorithms.

## 🎯 Project Objectives

- Develop a machine learning model for diabetes classification.
- Perform exploratory data analysis to understand the dataset.
- Apply data preprocessing and feature scaling.
- Train and evaluate multiple classification algorithms.
- Compare model performance using evaluation metrics.
- Develop an interactive Streamlit application for predictions and data exploration.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning and preprocessing |
| Streamlit | Interactive web application |
| Joblib | Model and scaler serialization |
| Matplotlib / Seaborn | Data visualization and analysis |
| Jupyter Notebook | Model development and experimentation |

## 📊 Dataset Description

The project uses `diabetes.csv`, which contains health-related measurements and a target variable named `Outcome`.

The eight input features are:

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skinfold thickness |
| Insulin | Two-hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Family-history-related diabetes score |
| Age | Age in years |

**Target variable:**

- `0` – Negative diabetes outcome.
- `1` – Positive diabetes outcome.

## ⚙️ Machine Learning Workflow

The project follows a structured machine learning workflow:

1. **Data Collection:** Load the diabetes dataset.
2. **Data Exploration:** Examine dataset structure, distributions, descriptive statistics and outcome distribution.
3. **Data Preprocessing:** Handle invalid zero values in selected medical measurements using missing-value treatment and median imputation.
4. **Feature Scaling:** Apply StandardScaler to standardize input features.
5. **Model Training:** Train and evaluate multiple classification algorithms.
6. **Model Evaluation:** Compare models using accuracy, precision, recall and F1-score.
7. **Model Selection:** Select the Decision Tree classifier based on the reported evaluation results.
8. **Deployment:** Integrate the trained model and scaler into a Streamlit application.

## 🤖 Models Evaluated

Four machine learning algorithms were evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

### Model Performance

The following are the previously reported test-set evaluation results.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 71.43% | 0.61 | 0.52 | 0.56 |
| Decision Tree | **79.22%** | **0.70** | **0.70** | **0.70** |
| Random Forest | 75.97% | 0.68 | 0.59 | 0.63 |
| SVM | 75.32% | 0.66 | 0.61 | 0.63 |

**Selected Model: Decision Tree Classifier**

The Decision Tree achieved the highest reported test accuracy of 79.22% among the four evaluated algorithms.

## 🖥️ Application Features

The Streamlit application contains three main sections:

### 1. Diabetes Classification
- Accepts eight health-related input variables.
- Applies the previously fitted feature scaler.
- Generates a classification prediction.
- Displays the predicted outcome.
- Displays the model-estimated positive-class probability when supported by the model.

### 2. Dataset Exploration
- Displays total dataset records.
- Shows the number of input features and dataset columns.
- Provides a dataset preview.
- Displays data types and missing-value information.
- Visualizes diabetes outcome distribution.
- Allows users to explore individual feature distributions.
- Presents descriptive statistics.

### 3. Model Performance
- Compares the four evaluated classification algorithms.
- Displays accuracy, precision, recall and F1-score.
- Provides a model accuracy comparison chart.
- Identifies the selected model.

## 📁 Repository Structure

```text
DIABETES_PREDICTION_MODEL/
│
├── app.py
├── diabetes.csv
├── model.joblib
├── scaler.joblib
├── Untitled.ipynb
├── UICT.jpg
└── README.md
```

| File | Description |
|---|---|
| `app.py` | Main Streamlit application |
| `diabetes.csv` | Dataset used in the project |
| `model.joblib` | Serialized trained classification model |
| `scaler.joblib` | Serialized fitted StandardScaler |
| `Untitled.ipynb` | Jupyter Notebook for model development |
| `UICT.jpg` | Institutional image used in the application |
| `README.md` | Project documentation |

## 🚀 Installation and Setup

### Prerequisites

Ensure that Python and pip are installed on your computer.

### Step 1: Clone the repository

```bash
git clone https://github.com/REAGANreya/DIABETES_PREDICTION_MODEL.git
```

### Step 2: Navigate into the project directory

```bash
cd DIABETES_PREDICTION_MODEL
```

### Step 3: Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it:

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### Step 4: Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

### Step 5: Run the application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

**Note:** The serialized model and scaler must be compatible with the installed scikit-learn environment. Use trusted model files when loading Joblib artifacts.

## ⚠️ Medical Disclaimer

This project is an educational machine learning prototype developed for data science learning and demonstration purposes.

It is not a clinically validated diagnostic system and has not been established as reliable for clinical use among Ugandan women or any other specific population.

Its predictions should not be interpreted as confirmed medical diagnoses or individual medical risk assessments. Users should consult qualified healthcare professionals for proper medical testing and advice.

## 👨‍💻 Developer

**YABIRAKU REAGAN**

Data Science Student | Machine Learning | Data Analytics | Python | SQL | Streamlit

GitHub: [@REAGANreya](https://github.com/REAGANreya)

## 🎓 Academic Context

Developed as a data science and machine learning project at the Uganda Institute of Information and Communications Technology (UICT).

---

*Building practical data science solutions through data analysis, machine learning and interactive applications.*

**If you find this project useful, consider giving the repository a ⭐.**
