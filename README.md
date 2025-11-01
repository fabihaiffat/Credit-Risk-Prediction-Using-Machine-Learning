# Credit-Risk-Prediction-Using-Machine-Learning
This project analyzes applicant and business data to classify credit risk as Good or Bad using an Extra Trees model. Includes full EDA, model training, and a Streamlit web app for real-time credit risk prediction

# Project Overview

Financial institutions face challenges in identifying applicants who are likely to default.  
This project uses achine Learning to analyze applicant data and predict creditworthiness,  helping lenders make more informed decisions

# Key Features
- Exploratory Data Analysis (EDA) with visual insights  
- Categorical & Numerical feature relationship analysis  
- Machine Learning model training with hyperparameter tuning  
- Model evaluation and comparison (Decision Tree, Random Forest, Extra Trees)  
- Streamlit app for easy user interaction and prediction

- # Dataset Description

The dataset contains various business and applicant-related attributes, including:

| Feature | Description |
|----------|--------------|
| `Business Duration (Years)` | How long the business has been operating |
| `Business Duration with RedX (Months)` | Duration of partnership with the logistics wing of the company- RedX |
| `Number of Employees` | Size of the business workforce |
| `Ownership Type` | Type of business ownership |
| `Sourcing Type` | How the applicant was sourced |
| `Risk` | Target variable — *Good* or *Bad* |

** Target distribution **  
  - Good applicant: 636  
  - Bad applicant: 364
    
# Machine Learning Workflow
1. **Data Preprocessing**
   - Encoded categorical variables using `LabelEncoder`
   - Handled numeric/categorical separation
   - Saved encoders and model via `joblib`

2. **Model Training**
   - Models Used:
     - Decision Tree  
     - Random Forest  
     - Extra Trees (Best Performing Model)
   - GridSearchCV used for hyperparameter tuning  
   - Evaluation based on Accuracy

3. **Best Model**
   - **Extra Trees Classifier**
   - Balanced class weights  
   - Saved as: `1extra_trees_credit_model.pkl`
# Model Performance

| Model | Accuracy | Best Parameters |
|--------|-----------|-----------------|
| Decision Tree | *77.0%* | {params_dt} |
| Random Forest | *83.0%* | {params_rf} |
| Extra Trees | *84.5%* | {params_et} ✅ |

**Final Model:** Extra Trees Classifier  
**Saved File:** `1extra_trees_credit_model.pkl`

# Streamlit App
A simple web app built with Streamlit- allows users to input applicant details and instantly predict credit risk.
