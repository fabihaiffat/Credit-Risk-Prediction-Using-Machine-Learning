import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("1extra_trees_credit_model.pkl") 

encoders = {
    "Ownership_type": joblib.load("Ownership type_encoder.pkl"),
    "Sex": joblib.load("Sex_encoder.pkl"),
    "Sourcing_Type": joblib.load("Sourcing Type_encoder.pkl"), 
    "Promotion_Channel": joblib.load("Promotion Channel_encoder.pkl"),
    "Business_Type": joblib.load("Business Type_encoder.pkl"),
    "Product_Type": joblib.load("Product Type_encoder.pkl"),
    "Marital_Status": joblib.load("Marital Status_encoder.pkl"),
    "Permanent_Resident": joblib.load("Permanent Resident_encoder.pkl"), 
    "Other_profession": joblib.load("Other profession_encoder.pkl"),
    "Business_Nature": joblib.load("Business Nature_encoder.pkl"),
}

st.title("Credit Risk Prediction App")
st.write("Enter applicant information to predict if credit risk is Good or Bad")

# --- Inputs ---
Trade_License = st.selectbox("Trade License", ["Yes", "No"])
Business_Duration_Years = st.number_input("Business Duration\n(Years)", min_value=0.6, max_value=30.0, value=2.0)
Availability_of_separate_office = st.selectbox("Availability of Separate Office", ["Yes", "No"])
Number_of_Business_Unit = st.number_input("Number of Business Units", min_value=1, max_value=7, value=3)
Number_of_Employee = st.number_input("Number of Employees", min_value=1, max_value=50, value=9)
Ownership_type = st.selectbox("Ownership Type", ["LTD", "Sole Proprietorships", "Corporations", "Partnerships"])
Sex = st.selectbox("Sex", ["male", "female"])
Sourcing_Type = st.selectbox("Sourcing Type", ["Foreign Source", "Local Source", "Own Product"])
Promotion_Channel = st.selectbox("Promotion Channel", ["Online only (FB + Other)", "Only Facebook", "Only Offline", "Online & Offline"])
Business_Type = st.selectbox("Business Type", ["E-commerce (website) only", "F-commerce only", "F-commerce+ E-commerce",
                                                "F-commerce+ E-commerce+ Wholesale", "Offline showroom+ F-commerce+ E-commerce",
                                                "Offline Showroom+ F-commerce+ E-commerce+ Wholesale"])
Product_Type = st.selectbox("Product Type", ["Clothing & Accessories", "Electronics & Gadget", "Cosmetics", "Food items", "Others"])
Marital_Status = st.selectbox("Marital Status", ["Married", "Unmarried"])
Permanent_Resident = st.selectbox("Permanent Resident", ["Inside Dhaka", "Outside Dhaka"])
Other_Profession = st.selectbox("Other Profession", ["YES", "NO"])
Business_Nature = st.selectbox("Business Nature", ["Seasonal Product", "Same Product line"])
Business_Duration_With_RedX = st.number_input("Business Duration with RedX\n(Months)", min_value=6, max_value=50, value=10)

# --- Prediction ---
if st.button("Predict Risk"):
    # Convert Yes/No to 0/1 for numeric features
    trade_license_val = 0 if Trade_License == "Yes" else 1
    availability_office_val = 0 if Availability_of_separate_office == "Yes" else 1
    other_profession_val = 0 if Other_Profession == "YES" else 1

    # Create input DataFrame
    input_df = pd.DataFrame({
        "Trade License": [trade_license_val],
        "Business Duration\n(Years)": [Business_Duration_Years],
        "Availability of separate office": [availability_office_val],
        "Number of Business Unit": [Number_of_Business_Unit],
        "Number of Employee": [Number_of_Employee],
        "Ownership type": [encoders["Ownership_type"].transform([Ownership_type])[0]],
        "Sex": [encoders["Sex"].transform([Sex])[0]],
        "Sourcing Type": [encoders["Sourcing_Type"].transform([Sourcing_Type])[0]],
        "Promotion Channel": [encoders["Promotion_Channel"].transform([Promotion_Channel])[0]],
        "Business Type": [encoders["Business_Type"].transform([Business_Type])[0]],
        "Product Type": [encoders["Product_Type"].transform([Product_Type])[0]],
        "Marital Status": [encoders["Marital_Status"].transform([Marital_Status])[0]],
        "Permanent Resident": [encoders["Permanent_Resident"].transform([Permanent_Resident])[0]],
        "Other profession": [other_profession_val],
        "Business Nature": [encoders["Business_Nature"].transform([Business_Nature])[0]],
        "Business Duration with RedX\n(Months)": [Business_Duration_With_RedX]
    })

    # Reorder columns exactly as the model expects
    input_df = input_df[model.feature_names_in_]

    # Make prediction
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success("The predicted credit risk is: **Good**")
    else:
        st.error("The predicted credit risk is: **BAD**")
