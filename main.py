import streamlit as st
import numpy as np
import pandas as pd

from prediction_helper import predict

# --- Streamlit Page Config ---
st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

# --- Light Background and Reduced Padding (including sidebar) ---
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #becff6;
    }
    /* Page padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #becff6;
    }
    /* Top navigation area fix */
    header[data-testid="stHeader"] {
        background-color: #becff6;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# --- Title ---
st.title("🧮 Credit Risk Prediction App")

# Create two main columns: Form on left, Results on right
form_col, result_col = st.columns([3, 2])

# ------------- Input Form Section -------------
with form_col:
    with st.form(key="prediction_form"):
        spacer1, col1, col2, col3, spacer2 = st.columns([1, 2, 2, 2, 1])

        with col1:
            number_of_open_accounts = st.number_input('Number of Open Accounts', min_value=1, max_value=4, value=2)
            sanction_amount = st.number_input('Sanction Amount', min_value=0, max_value=50000000, value=500000)
            income = st.number_input('Income', min_value=0, max_value=12000000, value=800000)
            total_loan_months = st.number_input('Total Loan Months', min_value=1, max_value=240, value=60)
            residence_type = st.selectbox('Residence Type', ['Owned', 'Mortgage', 'Rented'])

        with col2:
            credit_utilization_ratio = st.number_input('Credit Utilization Ratio (%)', min_value=0, max_value=99, value=30)
            loan_tenure_months = st.number_input('Loan Tenure (Months)', min_value=6, max_value=60, value=12)
            delinquent_months = st.number_input('Delinquent Months', min_value=0, max_value=24, value=0)
            total_dpd = st.number_input('Total DPD', min_value=0, max_value=200, value=0)
            loan_purpose = st.selectbox('Loan Purpose', ['Auto', 'Home', 'Personal', 'Education'])

        with col3:
            age = st.number_input('Age', min_value=18, max_value=70, value=30)
            loan_amount = st.number_input('Loan Amount', min_value=0, max_value=50000000, value=400000)
            loan_type = st.selectbox('Loan Type', ['Secured', 'Unsecured'])

        # Predict Button
        center_col1, center_col2, center_col3 = st.columns([2, 1, 2])
        with center_col2:
            submit_button = st.form_submit_button(label='Predict', use_container_width=True)

# ------------- Prediction Result Section -------------
with result_col:
    st.subheader("📈 Prediction Results")
    if submit_button:
        
        # Collect inputs
        input_dict = {
            'number_of_open_accounts' : number_of_open_accounts,
            'credit_utilization_ratio' : credit_utilization_ratio,
            'age' : age,
            'sanction_amount' : sanction_amount,
            'loan_tenure_months' : loan_tenure_months,
            'loan_amount' : loan_amount,
            'income' : income,
            'delinquent_months' : delinquent_months,
            'total_loan_months' : total_loan_months,
            'total_dpd' : total_dpd,
            'residence_type' : residence_type, 
            'loan_purpose' : loan_purpose,
            'loan_type' : loan_type
        }


        # Predict
        prob_default, credit_score, rating = predict(input_dict)

        # Results
        st.info(f"**Default Probability / Risk Probability:** {round(prob_default * 100, 2)}%")
        st.info(f"**Credit Score:** {credit_score}")
        st.info(f"**Rating:** {rating}")
    else:
        st.info("🖐️ Fill the form and click Predict to see the results.")

