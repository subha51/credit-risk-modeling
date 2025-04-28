
import pandas as pd
import joblib

model=joblib.load("artifacts/model_data.joblib")

def preprocess_input(input_dict):
    input_df=pd.DataFrame([input_dict])
    input_df['loan_to_income']=input_df['loan_amount']/input_df['income']
    input_df['delinquency_ratio']=input_df['delinquent_months']/input_df['total_loan_months']
    if input_df['delinquent_months'].sum()!=0:
        input_df['avg_dpd_per_delinquency']=input_df['total_dpd']/input_df['delinquent_months']
    else:
        input_df['avg_dpd_per_delinquency']=0
    input_df['credit_utilization_per_LTI']=input_df['credit_utilization_ratio']/input_df['loan_to_income']

    input_df=input_df.drop(columns=['loan_amount','income','delinquent_months','total_loan_months','total_dpd'])

    categorical_cols=['residence_type','loan_purpose','loan_type']
    input_df['residence_type_Owned']=0
    input_df['residence_type_Rented']=0
    input_df['loan_purpose_Education']=0
    input_df['loan_purpose_Home']=0
    input_df['loan_purpose_Personal']=0
    input_df['loan_type_Unsecured']=0

    for key,value in input_dict.items():
        if key in categorical_cols:
            if key=='residence_type':
                if value=='Owned':
                    input_df['residence_type_Owned']=1
                elif value=='Rented':
                    input_df['residence_type_Rented']=1
            elif key=='loan_purpose':
                if value=='Education':
                    input_df['loan_purpose_Education']=1
                elif value=='Home':
                    input_df['loan_purpose_Home']=1
                elif value=='Personal':
                    input_df['loan_purpose_Personal']=1
            elif key=='loan_type':
                if value=='Unsecured':
                    input_df['loan_type_Unsecured']=1


    input_df=input_df.drop(columns=categorical_cols)

    scaler=model['scaler']
    cols_to_scale=model['cols_to_scale']

    temp_df=pd.DataFrame(columns=cols_to_scale)

    for col in temp_df.columns:
        if col in input_df.columns:
            temp_df[col]=input_df[col]
        else:
            temp_df[col]=0


    temp_df[cols_to_scale]=scaler.transform(temp_df[cols_to_scale])

    for col in input_df.columns:
        if col in temp_df.columns:
            input_df[col]=temp_df[col]

    # print(input_df)
    # print(input_df.iloc[0].tolist())

    return input_df

def get_rating(credit_score):
    if credit_score >= 750:
        rating = "Excellent"
    elif credit_score >= 700:
        rating = "Good"
    elif credit_score >= 650:
        rating = "Fair"
    else:
        rating = "Good"
    return rating


def predict(input_dict):
    input_df=preprocess_input(input_dict)
    best_model=model['model']
    # print(best_model)
    # print(input_df)
    probability_of_default=best_model.predict_proba(input_df)[:,1][0]
    probability_of_not_default=1-probability_of_default
    credit_score=300+600*probability_of_not_default
    rating=get_rating(credit_score)
    return probability_of_default,credit_score,rating
