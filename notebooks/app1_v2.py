import streamlit as st
import numpy as np
from catboost import CatBoostClassifier
import joblib

# Load the trained CatBoost model
model = CatBoostClassifier()
model.load_model("catboost_model_new.cbm")

def predict_insurance(features):
    # Perform prediction
    prediction = model.predict([features])[0]
    return prediction

def main():
    st.title("Insurance Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Insurance Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Dropdown lists for input features
    fault_options = ['Policy Holder','Third Party']
    base_policy_options = ['Liability','Collision','All Perils']
    age_policy_holder_options = ['26 to 30','31 to 35','41 to 50','51 to 65','21 to 25','36 to 40','16 to 17','over 65','18 to 20']
    age_vehicle_options = ['3 years','6 years','7 years','more than 7','5 years','new','4 years','2 years']
    address_change_options = ['1 year','no change','4 to 8 years','2 to 3 years','under 6 months']
    vehicle_category_options = ['Sport','Utility','Sedan']

    fault = st.selectbox('Fault', fault_options)
    base_policy = st.selectbox('Base Policy', base_policy_options)
    age_policy_holder = st.selectbox('Age of Policy Holder', age_policy_holder_options)
    age_vehicle = st.selectbox('Age of Vehicle', age_vehicle_options)
    address_change = st.selectbox('Address Change-Claim', address_change_options)
    vehicle_category = st.selectbox('Vehicle Category', vehicle_category_options)


    if st.button("Predict"):
        # Convert selected dropdown values to numerical representations
        features = {
            "Fault": fault_options.index(fault),
            "BasePolicy": base_policy_options.index(base_policy),
            "AgeOfPolicyHolder": age_policy_holder_options.index(age_policy_holder),
            "AgeOfVehicle": age_vehicle_options.index(age_vehicle),
            "AddressChange-Claim": address_change_options.index(address_change),
            "VehicleCategory": vehicle_category_options.index(vehicle_category)
        }
        # Perform prediction
        prediction = predict_insurance(list(features.values()))
        st.write(f"The predicted class is: {prediction}")

if __name__ == "__main__":
    main()
