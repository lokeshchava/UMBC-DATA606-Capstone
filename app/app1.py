import streamlit as st
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
    
    # Add input fields for each feature
    features = {}
    for column in ["Fault","BasePolicy","AgeOfPolicyHolder","AgeOfVehicle","AddressChange-Claim","VehicleCategory"]:
        features[column] = st.text_input(column)


    if st.button("Predict"):
        # Perform prediction
        input_features = list(features.values())
        prediction = predict_insurance(input_features)
        st.write(f"The predicted class is: {prediction}")

if __name__ == "__main__":
    main()
