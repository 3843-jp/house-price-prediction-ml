
import streamlit as st
import pickle
import numpy as np

# Load model and polynomial transformer
model = pickle.load(open("house_model.pkl", "rb"))
#poly = pickle.load(open("poly.pkl", "rb"))

st.title("🏠 House Price Prediction System")
st.write("Predict house prices using Polynomial Regression Model")

st.subheader("Enter House Details")

bedrooms = st.number_input("Bedrooms", 0, 10, 3)
bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 2.0)
sqft_living = st.number_input("Sqft Living Area", 0, 10000, 1500)
sqft_lot = st.number_input("Sqft Lot Area", 0, 50000, 5000)
floors = st.number_input("Floors", 0.0, 5.0, 1.0)
waterfront = st.selectbox("Waterfront", [0,1])
view = st.slider("View Rating (0-4)", 0, 4, 0)
condition = st.slider("Condition (1-5)", 1, 5, 3)
grade = st.slider("Grade (1-13)", 1, 13, 7)
sqft_above = st.number_input("Sqft Above Ground", 0, 10000, 1200)
sqft_basement = st.number_input("Sqft Basement", 0, 5000, 0)
yr_built = st.number_input("Year Built", 1900, 2024, 2000)

if st.button("Predict Price"):

    input_data = np.array([[bedrooms,bathrooms,sqft_living,sqft_lot,
                            floors,waterfront,view,condition,
                            grade,sqft_above,sqft_basement,yr_built]])

   # input_poly = poly.transform(input_data)

    prediction = model.predict(input_data)

    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")