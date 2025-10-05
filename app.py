
import streamlit as st
import pandas as pd
import pickle

import joblib
model = joblib.load("fuel_model.pkl")

# -------------------------------
# Load trained pipeline model
# -------------------------------
with open("fuel_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Fuel Efficiency Predictor", layout="centered")

# Header Section
st.markdown(
    """
    <div style="text-align:center; padding:15px; background-color:#f5f7fa; border-radius:12px;">
        <h1 style="color:#2c3e50;">🚗 Fuel Efficiency Prediction</h1>
        <p style="font-size:18px; color:#34495e;">
            Enter your car details below and find out its estimated mileage (km/l).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# -------------------------------
# Dropdowns for categorical features
# -------------------------------
st.subheader("🔧 Car Details")

car_name = st.selectbox("Car Name", [
    'Maruti Alto','Hyundai Grand','Hyundai i20','Ford Ecosport','Maruti Wagon R',
    'Hyundai i10','Hyundai Venue','Maruti Swift','Hyundai Verna','Renault Duster',
    'Mini Cooper','Maruti Ciaz','Mercedes-Benz C-Class','Toyota Innova','Maruti Baleno'
])

brand = st.selectbox("Brand", [
    'Maruti','Hyundai','Ford','Renault','Mini','Mercedes-Benz','Toyota','Volkswagen',
    'Honda','Mahindra','Tata','Kia','BMW','Audi','Land Rover','Jaguar','MG','Isuzu',
    'Porsche','Skoda','Volvo','Lexus','Jeep','Maserati','Bentley','Nissan','ISUZU',
    'Ferrari','Mercedes-AMG','Rolls-Royce','Force'
])

model_name = st.selectbox("Model", [
    'Alto','Grand','i20','Ecosport','Wagon R','i10','Venue','Swift','Verna','Duster',
    'Cooper','Ciaz','C-Class','Innova','Baleno','Swift Dzire','Vento','Creta','City',
    'Bolero','Fortuner','KWID','Amaze','Santro','XUV500','KUV100','Ignis','RediGO',
    'Scorpio','Marazzo','Aspire','Figo','Vitara','Tiago','Polo','Seltos','Celerio',
    'GO','5','CR-V','Endeavour','KUV','Jazz','3','A4','Tigor','Ertiga','Safari','Thar',
    'Hexa','Rover','Eeco','A6','E-Class','Q7','Z4','6','XF','X5','Hector','Civic','D-Max',
    'Cayenne','X1','Rapid','Freestyle','Superb','Nexon','XUV300','Dzire VXI','S90','WR-V',
    'XL6','Triber','ES','Wrangler','Camry','Elantra','Yaris','GL-Class','7','S-Presso',
    'Dzire LXI','Aura','XC','Ghibli','Continental','CR','Kicks','S-Class','Tucson','Harrier',
    'X3','Octavia','Compass','CLS','redi-GO','Glanza','Macan','X4','Dzire ZXI','XC90',
    'F-PACE','A8','MUX','GTC4Lusso','GLS','X-Trail','XE','XC60','Panamera','Alturas',
    'Altroz','NX','Carnival','C','RX','Ghost','Quattroporte','Gurkha'
])

seller_type = st.radio("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
fuel_type = st.radio("Fuel Type", ['Petrol','Diesel','CNG','LPG','Electric'])
transmission_type = st.radio("Transmission Type", ['Manual','Automatic'])

st.markdown("---")

# -------------------------------
# Numeric features
# -------------------------------
st.subheader("⚙️ Technical Specifications")

col1, col2 = st.columns(2)

with col1:
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=50, value=5, step=1)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000)
    engine = st.number_input("Engine (CC)", min_value=500, max_value=5000, value=1200, step=100)

with col2:
    max_power = st.number_input("Max Power (bhp)", min_value=20.0, max_value=500.0, value=80.0, step=1.0)
    seats = st.number_input("Seats", min_value=2, max_value=10, value=5, step=1)
    selling_price = st.number_input("Selling Price (₹)", min_value=10000, max_value=10000000, value=500000, step=10000)

# -------------------------------
# Prepare input dataframe
# -------------------------------
input_data = pd.DataFrame({
    "car_name": [car_name],
    "brand": [brand],
    "model": [model_name],
    "vehicle_age": [vehicle_age],
    "km_driven": [km_driven],
    "seller_type": [seller_type],
    "fuel_type": [fuel_type],
    "transmission_type": [transmission_type],
    "engine": [engine],
    "max_power": [max_power],
    "seats": [seats],
    "selling_price": [selling_price]
})

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Mileage"):
    prediction = model.predict(input_data)[0]

    st.markdown("### 🎯 Prediction Result")
    st.success(f"Estimated Fuel Efficiency: **{prediction:.2f} km/l**")

    # Balloons effect 🎈
    st.balloons()
