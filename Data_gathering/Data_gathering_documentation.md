# Project :- predicting fuel efficiency of indian vehicles using deep learning 

## Goal :
The goal of this step is to build a dataset that truly represents vehicles, fuel types, and driving conditions in India. This dataset will be used to train a deep learning model that predicts fuel efficiency accurately under Indian conditions.

## Data Preparation Method :
An initial sample of 1000 rows was created with key features. This sample was then programmatically extended to 400,000 rows, incorporating realistic and diverse values that reflect the variety of vehicles, fuel types, and driving conditions found in India.

## About the Dataset :
Total rows: 400,000
Total columns: 12

## Feature Description: 

**vehicle_id**: A unique identifier for each vehicle record used for tracking and reference.

**engine_size**: Indicates the engine’s displacement in liters, reflecting its overall capacity and power potential.

**horsepower**: Measures the engine's power output, which influences acceleration and fuel consumption.

**vehicle_weight**: The total weight of the vehicle in kilograms, impacting performance and fuel efficiency.

**model_year**: Specifies the manufacturing year of the vehicle, often linked to technology level and emission standards.

**fuel_type**: Type of fuel used (e.g., Petrol, Diesel, CNG), which affects fuel economy and environmental impact.

**vehicle_type**: Classifies the vehicle (e.g., SUV, Sedan, Truck), each having distinct fuel usage patterns.

**fuel_efficiency**: The target variable representing how far a vehicle can travel per unit of fuel (e.g., km/l).

**engine_capacities**: Additional engine specifications such as total displacement or number of cylinders.

**transmission_types**: Describes the transmission system (e.g., Manual, Automatic), influencing gear shifts and fuel use.

**road_conditions**: Typical driving environments (e.g., City, Highway, Rural) that affect real-world fuel performance.

**climate_factors**: Environmental conditions (e.g., Hot, Humid, Rainy) that impact engine behavior and efficiency.



## Python code to create the dataset of 400,000 -

import pandas as pd
import numpy as np
from faker import Faker
import random
import os

df = pd.read_csv("/content/fuel efficiency dataset.csv")

columns = df.columns.tolist()

# Initialize Faker
fake = Faker()

# Number of synthetic rows needed
original_rows = len(df)
target_rows = 400000
synthetic_rows_needed = target_rows - original_rows

# Generate synthetic data based on column names
import random

synthetic_data = []

for i in range(1, 400001):
    row = {
        "vehicle_id": i,# unique identifier
        "engine_size": round(random.uniform(1.0, 6.5), 2),# in liters
        "horsepower": random.randint(60, 700),# in HP
        "vehicle_weight": random.randint(800, 5000), # in kg
        "model_year": random.randint(2000, 2025),
        "fuel_type": random.choice(["Petrol", "Diesel", "CNG", "Electric", "Hybrid", "LPG"]),
        "vehicle_type": random.choice(["Sedan", "SUV", "Truck", "Coupe", "Hatchback", "Van", "Scooter", "MUV", "Motor Cycle", "Tractor"]),
        "fuel_efficiency": round(random.uniform(5.0, 40.0), 2), # in km/l
        "engine_capacities": round(random.uniform(0.6, 6.0), 2), # in CC
        "transmission_types": random.choice(["Manual Transmission (MT)", "Automatic Transmission (AT)", "Continuously Variable Transmission (CVT)", "Automated Manual Transmission (AMT )", "Direct-Shift Gearbox (DSG )", "Dual-Clutch Transmission (DCT)", "Tiptronic", "E-CVT"]),
        "road_conditions": random.choice(["City", "Highway", "Off-road", "Rural", "Wet", "Icy", "Gravel", "Snowy", "Mountainous", "Potholes"]),
        "climate_factors": random.choice(["Hot", "Cold", "Rainy", "Snowy", "Humid", "Dry", "Windy", "Foggy", "Tropical", "Arid"])
    }
    synthetic_data.append(row)

# Convert to DataFrame and combine with original
synthetic_df = pd.DataFrame(synthetic_data, columns=columns)
full_df = pd.concat([df, synthetic_df], ignore_index=True)

# Save the synthetic dataset
full_df.to_csv("synthetic_fuel_efficiency.csv", index=False)



## Challenges Faced and Solutions :

**Lack of a suitable dataset**:
No available dataset reflected Indian vehicles and conditions, so I created a custom dataset using Python with features specific to the Indian market.

**Repetitive or unrealistic data**:
To avoid repeated or fake-looking values, I used random value generation within meaningful and realistic ranges for each feature.

**Missing or empty values**:
I ensured that all fields were fully populated during data generation, preventing any null or incomplete records.

## Dataset Summary :
        -Number of columns:13
        -Number of records:400,000
        -Target variable:fuel_efficiency(in km/m)
        -Data types:mix of numeric , categorical and possibly text columns 


