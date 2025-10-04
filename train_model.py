# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("cleaned_fuel_efficiency.csv")

# Prepare features and target
X = df.drop(columns=["Unnamed: 0", "mileage", "selling_price"])  # drop target + irrelevant
y = df["mileage"]

# Identify categorical and numeric features
categorical_cols = ["car_name", "brand", "model", "seller_type", "fuel_type", "transmission_type"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing + pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf_model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import r2_score, mean_squared_error
y_pred = pipeline.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Save trained pipeline
joblib.dump(pipeline, "fuel_model.pkl")
print("Model saved as fuel_model.pkl")

