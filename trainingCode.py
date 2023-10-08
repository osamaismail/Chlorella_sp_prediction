import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib


# Load the uploaded CSV file
df = pd.read_csv("chlorella_data.csv")


# Splitting the data into training and testing sets (80% train, 20% test)
X = df["Day"].values.reshape(-1, 1)  # Independent variable (Days)

# For simplicity, let's start with predicting the growth rate of Chlorella sp. after treatment
y = df["Chlorella After"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
lr_model = LinearRegression()

# Train the model on the training data
lr_model.fit(X_train, y_train)

# Predict the growth rates on the testing data
y_pred = lr_model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(mse, r2)

# Dictionary to store model performance
model_performance = {
    "Linear Regression": r2  # We already have the R2 score for linear regression
}

# Polynomial Regression (degree=2)
poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)
model_performance["Polynomial Regression"] = r2_score(y_test, y_pred_poly)

# Decision Tree Regression
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
model_performance["Decision Tree"] = r2_score(y_test, y_pred_dt)

# Random Forest Regression
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
model_performance["Random Forest"] = r2_score(y_test, y_pred_rf)

# Gradient Boosting Regression
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
model_performance["Gradient Boosting"] = r2_score(y_test, y_pred_gb)

# Determine the best model based on R2 score
best_model_name = max(model_performance, key=model_performance.get)
best_model_r2 = model_performance[best_model_name]

print(best_model_name, best_model_r2, model_performance)

# Save the trained Random Forest model to a file
model_path = "random_forest_model.pkl"
joblib.dump(rf_model, model_path)
