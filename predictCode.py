import joblib

# Load the saved Random Forest model
model_path = "random_forest_model.pkl"
loaded_model = joblib.load(model_path)

# Get user input for the day
day = input("Enter the day for which you want to predict the growth rate of Chlorella sp. after treatment: ")
day = float(day)  # Convert the input to a float

# Predict the growth rate using the loaded model
predicted_growth_rate = loaded_model.predict([[day]])

print(f"The predicted growth rate of Chlorella sp. after treatment for day {day} is: {predicted_growth_rate[0]}")

