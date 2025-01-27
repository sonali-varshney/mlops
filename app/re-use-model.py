import mlflow.sklearn

# Load the model
model_uri = "runs:/<run_id>/linear_regression_model"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Use the model for predictions
new_data = [[1200]]  # Example input
prediction = loaded_model.predict(new_data)
print(f"Predicted Price: {prediction[0]:.2f} Lakhs")
