import requests
import json

# Define API endpoint
url = "http://35.193.209.7:5001/invocations"

# Prepare input data
data = {
    "instances": [[1200], [2000]]      # Input data
}

# Send POST request
response = requests.post(url, json=data)

# Print response
print("Predictions:", response.json())
