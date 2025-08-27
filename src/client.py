import requests
import json
import numpy as np
from tensorflow import keras
import random
import os

AUTH_USER = os.getenv("API_USER", "admin")
AUTH_PASS = os.getenv("API_PASS", "model")

MODEL_NAME = "mnist"
HOST_PORT = 8080
url = f"http://localhost:{HOST_PORT}/v1/models/{MODEL_NAME}:predict"

(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test / 255.0

test_index = random.randint(0, len(x_test) - 1)
test_image = x_test[test_index]
true_label = y_test[test_index]
test_image_batch = np.expand_dims(test_image, axis=0)

data = json.dumps({
    "instances": test_image_batch.tolist()
})

print(f"Sending prediction request to secured endpoint: {url}")

try:
    response = requests.post(
        url,
        data=data,
        auth=(AUTH_USER, AUTH_PASS)
    )
    response.raise_for_status()

    predictions = response.json()['predictions'][0]
    predicted_label = np.argmax(predictions)

    print(f"\nTrue Label:      {true_label}")
    print(f"Predicted Label: {predicted_label}")

    if predicted_label == true_label:
        print("\nPrediction was CORRECT!")
    else:
        print("\nPrediction was INCORRECT.")

except requests.exceptions.HTTPError as e:
    if e.response.status_code == 401:
        print("\nError: Authentication failed (401 Unauthorized). Check your username and password.")
    else:
        print(f"\nAn HTTP error occurred: {e}")
except requests.exceptions.RequestException as e:
    print(f"\nAn error occurred while sending the request: {e}")