import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

MODEL_DIR = os.getenv("MODEL_BASE_PATH", "/app/models")
MODEL_NAME = os.getenv("MODEL_NAME", "mnist")

def create_and_save_model():
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)

    model = keras.Sequential(
        [
            keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=1, batch_size=128)

    base_path = os.path.join(MODEL_DIR, MODEL_NAME, "1")
    os.makedirs(base_path, exist_ok=True)

    tf.saved_model.save(model, base_path)
    print(f"Initial model saved at {base_path}")

if __name__ == "__main__":
    create_and_save_model()
