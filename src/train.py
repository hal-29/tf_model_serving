import tensorflow as tf
from tensorflow import keras

def train_and_evaluate():
    print("Starting model training and evaluation...")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=2)

    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Model evaluation complete. Test Accuracy: {accuracy:.4f}")

    return model, accuracy