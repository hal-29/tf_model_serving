import tensorflow as tf
from tensorflow import keras
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_evaluate():
    logger.info("Starting model training and evaluation...")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
    ]

    history = model.fit(
        x_train, y_train, 
        epochs=10, 
        validation_split=0.2,
        callbacks=callbacks,
        verbose=2
    )

    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"Model evaluation complete. Test Accuracy: {accuracy:.4f}")

    return model, accuracy
