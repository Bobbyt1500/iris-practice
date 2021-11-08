import tensorflow as tf
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    # Create neural network model
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu", input_shape=(4,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation="softmax")
    ])



    model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
    )

    model.summary()

    # Get data
    dataset = datasets.load_iris()

    x_train, x_test, y_train, y_test = train_test_split(dataset["data"], dataset["target"], test_size=0.2)

    y_train = tf.one_hot(y_train, 3)
    y_test = tf.one_hot(y_test, 3)

    model.fit(x_train, y_train, epochs=500)

    model.evaluate(x_test, y_test, verbose=2)

    
