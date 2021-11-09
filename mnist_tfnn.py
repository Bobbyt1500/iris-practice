import tensorflow as tf

from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(8, (3, 3), activation="relu", input_shape=(28,28, 1)),
        tf.keras.layers.SpatialDropout2D(0.2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(64),

        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
    )

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test, verbose=2)