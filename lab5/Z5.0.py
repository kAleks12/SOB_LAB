import os

from keras.datasets.fashion_mnist import load_data
from keras.src.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras import Sequential, layers
from matplotlib import pyplot as plt
from keras.models import load_model

NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
EPOCHS = 25
BATCH_SIZE = 128


def load_train_data():
    (X_train, y_train), (X_test, y_test) = load_data()

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)).astype(float) / 255.0
    X_test = X_test.reshape((X_test.shape[0], X_train.shape[1], X_train.shape[2], 1)).astype(float) / 255.0

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test


def compile_and_fit(data):
    X_train, y_train, X_test, y_test = data
    model = Sequential(
        [
            layers.Input(shape=INPUT_SHAPE),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dense(NUM_CLASSES, activation="softmax")
        ]
    )

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        batch_size=BATCH_SIZE,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2)]
    )
    model.save('fashion_mnist_model.keras')

    print(model.summary())
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))

    ax[0].set_title('Accuracy')
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('accuracy')
    ax[0].legend(['train', 'validation'], loc='upper left')

    ax[1].set_title('Loss')
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['train', 'validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('Z5.png')
    plt.show()

    return model


def evaluate_model(data, train_model):
    if train_model is True:
        model = compile_and_fit(data)
    else:
        model = load_model('fashion_mnist_model.keras')

    result = model.evaluate(data[-2], data[-1])
    print('Test loss:', result[0])
    print('Test accuracy:', result[1])


if __name__ == "__main__":
    train_data = load_train_data()
    evaluate_model(train_data, train_model=True)
