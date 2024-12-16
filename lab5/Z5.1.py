from keras import Sequential
from keras.datasets.fashion_mnist import load_data
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier

NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
IMG_SIZE = 28


def load_train_data():
    (X_train, y_train), (X_test, y_test) = load_data()

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)).astype(float) / 255.0
    X_test = X_test.reshape((X_test.shape[0], X_train.shape[1], X_train.shape[2], 1)).astype(float) / 255.0

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test


def evaluate_models(base_model, data):
    X_train, y_train, X_test, y_test = data
    cnn_res = base_model.evaluate(X_test, y_test)

    encoder = Sequential(base_model.layers[:-2])
    X_encoded_train = encoder.predict(X_train, batch_size=128)
    X_encoded_test = encoder.predict(X_test, batch_size=128)

    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X_encoded_train, y_train)
    clf_res = clf.score(X_encoded_test, y_test)

    print(f'Random forest classifier with cnn accuracy: {clf_res}')
    print(f'CNN accuracy: {cnn_res[1]}; loss: {cnn_res[0]}')


if __name__ == "__main__":
    feature_extractor = load_model('fashion_mnist_model.keras')
    data = load_train_data()
    evaluate_models(feature_extractor, data)
