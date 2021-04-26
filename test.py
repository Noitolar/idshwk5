import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# import tensorflow as tf
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
from tensorflow.keras import layers
from GetData import GetRawInfo
from GetData import FigureExtract


if __name__ == "__main__":
    domain_name_train, label_train = GetRawInfo("train.txt", is_train_set=True)
    x_train, y_train = FigureExtract(domain_name_train), label_train
    domain_name_test = GetRawInfo("test.txt", is_train_set=False)
    x_test = FigureExtract(domain_name_test)

    model = keras.Sequential()
    model.add(keras.Input(shape=5))
    model.add(layers.Dense(units=256, activation="relu"))
    model.add(layers.Dense(units=512, activation="relu"))
    model.add(layers.Dense(units=2, activation="softmax"))
    # print(model.summary())

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5)
    # model.save("model.h5")
    prediction = model.predict(x_test)
    result = [""]*len(prediction)
    for index in range(len(prediction)):
        if prediction[index][0] > prediction[index][1]:
            result[index] = domain_name_test[index] + ",notdga"
        else:
            result[index] = domain_name_test[index] + ",dga"

    with open("result.txt", "w") as file:
        file.write("\n".join(result))
