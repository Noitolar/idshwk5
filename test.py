import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# import tensorflow as tf
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
from tensorflow.keras import layers
from numpy import log2
from numpy import max
from numpy import min
from numpy import array


# Figure 01: root_domain_name
# Figure 02: length
# Figure 03: entropy
# Figure 04: vowel_ratio
# Figure 05: number_ratio


def SplitByComma(string):
    return string.split(",")


def RootDomainName(string):
    return string.split(".")[-1]


def IsDga(string):
    if string == "notdga":
        return 0
    elif string == "dga":
        return 1


def Normalize(data):
    data = array(data)
    norm = (data - min(data)) / (max(data) - min(data))
    return list(norm)


def GetRawInfo(file_path, is_train_set):
    with open(file_path, "r") as file:
        lines = file.read().splitlines()
        if is_train_set:
            (domain_name, label) = tuple(zip(*map(SplitByComma, lines)))
            label = tuple(map(IsDga, label))
            return domain_name, label
        else:
            return lines


def RootDomainNameClassify(domain_name_list):
    root_domain_name_set = set(map(RootDomainName, domain_name_list))
    root_dict = dict(zip(root_domain_name_set, range(len(root_domain_name_set))))
    return root_dict


def InfoEntropy(domain_name):
    pure_text = domain_name.rstrip(".")
    character_set = set((pure_text[index] for index in range(len(pure_text))))
    probability_list = (float(pure_text.count(character) / len(pure_text)) for character in character_set)
    info_entropy = 0
    for probability in probability_list:
        info_entropy -= probability * log2(probability)
    return info_entropy


def VowelRatio(domain_name):
    main_name = domain_name.strip(RootDomainName(domain_name))
    vowel_list = ("a", "e", "i", "o", "u")
    vowel_count = 0
    for vowel in vowel_list:
        vowel_count += main_name.count(vowel)
    vowel_ratio = vowel_count / len(main_name)
    return vowel_ratio


def NumberRatio(domain_name):
    main_name = domain_name.strip(RootDomainName(domain_name))
    number_list = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    number_count = 0
    for number in number_list:
        number_count += main_name.count(number)
    number_ratio = number_count / len(main_name)
    return number_ratio


def FigureExtract(domain_name_list):
    figure_root_domain_name = []
    figure_length = []
    figure_info_entropy = []
    figure_vowel_ratio = []
    figure_number_ratio = []
    root_dict = RootDomainNameClassify(domain_name_list)
    for domain_name in domain_name_list:
        figure_root_domain_name.append(root_dict[RootDomainName(domain_name)])
        figure_length.append(len(domain_name))
        figure_info_entropy.append(InfoEntropy(domain_name))
        figure_vowel_ratio.append(VowelRatio(domain_name))
        figure_number_ratio.append(NumberRatio(domain_name))
    figure_root_domain_name = Normalize(figure_root_domain_name)
    figure_length = Normalize(figure_length)
    figure_info_entropy = Normalize(figure_info_entropy)
    figure_vowel_ratio = Normalize(figure_vowel_ratio)
    figure_number_ratio = Normalize(figure_number_ratio)
    figure = list(
        zip(figure_root_domain_name, figure_length, figure_info_entropy, figure_vowel_ratio, figure_number_ratio))
    return figure


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
    result = [""] * len(prediction)
    for index in range(len(prediction)):
        if prediction[index][0] > prediction[index][1]:
            result[index] = domain_name_test[index] + ",notdga"
        else:
            result[index] = domain_name_test[index] + ",dga"

    with open("result.txt", "w") as file:
        file.write("\n".join(result))
