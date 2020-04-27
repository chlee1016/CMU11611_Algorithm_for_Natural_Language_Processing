# $ sh hw2.sh ../dev_text.txt ../dev_label.txt ../heldout_text.txt ../heldout_pred_nb.txt
import numpy as np
import time
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D

vocab_size = 10000
embedding_vector_length = 100
MAX_SEQUENCE_LENGTH = 1689

def load_data(data_text, data_label=None):
    f_train = open(data_text, encoding='UTF8')
    train_lines = f_train.readlines()
    f_train.close()
    X_train = train_lines

    if data_label is not None:
        f_train_label = open(data_label, encoding='UTF8')
        train_labels = f_train_label.readlines()
        f_train_label.close()
        y_train = label2int(train_labels)
    else:
        y_train = None
    return X_train, y_train


def train_val_split(data, label, portion):
    # I = np.arange(round(len(data) * portion))
    X_train = data[:round(len(data) * portion)]
    y_train = label[:round(len(data) * portion)]
    X_val = data[round(len(data) * portion):]
    y_val = label[round(len(data) * portion):]
    return X_train, y_train, X_val, y_val


def label2int(y_train):
    # replace neg to 0 and pos to 1
    for i in range(len(y_train)):
        if y_train[i] == 'pos\n':
            y_train[i] = 1
        else:
            y_train[i] = 0
    return y_train


def int2label(predicted_list):
    # replace 0 to neg and 1 to pos
    predicted_list = list(map(int, predicted_list))
    for i in range(len(predicted_list)):
        if predicted_list[i] == 1:
            predicted_list[i] = 'pos'
        else:
            predicted_list[i] = 'neg'
    return predicted_list


def cnn():
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=MAX_SEQUENCE_LENGTH, trainable=True))
    model.add(Conv1D(100, 3, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def text2seq(X_train, y_train, X_val, y_val, X_test):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_val = tokenizer.texts_to_sequences(X_val)
    X_test = tokenizer.texts_to_sequences(X_test)

    X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    X_val = pad_sequences(X_val, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    print('X_train', X_train.shape)
    print('X_val', X_val.shape)
    print('X_test', X_test.shape)

    print('y_train', y_train.shape)
    print('y_val', y_val.shape)
    return X_train, y_train, X_val, y_val, X_test


def main():
    print('this is main')
    train_text_path = "../dev_text.txt"
    train_label_path = "../dev_label.txt"
    test_text_path = "../heldout_text.txt"
    test_label_path = "../heldout_pred_nn.txt"

    X_train, y_train = load_data(train_text_path, train_label_path)
    X_test, _ = load_data(test_text_path)
    print('the number of pos class :', sum(y_train))
    print('the number of neg class :', len(y_train) - sum(y_train))

    X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, 0.7)
    X_train, y_train, X_val, y_val, X_test = text2seq(X_train, y_train, X_val, y_val, X_test)

    model = cnn()
    start_time = time.time()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), shuffle=True, epochs=5, batch_size=4)
    print("training time for cnn:", time.time() - start_time)

    scores = model.evaluate(X_val, y_val, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    predicted_list = model.predict_classes(X_test)
    predicted_list = int2label(predicted_list)

    f = open(test_label_path, 'w')
    for i in range(len(predicted_list)):
        f.write(str(np.squeeze(predicted_list[i])) + '\n')
    f.close()


if __name__ == '__main__':
    main()



