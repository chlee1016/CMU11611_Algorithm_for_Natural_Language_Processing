import numpy as np
import sys
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


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



class NaiveBayesClassifier(object):
    def __init__(self):
        self.defaultProb = 0.0001
        self.prior_classes = [0,1]
        self.likelihood = None
        self.vectorizer = CountVectorizer(max_features=10000)

        self.prior = np.zeros(len(self.prior_classes))
        self.likelihood = np.zeros((10000, len(self.prior_classes)))

    def getPrior(self, y_train):
        for i in range(len(self.prior_classes)):
            self.prior[i] = (sum(y_train == self.prior_classes[i]) / len(y_train)).astype('float64')
        return self.prior

    def getLikelihood(self, X_train, y_train):

        for i in range(len(self.prior_classes)):
            self.likelihood[:, i] = (np.sum((X_train[y_train == self.prior_classes[i], :]).toarray(), axis=0)/10000.0).astype('float64')
        return self.likelihood

    def fit(self, X_train, y_train):
        self.vectorizer.fit(X_train)
        X_train = self.vectorizer.transform(X_train)
        y_train = np.asarray(y_train)

        self.prior_classes = np.unique(np.asarray(y_train))
        self.likelihood = self.getLikelihood(X_train, y_train)
        self.prior = self.getPrior(y_train)

    def predict(self, X_test):
        predicted_list = []

        for i in range(len(X_test)):
            predicted_prob = np.ones(len(self.prior_classes))

            for _, word in enumerate(X_test[i].split()):
                if word in self.vectorizer.vocabulary_.keys():
                    idx = self.vectorizer.vocabulary_[word]

                    for j in range(len(self.prior_classes)):
                        predicted_prob[j] = predicted_prob[j] * self.likelihood[idx, j]

                else:
                    # print('OOV')
                    for j in range(len(self.prior_classes)):
                        predicted_prob[j] = predicted_prob[j] * 1

            predicted = np.argmax([predicted_prob[0] * self.prior[0], predicted_prob[1] * self.prior[1]])
            # print('self.predicted_prob', self.predicted_prob)
            predicted_list.append(predicted)
        return predicted_list

    def evaluate(self, predicted_list, label):
        correct = 0
        for i in range(len(predicted_list)):
            if predicted_list[i] == label[i]:
                correct += 1
        accuracy = correct / len(predicted_list)
        return accuracy


def main():
    print("run naivebayes.py")

    # train_text_path = "../dev_text.txt"
    # train_label_path = "../dev_label.txt"
    # test_text_path = "../heldout_text.txt"
    # test_label_path = "../heldout_pred_nb.txt"

    train_text_path = sys.argv[1]
    train_label_path = sys.argv[2]
    test_text_path = sys.argv[3]
    test_label_path = sys.argv[4]

    X_train, y_train = load_data(train_text_path, train_label_path)

    X_test, _ = load_data(test_text_path)
    print('the number of pos class :', sum(y_train))
    print('the number of neg class :', len(y_train) - sum(y_train))

    X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, 0.7)

    NB = NaiveBayesClassifier()
    print('training is started')
    NB.fit(X_train, y_train)
    print('training is finished')

    predicted_list = NB.predict(X_val)

    f = open(test_label_path, 'w')
    for i in range(len(predicted_list)):
        f.write(str(predicted_list[i])+'\n')
    f.close()

    accuracy = NB.evaluate(predicted_list, y_val)
    print('Accuracy: ', accuracy*100, '%')


if __name__ == '__main__':
    main()





# def preprocessing(sentence, remove_stopwords = False):
#     review_text = re.sub("[^a-zA-Z]", " ", sentence)
#     words = review_text.lower().split()
#     if remove_stopwords:
#         stops = set(stopwords.words("english"))
#         words = [w for w in words if not w in stops]
#         clean_review = ' '.join(words)
#     else:
#         clean_review = ' '.join(words)
#     return clean_review

























# tokenizer = Tokenizer(num_words=10000)
# tokenizer.fit_on_texts(train_lines)
# train_text_sequences = tokenizer.texts_to_sequences(train_lines)
# print('len of train_text_sequences', len(train_text_sequences))
# print('train_text_sequences[0] : ', train_text_sequences[0])
#
#
# # print('tokenizer.word_index', tokenizer.word_index)
# print('len(tokenizer.word_index)', len(tokenizer.word_index))
# # print('tokenizer.word_index.values()', tokenizer.word_index.values())
# print('zip(tokenizer.word_index.keys(), tokenizer.word_index.values())', zip(list(tokenizer.word_index.keys()), list(tokenizer.word_index.values())))
#
# print(len(tokenizer.word_index.keys()))
# print(len(tokenizer.word_index.values()))
# # print(list(zip(list(tokenizer.word_index.keys()), list(tokenizer.word_index.values()))))
#
# bow = list(zip(list(tokenizer.word_index.keys()), list(tokenizer.word_index.values())))
# print(len(bow[:10000]))
# print(bow[0][0])
# print(bow[0][1])
#
#
# def get_sentence_data(train_path_list, test_path_list):
#     train_sentence_list = get_sentence_list(train_path_list)
#     train_data = pd.DataFrame({'sentence' : train_sentence_list, 'label' : [0]*1000 + [1]*1000})
#
#     test_sentence_list = get_sentence_list(test_path_list)
#     test_data = pd.DataFrame({'sentence' : test_sentence_list, 'label' : [0]*1000 + [1]*1000})
#
#
#     clean_train_sentences = []
#     for sentence in train_data['sentence']:
#         clean_train_sentences.append(preprocessing(sentence, remove_stopwords=True))
#
#     clean_test_sentences = []
#     for sentence in test_data['sentence']:
#         clean_test_sentences.append(preprocessing(sentence, remove_stopwords=True))
#
#     tokenizer = Tokenizer(num_words=10000)
#     tokenizer.fit_on_texts(clean_train_sentences)
#     train_text_sequences = tokenizer.texts_to_sequences(clean_train_sentences)
#     test_text_sequences = tokenizer.texts_to_sequences(clean_test_sentences)
#
#     MAX_SEQUENCE_LENGTH = 3817
#
#     X_train = pad_sequences(train_text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
#     X_test = pad_sequences(test_text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
#
#     # clean_train_df = pd.DataFrame({'sentence': clean_train_sentences, 'label': train_data['label']})
#     # clean_test_df = pd.DataFrame({'sentence': clean_test_sentences, 'label': test_data['label']})
#
#     y_train = np.array(train_data['label'])
#     print('Shape of X_train: ', X_train.shape)
#     print('Shape of y_train: ', y_train.shape)
#     np.save(data_path + 'X_train', X_train)
#     np.save(data_path + 'y_train', y_train)
#
#     y_test = np.array(test_data['label'])
#     print('Shape of X_test: ', X_test.shape)
#     print('Shape of y_test: ', y_test.shape)
#     np.save(data_path + 'X_test', X_test)
#     np.save(data_path + 'y_test', y_test)
#     print('finished saving data')
#     ###################################
#     return tokenizer

