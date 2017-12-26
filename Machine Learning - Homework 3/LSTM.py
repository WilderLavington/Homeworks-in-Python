import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Conv2D, MaxPooling2D
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence
from keras.layers import Flatten

class RNN:
    '''
    RNN classifier
    '''
    def __init__(self, train_x, train_y, test_x, test_y, dict_size=20000, example_length=500, embedding_length=128, epoches=15, batch_size=128):
        '''
        initialize RNN model
        :param train_x: training data
        :param train_y: training label
        :param test_x: test data
        :param test_y: test label
        :param epoches:
        :param batch_size:
        '''
        self.batch_size = batch_size
        self.epoches = epoches
        self.example_len = example_length
        self.dict_size = dict_size
        self.embedding_len = embedding_length

        # TODO:preprocess training data
        train_x = sequence.pad_sequences(train_x, maxlen=example_length)
        test_x = sequence.pad_sequences(test_x, maxlen=example_length)
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y

        # TODO:build model
        self.model = Sequential()
        self.model.add(Embedding(dict_size, 128))
        self.model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3))
        self.model.add(Dense(1, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        self.model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    def train(self):
        '''
        fit in data and train model
        :return:
        '''
        # TODO: fit in data to train your model
        self.model.fit(self.train_x, self.train_y, epochs=self.epoches, batch_size = self.batch_size)

    def evaluate(self):
        '''
        evaluate trained model
        :return:
        '''
        return self.model.evaluate(self.test_x, self.test_y)


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=5000)
    rnn = RNN(train_x, train_y, test_x, test_y)
    rnn.train()
    print(rnn.evaluate())

