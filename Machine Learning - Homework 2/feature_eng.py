import os
import json
from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score


SEED = 5

# Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
import nltk


'''
The ItemSelector class was created by Matt Terry to help with using
Feature Unions on Heterogeneous Data Sources
All credit goes to Matt Terry for the ItemSelector class below
For more information:
http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
'''
class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


"""
This is an example of a custom feature transformer. The constructor is used
to store the state (e.g like if you need to store certain words/vocab), the
fit method is used to update the state based on the training data, and the
transform method is used to transform the data into the new feature(s). In
this example, we simply use the length of the movie review as a feature. This
requires no state, so the constructor and fit method do nothing.
"""

class TextLengthTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            features[i, 0] = len(ex)
            i += 1
        return features

# TODO: Add custom feature transformers for the movie review data

class average_word_length(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            # remove grammer strings . : ; , ? ! . - ( ) " '
            ex = ex.replace(".", ""); ex =  ex.replace(",", ""); ex =  ex.replace(":", "")
            ex = ex.replace(";", ""); ex = ex.replace("?", ""); ex = ex.replace("'", "")
            ex = ex.replace("!", ""); ex = ex.replace("(", "")
            ex = ex.replace(")", ""); ex = ex.replace('"', '')
            # split string on space
            ex = ex.split()
            average = sum([len(word) for word in ex])/len(ex)
            features[i, 0] = average
            i += 1

        return features

class past_tense(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            # remove grammer strings . : ; , ? ! . - ( ) " '
            ex = ex.replace(".", ""); ex = ex.replace(",", ""); ex = ex.replace(":", "")
            ex = ex.replace(";", ""); ex = ex.replace("?", ""); ex = ex.replace("'", "")
            ex = ex.replace("!", ""); ex = ex.replace("(", "")
            ex = ex.replace(")", ""); ex = ex.replace('"', '')
            # find number of ed in sentence to guess number of uses of the past tense
            past_tense_use = ex.count('ed '); past_tense_use += ex.count('was')
            features[i, 0] = past_tense_use
            i += 1

        return features

class present_tense(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            # remove grammer strings . : ; , ? ! . - ( ) " '
            ex = ex.replace(".", ""); ex = ex.replace(",", ""); ex = ex.replace(":", "")
            ex = ex.replace(";", ""); ex = ex.replace("?", ""); ex = ex.replace("'", "")
            ex = ex.replace("!", ""); ex = ex.replace("(", "");
            ex = ex.replace(")", ""); ex = ex.replace('"', '')
            # find number of ed in sentence to guess number of uses of the past tense
            present_tense_use = ex.count('ing'); present_tense_use += ex.count('is')

            features[i, 0] = present_tense_use
            i += 1

        return features

class grammer_quant(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            # count grammer strings . : ; , ? ! . - ( ) " '
            grammer = ex.count("."); grammer += ex.count(","); grammer += ex.count(":")
            grammer += ex.count(";"); grammer += ex.count("?"); grammer += ex.count("'")
            grammer += ex.count("!"); grammer += ex.count("(");
            grammer += ex.count(")"); grammer += ex.count('"')

            features[i, 0] = grammer
            i += 1

        return features

class number_of_sentences(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            # count grammer strings . : ; , ? ! . - ( ) " '
            sentences = ex.count(". "); sentences += ex.count("? ")
            sentences += ex.count("! ")
            features[i, 0] = sentences
            i += 1

        return features

class impossibilty_verbs(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            # count grammer strings . : ; , ? ! . - ( ) " '
            grammer = ex.count("doesn't"); grammer += ex.count("isn't"); grammer += ex.count("wasn't")
            grammer += ex.count("cant"); grammer += ex.count("wont"); grammer += ex.count("will not")

            features[i, 0] = grammer
            i += 1

        return features

class negative_words(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        # read in txt file of negative words
        file_n = open("negative_words.txt", "r")
        words = [line.split(maxsplit=1)[0] for line in file_n if line.strip()]
        #initialize
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            # count number of instances from each sentance that occur in a given word
            neg = 0
            for word in words:
                neg += ex.count(word)
            features[i, 0] = neg
            i += 1

        return features

class posative_words(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        # read in txt file of negative words
        file_n = open("posative_words.txt", "r")
        words = [line.split(maxsplit=1)[0] for line in file_n if line.strip()]
        #initialize
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            # count number of instances from each sentance that occur in a given word
            pos = 0
            for word in words:
                pos += ex.count(word)
            features[i, 0] = pos
            i += 1

        return features

class questions(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            # count questions
            features[i, 0] = ex.count("?")
            i += 1

        return features

class excitement(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            # count questions
            features[i, 0] = ex.count("!")
            i += 1

        return features

class compounded_ideas(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            # count questions
            features[i, 0] = ex.count(",")
            i += 1

        return features


class Featurizer:
    def __init__(self):
        # To add new features, just add a new pipeline to the feature union
        # The ItemSelector is used to select certain pieces of the input data
        # In this case, we are selecting the plaintext of the input data

        # TODO: Add any new feature transformers or other features to the FeatureUnion
        self.all_features = FeatureUnion([
            ('text_stats_1', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('text_length', TextLengthTransformer())
            ])),
            ('text_stats_2', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('past_tense', past_tense())
            ])),
            ('text_stats_3', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('present_tense', present_tense())
            ])),
            ('text_stats_4', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('questions', questions())
            ])),
            ('text_stats_5', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('excitement', excitement())
            ])),
            ('text_stats_6', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('compounded_ideas', compounded_ideas())
            ])),
            # ('text_stats_6', Pipeline([
            #     ('selector', ItemSelector(key='text')),
            #     ('negative_words', negative_words())
            # ])),
            # ('text_stats_7', Pipeline([
            #     ('selector', ItemSelector(key='text')),
            #     ('posative_words', posative_words())
            # ])),
            ('text_stats_8', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('grams', CountVectorizer(ngram_range=(1,7),token_pattern=r'\b\w+\b', min_df=12))
            ]))
        ])

    def train_feature(self, examples):
        return self.all_features.fit_transform(examples)

    def test_feature(self, examples):
        return self.all_features.transform(examples)

if __name__ == "__main__":

    # Read in data

    dataset_x = []
    dataset_y = []

    with open('/Users/wilder/PycharmProjects/Machine_learning_HW2/movie_review_data.json') as f:
        data = json.load(f)
        for d in data['data']:
            dataset_x.append(d['text'])
            dataset_y.append(d['label'])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.3, random_state=SEED)

    feat = Featurizer()

    labels = []
    for l in y_train:
        if not l in labels:
            labels.append(l)

    print("Label set: %s\n" % str(labels))

    # Here we collect the train features
    # The inner dictionary contains certain pieces of the input data that we
    # would like to be able to select with the ItemSelector
    # The text key refers to the plaintext
    feat_train = feat.train_feature({
        'text': [t for t in X_train]
    })
    # Here we collect the test features
    feat_test = feat.test_feature({
        'text': [t for t in X_test]
    })

    #print(feat_train)
    #print(set(y_train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, max_iter=15000, shuffle=True, verbose=2)

    lr.fit(feat_train, y_train)
    y_pred = lr.predict(feat_train)
    accuracy = accuracy_score(y_pred, y_train)
    print("Accuracy on training set =", accuracy)
    y_pred = lr.predict(feat_test)
    accuracy = accuracy_score(y_pred, y_test)
    print("Accuracy on test set =", accuracy)

    # EXTRA CREDIT: Replace the following code with scikit-learn cross validation
    # and determine the best 'alpha' parameter for regularization in the SGDClassifier

